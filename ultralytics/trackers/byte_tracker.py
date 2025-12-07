# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import LOGGER  # 日志工具
from ..utils.ops import xywh2ltwh  # 坐标转换：xywh(中心+宽高) → tlwh(左上角+宽高)
from .basetrack import BaseTrack, TrackState  # 基础跟踪类、跟踪状态枚举
from .utils import matching  # 匹配算法（匈牙利算法、IOU距离计算等）
from .utils.kalman_filter import KalmanFilterXYAH  # 针对XYAH格式的卡尔曼滤波器


class STrack(BaseTrack):
    """Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (Any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.
        angle (float | None): Optional angle information for oriented bounding boxes.

    Methods:
        predict: Predict the next state of the object using Kalman filter.
        multi_predict: Predict the next states for multiple tracks.
        multi_gmc: Update multiple track states using a homography matrix.
        activate: Activate a new tracklet.
        re_activate: Reactivate a previously lost tracklet.
        update: Update the state of a matched track.
        convert_coords: Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah: Convert tlwh bounding box to xyah format.

    Examples:
        Initialize and activate a new track
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """

    """
    单目标跟踪轨迹类（Single Track），基于卡尔曼滤波实现状态估计
    每个STrack实例对应一个被跟踪的目标，存储其坐标、ID、状态、卡尔曼滤波参数等
    """
    # 所有STrack实例共享的卡尔曼滤波器（批量预测时用）
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: list[float], score: float, cls: Any):
        """Initialize a new STrack instance.

        Args:
            xywh (list[float]): Bounding box coordinates and dimensions in the format (x, y, w, h, [a], idx), where (x,
                y) is the center, (w, h) are width and height, [a] is optional aspect ratio, and idx is the id.
            score (float): Confidence score of the detection.
            cls (Any): Class label for the detected object.

        Examples:
            >>> xywh = [100.0, 150.0, 50.0, 75.0, 1]
            >>> score = 0.9
            >>> cls = "person"
            >>> track = STrack(xywh, score, cls)
        """
        super().__init__()  # 调用父类BaseTrack初始化（主要是track_id生成逻辑）
        # xywh+idx or xywha+idx     校验输入长度：必须是5个（xywh+idx）或6个（xywha+idx）值
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        # 转换坐标：xywh(中心) → tlwh(左上角)，并转为float32数组
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None  # 当前轨迹的卡尔曼滤波器实例（激活时赋值）
        self.mean, self.covariance = None, None  # 卡尔曼滤波的均值（状态）和协方差（不确定性）
        self.is_activated = False  # 轨迹是否已激活（初始为False，激活后参与跟踪）

        self.score = score  # 目标置信度
        self.tracklet_len = 0  # 轨迹持续帧数
        self.cls = cls  # 目标类别
        self.idx = xywh[-1]  # 检测框索引
        # 如果输入是6个值，取第5个作为旋转角度（用于旋转框跟踪）
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        """Predict the next state (mean and covariance) of the object using the Kalman filter."""
        """
        用卡尔曼滤波器预测目标下一帧的状态（均值+协方差）
        核心作用：基于历史运动规律，预估当前帧目标位置，减少检测噪声影响
        """
        mean_state = self.mean.copy()   # 复制当前状态均值
        # 如果轨迹不是「跟踪中」状态，将速度项置0（停止运动预测）
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        # 卡尔曼滤波预测：输入当前状态，输出预测状态
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[STrack]):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
        """
        批量预测多个轨迹的下一帧状态（优化效率，避免单轨迹循环预测）
        Args:
            stracks (list[STrack]): 待预测的轨迹列表
        """
        if len(stracks) <= 0:
            return
        # 提取所有轨迹的均值和协方差，组成批量数组
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        # 非跟踪中轨迹，速度项置0
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        # 共享卡尔曼滤波器批量预测
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        # 将预测结果赋值回每个轨迹
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks: list[STrack], H: np.ndarray = np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix for multiple tracks."""
        """
        基于单应性矩阵（H）校正轨迹位置（处理摄像头运动，如平移/旋转）
        Args:
            stracks (list[STrack]): 待校正的轨迹列表
            H (np.ndarray): 单应性矩阵（2x3），默认单位矩阵（无校正）
        """
        if stracks:
            # 提取批量均值和协方差
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            # 分解单应性矩阵：旋转矩阵R + 平移向量t
            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)  # 扩展为8x8矩阵（适配卡尔曼状态维度）
            t = H[:2, 2]

            # 逐轨迹校正：旋转+平移
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)  # 旋转
                mean[:2] += t  # 平移
                cov = R8x8.dot(cov).dot(R8x8.transpose())  # 协方差校正

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        """
        激活新轨迹（首次检测到目标时调用）
        Args:
            kalman_filter (KalmanFilterXYAH): 卡尔曼滤波器实例
            frame_id (int): 当前帧ID
        """
        self.kalman_filter = kalman_filter  # 绑定卡尔曼滤波器
        self.track_id = self.next_id()  # 生成唯一跟踪ID（父类BaseTrack的方法）
        # 初始化卡尔曼滤波的均值和协方差（输入当前检测框坐标）
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0  # 重置轨迹长度
        self.state = TrackState.Tracked  # 标记为「跟踪中」
        if frame_id == 1:
            self.is_activated = True  # 第一帧直接激活
        self.frame_id = frame_id  # 记录当前帧ID
        self.start_frame = frame_id  # 记录轨迹起始帧

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Reactivate a previously lost track using new detection data and update its state and attributes."""
        """
        重新激活丢失的轨迹（当丢失的目标被重新检测到时调用）
        Args:
            new_track (STrack): 新检测到的轨迹（包含最新坐标）
            frame_id (int): 当前帧ID
            new_id (bool): 是否生成新ID（默认False：保留原ID，保证跟踪连续性）
        """
        # 卡尔曼滤波更新：用新检测框修正轨迹状态
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0  # 重置轨迹长度
        self.state = TrackState.Tracked  # 标记为「跟踪中」
        self.is_activated = True  # 激活轨迹
        self.frame_id = frame_id  # 更新帧ID
        if new_id:
            self.track_id = self.next_id()  # 可选：生成新ID
        # 更新置信度、类别、角度、索引等属性
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track: STrack, frame_id: int):
        """Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.

        Examples:
            Update the state of a track with new detection information
            >>> track = STrack([100, 200, 50, 80, 0.9, 1])
            >>> new_track = STrack([105, 205, 55, 85, 0.95, 1])
            >>> track.update(new_track, 2)
        """
        """
        更新匹配成功的轨迹（跟踪中目标，每帧调用）
        Args:
            new_track (STrack): 新检测到的轨迹（包含最新坐标）
            frame_id (int): 当前帧ID
        """
        self.frame_id = frame_id  # 更新帧ID
        self.tracklet_len += 1  # 轨迹长度+1

        new_tlwh = new_track.tlwh  # 新检测框的tlwh坐标
        # 卡尔曼滤波更新：融合新检测框，修正轨迹状态
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked  # 保持「跟踪中」
        self.is_activated = True  # 确保激活状态

        # 更新置信度、类别、角度、索引等属性
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        """
        坐标转换：tlwh(左上角+宽高) → xyah(中心x, 中心y, 宽高比, 高)
        适配卡尔曼滤波器的输入格式
        """
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        """Get the bounding box in top-left-width-height format from the current state estimate."""
        """
        只读属性：获取当前轨迹的tlwh格式坐标（从卡尔曼均值中还原）
        返回：[top_x, top_y, width, height]
        """
        if self.mean is None:
            return self._tlwh.copy()  # 未初始化时返回原始检测框
        ret = self.mean[:4].copy()  # 卡尔曼均值前4位是xyah格式
        ret[2] *= ret[3]  # 宽高比*高 → 宽度
        ret[:2] -= ret[2:] / 2  # 中心坐标(x,y) - (w/2,h/2) → 左上角坐标
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        """
        只读属性：转换tlwh坐标为xyxy格式（min_x, min_y, max_x, max_y）
        用于可视化、IOU计算等场景
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]  # 左上角+宽高 → 右下角坐标
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        """
        静态方法：tlwh → xyah（卡尔曼滤波标准格式）
        xyah: [center_x, center_y, aspect_ratio(w/h), height]
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2  # 左上角 → 中心坐标
        ret[2] /= ret[3]  # 宽度/高度 → 宽高比
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get the current position of the bounding box in (center x, center y, width, height) format."""
        """只读属性：获取xywh格式坐标（center_x, center_y, width, height）"""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2  # 左上角 → 中心坐标
        return ret

    @property
    def xywha(self) -> np.ndarray:
        """Get position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        """
        只读属性：获取xywha格式坐标（center_x, center_y, width, height, angle）
        用于旋转框跟踪，无角度时返回xywh并报警
        """
        if self.angle is None:
            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """Get the current tracking results in the appropriate bounding box format."""
        """
        只读属性：获取当前轨迹的最终输出结果
        返回格式：[坐标(xyxy/xywha), track_id, score, cls, idx]
        """
        # 有角度返回xywha，无角度返回xyxy
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """Return a string representation of the STrack object including start frame, end frame, and track ID."""
        """轨迹的字符串表示（便于调试）：OT_跟踪ID_(起始帧-结束帧)"""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

    This class encapsulates the functionality for initializing, updating, and managing the tracks for detected objects
    in a video sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman
    filtering for predicting the new object locations, and performs data association.

    Attributes:
        tracked_stracks (list[STrack]): List of successfully activated tracks.
        lost_stracks (list[STrack]): List of lost tracks.
        removed_stracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (Namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (KalmanFilterXYAH): Kalman Filter object.

    Methods:
        update: Update object tracker with new detections.
        get_kalmanfilter: Return a Kalman filter object for tracking bounding boxes.
        init_track: Initialize object tracking with detections.
        get_dists: Calculate the distance between tracks and detections.
        multi_predict: Predict the location of tracks.
        reset_id: Reset the ID counter of STrack.
        reset: Reset the tracker by clearing all tracks.
        joint_stracks: Combine two lists of stracks.
        sub_stracks: Filter out the stracks present in the second list from the first list.
        remove_duplicate_stracks: Remove duplicate stracks based on IoU.

    Examples:
        Initialize BYTETracker and update with detection results
        >>> tracker = BYTETracker(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """
    """
    ByteTrack核心跟踪器类：管理所有轨迹的生命周期（跟踪中/丢失/移除）
    核心逻辑：分层匹配（高置信度→低置信度）+ 卡尔曼预测 + 状态管理
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize a BYTETracker instance for object tracking.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video sequence.

        Examples:
            Initialize BYTETracker with command-line arguments and a frame rate of 30
            >>> args = Namespace(track_buffer=30)
            >>> tracker = BYTETracker(args, frame_rate=30)
        """
        """
        初始化ByteTrack跟踪器
        Args:
            args (Namespace): 跟踪参数（如track_high_thresh、track_buffer等）
            frame_rate (int): 视频帧率（用于计算最大丢失帧数）
        """
        self.tracked_stracks = []  # type: list[STrack]     # 跟踪中轨迹列表
        self.lost_stracks = []  # type: list[STrack]        # 丢失轨迹列表（可重新激活）
        self.removed_stracks = []  # type: list[STrack]     # 移除轨迹列表（彻底删除）

        self.frame_id = 0  # 当前处理的帧ID（从0开始）
        self.args = args  # 跟踪参数
        # 最大丢失帧数：track_buffer是秒数，转换为帧数（默认30帧/秒）
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()  # 初始化卡尔曼滤波器
        self.reset_id()  # 重置轨迹ID计数器

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update the tracker with new detections and return the current list of tracked objects."""
        """
        核心更新方法：输入新帧检测结果，更新所有轨迹状态，返回当前跟踪目标
        Args:
            results: 检测结果（包含xywh/xywhr、conf、cls等）
            img (np.ndarray | None): 原始图像（用于GMC校正）
            feats (np.ndarray | None): ReID特征（可选，用于距离计算）
        Returns:
            np.ndarray: 所有激活的跟踪目标结果，格式为N×(坐标+ID+分数+类别+索引)
        """
        """
        核心更新方法：输入新帧检测结果，更新所有轨迹状态，返回当前跟踪目标
        Args:
            results: 检测结果（包含xywh/xywhr、conf、cls等）
            img (np.ndarray | None): 原始图像（用于GMC校正）
            feats (np.ndarray | None): ReID特征（可选，用于距离计算）
        Returns:
            np.ndarray: 所有激活的跟踪目标结果，格式为N×(坐标+ID+分数+类别+索引)
        """
        self.frame_id += 1  # 帧ID自增
        # 初始化临时列表：分类管理不同状态的轨迹
        activated_stracks = []  # 本次激活的轨迹（新轨迹/重新激活）
        refind_stracks = []     # 丢失后重新找到的轨迹
        lost_stracks = []       # 本次丢失的轨迹
        removed_stracks = []    # 本次移除的轨迹

        # ===================== 步骤1：检测结果分层（ByteTrack核心）=====================
        scores = results.conf  # 所有检测框的置信度
        # 高置信度检测框索引（核心匹配用）
        remain_inds = scores >= self.args.track_high_thresh
        # 低置信度检测框索引（补充匹配用：low < score < high）
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh
        inds_second = inds_low & inds_high  # 低置信度索引（介于高低阈值之间）

        # 拆分检测结果：高置信度（主匹配）、低置信度（辅匹配）
        results_second = results[inds_second]  # 低置信度检测结果
        results = results[remain_inds]         # 高置信度检测结果

        # 拆分特征（如果有ReID特征）
        feats_keep = feats_second = img
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]    # 高置信度特征
            feats_second = feats[inds_second]  # 低置信度特征

        # 初始化检测结果为STrack实例（高置信度）
        detections = self.init_track(results, feats_keep)
        # ===================== 步骤2：整理历史轨迹 =====================
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []    # 未确认轨迹（刚初始化，仅1帧）
        tracked_stracks = []  # type: list[STrack] # # 已激活的跟踪中轨迹
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)# 未激活轨迹加入未确认列表
            else:
                tracked_stracks.append(track)# 已激活轨迹加入跟踪列表
        # Step 2: First association, with high score detection boxes         # 合并「跟踪中」和「丢失」轨迹，作为待匹配的历史轨迹池

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF # 卡尔曼滤波批量预测：预估所有历史轨迹在当前帧的位置
        self.multi_predict(strack_pool)

        # ===================== 步骤3：可选GMC校正（处理摄像头运动）=====================
        if hasattr(self, "gmc") and img is not None:
            # use try-except here to bypass errors from gmc module
            try:
                # 计算单应性矩阵（基于检测框和图像）
                warp = self.gmc.apply(img, results.xyxy)
            except Exception:
                # 异常时用单位矩阵（无校正）
                warp = np.eye(2, 3)
            # 校正历史轨迹和未确认轨迹的位置
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        # ===================== 步骤4：第一轮匹配（高置信度检测框）=====================
        # 计算历史轨迹与高置信度检测框的距离（IOU/融合特征）
        dists = self.get_dists(strack_pool, detections)
        # 线性分配（匈牙利算法）：匹配轨迹和检测框，返回匹配对、未匹配轨迹、未匹配检测框
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        # 处理匹配成功的轨迹
        for itracked, idet in matches:
            track = strack_pool[itracked]  # 匹配的历史轨迹
            det = detections[idet]         # 匹配的检测框
            if track.state == TrackState.Tracked:
                # 跟踪中轨迹：更新状态（融合新检测框）
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                # 丢失轨迹：重新激活（保留原ID）
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # ===================== 步骤5：第二轮匹配（低置信度检测框补充）=====================
        # 初始化低置信度检测框为STrack
        # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
        detections_second = self.init_track(results_second, feats_second)
        # 提取第一轮未匹配的「跟踪中」轨迹（仅跟踪中轨迹参与补充匹配）
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # 计算未匹配轨迹与低置信度检测框的IOU距离
        # TODO
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        # 线性分配（阈值0.5，更宽松）
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        # 处理第二轮匹配结果
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 第二轮仍未匹配的轨迹：标记为丢失
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()  # 标记为丢失
                lost_stracks.append(track)

        # ===================== 步骤6：处理未确认轨迹（刚初始化的轨迹）=====================
        # 提取第一轮未匹配的高置信度检测框
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        # 计算未确认轨迹与未匹配检测框的距离
        dists = self.get_dists(unconfirmed, detections)
        # 线性分配（阈值0.7，更严格）
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        # 匹配成功：更新未确认轨迹
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        # 匹配失败：标记为移除
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()  # 标记为移除
            removed_stracks.append(track)

        # ===================== 步骤7：初始化新轨迹 =====================
        # 遍历最终未匹配的检测框：作为新轨迹初始化
        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            # 低于新轨迹阈值：跳过（过滤噪声）
            if track.score < self.args.new_track_thresh:
                continue
            # 激活新轨迹（分配ID、初始化卡尔曼滤波）
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        # Step 5: Update state         # ===================== 步骤8：清理长时间丢失的轨迹 =====================

        for track in self.lost_stracks:
            # 丢失帧数超过阈值：标记为移除
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ===================== 步骤9：更新跟踪器全局状态 =====================
        # 1. 过滤跟踪中轨迹（仅保留状态为Tracked的）
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        # 2. 合并激活轨迹、重新找到的轨迹到跟踪中列表
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        # 3. 更新丢失轨迹列表：移除已找回的、已移除的，添加本次丢失的
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        # 4. 去重：避免同一目标同时出现在跟踪中/丢失列表
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # 5. 更新移除轨迹列表（限制长度，避免内存溢出）
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]  # clip remove stracks to 1000 maximum 最多保留999个

        # ===================== 步骤10：返回最终跟踪结果 =====================
        # 仅返回已激活的跟踪中轨迹，转为float32数组
        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self) -> KalmanFilterXYAH:
        """Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
        """Initialize object tracking with given detections, scores, and class labels using the STrack algorithm."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        return [STrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Calculate the distance between tracks and detections using IoU and optionally fuse scores."""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: list[STrack]):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    @staticmethod
    def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Filter out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]:
        """Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb

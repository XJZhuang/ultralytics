#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：hf_test.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/21 9:35 
@explain : 
'''

import cv2
import numpy as np
import os


def detect_scale_lines_in_roi(roi_image, angle_thresh=5, std_thresh=1.0):
    """
    在液位计ROI图像中检测水平刻度线，并在ROI上绘制结果

    参数：
        roi_image: 输入的液位计ROI图像（BGR格式，已裁剪好的玻璃管区域）
        angle_thresh: 水平直线的角度阈值（度），过滤非水平线
        std_thresh: 刻度线间距的标准差阈值，筛选等间距线

    返回：
        scale_line_relative_coords: 刻度线相对ROI的坐标列表，每个元素为((x1,y1), (x2,y2))
        roi_with_scale: 绘制了刻度线的ROI图像（BGR格式）
    """
    # -------------------------- 1. 图像预处理（基于ROI） --------------------------
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)  # 灰度化
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊去噪
    edges = cv2.Canny(blur, 50, 150)  # Canny边缘检测

    # -------------------------- 2. 霍夫直线检测 --------------------------
    lines = cv2.HoughLinesP(
        edges,
        rho=1,  # 距离分辨率（像素）
        theta=np.pi / 180,  # 角度分辨率（1度步长）
        threshold=50,  # 累加器阈值（控制检测到的直线数量）
        minLineLength=10,  # 最小直线长度（过滤短线段）
        maxLineGap=5  # 最大间隙（合并断开的直线）
    )

    # -------------------------- 3.绘制所有霍夫直线（调试关键！） --------------------------
    roi_with_all_lines = roi_image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 绘制所有检测到的直线（红色，线宽1）
            cv2.line(roi_with_all_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # 保存中间结果：所有霍夫直线
        cv2.imwrite("roi_with_all_hough_lines.jpg", roi_with_all_lines)
        print(f"霍夫直线检测完成，共检测到{len(lines)}条直线，已保存到roi_with_all_hough_lines.jpg")
    else:
        print("霍夫直线检测未找到任何直线！")
        return [], roi_image.copy()

    # if lines is None:
    #     print("警告：未检测到任何直线！")
    #     return [], roi_image.copy()
    #
    # # -------------------------- 3. 筛选水平刻度线（相对ROI坐标） --------------------------
    # horizontal_lines = []
    # for line in lines:
    #     x1_l, y1_l, x2_l, y2_l = line[0]
    #     dx, dy = x2_l - x1_l, y2_l - y1_l
    #
    #     # 计算直线角度（度），过滤非水平线（允许轻微倾斜）
    #     angle = np.degrees(np.arctan2(dy, dx))
    #     if (-angle_thresh <= angle <= angle_thresh) or (180 - angle_thresh <= angle <= 180 + angle_thresh):
    #         y_center = (y1_l + y2_l) / 2  # 水平直线的中心y坐标（相对ROI）
    #         horizontal_lines.append({
    #             "line": line[0],  # 原始直线坐标（相对ROI）
    #             "y_center": y_center,  # 中心y坐标（相对ROI）
    #             "abs_y_center": y_center  # 无需转换，已为ROI相对坐标
    #         })
    #
    # if not horizontal_lines:
    #     print("警告：未检测到水平直线！")
    #     return [], roi_image.copy()
    #
    # # 按y坐标排序（从上到下，适配竖直玻璃管）
    # horizontal_lines.sort(key=lambda x: x["y_center"])
    #
    # # -------------------------- 4. 筛选等间距刻度线 --------------------------
    # y_centers = [line["y_center"] for line in horizontal_lines]
    # diffs = np.diff(y_centers)  # 相邻刻度线的间距（相对ROI）
    #
    # if len(diffs) < 2:
    #     print("警告：刻度线数量不足，无法筛选等间距！")
    #     return horizontal_lines, roi_image.copy()
    #
    # # 计算间距的统计特征（均值+标准差）
    # mean_diff = np.mean(diffs)
    # std_diff = np.std(diffs)
    #
    # if std_diff < std_thresh:
    #     # 间距足够均匀，保留所有直线
    #     valid_lines = horizontal_lines
    # else:
    #     # 找出最长连续等间距段（容错处理）
    #     max_len = 0
    #     curr_len = 1
    #     start_idx = 0
    #
    #     for i in range(1, len(diffs)):
    #         if abs(diffs[i] - diffs[i - 1]) < 1.0:  # 相邻间距变化小于1px（容错）
    #             curr_len += 1
    #         else:
    #             if curr_len > max_len:
    #                 max_len = curr_len
    #                 start_idx = i - curr_len
    #             curr_len = 1
    #
    #     # 处理最后一个段
    #     if curr_len > max_len:
    #         max_len = curr_len
    #         start_idx = len(diffs) - curr_len
    #
    #     # 转换为直线索引（n个间距对应n+1条直线）
    #     valid_lines = horizontal_lines[start_idx: start_idx + max_len + 1]
    #
    # # -------------------------- 5. 在ROI上绘制刻度线 & 整理结果 --------------------------
    # roi_with_scale = roi_image.copy()  # 避免修改原始ROI
    # scale_line_relative_coords = []
    #
    # for line_info in valid_lines:
    #     line = line_info["line"]
    #     # 在ROI上绘制刻度线（绿色，线宽2）
    #     cv2.line(roi_with_scale, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
    #     # 记录相对ROI的坐标
    #     scale_line_relative_coords.append(((line[0], line[1]), (line[2], line[3])))
    #
    # print(f"成功检测到{len(valid_lines)}条等间距刻度线（ROI内）！")
    # return scale_line_relative_coords, roi_with_scale


# -------------------------- 示例调用：处理ROI图像并保存结果 --------------------------
if __name__ == "__main__":
    # -------------------------- 1. 配置路径与参数 --------------------------
    roi_image_path = "../../images/liquid/3.png"  # 替换为你的ROI图像路径（已裁剪好的玻璃管区域）
    output_path = "../../images/liquid/3-res.png"  # 结果保存路径
    angle_thresh = 5  # 水平线角度阈值（可根据实际情况调整）
    std_thresh = 1.0  # 刻度线间距标准差阈值（可根据实际情况调整）

    # -------------------------- 2. 加载ROI图像 --------------------------
    if not os.path.exists(roi_image_path):
        raise FileNotFoundError(f"ROI图像未找到：{roi_image_path}")
    roi_image = cv2.imread(roi_image_path)
    if roi_image is None:
        raise ValueError(f"无法读取ROI图像：{roi_image_path}")

    # -------------------------- 3. 检测刻度线并保存结果 --------------------------
    scale_coords, roi_with_scale = detect_scale_lines_in_roi(
        roi_image, angle_thresh=angle_thresh, std_thresh=std_thresh
    )

    # 保存带刻度线的ROI图像
    cv2.imwrite(output_path, roi_with_scale)
    print(f"结果已保存至：{output_path}")

    # 打印刻度线相对ROI的坐标（可选）
    print("刻度线相对ROI的坐标（格式：((x1, y1), (x2, y2))）：")
    for idx, coords in enumerate(scale_coords):
        print(f"刻度线{idx + 1}: {coords}")

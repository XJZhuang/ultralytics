#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：detect_track_predict_visual.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/10 14:33 
@explain : 逐帧目标跟踪，绘制轨迹，手动保存视频
'''
from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "../videos/video_01.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频的基本信息
# 帧率
fps = int(cap.get(cv2.CAP_PROP_FPS))
# 宽度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# 高度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 总帧数（可选）
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 定义视频编码器（根据系统选择合适的编码器）
# Windows: cv2.VideoWriter_fourcc(*'XVID') 或 *'MP4V'
# Linux/Mac: cv2.VideoWriter_fourcc(*'mp4v') 或 *'avc1'
fourcc = cv2.VideoWriter_fourcc(*'MP4V')

# 设置输出视频路径和VideoWriter对象
output_path = "../videos/video_01_result01.mp4"
# 参数：输出路径、编码器、帧率、(宽度, 高度)
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 检查VideoWriter是否初始化成功
if not video_writer.isOpened():
    raise ValueError("无法初始化视频写入器，请检查输出路径或编码器设置")

# Store the track history
track_history = defaultdict(lambda: [])

# 进度计数（可选，用于显示处理进度）
processed_frames = 0

try:
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            result = model.track(
                frame,
                persist=True,
                # save=True,    # 只保存图片
            )[0]

            # Get the boxes and track IDs
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Visualize the result on the frame
                frame = result.plot()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # 将处理后的帧写入输出视频
            video_writer.write(frame)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", frame)

            # 打印处理进度（可选）
            processed_frames += 1
            if processed_frames % 10 == 0:
                print(f"处理进度: {processed_frames}/{total_frames} 帧")

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
finally:
    # 释放资源（关键：必须释放VideoWriter，否则视频文件会损坏）
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print(f"跟踪结果已保存至: {output_path}")

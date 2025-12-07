#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：detect_track_predict.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/10 14:13 
@explain : 目标跟踪（stream推理），自动保存结果（无轨迹）
'''


from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.track(
    source="../videos/video_01.mp4",
    stream=True,
    show=True,
    save=True,
    tracker="bytetrack.yaml",
    project="yolo-track",
)  # with ByteTrack

for r in results:
    boxes = r.boxes
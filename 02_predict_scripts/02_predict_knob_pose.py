#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：02_predict_pose.py
@Author  ：zhuangxujun
@Date    ：2025-9-17 17:29 
@explain : 
'''

import os
import sys
ultralytics_path = os.path.abspath('../ultralytics-switchgear')
if ultralytics_path not in sys.path:
    sys.path.append(ultralytics_path)

# from ultralytics import YOLO
from ultralytics import YOLO

if __name__ == '__main__':

    # Load a pretrained YOLO11n model
    # model = YOLO(r"D:\1_Python\ultralytics\01_train_scripts\yolov8s-pose.pt")
    # model = YOLO(r"D:\1_Python\ultralytics\01_train_scripts\knob\train\weights\best.pt")
    model = YOLO(r"/home/zxj/ultralytics-switchgear/01_train_scripts/knob/train/weights/best.pt")

    # Define path to directory containing images and videos for inference
    source = r"/home/zxj/ultralytics-switchgear/04_demo/pose_knob"

    # Run inference on the source
    results = model(source, save=True, stream=True, project="knob", task_type="pose_knob")  # generator of Results objects

    for result in results:
        box = result.boxes
        xy = result.keypoints.xy  # x and y coordinates
        xyn = result.keypoints.xyn  # normalized
        kpts = result.keypoints.data  # x, y, visibility (if available)


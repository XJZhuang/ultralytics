#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：pose_predict.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/11 17:02 
@explain : 
'''
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov8s-pose.pt")

    source = r"../ultralytics/assets/zidane.jpg"

    results = model(
        source,
        save=True,
        save_json=True,
        save_txt=True,
        stream=True,
        project="yolo-pose",
    )  # generator of Results objects

    for result in results:
        box = result.boxes
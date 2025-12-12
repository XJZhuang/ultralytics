#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：seg_predict.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/6 14:43 
@explain : 
'''

from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov8s-seg.pt")
    # model = YOLO("yolo11s-seg.pt")

    source = r"../ultralytics/assets/bus.jpg"

    results = model(source, save=True, stream=True, project="yolo-seg")

    for result in results:
        box = result.boxes

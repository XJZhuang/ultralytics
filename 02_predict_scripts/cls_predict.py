#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：cls_predict.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/11 17:20 
@explain : 
'''
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO("yolov8s-cls.pt")

    source = r"../ultralytics/assets/bus.jpg"

    results = model(source, save=True, stream=True, project="yolo-cls")

    for result in results:
        box = result.boxes
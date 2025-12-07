#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：predict_lights.py
@Author  ：zhuangxujun
@Date    ：2025-9-10 11:01 
@explain : 
'''

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a pretrained YOLO11n model
    model = YOLO(r"D:\1_Python\ultralytics\0_train_scripts\lights\train\weights\best.pt")

    # Define path to directory containing images and videos for inference
    source = r"D:\1_Python\datasets\lights\images\test"

    # Run inference on the source
    results = model(source, save=True, stream=True, project="lights")  # generator of Results objects

    for result in results:
        box = result.boxes

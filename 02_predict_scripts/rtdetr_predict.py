#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：rtdetr_predict.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/5 15:44 
@explain : 
'''


from ultralytics import YOLO

if __name__ == '__main__':

    # Load a pretrained YOLO11n model
    model = YOLO("rtdetr-resnet50.pt")

    # Define path to directory containing images and videos for inference
    source = r"../ultralytics/assets/bus.jpg"

    # Run inference on the source
    results = model(source, save=True, stream=True, project="rtdetr")  # generator of Results objects

    for result in results:
        box = result.boxes

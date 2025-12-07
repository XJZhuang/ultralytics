#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：01_predict_switchgear.py
@Author  ：zhuangxujun
@Date    ：2025-9-18 14:26 
@explain : 对一个文件夹下的多张图片进行推理
'''

from ultralytics import YOLO

if __name__ == '__main__':

    # Load a pretrained YOLO11n model
    model = YOLO(r"../01_train_scripts/switchgear/train/weights/best.pt")

    # Define path to directory containing images and videos for inference
    source = r"/home/zxj/datasets/switchgear/images/val1"

    # Run inference on the source
    # results = model(source, save=True, stream=True, project="switchgear")  # generator of Results objects
    results = model(source, save=True, stream=True, project="switchgear2")  # generator of Results objects

    for result in results:
        box = result.boxes

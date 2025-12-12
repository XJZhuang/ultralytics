#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：detect_train.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/27 11:36 
@explain : 
'''
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")  # build a new model from YAML


# Train the model
results = model.train(
    data="coco8.yaml",
    batch=1,
    epochs=100,
    imgsz=640,
    project="yolo-detect",
)

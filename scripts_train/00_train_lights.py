#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：train_lights.py
@Author  ：zhuangxujun
@Date    ：2025-9-10 10:40 
@explain : 
'''

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"D:\1_Python\ultralytics\ultralytics\cfg\models\v8\yolov8-lights.yaml").load("yolov8s.pt")

    # train the model
    results = model.train(
        data=r"D:\1_Python\ultralytics\ultralytics\cfg\datasets\lights.yaml",
        epochs=1000,
        imgsz=640,
        batch=4,
        device=0,
        project="lights"
    )

    # model = YOLO(r"D:\1_Python\ultralytics-main\0_train_scripts\runs\detect\train\weights\last.pt")
    # result = model.train(resume=True)

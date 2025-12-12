#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：train_lights.py
@Author  ：zhuangxujun
@Date    ：2025-9-10 10:40 
@explain : 目标检测模型
'''

import os
import sys

# 添加项目路径到Python搜索路径
ultralytics_path = os.path.abspath('../ultralytics-switchgear')
if ultralytics_path not in sys.path:
    sys.path.append(ultralytics_path)

try:  # 导入ultralytics相关模块
    from ultralytics import YOLO  # 根据实际模块结构调整
    from ultralytics.utils import LOGGER
except ImportError as e:
    print(f"导入ultralytics失败: {e}")
    sys.exit(1)


if __name__ == '__main__':
    # Load a model
    # model = YOLO(r"../ultralytics/cfg/models/v8/yolov8-switchgear.yaml").load("yolov8s.pt")
    model = YOLO(r"../ultralytics/cfg/models/v8/yolov8-switchgear.yaml")\
        .load("/home/zxj/ultralytics-switchgear/01_train_scripts/switchgear/train4/weights/best.pt")

    # train the model
    results = model.train(
        data=r"../ultralytics/cfg/datasets/switchgear.yaml",
        epochs=1000,
        # epochs=2,
        imgsz=640,
        batch=64,
        device=0,
        project="switchgear"
    )

    # model = YOLO(r"D:\1_Python\ultralytics-main\0_train_scripts\runs\detect\train\weights\last.pt")
    # result = model.train(resume=True)

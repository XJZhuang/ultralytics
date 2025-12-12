#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：save_txt_labels.py
@Author  ：zhuangxujun
@Date    ：2025-9-28 17:19 
@explain : 只支持 目标检测 任务
'''

import os
import sys
import glob

from ultralytics import YOLO  # 根据实际模块结构调整
from ultralytics.utils import LOGGER

if __name__ == '__main__':
    # 加载现有的xxx.pt模型
    # yolo_model_path = r"/home/zxj/ultralytics-switchgear/01_train_scripts/switchgear/train2/weights/best.pt"
    yolo_model_path = r'D:\1_Python\fire-security-ai\fire_security\security\train_03\weights\best.pt'
    # model = YOLO("yolo11x.pt")
    model = YOLO(yolo_model_path)

    # 指定图片文件夹的路径
    image_folder = r"D:\1_Python\datasets\fire_security\images\train05"

    # 批量对图片进行预测，并保存标签文件
    results = model.predict(
        source=image_folder,
        project=image_folder,
        name="labels_txt",
        # classes=[0],    # person 的 类别ID
        exist_ok=True,
        save_txt=True,
        conf=0.8,    # 高置信度，只保留预测的准的框
    )

    # 对每张图片进行预测，并保存标签文件（效率低）
    # for image_path in glob.glob(image_folder + "*.jpg"):
    #     # 对图片进行预测，并保存标签文件
    #     results = model.predict(
    #         source=image_path,
    #         project=image_folder,  # 保存到图片文件夹
    #         name="labels_txt",  # 子文件夹名（可省略）
    #         exist_ok=True,  # 覆盖现有文件
    #         save_txt=True
    #     )
    #     print('done..........', image_path)

    print('done.....................overall')

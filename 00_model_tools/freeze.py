#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：freeze.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/19 11:44 
@explain : 冻结
'''
from ultralytics import RTDETR

if __name__ == '__main__':
    # model = RTDETR(r"./cfg/rtdetr-l.yaml").load("rtdetr-l.pt")
    model = RTDETR(r"./cfg/yolov8-rtdetr.yaml")
    model.load("../src/resources/predict_model/yolov8s.pt")

    print(model.model)
    # 3. （可选）冻结 backbone，只训 head
    for k, v in model.named_parameters():
        if "model.22" in k:  # 或 "22."（根据实际层名）
            v.requires_grad = True
        else:
            v.requires_grad = False

    # 冻结后检查
    trainable_names = []
    frozen_names = []

    for k, v in model.named_parameters():
        if v.requires_grad:
            trainable_names.append(k)
        else:
            frozen_names.append(k)

    print(f"Trainable params: {len(trainable_names)}")
    print(f"Example trainable: {trainable_names[0] if trainable_names else 'None'}")
    print(f"Frozen params: {len(frozen_names)}")

    print("All trainable modules/parameters:")
    for trainable_name in trainable_names:
        print(trainable_name)

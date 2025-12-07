#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：load_resnet_2_rtdetr.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/19 20:13 
@explain : 加载失败
'''
import torch
from ultralytics import RTDETR
model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml')
resnet_weights = torch.load(r"D:\1_Python\ultralytics\weights\resnet50-19c8e357.pth")
adapted_weights = {k if 'backbone' in k else 'backbone.' + k: v for k, v in resnet_weights.items()}
model.model.load_state_dict(adapted_weights, strict=False)
# Now you can proceed with training
# model.train(data='path_to_coco128.yaml', epochs=100)

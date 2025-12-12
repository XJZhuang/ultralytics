# -*- coding: utf-8 -*-
'''

@Project:   ultralytics-main
@File:      export_onnx.py
@Author:    123
@Date:      2025/1/17 16:27
@Explain:   导出onnx格式的权重
'''

from ultralytics import YOLO

# 加载模型
model = YOLO(r"D:\1_Python\ultralytics-main\try\runs\detect\train8\weights\best.pt")  # 加载自定义训练模型

# 导出模型
model.export(format="onnx")




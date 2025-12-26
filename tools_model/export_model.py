#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from ultralytics import YOLO

# Load the YOLO11 model
# model = YOLO("yolo11n.pt")
#
# # Export the model to ONNX format
# model.export(format="onnx")  # creates 'yolo11n.onnx'

# Load the exported ONNX model
onnx_model = YOLO(r"D:\1_Python\ultralytics\weights\yolo11n.onnx")

# Run inference
results = onnx_model(r"D:\1_Python\ultralytics\ultralytics\assets\bus.jpg")

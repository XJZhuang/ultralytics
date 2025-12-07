#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：loaded.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/19 17:36 
@explain : 验证 转换后的权重 中，是否加载成功
'''
from ultralytics import RTDETR
import torch

def mean_of_source_weights():
    # 1. 加载模型和转换后的权重
    model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml")
    converted_weights = torch.load("rtdetr-resnet50_converted.pt")

    # 2. 直接给模型层赋值（遍历转换后的权重，逐个更新）
    for target_key, weight in converted_weights.items():
        layers = target_key.split(".")
        current_module = model.model
        for layer in layers[:-1]:
            if hasattr(current_module, layer):
                current_module = getattr(current_module, layer)
            else:
                print(f"跳过不存在的层：{layer}（键：{target_key}）")
                break
        else:
            param_name = layers[-1]
            if hasattr(current_module, param_name):
                setattr(current_module, param_name, weight)

    # 3. 正确验证：结合“均值一致性”和“最大值有效性”
    # conv1_weight = model.model.model[0].layer[0].conv.weight
    conv1_weight = model.model.model[1].layer[0].cv1.conv.weight
    conv1_mean = conv1_weight.mean().item()
    conv1_max = conv1_weight.max().item()

    # 加载源权重对比（确认一致性）
    source_weights = torch.load(r"D:\1_Python\ultralytics\weights\resnet50-19c8e357.pth")
    # source_conv1_mean = source_weights["conv1.weight"].mean().item()
    source_conv1_mean = source_weights["layer1.0.conv1.weight"].mean().item()

    print(f"📊 加载后conv1均值：{conv1_mean:.6f}")     # -0.000495
    print(f"📊 源ResNet50conv1均值：{source_conv1_mean:.6f}")    # -0.000495
    print(f"📊 加载后conv1最大值：{conv1_max:.6f}")     # 0.781415

    # 正确判断条件：均值与源权重一致 + 最大值>0（排除全零权重）
    if abs(conv1_mean - source_conv1_mean) < 1e-6 and conv1_max > 0:
        print("✅ 权重100%加载成功！")
    else:
        print("❌ 加载失败")


def mean_of_converted_weights():
    # 加载转换后的权重文件
    converted_weights = torch.load("rtdetr-resnet50_converted.pt")

    # 查看conv1的权重均值
    conv1_weight = converted_weights["model.0.layer.0.conv.weight"]
    print(f"转换后权重文件中conv1的均值：{conv1_weight.mean().item():.6f}")
    print(f"转换后权重文件中conv1的最大值：{conv1_weight.max().item():.6f}")

    # 同时查看源ResNet50的conv1均值，对比是否一致
    source_weights = torch.load(r"D:\1_Python\ultralytics\weights\resnet50-19c8e357.pth")
    print(f"源ResNet50权重中conv1的均值：{source_weights['conv1.weight'].mean().item():.6f}")


if __name__ == '__main__':
     # mean_of_converted_weights()
     mean_of_source_weights()


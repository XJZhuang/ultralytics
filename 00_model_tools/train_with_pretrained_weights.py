#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：train_with_pretrained_weights.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/19 19:38 
@explain : 
'''
from ultralytics import RTDETR
import torch

# -------------------------- 1. 配置参数 --------------------------
# 数据集配置文件路径
DATASET_YAML = "path/to/your/dataset.yaml"
# 我们刚刚转换好的权重文件路径
PRETRAINED_BACKBONE_PATH = "rtdetr_resnet50_backbone.pt"
# 训练轮次
EPOCHS = 100
# 批次大小
BATCH = 8
# 图像尺寸
IMGSZ = 640
# 训练设备 (0 for GPU, 'cpu' for CPU)
DEVICE = 0

# -------------------------- 2. 创建模型并加载权重 --------------------------
print("正在创建 RT-DETR 模型...")
# 创建一个基于resnet50的RT-DETR模型，不加载预训练权重（我们将手动加载）
# model = RTDETR('rtdetr-resnet50.yaml').model  # 这会返回nn.Module对象
# 或者，更简单的方式是直接使用RTDETR类，并在之后加载权重
model = RTDETR('rtdetr-resnet50.yaml')

print(f"正在加载自定义骨干网权重: {PRETRAINED_BACKBONE_PATH}")
# 加载我们转换好的权重字典
backbone_weights = torch.load(PRETRAINED_BACKBONE_PATH, map_location='cpu')

# 使用 strict=False 是因为我们只加载骨干网权重，而模型的头部（neck, head）权重不存在于我们的文件中
# 这会自动跳过那些无法匹配的键，保留头部的随机初始化权重
model.model.load_state_dict(backbone_weights, strict=False)

print("✅ 自定义骨干网权重加载成功！")

# -------------------------- 3. (可选) 冻结骨干网进行训练 --------------------------
# 为了快速收敛，你可以选择先冻结骨干网（只训练头部）
# 骨干网对应的模块是 model.model[0] 到 model.model[4]
FREEZE_BACKBONE = False
if FREEZE_BACKBONE:
    print("正在冻结骨干网权重...")
    for i in range(5): # model.model[0] to model.model[4] are backbone layers
        for param in model.model.model[i].parameters():
            param.requires_grad = False
    print("✅ 骨干网权重已冻结，将只训练模型头部。")

# -------------------------- 4. 开始训练 --------------------------
print("\n🚀 开始训练！")
results = model.train(
    data=DATASET_YAML,
    epochs=EPOCHS,
    batch=BATCH,
    imgsz=IMGSZ,
    device=DEVICE,
    project='rtdetr_training',
    name='exp_with_resnet50_backbone',
    exist_ok=True,
    # 你可以添加更多训练参数，如学习率、权重衰减等
    # lr0=0.001,
    # lrf=0.01,
    # momentum=0.937,
    # weight_decay=0.0005,
)

print("🏁 训练完成！")
print(f"训练结果保存在: {results.save_dir}")
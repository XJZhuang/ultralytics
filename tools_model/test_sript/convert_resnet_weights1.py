#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：convert_resnet_weights.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/19 16:54 
@explain : 
'''

import torch
from ultralytics import RTDETR

# -------------------------- 1. 配置路径 --------------------------
source_weights_path = r"D:\1_Python\ultralytics\weights\resnet50-19c8e357.pth"  # 你的标准ResNet50权重
output_weights_path = "rtdetr-resnet50_converted.pt"  # 转换后保存的权重文件名
model_config_path = "ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml"  # RT-DETR配置文件

# -------------------------- 2. 加载源权重和目标模型 --------------------------
# 加载标准ResNet50权重
source_weights = torch.load(source_weights_path)
print(f"成功加载源权重，共{len(source_weights.keys())}个键")

# 加载RT-DETR模型（仅用于获取目标键名和形状）
model = RTDETR(model_config_path)
target_state_dict = model.model.state_dict()
print(f"成功加载RT-DETR模型，共{len(target_state_dict.keys())}个键")

# -------------------------- 3. 构建自动映射规则 --------------------------
key_mapping = {}

# 3.1 映射conv1 + bn1（对应model.0）
key_mapping["conv1.weight"] = "model.0.layer.0.conv.weight"
for bn_field in ["weight", "bias", "running_mean", "running_var"]:
    source_key = f"bn1.{bn_field}"
    target_key = f"model.0.layer.0.bn.{bn_field}"
    if source_key in source_weights and target_key in target_state_dict:
        key_mapping[source_key] = target_key
    else:
        print(f"{source_key}-->{target_key}映射失败")

# 3.2 映射layer1-layer4（对应model.1-model.4）
# ResNet50的layer1-layer4分别对应RT-DETR的model.1-model.4
layer_mapping = {
    "layer1": "model.1",
    "layer2": "model.2",
    "layer3": "model.3",
    "layer4": "model.4"
}

# ResNet50每个block的conv/bn对应RT-DETR的cv1/cv2/cv3 + bn
conv_bn_mapping = {
    "conv1": "cv1.conv",
    "bn1": "cv1.bn",
    "conv2": "cv2.conv",
    "bn2": "cv2.bn",
    "conv3": "cv3.conv",
    "bn3": "cv3.bn"
}

# 遍历每个layer（layer1-layer4）
for resnet_layer, rtdetr_model in layer_mapping.items():
    # 遍历源权重中该layer的所有键（如layer1.0.conv1.weight）
    for source_key in source_weights.keys():
        if source_key.startswith(resnet_layer):
            # 拆分源键（示例：layer1.0.conv1.weight → ["layer1", "0", "conv1", "weight"]）
            source_parts = source_key.split(".")
            if len(source_parts) < 4:
                continue

            # 提取block索引（如0）、conv/bn类型（如conv1）、字段（如weight）
            block_idx = source_parts[1]
            conv_bn_type = source_parts[2]
            field = ".".join(source_parts[3:])  # 可能是weight/bias/running_mean等

            # 1. 处理普通conv和bn（如conv1→cv1.conv，bn1→cv1.bn）
            if conv_bn_type in conv_bn_mapping:
                target_conv_bn = conv_bn_mapping[conv_bn_type]
                target_key = f"{rtdetr_model}.layer.{block_idx}.{target_conv_bn}.{field}"

            # 2. 处理shortcut（downsample→shortcut.0）
            elif conv_bn_type == "downsample":
                downsample_type = source_parts[3]  # 0=conv，1=bn
                if downsample_type == "0":
                    target_key = f"{rtdetr_model}.layer.{block_idx}.shortcut.0.conv.{field}"
                elif downsample_type == "1":
                    target_key = f"{rtdetr_model}.layer.{block_idx}.shortcut.0.bn.{field}"
                else:
                    continue
            else:
                continue

            # 验证目标键是否存在于RT-DETR模型中
            if target_key in target_state_dict:
                key_mapping[source_key] = target_key

print(f"成功构建{len(key_mapping)}个键的映射关系")

# -------------------------- 4. 执行权重转换 --------------------------
converted_weights = {}
mismatched = []

for source_key, target_key in key_mapping.items():
    # 检查源权重和目标权重的形状是否匹配
    source_tensor = source_weights[source_key]
    target_tensor_shape = target_state_dict[target_key].shape

    if source_tensor.shape == target_tensor_shape:
        converted_weights[target_key] = source_tensor
        print(f"✅ 映射成功：{source_key} → {target_key}（形状：{source_tensor.shape}）")
    else:
        mismatched.append(f"❌ 形状不匹配：{source_key}（{source_tensor.shape}）→ {target_key}（{target_tensor_shape}）")

# 打印形状不匹配的键（若有）
if mismatched:
    print("\n".join(mismatched))

# -------------------------- 5. 保存转换后的权重 --------------------------
torch.save(converted_weights, output_weights_path)
print(f"\n📥 转换完成！权重已保存到：{output_weights_path}（共{len(converted_weights)}个有效键）")

# -------------------------- 6. 验证转换结果 --------------------------
print("\n🔍 开始验证转换结果...")
try:
    # 步骤1: 创建一个全新的、未经修改的RT-DETR模型用于验证
    # 关键在于这里！每次验证都用 fresh_model = RTDETR(...) 创建新实例
    fresh_model = RTDETR(model_config_path)

    # 步骤2: 记录全新模型的初始（随机）权重状态
    initial_conv1_mean = fresh_model.model.state_dict()["model.0.layer.0.conv.weight"].mean().item()
    print(f"📊 全新模型（随机初始化）的conv1权重均值：{initial_conv1_mean:.6f}")

    # 步骤3: 加载我们转换后的权重到这个全新模型中（这是第一次加载）
    converted_weights = torch.load(output_weights_path, map_location='cpu')
    fresh_model.model.load_state_dict(converted_weights, strict=False)
    print("✅ 转换后的权重已成功加载到全新的RT-DETR模型！")

    # 步骤4: 记录加载权重后的状态
    loaded_conv1_mean = fresh_model.model.state_dict()["model.0.layer.0.conv.weight"].mean().item()
    print(f"📊 加载转换权重后，conv1权重均值：{loaded_conv1_mean:.6f}")

    # 步骤5: 与源ResNet50权重的对应层进行终极对比
    source_conv1_mean = source_weights['conv1.weight'].mean().item()
    print(f"📊 源ResNet50模型的conv1权重均值：{source_conv1_mean:.6f}")

    # 最终判断
    if abs(loaded_conv1_mean - source_conv1_mean) < 1e-5:
        print("\n🎉 验证成功！")
        print("  - 加载的权重与源ResNet50权重高度一致。")
        print("  - 权重转换和加载过程完全正确！")
    elif abs(initial_conv1_mean - loaded_conv1_mean) > 0.01:
        print("\n✅ 验证成功！")
        print("  - 权重已成功加载，模型状态发生显著变化。")
        print("  - 建议检查映射规则以确保与源权重完全一致。")
    else:
        print("\n⚠️  验证警告：")
        print("  - 加载前后权重差异很小，可能存在问题。")
        print("  - 请检查转换脚本或权重文件。")

except Exception as e:
    print(f"\n❌ 验证过程中发生错误：{e}")
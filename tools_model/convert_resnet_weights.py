#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：convert_resnet_weights.py.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/19 19:37 
@explain : 加载 resnet50预训练权重，转换为RTDETR-restnet50模型结构中层名称相同的pt文件
'''
import torch

# -------------------------- 1. 配置路径 --------------------------
# 你的标准ResNet50权重文件路径（例如从torchvision下载）
source_weights_path = r"D:\1_Python\ultralytics\weights\resnet50-19c8e357.pth"
# 转换后权重文件的保存路径和名称
output_weights_path = "rtdetr_resnet50_backbone.pt"

# -------------------------- 2. 加载源权重 --------------------------
print(f"正在加载源权重: {source_weights_path}")
source_weights = torch.load(source_weights_path, map_location='cpu')
print(f"成功加载源权重，共包含 {len(source_weights.keys())} 个键。")

# -------------------------- 3. 定义映射规则 --------------------------
# 这个映射表是根据我们之前的详细分析创建的
# key_mapping = {
#     # Stem层
#     "conv1.weight": "model.0.layer.0.conv.weight",
#     "bn1.weight": "model.0.layer.0.bn.weight",
#     "bn1.bias": "model.0.layer.0.bn.bias",
#     "bn1.running_mean": "model.0.layer.0.bn.running_mean",
#     "bn1.running_var": "model.0.layer.0.bn.running_var",
#
#     # Layer1
#     "layer1.0.conv1.weight": "model.1.layer.0.cv1.conv.weight",
#     "layer1.0.bn1.weight": "model.1.layer.0.cv1.bn.weight",
#     "layer1.0.bn1.bias": "model.1.layer.0.cv1.bn.bias",
#     "layer1.0.bn1.running_mean": "model.1.layer.0.cv1.bn.running_mean",
#     "layer1.0.bn1.running_var": "model.1.layer.0.cv1.bn.running_var",
#     "layer1.0.conv2.weight": "model.1.layer.0.cv2.conv.weight",
#     "layer1.0.bn2.weight": "model.1.layer.0.cv2.bn.weight",
#     "layer1.0.bn2.bias": "model.1.layer.0.cv2.bn.bias",
#     "layer1.0.bn2.running_mean": "model.1.layer.0.cv2.bn.running_mean",
#     "layer1.0.bn2.running_var": "model.1.layer.0.cv2.bn.running_var",
#     "layer1.0.conv3.weight": "model.1.layer.0.cv3.conv.weight",
#     "layer1.0.bn3.weight": "model.1.layer.0.cv3.bn.weight",
#     "layer1.0.bn3.bias": "model.1.layer.0.cv3.bn.bias",
#     "layer1.0.bn3.running_mean": "model.1.layer.0.cv3.bn.running_mean",
#     "layer1.0.bn3.running_var": "model.1.layer.0.cv3.bn.running_var",
#     "layer1.0.downsample.0.weight": "model.1.layer.0.shortcut.0.conv.weight",
#     "layer1.0.downsample.1.weight": "model.1.layer.0.shortcut.0.bn.weight",
#     "layer1.0.downsample.1.bias": "model.1.layer.0.shortcut.0.bn.bias",
#     "layer1.0.downsample.1.running_mean": "model.1.layer.0.shortcut.0.bn.running_mean",
#     "layer1.0.downsample.1.running_var": "model.1.layer.0.shortcut.0.bn.running_var",
#
#     # Layer2, Layer3, Layer4 的映射规则类似，为简洁起见，这里使用循环自动生成
#     # ... (实际使用时，你需要将Layer2到Layer4的所有键都添加到映射表中，或者编写循环来生成)
# }


# 为了让脚本更简洁高效，我们可以编写循环来自动生成映射规则，而不是手动全部写出
def generate_mapping():
    mapping = {}
    # Stem层
    mapping["conv1.weight"] = "model.0.layer.0.conv.weight"
    for field in ["weight", "bias", "running_mean", "running_var"]:
        mapping[f"bn1.{field}"] = f"model.0.layer.0.bn.{field}"

    # Layer1-4
    layer_mapping = {"layer1": "model.1", "layer2": "model.2", "layer3": "model.3", "layer4": "model.4"}
    conv_bn_mapping = {"conv1": "cv1.conv", "bn1": "cv1.bn", "conv2": "cv2.conv", "bn2": "cv2.bn", "conv3": "cv3.conv",
                       "bn3": "cv3.bn"}

    for resnet_layer, rtdetr_model in layer_mapping.items():
        # 假设每个layer最多有6个block (layer3有6个)
        for block_idx in range(6):
            for resnet_conv_bn, rtdetr_cv_bn in conv_bn_mapping.items():
                for field in ["weight", "bias", "running_mean", "running_var"]:
                    # 对于conv层，没有running_mean和running_var
                    if "conv" in resnet_conv_bn and "running" in field:
                        continue
                    source_key = f"{resnet_layer}.{block_idx}.{resnet_conv_bn}.{field}"
                    target_key = f"{rtdetr_model}.layer.{block_idx}.{rtdetr_cv_bn}.{field}"
                    mapping[source_key] = target_key

            # 处理shortcut/downsample
            for field in ["weight", "bias", "running_mean", "running_var"]:
                source_key = f"{resnet_layer}.{block_idx}.downsample.0.{field}"
                target_key = f"{rtdetr_model}.layer.{block_idx}.shortcut.0.conv.{field}"
                if "running" in field: continue  # conv层没有
                mapping[source_key] = target_key

                source_key = f"{resnet_layer}.{block_idx}.downsample.1.{field}"
                target_key = f"{rtdetr_model}.layer.{block_idx}.shortcut.0.bn.{field}"
                mapping[source_key] = target_key
    return mapping


key_mapping = generate_mapping()

# -------------------------- 4. 执行转换 --------------------------
print("开始执行权重转换...")
converted_weights = {}
for source_key, target_key in key_mapping.items():
    if source_key in source_weights:
        converted_weights[target_key] = source_weights[source_key]
        # print(f"  映射成功: {source_key} -> {target_key}")

print(f"转换完成，共生成 {len(converted_weights.keys())} 个键。")

# -------------------------- 5. 保存转换后的权重 --------------------------
torch.save(converted_weights, output_weights_path)
print(f"转换后的权重已保存至: {output_weights_path}")

# -------------------------- 6. 验证转换结果 --------------------------
print("\n开始验证转换结果...")
try:
    # 加载转换后的权重
    converted_weights = torch.load(output_weights_path)
    print("✅ 转换后的权重文件可以正常加载。")

    # 检查几个关键键是否存在
    test_keys = [
        "model.0.layer.0.conv.weight",
        "model.1.layer.0.cv1.conv.weight",
        "model.2.layer.1.cv2.bn.weight"
    ]
    for key in test_keys:
        if key in converted_weights:
            print(f"✅ 关键键 '{key}' 存在于转换后的权重中。")
        else:
            print(f"❌ 关键键 '{key}' 不存在于转换后的权重中。")

except Exception as e:
    print(f"❌ 验证失败: {e}")
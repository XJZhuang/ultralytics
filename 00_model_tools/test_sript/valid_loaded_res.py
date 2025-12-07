#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：valid_loaded_res.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/19 19:42 
@explain : 
'''
from ultralytics import RTDETR
import torch
import warnings

warnings.filterwarnings('ignore')

# -------------------------- 1. 配置参数 --------------------------
DATASET_YAML = "path/to/your/dataset.yaml"
PRETRAINED_BACKBONE_PATH = "rtdetr_resnet50_backbone.pt"
EPOCHS = 100
BATCH = 8
IMGSZ = 640
DEVICE = 0


# -------------------------- 辅助函数：获取骨干网参数名称 --------------------------
def get_backbone_param_names(model):
    """
    遍历模型，获取所有属于 ResNet 骨干网 (model.model[0] to model.model[4]) 的参数名称。
    """
    backbone_param_names = []
    # 骨干网由 model.model[0] (stem) 和 model.model[1-4] (layer1-layer4) 组成
    for i in range(5):
        module = model.model.model[i]
        # 遍历模块的所有参数和缓冲区
        for name, _ in module.named_parameters(prefix=f"model.{i}", recurse=True):
            backbone_param_names.append(name)
        for name, _ in module.named_buffers(prefix=f"model.{i}", recurse=True):
            backbone_param_names.append(name)
    return backbone_param_names


# -------------------------- 2. 创建模型并执行全面验证 --------------------------
print("🚀 开始创建模型并执行权重加载验证...")

# 创建一个全新的、干净的模型用于验证
model = RTDETR('rtdetr-resnet50.yaml')

# 步骤 1: 获取骨干网所有参数的名称列表
print("🔍 正在扫描骨干网参数列表...")
backbone_param_names = get_backbone_param_names(model)
print(f"✅ 共识别出 {len(backbone_param_names)} 个骨干网相关参数/缓冲区。")

# 步骤 2: 记录加载前（随机初始化）的状态
print("\n📋 记录加载前（随机初始化）的参数状态...")
initial_state = {}
for param_name in backbone_param_names:
    # 使用 model.model.state_dict() 安全地获取参数/缓冲区的值
    try:
        # .clone() 确保我们得到的是一个副本，而不是引用
        initial_state[param_name] = model.model.state_dict()[param_name].clone()
    except KeyError:
        print(f"⚠️  警告：在模型中未找到参数 '{param_name}'。")

# 步骤 3: 加载转换后的权重
print(f"\n📥 正在加载转换后的权重: {PRETRAINED_BACKBONE_PATH}")
backbone_weights = torch.load(PRETRAINED_BACKBONE_PATH, map_location='cpu')
# strict=False 允许跳过模型中不存在的键（例如，模型头部的键）
model.model.load_state_dict(backbone_weights, strict=False)
print("✅ 权重文件已提交给模型加载。")

# 步骤 4: 记录加载后的状态
print("\n📋 记录加载后的参数状态...")
loaded_state = {}
for param_name in backbone_param_names:
    try:
        loaded_state[param_name] = model.model.state_dict()[param_name].clone()
    except KeyError:
        pass  # 忽略已警告过的缺失参数

# 步骤 5 & 6: 对比分析并生成报告
print("\n📊 开始对比分析，生成验证报告...")
report = {
    "total": len(backbone_param_names),
    "successfully_loaded": 0,
    "not_changed": 0,
    "missing_in_weights": 0,
    "mismatched_shape": 0
}

print("\n======================================================================")
print("                      权重加载全面验证报告")
print("======================================================================")

for param_name in backbone_param_names:
    if param_name not in initial_state or param_name not in loaded_state:
        print(f"❌ [缺失] 参数 '{param_name}' 在模型中未找到。")
        report["missing_in_weights"] += 1
        continue

    initial_tensor = initial_state[param_name]
    loaded_tensor = loaded_state[param_name]

    # 检查形状是否一致
    if initial_tensor.shape != loaded_tensor.shape:
        print(f"❌ [形状不匹配] 参数 '{param_name}'")
        print(f"    - 初始形状: {initial_tensor.shape}")
        print(f"    - 加载形状: {loaded_tensor.shape}")
        report["mismatched_shape"] += 1
        continue

    # 检查值是否发生了显著变化
    # 对于权重参数，我们检查均值是否有很大差异
    if 'weight' in param_name.lower() or 'bias' in param_name.lower():
        initial_mean = initial_tensor.float().mean().item()
        loaded_mean = loaded_tensor.float().mean().item()
        mean_diff = abs(initial_mean - loaded_mean)

        if mean_diff > 0.01:  # 差异阈值，可以根据需要调整
            # print(f"✅ [加载成功] 参数 '{param_name}' (均值差异: {mean_diff:.4f})")
            report["successfully_loaded"] += 1
        else:
            print(f"⚠️  [未变化] 参数 '{param_name}' 加载前后均值差异过小 ({mean_diff:.4f})，可能未加载成功。")
            report["not_changed"] += 1
    # 对于 BatchNorm 的 running_mean 和 running_var，我们直接检查所有元素是否完全一致
    elif 'running_mean' in param_name or 'running_var' in param_name:
        if not torch.allclose(initial_tensor, loaded_tensor):
            # print(f"✅ [加载成功] 缓冲区 '{param_name}'")
            report["successfully_loaded"] += 1
        else:
            print(f"⚠️  [未变化] 缓冲区 '{param_name}' 加载前后值完全一致，可能未加载成功。")
            report["not_changed"] += 1
    else:
        # 对于其他类型的缓冲区，也检查所有元素是否完全一致
        if not torch.allclose(initial_tensor, loaded_tensor):
            # print(f"✅ [加载成功] 缓冲区 '{param_name}'")
            report["successfully_loaded"] += 1
        else:
            print(f"⚠️  [未变化] 缓冲区 '{param_name}' 加载前后值完全一致，可能未加载成功。")
            report["not_changed"] += 1

print("======================================================================")
print("📈 验证结果汇总:")
print(f"   - 计划验证的骨干网参数/缓冲区总数: {report['total']}")
print(f"   - ✅ 成功加载并发生显著变化的数量: {report['successfully_loaded']}")
print(f"   - ⚠️  加载前后未发生变化的数量: {report['not_changed']}")
print(f"   - ❌ 形状不匹配的数量: {report['mismatched_shape']}")
print(f"   - ❌ 在模型中缺失的数量: {report['missing_in_weights']}")

if report['not_changed'] == 0 and report['mismatched_shape'] == 0 and report['missing_in_weights'] == 0:
    print("\n🎉 恭喜！所有骨干网参数均已成功加载！")
else:
    print("\n⚠️  验证完成，但发现一些问题，请查看上述详细报告。")
print("======================================================================")

# # -------------------------- 3. (可选) 冻结骨干网 --------------------------
# FREEZE_BACKBONE = False
# if FREEZE_BACKBONE:
#     print("\n❄️  正在冻结骨干网权重...")
#     for i in range(5):  # model.model[0] to model.model[4]
#         for param in model.model.model[i].parameters():
#             param.requires_grad = False
#     print("✅ 骨干网权重已冻结。")
#
# # -------------------------- 4. 开始训练 --------------------------
# print("\n🚀 启动训练流程...")
# results = model.train(
#     data=DATASET_YAML,
#     epochs=EPOCHS,
#     batch=BATCH,
#     imgsz=IMGSZ,
#     device=DEVICE,
#     project='rtdetr_training',
#     name='exp_with_resnet50_backbone',
#     exist_ok=True,
#     # 可以添加更多训练参数...
# )
#
# print("\n🏁 训练完成！")
# print(f"训练结果保存在: {results.save_dir}")
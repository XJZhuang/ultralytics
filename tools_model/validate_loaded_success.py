#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：validate_loaded_success.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/19 19:46 
@explain : 详细验证每一层参数的加载情况
'''
from ultralytics import RTDETR
import torch

# -------------------------- 配置参数 --------------------------
PRETRAINED_BACKBONE_PATH = "rtdetr_resnet50_backbone.pt"    # 转换后的权重
ORIGINAL_RESNET50_PATH = r"D:\1_Python\ultralytics\weights\resnet50-19c8e357.pth"   # 原始 ResNet50 权重


# 只验证加载前后是否变化
def verify_weights_loaded_v1():
    print("🚀 开始详细验证可训练参数加载情况...\n")

    # 创建模型
    model = RTDETR('rtdetr-resnet50.yaml')
    # 加载转换后的权重
    backbone_weights = torch.load(PRETRAINED_BACKBONE_PATH, map_location='cpu')

    # 步骤1：获取所有「可训练参数（weight/bias）」的名称（按层分组）
    trainable_params = {}
    for name, param in model.model.named_parameters():
        if "weight" in name or "bias" in name:
            # 只保留骨干网（model.0~model.4）的参数
            if name.startswith(("model.0.", "model.1.", "model.2.", "model.3.", "model.4.")):
                # 按层分组（例如：model.0, model.1.layer.0 等）
                layer_key = name.split('.layer.')[0] if '.layer.' in name else name.split('.')[0] + '.' + \
                                                                               name.split('.')[1]
                if layer_key not in trainable_params:
                    trainable_params[layer_key] = []
                trainable_params[layer_key].append(name)

    print(f"✅ 共识别到骨干网可训练参数：{sum(len(params) for params in trainable_params.values())} 个")
    print(f"📊 按层分组如下：")
    for layer, params in trainable_params.items():
        print(f"   - {layer}: {len(params)} 个参数")
    print("\n" + "=" * 80 + "\n")

    # 步骤2：记录加载前的参数值（取前10个元素）
    initial_vals = {}
    print("📋 正在记录加载前的参数值...")
    for layer, params in trainable_params.items():
        for name in params:
            try:
                initial_vals[name] = model.model.state_dict()[name].flatten()[:10].clone()
            except KeyError:
                print(f"⚠️  警告：未找到参数 {name}")
    print("✅ 参数值记录完成！\n")

    # 步骤3：加载权重
    print("📥 正在加载转换后的权重...")
    model.model.load_state_dict(backbone_weights, strict=False)
    print("✅ 权重加载完成！\n")

    # 步骤4：逐层逐参数验证加载情况
    print("🔍 开始逐层逐参数验证...\n")
    total_success = 0
    total_failed = 0
    layer_results = {}

    for layer, params in trainable_params.items():
        print(f"📊 正在验证层：{layer}")
        layer_success = 0
        layer_failed = 0
        for name in params:
            if name not in initial_vals:
                print(f"   - ❌ 跳过参数 {name}（未找到初始值）")
                continue

            initial = initial_vals[name]
            loaded = model.model.state_dict()[name].flatten()[:10]

            # 判断是否加载成功（允许微小浮点误差）
            if not torch.allclose(initial, loaded, atol=1e-6):
                print(f"   - ✅ 加载成功：{name}")
                layer_success += 1
                total_success += 1
            else:
                print(f"   - ❌ 加载失败：{name}（值未变化）")
                layer_failed += 1
                total_failed += 1

        layer_results[layer] = (layer_success, layer_failed)
        print(f"   📈 该层验证结果：成功 {layer_success} 个，失败 {layer_failed} 个\n")

    # 步骤5：输出总体验证报告
    print("=" * 80)
    print("                      权重加载详细验证报告")
    print("=" * 80)
    print("\n📊 各层验证结果汇总：")
    for layer, (success, failed) in layer_results.items():
        total = success + failed
        if total > 0:
            rate = success / total * 100
            print(f"   - {layer}：成功 {success}/{total} 个 ({rate:.2f}%)")
        else:
            print(f"   - {layer}：无参数需要验证")

    print(f"\n📈 总体验证结果：")
    total = total_success + total_failed
    if total > 0:
        success_rate = total_success / total * 100
        print(f"   - 成功加载：{total_success} 个")
        print(f"   - 加载失败：{total_failed} 个")
        print(f"   - 总体成功率：{success_rate:.2f}%")
    else:
        print("   - 无参数需要验证")

    print("\n" + "=" * 80)
    if total_failed == 0:
        print("🎉 恭喜！所有可训练参数均已成功加载！")
    else:
        print("⚠️  存在加载失败的参数，请检查映射规则或权重文件！")
    print("=" * 80)


# -------------------------- 核心工具函数 --------------------------
def get_resnet50_to_rtdetr_mapping():
    '''
    定义「原始 ResNet50 参数名」到「RT-DETR 参数名」的映射规则
    （与 convert_resnet_weights.py 的映射规则完全一致，确保反向验证的正确性）
    '''
    mapping = {}

    # 1. Stem 层（对应 model.0）
    mapping["conv1.weight"] = "model.0.layer.0.conv.weight"
    mapping["bn1.weight"] = "model.0.layer.0.bn.weight"
    mapping["bn1.bias"] = "model.0.layer.0.bn.bias"
    mapping["bn1.running_mean"] = "model.0.layer.0.bn.running_mean"
    mapping["bn1.running_var"] = "model.0.layer.0.bn.running_var"

    # 2. Layer1-layer4（对应 model.1-model.4）
    for resnet_layer, rtdetr_layer in [("layer1", "model.1"), ("layer2", "model.2"),
                                       ("layer3", "model.3"), ("layer4", "model.4")]:
        # 每个 layer 包含多个 block（0-5，根据 ResNet50 结构）
        for block_idx in range(6):  # 覆盖 ResNet50 的所有 block
            # 每个 block 包含 3 个 conv + bn（cv1/cv2/cv3）
            for conv_idx in ["1", "2", "3"]:
                # Conv 层
                resnet_conv = f"{resnet_layer}.{block_idx}.conv{conv_idx}.weight"
                rtdetr_conv = f"{rtdetr_layer}.layer.{block_idx}.cv{conv_idx}.conv.weight"
                mapping[resnet_conv] = rtdetr_conv

                # BN 层（weight/bias/running_mean/running_var）
                for bn_attr in ["weight", "bias", "running_mean", "running_var"]:
                    resnet_bn = f"{resnet_layer}.{block_idx}.bn{conv_idx}.{bn_attr}"
                    rtdetr_bn = f"{rtdetr_layer}.layer.{block_idx}.cv{conv_idx}.bn.{bn_attr}"
                    mapping[resnet_bn] = rtdetr_bn

            # Shortcut（仅当 block 有 shortcut 时存在）
            resnet_shortcut_conv = f"{resnet_layer}.{block_idx}.downsample.0.weight"
            rtdetr_shortcut_conv = f"{rtdetr_layer}.layer.{block_idx}.shortcut.0.conv.weight"
            mapping[resnet_shortcut_conv] = rtdetr_shortcut_conv

            for bn_attr in ["weight", "bias", "running_mean", "running_var"]:
                resnet_shortcut_bn = f"{resnet_layer}.{block_idx}.downsample.1.{bn_attr}"
                rtdetr_shortcut_bn = f"{rtdetr_layer}.layer.{block_idx}.shortcut.0.bn.{bn_attr}"
                mapping[resnet_shortcut_bn] = rtdetr_shortcut_bn

    return mapping


# -------------------------- 核心验证逻辑 --------------------------
def verify_weights_loaded():
    print("🚀 开始详细验证（含原始 ResNet50 数值对比）...\n")

    # ====================== 准备工作 ======================
    # 1. 创建 RT-DETR 模型
    model = RTDETR('rtdetr-resnet50.yaml')
    # 2. 加载转换后的权重（待验证）
    backbone_weights = torch.load(PRETRAINED_BACKBONE_PATH, map_location='cpu')
    # 3. 加载原始 ResNet50 权重（用于对比）
    try:
        original_resnet_weights = torch.load(ORIGINAL_RESNET50_PATH, map_location='cpu')
        print(f"✅ 成功加载原始 ResNet50 权重：{ORIGINAL_RESNET50_PATH}")
    except Exception as e:
        print(f"❌ 加载原始 ResNet50 权重失败！请检查路径：{ORIGINAL_RESNET50_PATH}")
        print(f"   错误信息：{str(e)}")
        return
    # 4. 获取「ResNet50 → RT-DETR」的映射规则（用于反向查找）
    resnet_to_rtdetr = get_resnet50_to_rtdetr_mapping()
    # 反向映射：RT-DETR 参数名 → 原始 ResNet50 参数名（用于快速查找）
    rtdetr_to_resnet = {v: k for k, v in resnet_to_rtdetr.items()}

    # ====================== 步骤1：筛选 RT-DETR 骨干网参数 ======================
    trainable_params = {}
    for name, param in model.model.named_parameters():
        if "weight" in name or "bias" in name:
            if name.startswith(("model.0.", "model.1.", "model.2.", "model.3.", "model.4.")):
                # 按层分组（便于输出）
                layer_key = name.split('.layer.')[
                    0] if '.layer.' in name else f"{name.split('.')[0]}.{name.split('.')[1]}"
                if layer_key not in trainable_params:
                    trainable_params[layer_key] = []
                trainable_params[layer_key].append(name)

    print(f"\n✅ 共识别到 RT-DETR 骨干网可训练参数：{sum(len(params) for params in trainable_params.values())} 个")
    print(f"📊 按层分组如下：")
    for layer, params in trainable_params.items():
        print(f"   - {layer}: {len(params)} 个参数")
    print("\n" + "=" * 80 + "\n")

    # ====================== 步骤2：记录加载前的参数值（用于「是否变化」验证） ======================
    initial_vals = {}
    print("📋 正在记录 RT-DETR 加载前的参数值...")
    for layer, params in trainable_params.items():
        for name in params:
            try:
                initial_vals[name] = model.model.state_dict()[name].flatten()[:10].clone()
            except KeyError:
                print(f"⚠️  警告：RT-DETR 中未找到参数 {name}")
    print("✅ 加载前参数值记录完成！\n")

    # ====================== 步骤3：加载转换后的权重 ======================
    print("📥 正在加载转换后的权重到 RT-DETR...")
    model.model.load_state_dict(backbone_weights, strict=False)
    print("✅ 权重加载完成！\n")

    # ====================== 步骤4：核心验证（3个维度） ======================
    print("🔍 开始逐层逐参数验证（3个维度：是否变化 + 数值是否匹配原始 ResNet50）...\n")
    # 统计指标
    total = 0
    changed = 0  # 加载前后是否变化
    matched_resnet = 0  # 数值是否匹配原始 ResNet50
    no_resnet_mapping = 0  # 无对应 ResNet50 参数（正常情况应极少）
    mismatch_resnet = 0  # 数值与 ResNet50 不匹配（需关注）

    for layer, params in trainable_params.items():
        print(f"📊 正在验证层：{layer}")
        layer_total = len(params)
        layer_changed = 0
        layer_matched = 0
        layer_no_mapping = 0
        layer_mismatch = 0

        for name in params:
            total += 1
            # 1. 验证「加载前后是否变化」
            if name not in initial_vals:
                print(f"   - ❌ 跳过参数 {name}（未找到加载前的值）")
                continue
            initial = initial_vals[name]
            loaded = model.model.state_dict()[name].flatten()[:10]
            is_changed = not torch.allclose(initial, loaded, atol=1e-6)
            if is_changed:
                layer_changed += 1
                changed += 1

            # 2. 验证「数值是否匹配原始 ResNet50」
            is_matched = False
            if name in rtdetr_to_resnet:
                # 找到对应的原始 ResNet50 参数名
                resnet_name = rtdetr_to_resnet[name]
                if resnet_name in original_resnet_weights:
                    # 提取原始 ResNet50 的数值（取前10个元素，与加载后对齐）
                    original_resnet_val = original_resnet_weights[resnet_name].flatten()[:10]
                    # 对比加载后的值与原始 ResNet50 的值
                    is_matched = torch.allclose(loaded, original_resnet_val, atol=1e-6)
                    if is_matched:
                        layer_matched += 1
                        matched_resnet += 1
                    else:
                        layer_mismatch += 1
                        mismatch_resnet += 1
                else:
                    # 原始 ResNet50 中无此参数（异常）
                    layer_no_mapping += 1
                    no_resnet_mapping += 1
            else:
                # 无对应的 ResNet50 映射（异常）
                layer_no_mapping += 1
                no_resnet_mapping += 1

            # 3. 输出详细结果
            status_parts = []
            status_parts.append("✅ 变化" if is_changed else "❌ 未变化")
            if is_matched:
                status_parts.append("✅ 匹配原始 ResNet50")
            elif name in rtdetr_to_resnet and rtdetr_to_resnet[name] in original_resnet_weights:
                status_parts.append("❌ 与原始 ResNet50 不匹配")
            else:
                status_parts.append("⚠️  无对应 ResNet50 参数")
            print(f"   - {name}：{', '.join(status_parts)}")

        # 输出该层统计
        print(
            f"   📈 该层统计：总数 {layer_total} | 变化 {layer_changed} | 匹配 ResNet50 {layer_matched} | 不匹配 {layer_mismatch} | 无映射 {layer_no_mapping}\n")

    # ====================== 步骤5：输出总体验证报告 ======================
    print("=" * 80)
    print("                      权重加载验证报告（含原始 ResNet50 对比）")
    print("=" * 80)
    print(f"\n📊 核心指标统计：")
    print(f"   - 骨干网参数总数：{total} 个")
    print(f"   - 加载前后有变化：{changed} 个（{changed / total * 100:.2f}%）")
    print(f"   - 数值匹配原始 ResNet50：{matched_resnet} 个（{matched_resnet / total * 100:.2f}%）")
    print(f"   - 数值与 ResNet50 不匹配：{mismatch_resnet} 个（需检查映射规则！）")
    print(f"   - 无对应 ResNet50 参数：{no_resnet_mapping} 个（需检查映射规则！）")

    print("\n" + "=" * 80)
    # 最终结论
    if mismatch_resnet == 0 and no_resnet_mapping == 0 and changed == total:
        print("🎉 验证通过！所有参数均成功加载，且数值与原始 ResNet50 完全匹配！")
    else:
        print("⚠️  验证未通过！存在以下问题：")
        if mismatch_resnet > 0:
            print(f"   - {mismatch_resnet} 个参数数值与原始 ResNet50 不匹配，请检查 convert_resnet_weights.py 的映射规则！")
        if no_resnet_mapping > 0:
            print(f"   - {no_resnet_mapping} 个参数无对应 ResNet50 映射，请检查 get_resnet50_to_rtdetr_mapping() 函数！")
        if changed < total:
            print(f"   - {total - changed} 个参数加载前后无变化，请检查权重文件 {PRETRAINED_BACKBONE_PATH}！")
    print("=" * 80)


if __name__ == '__main__':
    verify_weights_loaded()
    # verify_weights_loaded_v1()

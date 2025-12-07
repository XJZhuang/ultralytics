#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：attr_count.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/4 16:53 
@explain : 统计 安全帽和抽烟属性 的数量。
'''

import os
import json
import argparse
from pathlib import Path
import numpy as np

# ========== 全局配置项 ==========
ANNOTATION_TYPE = "yolo"  # 标注类型：yolo(txt)/json
SMOKE_ATTR_IDX = -2  # YOLO格式中smoke属性列索引（最后第二列）
NO_HELMET_ATTR_IDX = -1  # YOLO格式中no_helmet属性列索引（最后一列）
HEAD_CLASS_ID = 1  # 头部类别ID（仅统计头部的属性）


# ========== 单目录统计核心函数 ==========
def count_single_dir(annotation_dir, annotation_type="yolo"):
    """统计单个标注目录的属性组合数量"""
    count_dict = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    total_samples = 0

    # 获取标注文件列表
    if annotation_type == "yolo":
        # ann_files = list(Path(annotation_dir).glob("*.txt"))
        # 只匹配数字/字母命名的标注txt，排除classes.txt
        ann_files = []
        for file in Path(annotation_dir).glob("*.txt"):
            if file.name.lower() != "classes.txt":  # 过滤classes.txt
                ann_files.append(file)
    elif annotation_type == "json":
        ann_files = list(Path(annotation_dir).glob("*.json"))
    else:
        raise ValueError(f"不支持的标注类型：{annotation_type}")

    if not ann_files:
        print(f"⚠️  目录 {annotation_dir} 未找到标注文件！")
        return count_dict, total_samples

    # 遍历所有标注文件
    for ann_file in ann_files:
        try:
            if annotation_type == "yolo":
                with open(ann_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()

                        # 过滤非头部类别
                        if int(parts[0]) != HEAD_CLASS_ID:
                            continue

                        # 提取属性值（兼容浮点型标注，如1.0/0.0）
                        try:
                            smoke = int(float(parts[SMOKE_ATTR_IDX]))
                            no_helmet = int(float(parts[NO_HELMET_ATTR_IDX]))
                        except (IndexError, ValueError):
                            print(f"❌ 文件 {ann_file.name} 行 '{line}' 解析失败，跳过")
                            continue

                        # 校验属性值合法性（仅0/1）
                        if smoke not in [0, 1] or no_helmet not in [0, 1]:
                            print(f"⚠️  文件 {ann_file.name} 属性值非法：smoke={smoke}, no_helmet={no_helmet}")
                            continue

                        count_dict[(smoke, no_helmet)] += 1
                        total_samples += 1

            elif annotation_type == "json":
                with open(ann_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for ann in data.get("annotations", []):
                        # 过滤非头部类别
                        if ann.get("category_id") != HEAD_CLASS_ID:
                            continue
                        # 提取属性
                        attrs = ann.get("attributes", {})
                        smoke = int(attrs.get("smoke", 0))
                        no_helmet = int(attrs.get("no_helmet", 0))
                        # 校验属性值
                        if smoke not in [0, 1] or no_helmet not in [0, 1]:
                            print(f"⚠️  文件 {ann_file.name} 属性值非法：smoke={smoke}, no_helmet={no_helmet}")
                            continue
                        count_dict[(smoke, no_helmet)] += 1
                        total_samples += 1

        except Exception as e:
            print(f"❌ 解析 {ann_file} 出错：{str(e)}")
            continue

    return count_dict, total_samples


# ========== 批量统计指定子集 ==========
def batch_count_specified_subsets(root_dir, subset_names, annotation_type="yolo"):
    """
    统计指定名称的子集
    :param root_dir: 数据集根目录（包含所有子集目录）
    :param subset_names: 要统计的子集名称列表（如 ["train01", "train03", "val01"]）
    :param annotation_type: 标注类型
    :return: 子集结果、汇总结果、总样本数
    """
    subset_results = {}
    total_count = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    total_all = 0

    # 遍历指定的每个子集
    for subset_name in subset_names:
        # 拼接子集目录
        subset_dir = Path(root_dir) / subset_name
        if not subset_dir.exists():
            print(f"⚠️  子集 {subset_name} 目录不存在，跳过")
            continue
        if not subset_dir.is_dir():
            print(f"⚠️  {subset_name} 不是目录，跳过")
            continue

        # 拼接标注目录（默认子集下有labels子目录）
        ann_dir = subset_dir


        # 统计当前子集
        print(f"\n📊 正在统计子集：{subset_name}...")
        count_dict, total = count_single_dir(ann_dir, annotation_type)
        subset_results[subset_name] = (count_dict, total)

        # 汇总到全局统计
        for key in total_count.keys():
            total_count[key] += count_dict[key]
        total_all += total

    return subset_results, total_count, total_all


# ========== 格式化打印统计结果 ==========
def print_statistics(subset_results, total_count, total_all, subset_type="指定子集"):
    print(f"\n==================== {subset_type} 统计结果 ====================")

    # 打印每个指定子集的明细
    for subset_name, (count_dict, total) in subset_results.items():
        if total == 0:
            print(f"\n【{subset_name}】无有效头部标注数据")
            continue
        print(f"\n【{subset_name}】")
        print(f"  总头部目标数量：{total}")
        print(f"  不吸烟 + 戴安全帽 (0,0)：{count_dict[(0, 0)]} 个（占比：{count_dict[(0, 0)] / total * 100:.2f}%）")
        print(f"  不吸烟 + 未戴安全帽 (0,1)：{count_dict[(0, 1)]} 个（占比：{count_dict[(0, 1)] / total * 100:.2f}%）")
        print(f"  吸烟 + 戴安全帽 (1,0)：{count_dict[(1, 0)]} 个（占比：{count_dict[(1, 0)] / total * 100:.2f}%）")
        print(f"  吸烟 + 未戴安全帽 (1,1)：{count_dict[(1, 1)]} 个（占比：{count_dict[(1, 1)] / total * 100:.2f}%）")

    # 打印汇总结果
    if total_all == 0:
        print(f"\n【{subset_type} 汇总】无有效数据")
        return
    print(f"\n【{subset_type} 汇总】")
    print(f"  总头部目标数量：{total_all}")
    print(f"  不吸烟 + 戴安全帽 (0,0)：{total_count[(0, 0)]} 个（占比：{total_count[(0, 0)] / total_all * 100:.2f}%）")
    print(f"  不吸烟 + 未戴安全帽 (0,1)：{total_count[(0, 1)]} 个（占比：{total_count[(0, 1)] / total_all * 100:.2f}%）")
    print(f"  吸烟 + 戴安全帽 (1,0)：{total_count[(1, 0)]} 个（占比：{total_count[(1, 0)] / total_all * 100:.2f}%）")
    print(f"  吸烟 + 未戴安全帽 (1,1)：{total_count[(1, 1)]} 个（占比：{total_count[(1, 1)] / total_all * 100:.2f}%）")

    # 单属性汇总
    smoke_1 = total_count[(1, 0)] + total_count[(1, 1)]
    no_helmet_1 = total_count[(0, 1)] + total_count[(1, 1)]
    print(f"\n【{subset_type} 单属性汇总】")
    print(f"  吸烟(1)：{smoke_1} 个（占比：{smoke_1 / total_all * 100:.2f}%），不吸烟(0)：{total_all - smoke_1} 个")
    print(f"  未戴安全帽(1)：{no_helmet_1} 个（占比：{no_helmet_1 / total_all * 100:.2f}%），戴安全帽(0)：{total_all - no_helmet_1} 个")
    print("=" * 70)


# ========== 主函数（命令行调用） ==========
def main():
    parser = argparse.ArgumentParser(description="统计指定子集的吸烟/未戴安全帽属性分布")
    parser.add_argument("--root_dir", default=r"D:\1_Python\datasets\fire_security\labels_yolo+attr",
                        help="数据集根目录（包含所有子集目录，如train01、train03）")
    parser.add_argument("--train_subsets", nargs="+", default=["train01", "train03", "train04"],
                        help="要统计的训练集子集名称列表（如 --train_subsets train01 train03）")
    parser.add_argument("--val_subsets", nargs="+", default=["val01", "val03", "val04"],
                        help="要统计的验证集子集名称列表（如 --val_subsets val01 val05）")
    parser.add_argument("--ann_type", default="yolo",
                        help="标注类型：yolo/json（默认yolo）")

    args = parser.parse_args()

    # 1. 统计指定的训练集子集
    train_subsets, train_total_count, train_total_all = batch_count_specified_subsets(
        args.root_dir, args.train_subsets, args.ann_type
    )
    if train_subsets:
        print_statistics(train_subsets, train_total_count, train_total_all, "训练集")
    else:
        print("\n⚠️  未找到有效训练集子集！")

    # 2. 统计指定的验证集子集
    val_subsets, val_total_count, val_total_all = batch_count_specified_subsets(
        args.root_dir, args.val_subsets, args.ann_type
    )
    if val_subsets:
        print_statistics(val_subsets, val_total_count, val_total_all, "验证集")
    else:
        print("\n⚠️  未找到有效验证集子集！")

    # 3. 统计整体（训练+验证）
    if train_subsets and val_subsets:
        all_total_count = {
            key: train_total_count[key] + val_total_count[key]
            for key in train_total_count.keys()
        }
        all_total_all = train_total_all + val_total_all
        print_statistics({}, all_total_count, all_total_all, "整体数据集")


if __name__ == "__main__":
    main()
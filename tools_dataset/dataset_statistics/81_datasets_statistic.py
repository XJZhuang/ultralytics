#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：81_datasets_statistic.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/10 14:40 
@explain : 统计指定数据集的标签数量
'''

import os
from collections import defaultdict
import glob


def count_dataset(images_root, labels_root, class_names, specific_folders=None):
    """
    统计数据集指定子文件夹中的图片和标注信息，显示实际类别名称

    参数:
        images_root: 图片根目录
        labels_root: 标签根目录
        class_names: 类别名称列表，索引对应类别ID
        specific_folders: 要统计的特定子文件夹列表，如["train03", "train07"]
    """
    # 获取所有子文件夹
    all_subfolders = [f for f in os.listdir(images_root)
                      if os.path.isdir(os.path.join(images_root, f))]

    # 确定需要统计的子文件夹
    if specific_folders:
        subfolders = [f for f in specific_folders if f in all_subfolders]
        missing = [f for f in specific_folders if f not in all_subfolders]
        if missing:
            print(f"警告: 以下文件夹在图片目录中不存在，将被跳过: {', '.join(missing)}")
    else:
        subfolders = all_subfolders

    if not subfolders:
        print("没有找到可统计的文件夹")
        return {}

    # 存储所有文件夹的统计结果
    all_stats = {}

    # 遍历每个子文件夹
    for folder in subfolders:
        print(f"正在处理 {folder}...")

        # 初始化统计字典
        stats = {
            'total_images': 0,
            'total_instances': 0,
            'class_instances': defaultdict(int),  # 每个类别的实例数量
            'class_images': defaultdict(int)  # 每个类别出现的图片数
        }

        # 图片和标签文件夹路径
        img_folder = os.path.join(images_root, folder)
        label_folder = os.path.join(labels_root, folder)

        if not os.path.exists(label_folder):
            print(f"警告: 标签文件夹 {label_folder} 不存在，将跳过该文件夹")
            continue

        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(img_folder, ext), recursive=False))

        # 统计图片数量
        stats['total_images'] = len(image_files)

        # 处理每张图片对应的标签文件
        for img_path in image_files:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_folder, f"{img_name}.txt")

            # 记录当前图片中出现的类别
            classes_in_image = set()

            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # 处理每一行标注
                for line in lines:
                    line = line.strip()
                    if line:  # 跳过空行
                        parts = line.split()
                        if len(parts) >= 1:  # 确保至少有类别信息
                            try:
                                # 将类别ID转换为整数
                                class_id = int(parts[0])
                                # 获取实际类别名称
                                class_name = class_names[class_id]

                                stats['total_instances'] += 1
                                stats['class_instances'][class_name] += 1
                                classes_in_image.add(class_name)
                            except (ValueError, IndexError):
                                print(f"警告: 标签文件 {label_path} 中存在无效的类别ID: {parts[0]}")

            # 更新每个类别出现的图片数
            for cls in classes_in_image:
                stats['class_images'][cls] += 1

        # 保存当前文件夹的统计结果
        all_stats[folder] = stats
        print(f"完成处理 {folder}\n")

    return all_stats


def print_statistics(all_stats, class_names):
    """打印统计结果，按指定类别顺序显示"""
    if not all_stats:
        return

    # 首先打印每个文件夹的统计
    for folder, stats in all_stats.items():
        print(f"===== {folder} 统计结果 =====")
        print(f"图片总数: {stats['total_images']}")
        print(f"实例总数: {stats['total_instances']}")

        print("\n类别实例分布:")
        # 按预设的类别顺序显示
        for class_name in class_names:
            count = stats['class_instances'].get(class_name, 0)
            img_count = stats['class_images'].get(class_name, 0)
            print(f"  {class_name}: {count} 个实例，出现在 {img_count} 张图片中")

        print("\n" + "=" * 40 + "\n")

    # 计算并打印总计结果
    total_stats = {
        'total_images': 0,
        'total_instances': 0,
        'class_instances': defaultdict(int),
        'class_images': defaultdict(int)
    }

    for stats in all_stats.values():
        total_stats['total_images'] += stats['total_images']
        total_stats['total_instances'] += stats['total_instances']

        for cls, count in stats['class_instances'].items():
            total_stats['class_instances'][cls] += count

        for cls, count in stats['class_images'].items():
            total_stats['class_images'][cls] += count

    print("===== 所有指定文件夹总计结果 =====")
    print(f"总图片数: {total_stats['total_images']}")
    print(f"总实例数: {total_stats['total_instances']}")

    print("\n类别总实例分布:")
    # 按预设的类别顺序显示
    for class_name in class_names:
        count = total_stats['class_instances'].get(class_name, 0)
        img_count = total_stats['class_images'].get(class_name, 0)
        print(f"  {class_name}: {count} 个实例，出现在 {img_count} 张图片中")


if __name__ == "__main__":
    # 设置图片和标签的根目录
    IMAGES_ROOT = os.path.join(r"D:\1_Python\datasets\switchgear", "images")  # 图片根目录
    LABELS_ROOT = os.path.join(r"D:\1_Python\datasets\switchgear", "labels")  # 标签根目录

    # 定义类别名称，索引对应类别ID
    # 例如：class_names[0] = "light" 表示ID为0的类别是light
    CLASS_NAMES = [
        "light",
        "knob",
        "pointer",
        "switch",
        "breaker",
        "liquid"
    ]

    # 指定要统计的子文件夹，如["train03", "train07"] 若要统计所有文件夹，可设置为 None 或空列表 []
    SPECIFIC_FOLDERS = ["train01", "train02", "train03", "train04", "val01", "val02", "val03", "val04", ]
    # SPECIFIC_FOLDERS = ["train01", "train02", "train03", "train04", ]
    # SPECIFIC_FOLDERS = ["val01", "val02", "val03", "val04", ]

    # 检查目录是否存在
    if not os.path.exists(IMAGES_ROOT):
        print(f"错误: 图片根目录 {IMAGES_ROOT} 不存在")
        exit(1)

    if not os.path.exists(LABELS_ROOT):
        print(f"错误: 标签根目录 {LABELS_ROOT} 不存在")
        exit(1)

    # 统计数据集
    stats = count_dataset(IMAGES_ROOT, LABELS_ROOT, CLASS_NAMES, SPECIFIC_FOLDERS)

    # 打印统计结果
    print_statistics(stats, CLASS_NAMES)

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：yolov9-main 
@File    ：22_divide_datasets.py
@IDE     ：PyCharm 
@Author  ：zhuangxujun
@Date    ：2024/7/19 16:49 
@explain : 划分数据集为 训练集 和 验证集。（同时划分图片和标签）
'''

import os
import random
import shutil
from typing import List


def move_images(src_img_dir: str, dst_img_dir: str, val_ratio: float = 0.2) -> List[str]:
    """
    从源图片目录随机抽取指定比例的图片移动到目标图片目录

    参数:
        src_img_dir: 源图片目录（训练集图片所在路径）
        dst_img_dir: 目标图片目录（验证集图片目标路径）
        val_ratio: 验证集占比，默认0.2

    返回:
        移动到目标目录的图片文件名列表
    """
    # 检查源目录是否存在
    if not os.path.isdir(src_img_dir):
        raise NotADirectoryError(f"源图片目录不存在: {src_img_dir}")

    # 创建目标目录（若不存在）
    os.makedirs(dst_img_dir, exist_ok=True)

    # 获取源目录中所有文件
    all_files = os.listdir(src_img_dir)
    total_files = len(all_files)

    if total_files == 0:
        print(f"警告: 源图片目录 {src_img_dir} 中没有文件")
        return []

    # 计算需要抽取的文件数量
    pick_number = max(1, int(total_files * val_ratio))  # 确保至少抽取1个文件
    # 随机抽取文件
    selected_files = random.sample(all_files, pick_number)

    # 移动选中的文件
    moved_files = []
    for filename in selected_files:
        src_path = os.path.join(src_img_dir, filename)
        dst_path = os.path.join(dst_img_dir, filename)

        # 跳过目录，只处理文件
        if not os.path.isfile(src_path):
            print(f"跳过非文件: {filename}")
            continue

        try:
            shutil.move(src_path, dst_path)
            moved_files.append(filename)
            print(f"已移动图片: {filename}")
        except Exception as e:
            print(f"移动图片 {filename} 失败: {str(e)}")

    return moved_files


def move_labels(file_list, train_label_dir, val_label_dir):
    """
    根据图片文件列表移动对应的标签文件

    参数:
        file_list: 图片文件名列表（如['img1.jpg', 'img2.png']）
        train_label_dir: 训练集标签文件所在目录
        val_label_dir: 验证集标签文件目标目录
    """

    # 创建目标目录（若不存在）
    os.makedirs(val_label_dir, exist_ok=True)

    # 支持的图片后缀（统一转为小写处理）
    IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}

    for img_filename in file_list:
        # 分离文件名和后缀（自动处理不同长度的后缀）
        img_name, img_ext = os.path.splitext(img_filename)
        # 统一后缀为小写，增强兼容性
        img_ext_lower = img_ext.lower()

        # 判断是否为支持的图片格式
        if img_ext_lower in IMAGE_SUFFIXES:
            # 构建标签文件路径（使用os.path.join增强跨平台性）
            label_filename = f"{img_name}.txt"
            src_label_path = os.path.join(train_label_dir, label_filename)

            # 检查标签文件是否存在
            if os.path.exists(src_label_path):
                try:
                    # 移动标签文件到目标目录
                    dst_label_path = os.path.join(val_label_dir, label_filename)
                    shutil.move(src_label_path, dst_label_path)
                    print(f"成功移动标签: {label_filename} (对应图片: {img_filename})")
                except Exception as e:
                    print(f"移动标签失败 {label_filename}: {str(e)}")
            else:
                print(f"未找到对应标签文件: {label_filename} (对应图片: {img_filename})")


def main():
    # 配置参数
    # root_path = r'D:\1_Python\datasets\switchgear'
    # root_path = r'D:\1_Python\datasets\knobs'
    root_path = r'D:\1_Python\datasets\liquids'
    # root_path = r'D:\1_Python\datasets\fire_security'
    # group_index = ['06']
    # group_index = ['01', '03', '04']  # 分组索引
    group_index = ['02']  # 分组索引
    val_ratio = 0.2  # 验证集比例

    for group in group_index:
        print(f"\n===== 处理分组: {group} =====")

        # 构建图片目录路径
        src_img_dir = os.path.join(root_path, "images", "train"+group)
        dst_img_dir = os.path.join(root_path, "images", "val"+group)

        # 移动图片并获取移动的图片列表
        moved_images = move_images(src_img_dir, dst_img_dir, val_ratio)
        if not moved_images:
            print(f"分组 {group} 没有移动任何图片，跳过标签处理")
            continue

        # 构建标签目录路径
        src_label_dir = os.path.join(root_path, "labels", "train"+group)
        dst_label_dir = os.path.join(root_path, "labels", "val"+group)

        # 移动对应的标签文件
        move_labels(moved_images, src_label_dir, dst_label_dir)

    print("\n===== 所有操作完成 =====")


if __name__ == '__main__':
    main()


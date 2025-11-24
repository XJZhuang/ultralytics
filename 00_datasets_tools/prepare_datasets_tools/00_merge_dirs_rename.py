#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：00_merge_dirs_rename.py
@Author  ：DELL
@Date    ：2025/9/30 9:54 
@explain : 合并文件夹中的图片，并重命名
'''
import os
import shutil

# 源文件夹路径
base_dir = r"D:\1_Python\datasets\switchgear\images"

# 目标文件夹路径
target_dir = os.path.join(base_dir, "train03")

# 创建目标文件夹（如果不存在）
os.makedirs(target_dir, exist_ok=True)

# 需要处理的文件夹范围：train03-1到train03-3
for folder_num in range(1, 4):
    # 构建源文件夹路径
    folder_name = f"train03-{folder_num}"
    source_folder = os.path.join(base_dir, folder_name)

    # 检查源文件夹是否存在
    if not os.path.exists(source_folder) or not os.path.isdir(source_folder):
        print(f"警告：文件夹 {source_folder} 不存在，已跳过")
        continue

    # 获取源文件夹中的所有文件并排序
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    files.sort()  # 排序确保序号稳定

    # 遍历文件并复制重命名
    for file_idx, filename in enumerate(files, start=1):  # 序号从1开始
        source_path = os.path.join(source_folder, filename)

        # 获取文件扩展名
        _, ext = os.path.splitext(filename)

        # 构建新文件名：output_3_1_{序号} 格式
        new_filename = f"output_3_{folder_num}_{file_idx}{ext}"
        target_path = os.path.join(target_dir, new_filename)

        # 复制文件
        shutil.copy2(source_path, target_path)
        print(f"已复制: {source_path} -> {target_path}")

print("文件复制和重命名操作完成")

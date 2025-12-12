#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：00_merge_dirs.py
@Author  ：DELL
@Date    ：2025/9/30 9:45 
@explain : 合并多个文件夹里的文件
'''
import os
import shutil

# 源文件夹路径
base_dir = r"D:\1_Python\datasets\switchgear\images"

# 目标文件夹路径
target_dir = os.path.join(base_dir, "train04")

# 创建目标文件夹（如果不存在）
os.makedirs(target_dir, exist_ok=True)

# 需要处理的文件夹范围：train04-01到train04-22
for i in range(1, 23):
    folder_name = f"train04-{i}"
    source_folder = os.path.join(base_dir, folder_name)

    # 检查源文件夹是否存在
    if not os.path.exists(source_folder) or not os.path.isdir(source_folder):
        print(f"警告：文件夹 {source_folder} 不存在，已跳过")
        continue

    # 获取源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)

        # 只处理文件，不处理子文件夹
        if os.path.isfile(source_path):
            target_path = os.path.join(target_dir, filename)

            # 检查目标文件是否已存在
            if os.path.exists(target_path):
                print(f"错误：文件 {filename} 已存在，已跳过")
                continue

            # 复制文件
            shutil.copy2(source_path, target_path)
            print(f"已复制: {source_path} -> {target_path}")

print("文件复制操作完成")

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：71_knob_gt.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/14 19:24 
@explain : 
'''

import os
import csv


def get_image_files(base_dir, specific_folders, image_extensions=['.jpg', '.jpeg', '.png', '.gif', '.bmp']):
    """
    获取指定文件夹中的所有图片文件

    参数:
        base_dir: 基础目录
        specific_folders: 需要读取的文件夹列表
        image_extensions: 图片文件扩展名列表

    返回:
        图片文件路径列表
    """
    image_files = []

    # 遍历每个指定的文件夹
    for folder in specific_folders:
        # 构建完整的文件夹路径
        folder_path = os.path.join(base_dir, folder)

        # 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            print(f"警告: 文件夹 {folder_path} 不存在，已跳过")
            continue

        # 遍历文件夹中的所有文件
        for file in os.listdir(folder_path):
            # 检查文件扩展名是否为图片格式
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_path = f"{folder}/{file}"
                image_files.append(image_path)
            else:
                print(f"error-->{file.lower()}")

    return image_files


def save_to_csv(image_files, output_file, header='image_name'):
    """
    将图片文件名保存到CSV文件

    参数:
        image_files: 包含文件夹的图片路径列表
        output_file: 输出CSV文件路径
        header: CSV文件表头
    """
    # 检查文件是否存在
    file_exists = os.path.isfile(output_file)

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([header])
            print(f"创建新文件 {output_file} 并写入表头")
        else:
            print(f"文件 {output_file} 已存在，将追加 {len(image_files)} 条记录")
        # 写入每个图片路径
        for file in image_files:
            writer.writerow([file])


if __name__ == "__main__":
    # 基础目录
    base_directory = r"D:\1_Python\datasets\knobs\images"

    # 在这里自定义指定需要读取的文件夹
    # specific_folders = ["train01", "train04", ]
    specific_folders = ["val01", "val04", ]

    # 输出CSV文件路径
    output_csv = "val_gt.csv"

    # 获取所有图片文件
    print(f"正在读取 {base_directory} 目录下的 {specific_folders} 文件夹中的图片...")
    image_files = get_image_files(base_directory, specific_folders)

    if image_files:
        # 保存到CSV
        save_to_csv(image_files, output_csv)
        print(f"成功将 {len(image_files)} 个图片路径（包含文件夹）保存到 {output_csv}")
    else:
        print("未找到任何图片文件")

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：15_order_class_id.py
@Author  ：DELL
@Date    ：2025/10/9 18:43 
@explain : 使用labelimg对图片标注后的txt文件，里面每一行代表一个物体的标注信息，顺序时标注的顺序。
使用python代码对每一行的信息进行重新排序，根据类型ID降序，ID相同时再根据位置排序。代码的输入是txt文件的路径(对文件夹内的路径做批量处理)。
'''

import os
import sys

def sort_labelimg_txt(txt_path):
    """
    对单个LabelImg生成的txt标注文件进行排序
    排序规则：先按类型ID降序，ID相同则按位置排序
    返回值：如果内容有变化返回True，否则返回False
    """
    # 检查文件是否存在
    if not os.path.exists(txt_path):
        print(f"错误：文件 {txt_path} 不存在，已跳过")
        return False

    # 读取原始内容
    with open(txt_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()
        original_content = ''.join(original_lines)  # 保存原始内容用于比较

    # 解析每一行并存储为元组
    annotations = []
    for line in original_lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            print(f"警告：{txt_path} 中存在无效格式 - {line}，已跳过")
            continue
        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 存储解析结果，用于排序
            annotations.append((
                class_id,  # 使用负号实现降序排序
                x_center,  # 位置排序：先按x中心
                y_center,  # 再按y中心
                width,  # 然后按宽度
                height,  # 最后按高度
                line  # 原始行内容
            ))
        except ValueError:
            print(f"警告：{txt_path} 中无法解析 - {line}，已跳过")
            continue

    # 排序标注
    annotations.sort()

    # 提取排序后的行
    sorted_lines = [anno[-1] + '\n' for anno in annotations]
    sorted_content = ''.join(sorted_lines)  # 排序后的内容

    # 比较排序前后内容是否有变化
    if sorted_content == original_content:
        return False  # 无变化

    # 内容有变化，写入新内容
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.writelines(sorted_lines)

    return True  # 有变化

def batch_sort_folder(folder_path):
    """
    批量处理文件夹中所有的txt文件（跳过classes.txt）
    只显示排序后内容有变化的文件
    """
    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：{folder_path} 不是有效的文件夹路径")
        return

    # 统计变量
    processed_count = 0
    changed_count = 0
    skipped_count = 0

    print("开始处理文件...\n")

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 跳过classes.txt文件
        if filename.lower() == "classes.txt":
            print(f"已跳过：{os.path.join(folder_path, filename)}")
            skipped_count += 1
            continue

        # 只处理其他txt文件
        # 只处理txt文件
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            # 确保是文件而不是文件夹
            if os.path.isfile(file_path):
                processed_count += 1
                # 排序并检查是否有变化
                has_changed = sort_labelimg_txt(file_path)
                if has_changed:
                    changed_count += 1
                    print(f"已更新（排序有变化）：{file_path}")

    print(f"\n批量处理完成")
    print(f"总计处理：{processed_count} 个txt文件")
    print(f"排序后有变化：{changed_count} 个文件")
    print(f"跳过 classes.txt：{skipped_count} 个文件")


if __name__ == "__main__":
    txt_file_path = r"D:\1_Python\datasets\switchgear\labels\train03"
    batch_sort_folder(txt_file_path)

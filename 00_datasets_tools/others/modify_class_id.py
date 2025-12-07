#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：modify_class_id.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/25 13:43 
@explain : 
'''

import os
import argparse


def modify_yolo_labels(input_dir, output_dir=None):
    """
    批量修改YOLO格式标注文件中的类别ID

    Args:
        input_dir: 输入文件夹路径，包含原始的.txt标注文件
        output_dir: 输出文件夹路径，用于保存修改后的标注文件
                   如果为None，则直接在原文件上修改
    """

    # 如果输出目录不存在且不是None，则创建输出目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_dir, filename)

            # 确定输出路径
            if output_dir:
                output_path = os.path.join(output_dir, filename)
            else:
                output_path = input_path

            # 读取并修改标注内容
            with open(input_path, 'r', encoding='utf-8') as f_in, \
                    open(output_path, 'w', encoding='utf-8') as f_out:

                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue

                    # 分割YOLO格式的标注行：class_id x_center y_center width height
                    parts = line.split()
                    if len(parts) != 5:
                        # 格式不正确的行直接保留
                        f_out.write(line + '\n')
                        continue

                    class_id = int(parts[0])
                    coordinates = parts[1:]

                    # 根据需求修改类别ID
                    if class_id == 3:
                        # 删除ID为3的标签（不写入文件）
                        continue
                    elif class_id == 1:
                        # 将ID为1的标签改为3
                        new_line = f"3 {' '.join(coordinates)}\n"
                        f_out.write(new_line)
                    else:
                        # 其他标签保持不变
                        f_out.write(line + '\n')

            print(f"已处理: {filename}")

    print(f"\n处理完成！{'结果已保存到输出目录' if output_dir else '已直接修改原文件'}")


def main():
    input_dir = r"D:\1_Python\datasets\fire_security\labels\train01"
    output_dir = r"D:\1_Python\datasets\fire_security\labels\train01-1"
    # 执行修改
    modify_yolo_labels(input_dir, output_dir)


if __name__ == "__main__":
    main()
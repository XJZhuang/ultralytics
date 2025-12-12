# -*- coding: utf-8 -*-
'''
@Project:   ultralytics-main
@File:      3_undivided_val.py
@Author:    123
@Date:      2025/1/20 15:34
@Explain:   按班组划分的验证集恢复到训练集，重新标注，添加图片，最后再划分
'''

import os
import shutil


if __name__ == '__main__':
    # 源文件夹
    # source_dir = r'D:\1_Python\datasets\1_detect_electricity_yantian\images\val'
    # source_dir = r'D:\1_Python\datasets\2_detect_electricity_futian\images\val'
    source_dir = r'D:\1_Python\datasets\0_detect_safety\images\val'
    target_prefix = r'D:\1_Python\datasets\0_detect_safety\images\train_'

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_dir):
        # 检查文件是否是图片（可以根据扩展名来判断，例如.jpg, .png等）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # 根据文件名前缀来确定目标文件夹
            group_name = filename.split('_')[0]  # 假设文件名以类别前缀_开头
            target_dir = f'{target_prefix}{group_name}'

            # 创建目标文件夹（如果不存在）
            # os.makedirs(target_dir, exist_ok=True)
            if not os.path.exists(target_dir):
                print(target_dir)
                raise Exception

            # 构建完整的源文件路径和目标文件路径
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)

            # 移动文件到目标文件夹
            shutil.move(source_file, target_file)
            print(f'Moved {source_file} to {target_file}')

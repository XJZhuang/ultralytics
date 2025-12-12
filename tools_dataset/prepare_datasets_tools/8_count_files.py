#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-main 
@File    ：8_count_files.py
@IDE     ：PyCharm 
@Author  ：zhuangxujun
@Date    ：2024/12/6 14:14 
@explain : 统计每个文件夹下的文件数量
'''

import os


def count_files_in_directory(directory):
    # 初始化文件计数器
    file_count = 0
    # 遍历目录内容
    for root, dirs, files in os.walk(directory):
        # 统计当前文件夹下的文件数量
        file_count += len(files)
        # 打印当前文件夹路径和文件数量
        print(f"Directory: {root}, File Count: {len(files)}")
    # 返回总文件数量（如果你需要的话）
    return file_count


if __name__ == '__main__':
    # 指定要统计的目录路径
    directory_to_count = r'D:\1_Python\datasets\2_detect_electricity_futian\images'

    # 调用函数并获取总文件数量
    total_files = count_files_in_directory(directory_to_count)
    print(f"Total files in the directory and its subdirectories: {total_files}")
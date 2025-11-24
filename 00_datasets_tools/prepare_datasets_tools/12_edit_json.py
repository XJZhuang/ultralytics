#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-main 
@File    ：12_edit_json.py
@IDE     ：PyCharm 
@Author  ：zhuangxujun
@Date    ：2024/11/25 10:58 
@explain : 编辑txt或json文件
'''

import os
import json


def modify_image_path(json_file):
    """修改JSON文件中的imagePath，将train1改为train01"""
    try:
        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 检查是否有imagePath字段
        if "imagePath" in data:
            original_path = data["imagePath"]
            # 替换train1为train01
            modified_path = original_path.replace("train1", "train01")

            # 如果路径有变化才进行修改
            if modified_path != original_path:
                data["imagePath"] = modified_path

                # 写回修改后的内容
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"已修改: {json_file}")
            else:
                print(f"无需修改: {json_file}")
        else:
            print(f"警告: {json_file} 中未找到imagePath字段")

    except Exception as e:
        print(f"处理 {json_file} 时出错: {str(e)}")


def process_json_files(directory):
    """处理指定目录下的所有JSON文件"""
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误: 目录 {directory} 不存在")
        return

    # 遍历目录下的所有文件
    for filename in os.listdir(directory):
        # 只处理JSON文件
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            modify_image_path(file_path)


if __name__ == "__main__":
    # 指定JSON文件所在目录
    json_directory = r"D:\1_Python\datasets\knobs\labels_json\train1"

    # 处理目录下的所有JSON文件
    print(f"开始处理目录: {json_directory}")
    process_json_files(json_directory)
    print("处理完成")

# -*- coding: utf-8 -*-
"""
@Project:   ultralytics-main
@File:      93_generate_socket.py
@Author:    123
@Date:      2024/12/6 14:14
@Explain:   生成相同坐标位置的JSON文件
"""

import os
import json

# 指定图片文件夹路径
image_folder = r'D:\1_Python\datasets\5_detect_electricity_shenshan\images\train_xm'
# 指定JSON文件夹路径，用于存储生成的JSON文件
json_folder = r'D:\1_Python\datasets\5_detect_electricity_shenshan\labels_json\train_xm'

# 确保JSON文件夹存在
os.makedirs(json_folder, exist_ok=True)

# 遍历图片文件夹中的所有文件
for filename in os.listdir(image_folder):
    # 检查文件是否为图片（这里假设图片文件的后缀为.jpg或.png）
    if filename.endswith(('.jpg', '.png')):
        # 构造图片的完整路径
        image_path = os.path.join(image_folder, filename)
        # 构造JSON文件的完整路径
        json_filename = os.path.splitext(filename)[0] + '.json'
        json_path = os.path.join(json_folder, json_filename)

        # 构造JSON内容
        json_content = {
            "version": "5.5.0",
            "flags": {},
            "shapes": [
                {
                    "label": "socket",
                    "points": [
                        [
                            2192.5641025641025,
                            1377.5213675213674
                        ],
                        [
                            2342.136752136752,
                            1589.4871794871794
                        ]
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None
                }
            ],
            "imagePath": os.path.relpath(image_path, json_folder),
            "imageData": None,
            "imageHeight": 1620,
            "imageWidth": 2880
        }

        # 写入JSON文件
        with open(json_path, 'w') as json_file:
            json.dump(json_content, json_file, indent=2)

print("JSON files have been created successfully.")
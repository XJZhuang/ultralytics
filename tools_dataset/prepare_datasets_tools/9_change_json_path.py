#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-main 
@File    ：9_change_json_path.py
@IDE     ：PyCharm 
@Author  ：zhuangxujun
@Date    ：2024/12/4 10:02 
@explain : 
'''

import os
import json


# current_dir = os.getcwd()
current_dir = r"D:\1_Python\datasets\2_detect_electricity_futian\labels_json\train_ml"
for file_name in os.listdir(current_dir):
    if file_name.endswith('.json'):
        file_path = os.path.join(current_dir, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
            if "imagePath" in data:
                old_path = data["imagePath"]
                filename = old_path.split("\\")[-1]
                new_path = f"..\\..\\images\\train_ytg\\{filename}"
                data["imagePath"] = new_path
            with open(file_path, 'w', encoding='utf-8') as json_file:
                json.dump(data, json_file, indent=4, ensure_ascii=False)
            print(f"更新路径: {file_name}")
        except Exception as e:
            print(f"错误:{file_name}: {e}")

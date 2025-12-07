#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：00_import_test.py
@Author  ：zhuangxujun
@Date    ：2025-9-25 13:51 
@explain : 
'''

import sys
import os

# 添加项目路径到Python搜索路径
# 假设ultralytics项目在'ultralytics_project'文件夹下
ultralytics_path = os.path.abspath('../ultralytics-switchgear')
if ultralytics_path not in sys.path:
    sys.path.append(ultralytics_path)

# 导入ultralytics相关模块
try:
    from ultralytics import YOLO  # 根据实际模块结构调整
except ImportError as e:
    print(f"导入ultralytics失败: {e}")
    sys.exit(1)

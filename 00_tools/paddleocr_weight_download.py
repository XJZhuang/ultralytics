#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：paddleocr_weight_download.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/20 15:24 
@explain : 
'''

from paddlex import create_model
model = create_model("PP-OCRv5_mobile_det")   # 或 PP-OCRv5_mobile_det
print(model.model_dir)        # 第一次运行会自动下载并解压到
model = create_model("PP-OCRv5_mobile_rec")   # 或 PP-OCRv5_mobile_det
print(model.model_dir)        # 第一次运行会自动下载并解压到
# ~/.paddlex/official_models/
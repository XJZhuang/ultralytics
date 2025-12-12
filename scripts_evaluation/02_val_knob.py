#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：01_val_knob.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/14 10:58
@explain : 
'''

import os
import sys

# 添加项目路径到Python搜索路径
ultralytics_path = os.path.abspath('../ultralytics-switchgear')
if ultralytics_path not in sys.path:
    sys.path.append(ultralytics_path)

try:  # 导入ultralytics相关模块
    from ultralytics import YOLO  # 根据实际模块结构调整
    from ultralytics.utils import LOGGER
except ImportError as e:
    print(f"导入ultralytics失败: {e}")
    sys.exit(1)

if __name__ == '__main__':

    # Load a model
    model = YOLO(
        "/home/zxj/ultralytics-switchgear/01_train_scripts/knob/train/weights/best.pt",
    )

    # Customize validation settings
    metrics = model.val(
        data="../ultralytics/cfg/datasets/knob-pose.yaml",
        imgsz=640,
        batch=5,
        # conf=0.25,
        # iou=0.6,
        device=0,
        save_json=True,
        project="knob",
        name="val_output",
        verbose=True,
        visualize=True,
        task_type="pose_knob",
    )

    LOGGER.info("_______________")
    LOGGER.info(f"{metrics.box.map}")
    LOGGER.info(f"{metrics.box.map50}")
    LOGGER.info(f"{metrics.box.map75}")
    LOGGER.info(f"{metrics.box.maps}")
    LOGGER.info("_______________")



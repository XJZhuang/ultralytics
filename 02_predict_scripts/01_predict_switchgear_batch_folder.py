#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：01_predict_switchgear_batch_folder.py
@Author  ：zhuangxujun
@Date    ：2025-9-19 16:48 
@explain : 对多个文件夹下的多张图片进行推理
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
    # 加载预训练模型
    model = YOLO(r"../01_train_scripts/switchgear/train4/weights/best.pt")

    # 定义需要进行推理的所有文件夹路径
    source_folders = [
        # r"/home/zxj/datasets/switchgear/images/val01",
        # r"/home/zxj/datasets/switchgear/images/val02",
        # r"/home/zxj/datasets/switchgear/images/val03",
        # r"/home/zxj/datasets/switchgear/images/val04",
        r"/home/zxj/datasets/switchgear/images/val05",
    ]

    # 推理参数配置
    project_name = "switchgear4"  # 结果保存的主项目文件夹
    save_results = True  # 是否保存结果

    # 遍历每个文件夹进行推理
    for folder in source_folders:
        # 检查文件夹是否存在
        if not os.path.exists(folder):
            print(f"警告: 文件夹 {folder} 不存在，已跳过")
            continue

        # 获取文件夹名称作为子目录，用于区分不同文件夹的结果
        folder_name = os.path.basename(folder)
        print(f"正在处理文件夹: {folder_name}")

        # 运行推理
        results = model(
            folder,
            save=save_results,
            stream=True,
            project=project_name,
            name=folder_name  # 每个文件夹的结果保存在独立的子目录中
        )

        # 可以在这里处理推理结果
        for result in results:
            boxes = result.boxes  # 边界框信息
            # 如需处理结果，可以在这里添加代码
            # 例如: print(f"检测到 {len(boxes)} 个目标 in {result.path}")

        print(f"文件夹 {folder_name} 处理完成\n")

    print("所有文件夹推理完成！")

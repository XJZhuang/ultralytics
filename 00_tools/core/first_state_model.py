#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：first_state_model.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/14 16:43 
@explain : 
'''

import os
import cv2
from pathlib import Path
from ultralytics import YOLO


# 示例处理函数 - 可以根据需要扩展
def light_detection(detection_info, cropped_img):
    """处理灯光类别的检测结果"""
    print(f"处理灯光检测: {detection_info['class_name']} 置信度: {detection_info['confidence']:.2f}")
    # 这里可以添加灯光类别的特定处理逻辑
    # 例如：灯光状态分析、亮度检测等


def knob_detection(detection_info, cropped_img):
    """处理旋钮类别的检测结果"""
    print(f"处理旋钮检测: {detection_info['class_name']} 置信度: {detection_info['confidence']:.2f}")
    # 这里可以添加旋钮类别的特定处理逻辑
    # 例如：旋钮位置识别、状态判断等


def default_detection(detection_info, cropped_img):
    """默认处理函数，用于处理没有特定处理函数的类别"""
    print(f"处理{detection_info['class_name']}检测: 置信度: {detection_info['confidence']:.2f}")
    # 默认处理逻辑


def run_first_state(source):
    """
    运行YOLO模型进行目标检测，根据类别调用相应处理函数，并返回所有检测信息

    参数:
        source: 输入图像/视频的路径，可以是目录、单张图片或视频文件

    返回值:
        list: 包含所有检测框信息的列表，每个元素为字典，包含:
            - image_path: 原始图像路径
            - class_name: 类别名称
            - class_id: 类别ID
            - confidence: 置信度
            - bbox: 边界框坐标 (x1, y1, x2, y2)
            - cropped_path: 裁剪图像保存路径
    """
    # 类别到处理函数的映射表，可根据需要扩展
    category_handlers = {
        "light": light_detection,
        "knob": knob_detection
        # 可以添加更多类别及其对应的处理函数
    }

    # 加载预训练模型
    model_path = Path("../01_train_scripts/switchgear/train4/weights/best.pt")
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = YOLO(model_path)

    # 定义输入源和输出目录
    output_dir = Path("cut_images")
    output_dir.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在

    # 存储所有检测结果
    detection_results = []

    # 运行推理
    results = model(source, save=True, stream=True, project="output")  # 生成Results对象的生成器

    for result in results:
        img = result.orig_img  # 获取原始图像
        boxes = result.boxes  # 处理每个检测到的边界框
        img_path = Path(result.path)  # 获取原始图像的路径
        img_name = img_path.name  # 获取文件名

        # 处理每个检测框
        for i, box in enumerate(boxes):
            # 获取类别信息
            class_id = int(box.cls)
            class_name = model.names[class_id]
            confidence = float(box.conf)  # 置信度

            # 获取边界框坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # 计算10%的边界冗余
            width = x2 - x1
            height = y2 - y1
            redundancy_w = width * 0.1
            redundancy_h = height * 0.1

            # 计算添加冗余后的新边界框，确保不超出图像范围
            new_x1 = max(0, x1 - redundancy_w)
            new_y1 = max(0, y1 - redundancy_h)
            new_x2 = min(img.shape[1], x2 + redundancy_w)
            new_y2 = min(img.shape[0], y2 + redundancy_h)

            # 转换为整数坐标
            new_x1, new_y1, new_x2, new_y2 = map(int, [new_x1, new_y1, new_x2, new_y2])

            # 截取图像
            cropped_img = img[new_y1:new_y2, new_x1:new_x2]

            # 生成输出文件名和路径
            base_name, ext = os.path.splitext(img_name)
            output_filename = f"{base_name}_{class_name}_{i}{ext}"
            output_path = output_dir / output_filename

            # 保存截取的图像
            try:
                cv2.imwrite(str(output_path), cropped_img)
                print(f"已保存截取的{class_name}图像到: {output_path}")
            except Exception as e:
                print(f"保存图像时出错 {output_path}: {str(e)}")
                continue

            # 记录检测信息
            detection_info = {
                "image_path": str(img_path),
                "class_name": class_name,
                "class_id": class_id,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),  # 原始边界框
                "expanded_bbox": (new_x1, new_y1, new_x2, new_y2),  # 扩展后的边界框
                "cropped_path": str(output_path)
            }
            detection_results.append(detection_info)

            # 根据类别调用相应的处理函数
            handler = category_handlers.get(class_name, default_detection)
            handler(detection_info, cropped_img)

    return detection_results


if __name__ == '__main__':

    run_first_state("test_images/1_1742957720126_1.jpeg")

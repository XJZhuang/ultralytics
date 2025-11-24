#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：01_cut_images.py
@Author  ：zhuangxujun
@Date    ：2025-9-22 18:34 
@explain : ROI（通过模型的推理框）
'''
import os
import cv2

from ultralytics import YOLO

if __name__ == '__main__':
    # Load a pretrained YOLO11n model
    model = YOLO(r"../01_train_scripts/switchgear/train/weights/best.pt")

    # Define path to directory containing images and videos for inference
    # source = r"/home/zxj/datasets/switchgear/images/val1"
    # source = r"../datasets/knobs/images/val1"
    source = r"D:\1_Python\datasets\switchgear\knobs\images\train_origin"
    # 创建输出目录用于保存截取的图片
    output_dir = r"D:\1_Python\datasets\switchgear\knobs\images\train1"
    os.makedirs(output_dir, exist_ok=True)

    # Run inference on the source
    # results = model(source, save=True, stream=True, project="switchgear")  # generator of Results objects
    results = model(source, save=True, stream=True, project="knob_pose")  # generator of Results objects

    # for result in results:
    #     box = result.boxes
    for result in results:
        img = result.orig_img  # 获取原始图像
        boxes = result.boxes  # 处理每个检测到的边界框
        img_path = result.path  # 获取原始图像的文件名

        img_name = os.path.basename(img_path)

        for i, box in enumerate(boxes):
            # 获取类别ID和类别名称
            class_id = int(box.cls)
            class_name = model.names[class_id]

            # 只处理类别为"knob"的目标
            if class_name == "knob":
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

                # 生成输出文件名
                base_name, ext = os.path.splitext(img_name)
                output_filename = f"{base_name}_knob_{i}{ext}"
                output_path = os.path.join(output_dir, output_filename)

                # 保存截取的图像
                cv2.imwrite(output_path, cropped_img)
                print(f"已保存截取的knob图像到: {output_path}")

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：02_cut_images_by_label.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/10 11:05 
@explain : 通过 标注文件 切割
'''

import os
import cv2
import shutil
from pathlib import Path


def create_directory(path):
    """创建目录，如果已存在则不做操作"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")
    else:
        print(f"目录已存在: {path}")


def parse_yolo_label(label_path):
    """解析YOLO格式的标签文件"""
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            # YOLO格式: class_id x_center y_center width height
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append((class_id, x_center, y_center, width, height))
            except ValueError:
                continue
    return labels


def crop_objects_from_image(image_path, labels, target_class, output_dir, existing_filenames):
    """从图像中裁剪出指定类别的物体并保存，每个图片中的物体从1开始计数"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return 0  # 返回本图处理的数量

    img_height, img_width = image.shape[:2]
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    # 检查是否同名
    if image_name in existing_filenames:
        print(f"警告: 检测到同名文件 '{image_name}'，可能会被覆盖")
        return 0

    # 将文件名加入已存在列表
    existing_filenames.add(image_name)

    # 筛选目标类别
    target_labels = [label for label in labels if label[0] == target_class]
    total_cropped = len(target_labels)

    if total_cropped == 0:
        return 0  # 没有目标类别，返回0

    # 处理每个目标，从1开始计数
    for obj_idx, (class_id, x_center, y_center, width, height) in enumerate(target_labels, 1):
        # 计算原始边界框坐标
        x_center_px = x_center * img_width
        y_center_px = y_center * img_height
        width_px = width * img_width
        height_px = height * img_height

        # 计算边界框左上角和右下角坐标
        x1 = int(x_center_px - width_px / 2)
        y1 = int(y_center_px - height_px / 2)
        x2 = int(x_center_px + width_px / 2)
        y2 = int(y_center_px + height_px / 2)

        # 计算10%的冗余量
        w_pad = int(width_px * 0.1)
        h_pad = int(height_px * 0.1)

        # 扩充边界框，确保不超出图像范围
        x1_pad = max(0, x1 - w_pad)
        y1_pad = max(0, y1 - h_pad)
        x2_pad = min(img_width, x2 + w_pad)
        y2_pad = min(img_height, y2 + h_pad)

        # 裁剪区域
        cropped = image[y1_pad:y2_pad, x1_pad:x2_pad]

        # 生成输出文件名 - 每个图片中的物体从1开始计数
        output_filename = f"{image_name}_1_{obj_idx}.jpg"

        output_path = os.path.join(output_dir, output_filename)

        # 保存裁剪后的图像
        cv2.imwrite(output_path, cropped)
        print(f"保存裁剪图像: {output_path}")

    return total_cropped


def process_selected_folders(images_root, labels_root, selected_folders, target_class, output_dir):
    """处理选中的文件夹"""
    # 创建输出目录
    create_directory(output_dir)

    total_count = 0  # 用于统计总处理数量
    existing_filenames = set()  # 用于跟踪已生成的文件名

    # 处理每个选中的文件夹
    for folder in selected_folders:
        print(f"\n处理文件夹: {folder}")

        # 构建图像和标签文件夹路径
        img_folder = os.path.join(images_root, folder)
        label_folder = os.path.join(labels_root, folder)

        # 检查路径是否存在
        if not os.path.isdir(img_folder):
            print(f"图像文件夹不存在: {img_folder}")
            continue
        if not os.path.isdir(label_folder):
            print(f"标签文件夹不存在: {label_folder}")
            continue

        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for file in os.listdir(img_folder):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)

        if not image_files:
            print(f"文件夹 {folder} 中没有找到图像文件")
            continue

        # 处理每个图像
        for img_file in image_files:
            img_path = os.path.join(img_folder, img_file)
            # 构建对应的标签文件路径
            img_name = os.path.splitext(img_file)[0]
            label_file = f"{img_name}.txt"
            label_path = os.path.join(label_folder, label_file)

            # 解析标签
            labels = parse_yolo_label(label_path)
            if not labels:
                continue  # 没有标签，跳过

            # 裁剪并保存目标物体，每个图片内从1开始计数
            img_count = crop_objects_from_image(
                img_path, labels, target_class, output_dir, existing_filenames
            )
            total_count += img_count

    print(f"\n处理完成，共保存 {total_count} 个裁剪图像")


def main():
    """
      0: light  # 指示灯
      1: knob   # 旋钮
      2: pointer  # 仪表盘指针
      3: switch # 压板开关
      4: breaker  # 断路器
      5: liquid  # 液位表
    """
    # 配置参数 - 可根据需要修改
    images_root = os.path.join(r"D:\1_Python\datasets\switchgear", "images")  # images根目录
    labels_root = os.path.join(r"D:\1_Python\datasets\switchgear", "labels")  # labels根目录
    selected_folders = ["train04", "train05", "val04", "val05"]  # 选择要处理的子文件夹
    # selected_folders = ["train01", "train02", "train03", "train04", "val01", "val02", "val03", "val04"]  # 选择要处理的子文件夹
    target_class = 5  # 要筛选的类别序号
    # output_dir = os.path.join(r"D:\1_Python\datasets\knobs", "images", selected_folders[0])  # 裁剪图像保存路径
    # output_dir = os.path.join(r"D:\1_Python\datasets\switchgear", "images", "cut_liquid")  # 裁剪图像保存路径
    output_dir = os.path.join(r"D:\1_Python\datasets\liquid", "images")  # 裁剪图像保存路径

    # 执行处理
    process_selected_folders(images_root, labels_root, selected_folders, target_class, output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：rename_images.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/12 9:16 
@explain : 
'''

import os
import shutil
import json
import re
from PIL import Image


def rename_files(input_dir):
    """
    批量重命名图片：提取时间、变电站编号、油位计编号，避免重复命名
    """
    # 创建备份文件夹（存储原文件，防止误操作）
    backup_dir = os.path.join(input_dir, "rename_backup")
    os.makedirs(backup_dir, exist_ok=True)

    # 正则表达式：提取关键信息
    # 时间：14位数字（年月日时分秒，如20251111143534）
    time_pattern = re.compile(r'(\d{14})')
    # 变电站编号：从"变电站X#"中提取X（X为数字）
    substation_pattern = re.compile(r'变电站(\d+)#')
    # 油位计编号：从"油位计-Y"中提取Y（Y为数字）
    oil_meter_pattern = re.compile(r'油位计-(\d+)')

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_dir):
        old_path = os.path.join(input_dir, filename)
        # 跳过文件夹，只处理文件
        if os.path.isdir(old_path):
            continue

        # 分离文件名和后缀（如"xxx.jpg" → "xxx"和".jpg"）
        name_part, ext_part = os.path.splitext(filename)
        # 只处理常见图片格式（可根据需要添加其他格式）
        # if ext_part.lower() not in ('.jpg', '.jpeg', '.bmp', '.png'):
        if ext_part.lower() not in '.txt':
            print(f"跳过非图片文件：{filename}")
            continue

        # 提取关键信息
        time_match = time_pattern.search(name_part)
        substation_match = substation_pattern.search(name_part)
        oil_meter_match = oil_meter_pattern.search(name_part)

        # 检查是否提取到所有信息
        if not (time_match and substation_match and oil_meter_match):
            print(f"提取信息失败（跳过）：{filename}")
            continue

        # 提取具体值
        time_str = time_match.group(1)
        substation_id = substation_match.group(1)
        oil_meter_id = oil_meter_match.group(1)

        # 构造新文件名
        # new_filename = f"{time_str}_{substation_id}_{oil_meter_id}{ext_part}"
        new_filename = f"{substation_id}_{oil_meter_id}_{time_str}{ext_part}"
        new_path = os.path.join(input_dir, new_filename)

        # 检查新文件名是否已存在
        if os.path.exists(new_path):
            print(f"警告：新文件名已存在（不重命名）：{new_filename}")
            continue

        # 执行重命名（先备份原文件）
        try:
            # 备份原文件到backup_dir
            shutil.copy2(old_path, os.path.join(backup_dir, filename))
            # 重命名
            os.rename(old_path, new_path)
            print(f"重命名成功：{filename} → {new_filename}")
        except Exception as e:
            print(f"重命名失败 {filename}：{str(e)}")


def extract_datetime_from_filename(filename):
    """
    从原始文件名中提取日期和时间信息。
    原始格式示例: 20250905T174928_613390_KZNZKG0A7R301AVT4M3CCD0602416QG30000000EBG4G00000000-f1.jpeg
    提取结果: 20250905174928
    """
    match = re.match(r'^(\d{8}T\d{6})', filename)
    if match:
        datetime_str = match.group(1).replace('T', '')
        return datetime_str
    return None


def copy_and_rename_for_fastcocoeval(input_dir, output_dir="coco_format_datetime_copy"):
    """
    复制图片，根据原始文件名中的日期时间重命名副本，并生成COCO标注文件。
    原始图片将被保留。
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    # 1. 创建输出目录结构
    image_output_dir = os.path.join(output_dir)
    os.makedirs(image_output_dir, exist_ok=True)
    print(f"输出目录：{os.path.abspath(output_dir)}")

    # 2. 遍历所有图片文件
    image_files = []
    print(f"正在扫描目录: {input_dir}")
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(image_extensions):
                # 避免处理输出目录自身的图片，防止循环
                # if output_dir in os.path.abspath(root):
                #     continue
                image_files.append(os.path.abspath(os.path.join(root, filename)))

    if not image_files:
        print("错误：在指定目录中未找到任何图片。")
        return

    print(f"共找到 {len(image_files)} 张图片，开始复制和重命名...")

    # 用于记录每个时间点的图片数量，以生成序号
    datetime_counter = {}

    coco_images = []
    image_id = 0

    for img_path in image_files:
        original_filename = os.path.basename(img_path)

        # 3. 提取日期时间
        datetime_str = extract_datetime_from_filename(original_filename)

        if not datetime_str:
            print(f"  警告: 无法从文件名 '{original_filename}' 中提取日期时间，已跳过。")
            continue

        # 4. 生成序号
        counter = datetime_counter.get(datetime_str, 0) + 1
        datetime_counter[datetime_str] = counter
        counter_str = f"{counter:02d}"

        # 5. 构建新文件名和新路径
        new_filename = f"{datetime_str}{counter_str}.jpg"
        new_img_path = os.path.join(image_output_dir, new_filename)

        try:
            # 6. 打开原始图片，处理格式并保存为副本（统一为JPG）
            with Image.open(img_path) as img:
                width, height = img.size
                # 处理透明通道，统一转为RGB模式
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                # 保存副本
                img.save(new_img_path, "JPEG")

            print(f"  已复制并命名: {original_filename} -> {new_filename}")

            # 7. 构建COCO图片条目
            coco_image_entry = {
                "id": image_id,
                "file_name": new_filename,
                "width": width,
                "height": height,
                "license": 1,
                "date_captured": "",
                "original_filename": original_filename,
                "original_path": img_path  # 记录原始路径，方便追溯
            }
            coco_images.append(coco_image_entry)

            image_id += 1

        except Exception as e:
            print(f"  处理失败: {original_filename} - 错误信息: {e}，已跳过。")

    # 8. 生成完整的COCO标注文件
    coco_data = {
        "info": {
            "description": "COCO format dataset (copied and renamed)",
            "version": "1.0",
            "year": 2024,
            "contributor": "Your Name",
            "date_created": ""
        },
        "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
        "images": coco_images,
        "annotations": [],
        "categories": []
    }

    annotation_path = os.path.join(output_dir, "instances.json")
    with open(annotation_path, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！")
    print(f" - 重命名后的图片副本已保存至: {os.path.abspath(image_output_dir)}")
    print(f" - COCO标注文件已生成: {os.path.abspath(annotation_path)}")
    print(f" - 原始图片在 '{input_dir}' 中保持不变。")


def main_01():
    # --- 配置区域 ---
    # 设置你要扫描的原始图片文件夹路径
    scan_directory = r'D:\1_Python\datasets\fire_security\images\train01_origin'
    # 设置输出目录（用于存放复制和重命名后的图片）
    output_directory = r"D:\1_Python\datasets\fire_security\images\train01"

    if not os.path.isdir(scan_directory):
        print(f"错误：目录 '{scan_directory}' 不存在或不是一个有效的目录。")
        return

    copy_and_rename_for_fastcocoeval(scan_directory, output_directory)


if __name__ == "__main__":
    main_01()

    # input_folder = r"D:\1_Python\datasets\switchgear\labels\train05"
    # rename_files(input_folder)
    # print("处理完成！原文件已备份至 rename_backup 文件夹")

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：txt2labelme_json.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/3 14:36 
@explain : 
'''
import os
import json
import argparse
from pathlib import Path
from PIL import Image


def yolo_to_labelme(yolo_txt_path, image_path, class_names, output_json_path=None):
    """
    将单个YOLO格式的txt文件转换为labelme格式的json文件

    参数:
        yolo_txt_path: YOLO格式txt文件的路径
        image_path: 对应的图片文件路径
        class_names: 类别名称列表（按YOLO的类别ID顺序）
        output_json_path: 输出json文件的路径，如果为None则与txt文件同目录
    """
    # 如果未指定输出路径，使用默认路径
    if output_json_path is None:
        output_json_path = os.path.splitext(yolo_txt_path)[0] + ".json"

    # 获取图片尺寸（这里需要PIL库来读取图片信息）
    try:
        img = Image.open(image_path)
        image_width, image_height = img.size
        img.close()
    except Exception as e:
        print(f"警告: 无法读取图片 {image_path} 的尺寸，使用默认尺寸(1920x1080)")
        image_width, image_height = 1920, 1080

    # 初始化labelme格式的字典
    labelme_data = {
        "version": "5.1.1",
        "flags": {},
        "shapes": [],
        "imagePath": "../../images/train01/" + os.path.basename(image_path),
        "imageData": None,  # 留空，labelme会自动处理
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # 读取YOLO格式的txt文件
    if os.path.exists(yolo_txt_path):
        with open(yolo_txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # YOLO格式：class_id x_center y_center width height (归一化值)
            parts = line.split()
            if len(parts) != 5:
                print(f"警告: 跳过无效行 '{line}' (格式不正确)")
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 检查类别ID是否有效
            if class_id < 0 or class_id >= len(class_names):
                print(f"警告: 未知的类别ID {class_id}，跳过该行")
                continue

            class_name = class_names[class_id]

            # 转换为绝对坐标
            x_center_abs = x_center * image_width
            y_center_abs = y_center * image_height
            width_abs = width * image_width
            height_abs = height * image_height

            # 计算labelme需要的两个点：左上角(x1, y1) 和 右下角(x2, y2)
            x1 = x_center_abs - width_abs / 2  # 左上角x坐标
            y1 = y_center_abs - height_abs / 2  # 左上角y坐标
            x2 = x_center_abs + width_abs / 2  # 右下角x坐标
            y2 = y_center_abs + height_abs / 2  # 右下角y坐标

            # 确保坐标在图片范围内（0 <= x <= 图片宽度，0 <= y <= 图片高度）
            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))

            # 确保x1 <= x2 和 y1 <= y2（防止异常情况）
            if x1 > x2:
                print(f"error x1 > x2 {x1} > {x2}")
                x1, x2 = x2, x1
            if y1 > y2:
                print(f"error y1 > y2 {y1} > {y2}")
                y1, y2 = y2, y1

            # 创建形状字典（labelme的矩形只需要两个点）
            shape = {
                "label": class_name,
                "points": [
                    [x1, y1],  # 第一个点：左上角
                    [x2, y2]   # 第二个点：右下角
                ],
                "group_id": None,
                "description": "",
                "shape_type": "rectangle",  # labelme的矩形类型
                "flags": {}
            }

            labelme_data["shapes"].append(shape)

    # 保存为json文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    print(f"已转换: {yolo_txt_path} -> {output_json_path}")


def batch_convert(yolo_dir, image_dir, output_dir=None, class_names=None):
    """
    批量转换YOLO格式txt文件到labelme格式json文件

    参数:
        yolo_dir: YOLO txt文件所在目录
        image_dir: 图片文件所在目录（图片文件名需与txt文件一致）
        output_dir: 输出json文件的目录，如果为None则与txt文件同目录
        class_names: 类别名称列表
    """
    # 默认类别名称（按ID顺序）
    if class_names is None:
        class_names = ["person", "head", "helmet", "cigarette"]

    # 确保输出目录存在
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # 获取所有YOLO txt文件
    yolo_files = list(Path(yolo_dir).glob("*.txt"))

    if not yolo_files:
        print(f"警告: 在目录 {yolo_dir} 中未找到任何txt文件")
        return

    print(f"找到 {len(yolo_files)} 个YOLO格式文件，开始转换...")

    for txt_file in yolo_files:
        # 获取文件名（不含扩展名）
        file_name = txt_file.stem
        if file_name == "classes":
            print(f"跳过 {file_name}.txt 文件")
            continue

        # 对应的图片路径
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        image_path = None
        for ext in image_extensions:
            img_path_candidate = os.path.join(image_dir, file_name + ext)
            if os.path.exists(img_path_candidate):
                image_path = img_path_candidate
                break

        if image_path is None:
            print(f"警告: 未找到 {file_name} 对应的图片文件，跳过该txt文件")
            continue

        # 输出json路径
        if output_dir is None:
            json_path = os.path.join(os.path.dirname(txt_file), file_name + ".json")
        else:
            json_path = os.path.join(output_dir, file_name + ".json")

        # 执行转换
        yolo_to_labelme(
            yolo_txt_path=str(txt_file),
            image_path=image_path,
            class_names=class_names,
            output_json_path=json_path
        )

    print("批量转换完成！")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='YOLO格式txt文件转换为labelme格式json文件')
    parser.add_argument('--yolo-dir', help='YOLO txt文件所在目录', default=r"D:\1_Python\datasets\fire_security\labels\train01")
    parser.add_argument('--image-dir', help='图片文件所在目录', default=r"D:\1_Python\datasets\fire_security\images\train01")
    parser.add_argument('--output-dir', help='输出json文件的目录（可选）', default=r"D:\1_Python\datasets\fire_security\labels_labelme\train01")
    parser.add_argument('--classes', nargs='*', default=["person", "head", "helmet", "cigarette"],
                        help='类别名称列表，按YOLO的类别ID顺序（默认: person head helmet cigarette）')

    args = parser.parse_args()

    # 执行批量转换
    batch_convert(
        yolo_dir=args.yolo_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        class_names=args.classes
    )


if __name__ == "__main__":

    main()

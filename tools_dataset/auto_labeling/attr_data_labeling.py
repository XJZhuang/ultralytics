#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：attr_data_labeling.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/27 9:28 
@explain : 
'''

import os
import cv2

# 数据集根目录，里面应包含 images 和 labels 两个文件夹
DATASET_PATH = r'D:\1_Python\datasets\fire_security'
# 你在YOLO中定义的“头部”类别的ID (例如，如果你的 classes.txt 里头部是第一个，ID就是0)
HEAD_CLASS_ID = 1

IMAGE_DIR = os.path.join(DATASET_PATH, 'images/val01')
LABEL_DIR = os.path.join(DATASET_PATH, 'labels_attr/val01')
OUTPUT_LABEL_DIR = os.path.join(DATASET_PATH, 'labels_with_attributes/val01')
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)    # 创建输出目录
# 在循环外创建一个固定的窗口
WINDOW_NAME = 'Attribute Annotation Tool'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) # 使用 WINDOW_NORMAL 可以让窗口大小可调

def process_annotations():
    """
    半自动化地为现有的YOLO标注添加属性。
    """
    # 遍历所有标注文件
    for label_filename in os.listdir(LABEL_DIR):
        if not label_filename.endswith('.txt'):
            continue

        label_path = os.path.join(LABEL_DIR, label_filename)
        image_base = os.path.splitext(label_filename)[0]
        image_path = os.path.join(IMAGE_DIR, image_base + '.jpg')

        if not os.path.exists(image_path):
            image_path = os.path.join(IMAGE_DIR, image_base + '.png')
            if not os.path.exists(image_path):
                print(f"警告: 找不到与 {label_filename} 对应的图像，已跳过。")
                continue

        output_label_path = os.path.join(OUTPUT_LABEL_DIR, label_filename)

        # 如果已经处理过，就跳过
        if os.path.exists(output_label_path):
            print(f"信息: {label_filename} 已处理，已跳过。")
            continue

        print(f"\n--- 正在处理: {label_filename} ---")

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法加载图像 {image_path}，已跳过。")
            continue
        h, w, _ = image.shape

        # 读取并处理标注
        with open(label_path, 'r') as f_in, open(output_label_path, 'w') as f_out:
            lines = f_in.readlines()
            head_instances = [] # 存储所有头部的信息 (原始行, 边界框坐标)

            # 第一次遍历：收集头部信息并绘制初始图像
            display_image = image.copy()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue

                class_id = int(parts[0])
                if class_id == HEAD_CLASS_ID:
                    # YOLO格式: class_id x_center y_center width height
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    width = float(parts[3]) * w
                    height = float(parts[4]) * h

                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    head_instances.append({'line': line, 'box': (x1, y1, x2, y2)})

            if not head_instances:
                print("信息: 此图像中没有检测到头部目标，直接复制标注文件。")
                f_out.writelines(lines)
                continue

            # 为每个头部目标询问属性
            for i, instance in enumerate(head_instances):
                temp_image = display_image.copy()
                x1, y1, x2, y2 = instance['box']

                # 高亮显示当前正在标注的头部
                cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 0, 255), 3) # 用红色粗框标出
                cv2.putText(temp_image, f'标注中: Head {i + 1}/{len(head_instances)}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 更新窗口内容
                cv2.imshow(WINDOW_NAME, temp_image)
                # 等待1ms，让窗口有机会刷新。这是OpenCV GUI正常工作所必需的。
                cv2.waitKey(1)

                # 请求用户输入
                print(f"\n请为 'Head {i + 1}/{len(head_instances)}' 输入属性 (0 或 1):")
                attr1 = None
                while attr1 not in [0, 1]:
                    try:
                        attr1 = int(input("属性1 (例如: 未佩戴头盔): ").strip())
                    except ValueError:
                        print("输入无效，请输入 0 或 1。")

                attr2 = None
                while attr2 not in [0, 1]:
                    try:
                        attr2 = int(input("属性2 (例如: 吸烟): ").strip())
                    except ValueError:
                        print("输入无效，请输入 0 或 1。")

                # 构建新的标注行并写入
                original_parts = instance['line'].strip().split()
                new_line_parts = original_parts + [str(attr1), str(attr2)]
                f_out.write(' '.join(new_line_parts) + '\n')

            # 处理非头部目标，直接写入
            for line in lines:
                parts = line.strip().split()
                if not parts or int(parts[0]) != HEAD_CLASS_ID:
                    f_out.write(line)

            print(f"--- 处理完成: {label_filename} ---")

    print("\n所有文件处理完毕！")

if __name__ == '__main__':
    try:
        process_annotations()
    finally:
        # 确保在程序结束时关闭所有OpenCV窗口，释放资源
        cv2.destroyAllWindows()
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：match_model.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/25 19:12 
@explain : 
'''
import os
import cv2
import subprocess
import platform
import tempfile

def open_image_with_default_viewer(image_path):
    """使用系统默认的图片查看器打开图片。"""
    try:
        if platform.system() == 'Windows':
            os.startfile(image_path)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', image_path])
        else:  # Linux
            subprocess.run(['xdg-open', image_path])
    except Exception as e:
        print(f"警告: 无法自动打开图片查看器。请手动查看临时文件: {image_path}")
        print(f"错误信息: {e}")

def convert_yolo_to_match_format(yolo_labels_dir, images_dir, output_labels_dir, head_class_id=1):
    """
    半自动化工具：将YOLO格式标注转换为MatchModel格式，并添加关系标签。

    :param yolo_labels_dir: 你的YOLO标注文件所在目录
    :param images_dir: 对应的图片所在目录
    :param output_labels_dir: MatchModel格式标注文件的输出目录
    :param head_class_id: YOLO标注中 'head' 对应的 class_id
    """
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    # 获取所有标注文件
    label_files = [f for f in os.listdir(yolo_labels_dir) if f.endswith('.txt')]

    for label_filename in label_files:
        base_filename = os.path.splitext(label_filename)[0]
        image_path = os.path.join(images_dir, base_filename + '.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(images_dir, base_filename + '.png')
            if not os.path.exists(image_path):
                print(f"警告: 找不到图片 {base_filename}.jpg 或 .png, 跳过。")
                continue

        yolo_label_path = os.path.join(yolo_labels_dir, label_filename)
        match_label_path = os.path.join(output_labels_dir, label_filename)

        # 如果输出文件已存在，询问是否跳过
        if os.path.exists(match_label_path):
            user_input = input(f"文件 {match_label_path} 已存在。是否跳过? (y/n, 默认y): ")
            if user_input.lower() != 'n':
                continue

        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法读取图片 {image_path}, 跳过。")
            continue
        h, w, _ = img.shape
        img_copy = img.copy() # 用于绘制的副本

        # 解析YOLO标注，提取head框
        head_boxes = []
        with open(yolo_label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                if class_id == head_class_id:
                    x_center, y_center = float(parts[1]), float(parts[2])
                    width, height = float(parts[3]), float(parts[4])
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)

                    head_boxes.append(((x1, y1), (x2, y2)))

        if not head_boxes:
            print(f"图片 {base_filename} 中未找到 head 标注，跳过。")
            # 创建一个空文件，表明已处理
            open(match_label_path, 'a').close()
            continue

        print(f"\n=============================================")
        print(f"开始处理图片: {base_filename}")
        print("将为每个头部生成预览图，请在图片查看器中查看。")
        print("格式: 吸烟(0/1) 未戴头盔(0/1)  (例如: 0 1 表示不吸烟，未戴头盔)")
        print("输入 'q' 可以随时退出。")
        print("=============================================")

        match_labels = []

        # 创建一个临时文件用于预览
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_image_path = temp_file.name

        try:
            for i, (pt1, pt2) in enumerate(head_boxes):
                # 绘制所有head框
                img_display = img_copy.copy()
                for j, (p1, p2) in enumerate(head_boxes):
                    color = (0, 255, 0) if j == i else (0, 0, 255)
                    cv2.rectangle(img_display, p1, p2, color, 2)

                # 高亮显示当前正在标注的head
                cv2.rectangle(img_display, pt1, pt2, (0, 255, 255), 4)
                cv2.putText(img_display, f"标注中: 头部 {i+1}/{len(head_boxes)}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 4)
                cv2.putText(img_display, "请在终端输入标签", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

                # 保存到临时文件
                cv2.imwrite(temp_image_path, img_display)

                # 用默认查看器打开
                print(f"\n正在打开头部 {i+1} 的预览图...")
                open_image_with_default_viewer(temp_image_path)

                # 等待用户输入
                while True:
                    user_input = input(f"\n请输入 头部 {i+1} 的标签 (smoking no_helmet): ")
                    user_input = user_input.strip()

                    if user_input.lower() == 'q':
                        print("用户退出。")
                        return

                    parts = user_input.split()
                    if len(parts) == 2 and all(p in ['0', '1'] for p in parts):
                        is_smoking = int(parts[0])
                        no_helmet = int(parts[1])
                        break
                    else:
                        print("输入无效，请输入两个数字 (0 或 1)，用空格隔开。例如: 1 0")

                # 存储标签
                x1_norm = pt1[0] / w
                y1_norm = pt1[1] / h
                x2_norm = pt2[0] / w
                y2_norm = pt2[1] / h
                match_labels.append(f"{x1_norm:.6f} {y1_norm:.6f} {x2_norm:.6f} {y2_norm:.6f} {is_smoking} {no_helmet}")
                print(f"  头部 {i+1} 标签已保存: 吸烟={is_smoking}, 未戴头盔={no_helmet}")

        finally:
            # 无论如何都删除临时文件
            if os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                    print(f"\n临时预览文件已删除: {temp_image_path}")
                except Exception as e:
                    print(f"警告: 无法删除临时文件 {temp_image_path}。请手动删除。")
                    print(f"错误信息: {e}")

        print(f"\n图片 {base_filename} 处理完毕。")

        # 保存到新的标注文件
        with open(match_label_path, 'w') as f:
            f.write('\n'.join(match_labels))

        print(f"标注文件已保存到: {match_label_path}")

    cv2.destroyAllWindows()
    print("\n所有图片处理完毕!")


if __name__ == '__main__':
    # --- 请在这里配置你的路径 ---
    # 你的YOLO标注文件所在目录
    # YOLO_LABELS_DIR = r'D:\1_Python\datasets\fire_security\labels_match_model\train01'
    YOLO_LABELS_DIR = r'D:\1_Python\datasets\fire_security\labels_match_model\val01'
    # 对应的图片所在目录
    # IMAGES_DIR = r'D:\1_Python\datasets\fire_security\images\train01'
    IMAGES_DIR = r'D:\1_Python\datasets\fire_security\images\val01'
    # MatchModel格式标注文件的输出目录
    # OUTPUT_MATCH_LABELS_DIR = r'D:\1_Python\datasets\fire_security\labels_match_model\train01-auto'
    OUTPUT_MATCH_LABELS_DIR = r'D:\1_Python\datasets\fire_security\labels_match_model\val01-auto'
    # YOLO标注中 'head' 类别的ID
    HEAD_CLASS_ID = 1
    # ----------------------------

    convert_yolo_to_match_format(YOLO_LABELS_DIR, IMAGES_DIR, OUTPUT_MATCH_LABELS_DIR, HEAD_CLASS_ID)
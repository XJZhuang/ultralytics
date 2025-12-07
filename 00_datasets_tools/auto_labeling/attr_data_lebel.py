#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：attr_data_lebel.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/27 9:39 
@explain : 
'''
import os
import cv2
import ctypes  # 用于获取屏幕分辨率（Windows系统）

# --- 配置区 ---
DATASET_PATH = r'D:\1_Python\datasets\fire_security'
HEAD_CLASS_ID = 1
# 窗口最大占屏幕的比例（避免窗口超出屏幕）
MAX_SCREEN_RATIO = 0.8
# --- 配置区结束 ---

# IMAGE_DIR = os.path.join(DATASET_PATH, 'images/val01')
IMAGE_DIR = os.path.join(DATASET_PATH, 'images/train01')
# LABEL_DIR = os.path.join(DATASET_PATH, 'labels_attr/val01')
LABEL_DIR = os.path.join(DATASET_PATH, 'labels_attr/train01')
# OUTPUT_LABEL_DIR = os.path.join(DATASET_PATH, 'labels_with_attributes/val01')
OUTPUT_LABEL_DIR = os.path.join(DATASET_PATH, 'labels_with_attributes/train01')

os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
WINDOW_NAME = 'Attribute Annotation Tool'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # 允许手动缩放窗口


def get_screen_resolution():
    """获取屏幕分辨率（Windows系统）"""
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)
    return screen_width, screen_height


def calculate_optimal_window_size(image_shape, screen_shape):
    """
    计算最优窗口大小（保持图像比例，不超过屏幕最大比例）
    :param image_shape: (h, w) 图像原始尺寸
    :param screen_shape: (screen_w, screen_h) 屏幕分辨率
    :return: (opt_w, opt_h) 最优窗口尺寸
    """
    img_h, img_w = image_shape
    screen_w, screen_h = screen_shape
    max_win_w = int(screen_w * MAX_SCREEN_RATIO)
    max_win_h = int(screen_h * MAX_SCREEN_RATIO)

    # 计算缩放比例（保持图像宽高比）
    scale = min(max_win_w / img_w, max_win_h / img_h)
    # 确保缩放后的尺寸不小于最小显示尺寸（避免太小）
    scale = max(scale, 0.5)  # 最小缩放比例为0.5（至少是原图的一半）

    opt_w = int(img_w * scale)
    opt_h = int(img_h * scale)
    return opt_w, opt_h


def process_annotations():
    """半自动化属性标注（窗口大小优化版）"""
    screen_w, screen_h = get_screen_resolution()
    label_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.txt')]
    window_resized_manually = False  # 标记用户是否手动调整过窗口大小

    for idx, label_filename in enumerate(label_files):
        print(f"\n--- 正在处理: {label_filename} ({idx + 1}/{len(label_files)}) ---")

        label_path = os.path.join(LABEL_DIR, label_filename)
        image_base = os.path.splitext(label_filename)[0]
        image_path = os.path.join(IMAGE_DIR, image_base + '.jpg')

        if not os.path.exists(image_path):
            image_path = os.path.join(IMAGE_DIR, image_base + '.png')
            if not os.path.exists(image_path):
                print(f"警告: 找不到与 {label_filename} 对应的图像，已跳过。")
                continue

        output_label_path = os.path.join(OUTPUT_LABEL_DIR, label_filename)
        if os.path.exists(output_label_path):
            print(f"信息: {label_filename} 已处理，已跳过。")
            continue

        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"警告: 无法加载图像 {image_path}，已跳过。")
            continue
        img_h, img_w, _ = image.shape

        # 收集头部目标信息
        with open(label_path, 'r') as f_in:
            lines = f_in.readlines()
            head_instances = []
            display_image = image.copy()

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if class_id == HEAD_CLASS_ID:
                    x_center = float(parts[1]) * img_w
                    y_center = float(parts[2]) * img_h
                    width = float(parts[3]) * img_w
                    height = float(parts[4]) * img_h
                    x1, y1, x2, y2 = map(int, [x_center - width / 2, y_center - height / 2, x_center + width / 2,
                                               y_center + height / 2])
                    head_instances.append({'line': line, 'box': (x1, y1, x2, y2)})
                    # 绘制所有头部框（绿色）
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_image, f'Head {len(head_instances)}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if not head_instances:
            print("信息: 无头部目标，直接复制标注文件。")
            with open(output_label_path, 'w') as f_out:
                f_out.writelines(lines)
            continue

        # 调整窗口大小（仅在用户未手动调整时自动优化）
        if not window_resized_manually:
            opt_w, opt_h = calculate_optimal_window_size((img_h, img_w), (screen_w, screen_h))
            cv2.resizeWindow(WINDOW_NAME, opt_w, opt_h)
            print(f"自动调整窗口大小为: {opt_w}x{opt_h}（按'F'全屏，按'ESC'退出）")

        # 标注每个头部目标
        with open(output_label_path, 'w') as f_out:
            # 先处理非头部目标
            non_head_lines = [line for line in lines if line.strip() and int(line.strip().split()[0]) != HEAD_CLASS_ID]
            f_out.writelines(non_head_lines)

            for i, instance in enumerate(head_instances):
                temp_image = display_image.copy()
                x1, y1, x2, y2 = instance['box']
                # 高亮当前标注目标（红色粗框）
                cv2.rectangle(temp_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(temp_image, f'标注中: Head {i + 1}/{len(head_instances)}', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                # 显示图像并处理快捷键
                cv2.imshow(WINDOW_NAME, temp_image)
                key = cv2.waitKey(1) & 0xFF  # 等待1ms，同时捕获按键

                # 快捷键处理
                if key == 27:  # ESC键退出
                    print("用户手动退出标注。")
                    cv2.destroyAllWindows()
                    return
                elif key == ord('f') or key == ord('F'):  # F键全屏
                    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    window_resized_manually = True
                elif key == ord('r') or key == ord('R'):  # R键恢复默认大小
                    opt_w, opt_h = calculate_optimal_window_size((img_h, img_w), (screen_w, screen_h))
                    cv2.resizeWindow(WINDOW_NAME, opt_w, opt_h)
                    window_resized_manually = False

                # 输入属性
                print(f"\nHead {i + 1}/{len(head_instances)} 属性输入:")
                attr1 = None
                while attr1 not in [0, 1]:
                    try:
                        attr1 = int(input("属性1（未佩戴头盔: 0=否/1=是）: ").strip())
                    except ValueError:
                        print("输入无效！请输入 0 或 1。")

                attr2 = None
                while attr2 not in [0, 1]:
                    try:
                        attr2 = int(input("属性2（吸烟: 0=否/1=是）: ").strip())
                    except ValueError:
                        print("输入无效！请输入 0 或 1。")

                # 写入带属性的标注行
                original_parts = instance['line'].strip().split()
                new_line = ' '.join(original_parts + [str(attr1), str(attr2)]) + '\n'
                f_out.write(new_line)

        print(f"✅ 处理完成: {label_filename}")

    print("\n🎉 所有文件标注完成！")


if __name__ == '__main__':
    try:
        process_annotations()
    finally:
        cv2.destroyAllWindows()
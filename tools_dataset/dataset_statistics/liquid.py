#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import numpy as np
from pathlib import Path
from PIL import Image


def analyze_specific_folders(root_dir, target_folders, target_height=640):
    """
    统计指定子文件夹内的图片长宽比，推荐 YOLOv8 imgsz。

    :param root_dir: 数据集根目录 (例如 'dataset/images')
    :param target_folders: 需要统计的子文件夹名称列表 (例如 ['train01', 'val02'])
    :param target_height: 训练时的目标高度 (默认640)
    """
    print(f"🚀 开始分析...")
    print(f"📂 根目录: {root_dir}")
    print(f"📂 指定文件夹: {target_folders}")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    widths = []
    heights = []
    ratios = []  # H / W

    total_images = 0

    # 遍历列表中的每一个文件夹
    for folder_name in target_folders:
        folder_path = Path(root_dir) / folder_name

        if not folder_path.exists():
            print(f"⚠️ 跳过不存在的文件夹: {folder_path}")
            continue

        print(f"   -> 正在扫描: {folder_name} ...")

        # 只扫描当前文件夹下的图片（不递归进入更深层级，如果需要递归当前子目录，可改为 rglob）
        current_folder_images = [
            p for p in folder_path.glob('*')
            if p.suffix.lower() in valid_extensions
        ]

        for img_path in current_folder_images:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    ratios.append(h / w)  # 计算高宽比
            except Exception as e:
                print(f"      ❌ 读取错误 {img_path.name}: {e}")

        total_images += len(current_folder_images)

    if total_images == 0:
        print("\n❌ 未找到任何图片，请检查路径配置。")
        return

    # 转为 numpy 进行计算
    ratios = np.array(ratios)

    # 核心统计
    mean_ratio = np.mean(ratios)  # 平均高宽比
    min_ratio = np.min(ratios)  # 最小高宽比 (对应最“胖/宽”的图片)
    max_ratio = np.max(ratios)  # 最大高宽比 (对应最“瘦/窄”的图片)

    # --- 推荐计算逻辑 ---
    # YOLO要求边长必须是32的倍数
    def align_32(x):
        return int(np.ceil(x / 32) * 32)

    # 逻辑：为了让所有图片在缩放到 target_height 时都不丢失宽度信息，
    # 我们应该依据“最胖”的那个图片（min_ratio）来设定宽度。
    # 宽度 = 高度 / 宽高比

    rec_width_conservative = align_32(target_height / min_ratio)  # 保守策略（推荐）
    rec_width_average = align_32(target_height / mean_ratio)  # 平均策略

    print("-" * 50)
    print(f"📊 统计结果 (共 {total_images} 张图片):")
    print(f"   平均高宽比 (Mean H/W): {mean_ratio:.2f}")
    print(f"   最小高宽比 (最胖 H/W): {min_ratio:.2f}  <-- 关键指标")
    print(f"   最大高宽比 (最瘦 H/W): {max_ratio:.2f}")
    print("-" * 50)
    print("💡 推荐 YOLOv8 imgsz 设置:")
    print(f"   目标高度: {target_height}")
    print(f"   推荐宽度 (基于最宽样本): {rec_width_conservative} (必须容纳最胖的液位计)")
    print(f"   推荐宽度 (基于平均样本): {rec_width_average}")

    print("\n✅ 最终建议配置 (复制到训练代码):")
    print(f"   imgsz=[{target_height}, {rec_width_conservative}]")
    print("-" * 50)


# --- 用户配置区域 ---
if __name__ == '__main__':
    # 1. 设置图片所在的根目录
    root_directory = r'D:\1_Python\datasets\liquids\images'

    # 2. 设置你想要统计的具体文件夹列表
    # 这里只写文件夹名字即可
    my_target_folders = [
        'train01',
        'train02',
        'val01',
        'val02'
    ]

    # 3. 运行分析 (假设你想把高度定为 640)
    analyze_specific_folders(root_directory, my_target_folders, target_height=640)

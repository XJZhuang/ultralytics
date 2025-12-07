#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：divide_by_images_divide.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/4 8:49 
@explain : 
'''
import os
import shutil
from pathlib import Path
from tqdm import tqdm


def get_image_basenames(image_dir):
    """
    获取指定图片目录下所有图片文件的basename（无扩展名）
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.JPG', '.JPEG', '.PNG'}
    basenames = set()

    for file in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file)
        if os.path.isfile(file_path) and Path(file).suffix in image_extensions:
            basename = os.path.splitext(file)[0]
            basenames.add(basename)

    return basenames


def move_matched_labels(src_label_dir, dst_label_dir, target_basenames):
    """
    从源标签目录移动匹配basename的标签文件到目标目录
    """
    # 创建目标目录（如果不存在）
    os.makedirs(dst_label_dir, exist_ok=True)

    moved_count = 0
    skipped_count = 0
    missing_count = 0

    # 遍历目标basename，匹配并移动标签
    for basename in tqdm(target_basenames, desc=f"处理 {os.path.basename(dst_label_dir)}"):
        src_label_path = os.path.join(src_label_dir, f"{basename}.txt")
        dst_label_path = os.path.join(dst_label_dir, f"{basename}.txt")

        if os.path.exists(src_label_path):
            # 避免覆盖已存在文件
            if os.path.exists(dst_label_path):
                print(f"\n跳过：{dst_label_path} 已存在")
                skipped_count += 1
                continue

            # 移动文件
            shutil.move(src_label_path, dst_label_path)
            moved_count += 1
        else:
            print(f"\n缺失：未找到标签文件 {src_label_path}")
            missing_count += 1

    return moved_count, skipped_count, missing_count


def main():
    # 基础路径配置（核心！根据你的实际路径修改）
    BASE_IMAGE_DIR = r'D:\1_Python\datasets\fire_security\images'
    BASE_LABEL_DIR = r'D:\1_Python\datasets\fire_security\labels_yolo+attr'

    # 定义需要匹配的子集映射（key: 图片子集目录, value: 标签源子集目录）
    # 例如：images/val01 对应 labels_yolo+attr/train01
    subset_mapping = {
        "val01": "train01",
        "val03": "train03",
        "val04": "train04",
        # 可添加更多映射，如 "test01": "train01"
    }

    # 总统计
    total_moved = 0
    total_skipped = 0
    total_missing = 0

    print("=" * 60)
    print("开始匹配并移动标签文件")
    print(f"图片根目录：{BASE_IMAGE_DIR}")
    print(f"标签根目录：{BASE_LABEL_DIR}")
    print("=" * 60)

    # 遍历每个子集映射
    for img_subset, label_src_subset in subset_mapping.items():
        # 构建完整路径
        img_subset_dir = os.path.join(BASE_IMAGE_DIR, img_subset)
        label_src_dir = os.path.join(BASE_LABEL_DIR, label_src_subset)
        label_dst_dir = os.path.join(BASE_LABEL_DIR, img_subset)  # 标签目标目录（如val01）

        # 验证路径是否存在
        if not os.path.exists(img_subset_dir):
            print(f"\n❌ 图片子集目录不存在：{img_subset_dir}，跳过该子集")
            continue

        if not os.path.exists(label_src_dir):
            print(f"\n❌ 标签源目录不存在：{label_src_dir}，跳过该子集")
            continue

        # 获取图片basename列表
        print(f"\n📌 处理子集：{img_subset}（图片） <- {label_src_subset}（标签源）")
        print(f"正在读取 {img_subset_dir} 中的图片文件名...")
        target_basenames = get_image_basenames(img_subset_dir)

        if not target_basenames:
            print(f"⚠️ {img_subset_dir} 中未找到图片文件，跳过")
            continue

        print(f"找到 {len(target_basenames)} 个图片文件，开始匹配标签...")

        # 移动匹配的标签
        moved, skipped, missing = move_matched_labels(
            label_src_dir, label_dst_dir, target_basenames
        )

        # 更新统计
        total_moved += moved
        total_skipped += skipped
        total_missing += missing

        # 输出子集统计
        print(f"\n📊 {img_subset} 统计：")
        print(f"  ✅ 成功移动：{moved} 个")
        print(f"  ⚠️  跳过（已存在）：{skipped} 个")
        print(f"  ❌ 缺失标签：{missing} 个")

    # 输出总统计
    print("\n" + "=" * 60)
    print("📈 总统计：")
    print(f"✅ 累计移动标签：{total_moved} 个")
    print(f"⚠️  累计跳过：{total_skipped} 个")
    print(f"❌ 累计缺失标签：{total_missing} 个")
    print("=" * 60)
    print("操作完成！")


if __name__ == "__main__":
    # 可选：安装进度条
    try:
        from tqdm import tqdm
    except ImportError:
        print("提示：安装tqdm可显示进度条（可选）：pip install tqdm")


        def tqdm(iterable, desc=""):
            return iterable

    main()

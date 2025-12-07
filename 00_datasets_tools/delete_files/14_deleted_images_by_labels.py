#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：14_deleted_images_by_labels.py
@Author  ：zhuangxujun
@Date    ：2025-9-17 14:45 
@explain : 根据 txt 标签 删除 多余的图片。 注意：建议保存部分无标签的背景图片
'''
import os
import shutil
from typing import List, Tuple


def get_image_files(img_dir: str) -> List[str]:
    """获取图片目录中所有支持的图片文件"""
    # 支持的图片后缀
    IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    image_files = []
    for filename in os.listdir(img_dir):
        # 检查是否为文件
        if not os.path.isfile(os.path.join(img_dir, filename)):
            continue

        # 检查是否为支持的图片格式
        _, ext = os.path.splitext(filename)
        if ext.lower() in IMAGE_SUFFIXES:
            image_files.append(filename)

    return image_files


def process_group(img_dir: str, label_dir: str, dest_dir: str) -> Tuple[int, int]:
    """
    处理单个分组的无标签图片

    参数:
        img_dir: 该分组图片所在目录
        label_dir: 该分组标签所在目录
        dest_dir: 无标签图片的目标目录

    返回:
        元组 (总图片数, 移动的无标签图片数)
    """
    # 检查目录是否存在
    for dir_path in [img_dir, label_dir]:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"目录不存在: {dir_path}")

    # 创建目标目录（如果不存在）
    os.makedirs(dest_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = get_image_files(img_dir)
    total_images = len(image_files)
    moved_count = 0

    if total_images == 0:
        print(f"⚠️ 分组图片目录 {img_dir} 中未找到图片文件")
        return (total_images, moved_count)

    # 检查每个图片是否有对应的标签
    for img_filename in image_files:
        # 获取图片文件名（不含后缀）
        img_name = os.path.splitext(img_filename)[0]
        # 对应的标签文件名
        label_filename = f"{img_name}.txt"
        label_path = os.path.join(label_dir, label_filename)

        # 检查标签文件是否存在
        if not os.path.exists(label_path):
            # 没有对应的标签，移动图片
            src_img_path = os.path.join(img_dir, img_filename)
            dest_img_path = os.path.join(dest_dir, img_filename)

            # 避免目标目录中已有同名文件
            if os.path.exists(dest_img_path):
                # 添加后缀避免覆盖
                name, ext = os.path.splitext(img_filename)
                dest_img_path = os.path.join(dest_dir, f"{name}_duplicate{ext}")
                print(f"⚠️ 目标目录已存在 {img_filename}，将重命名为 {os.path.basename(dest_img_path)}")

            try:
                shutil.move(src_img_path, dest_img_path)
                moved_count += 1
                print(f"✅ 已移动无标签图片: {img_filename}")
            except Exception as e:
                print(f"❌ 移动图片 {img_filename} 失败: {str(e)}")

    print(f"📊 分组处理完成: 共检查 {total_images} 张图片，移动 {moved_count} 张无标签图片\n")
    return total_images, moved_count


def main():
    # 配置基础路径和分组
    root_path = r"D:\1_Python\datasets\switchgear\switchgear"
    group_index = ['1', '2', '3']  # 要处理的分组

    total_all = 0
    moved_all = 0

    print(f"===== 开始处理分组: {group_index} =====")

    for group in group_index:
        print(f"\n----- 处理分组 {group} -----")

        # 构建该分组的图片和标签目录路径
        img_dir = os.path.join(root_path, "images", f"train{group}")
        label_dir = os.path.join(root_path, "labels", f"train{group}")
        dest_dir = os.path.join(root_path, "images", f"deleted{group}")  # 无标签图片目标目录

        # 处理当前分组
        total, moved = process_group(img_dir, label_dir, dest_dir)
        total_all += total
        moved_all += moved

    print(f"===== 所有分组处理完成 =====")
    print(f"总统计: 共检查 {total_all} 张图片，移动 {moved_all} 张无标签图片")
    print("注意：建议保存部分无标签的背景图片")


if __name__ == "__main__":
    main()

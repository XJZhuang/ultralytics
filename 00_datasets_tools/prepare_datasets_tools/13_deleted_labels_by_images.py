#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-main 
@File    ：13_deleted_labels_by_images.py
@Author  ：zhuangxujun
@Date    ：2024/12/3 11:22
@explain : 移动多余的txt标签文件（没有对应图片的标签）到指定路径，而非直接删除
'''

import os
import shutil
from typing import List, Tuple


def get_corresponding_image_paths(image_dir: str, base_name: str) -> List[str]:
    """
    获取与标签文件对应的所有可能的图片路径

    Args:
        image_dir: 图片所在目录
        base_name: 标签文件的基础名称（不含扩展名）

    Returns:
        可能存在的图片文件路径列表
    """
    # 支持的图片后缀
    # IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
    return [
        os.path.join(image_dir, f"{base_name}{ext}")
        for ext in IMAGE_SUFFIXES
    ]


def check_image_exists(image_paths: List[str]) -> Tuple[bool, str]:
    """
    检查图片是否存在

    Args:
        image_paths: 图片路径列表

    Returns:
        元组 (是否存在, 存在的图片路径)
    """
    for path in image_paths:
        if os.path.exists(path) and os.path.isfile(path):
            return True, path
    return False, ""


def move_orphaned_label_files(image_dir: str, label_dir: str, dest_dir: str, label_suffix: str) -> Tuple[int, int]:
    """
    移动没有对应图片的标签文件到指定目录

    Args:
        image_dir: 图片所在目录
        label_dir: 标签文件所在目录
        dest_dir: 要移动的标签文件的目标目录
        label_suffix: 标签文件后缀

    Returns:
        元组 (总标签文件数, 移动的标签文件数)
    """
    # 检查目录是否存在
    for dir_path in [image_dir, label_dir]:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"目录不存在: {dir_path}")

    # 创建目标目录（如果不存在）
    os.makedirs(dest_dir, exist_ok=True)

    # 获取所有标签文件
    label_files = [f for f in os.listdir(label_dir)
                   if os.path.isfile(os.path.join(label_dir, f))
                   and f.endswith(label_suffix)
                   and f not in "classes.txt"  # 排除指定文件
                   ]

    total_labels = len(label_files)
    moved_count = 0

    if total_labels == 0:
        print(f"⚠️ 标签目录 {label_dir} 中未找到标签文件")
        return total_labels, moved_count

    # 检查每个标签文件是否有对应的图片
    for label_filename in label_files:
        # 获取标签文件的基础名称（不含扩展名）
        base_name = os.path.splitext(label_filename)[0]

        # 获取可能的图片路径
        image_paths = get_corresponding_image_paths(image_dir, base_name)

        # 检查图片是否存在
        image_exists, image_path = check_image_exists(image_paths)

        if not image_exists:
            # 没有对应的图片，移动标签文件
            src_label_path = os.path.join(label_dir, label_filename)
            dest_label_path = os.path.join(dest_dir, label_filename)

            # 避免目标目录中已有同名文件
            if os.path.exists(dest_label_path):
                # 添加后缀避免覆盖
                name, ext = os.path.splitext(label_filename)
                dest_label_path = os.path.join(dest_dir, f"{name}_duplicate{ext}")
                print(f"⚠️ 目标目录已存在 {label_filename}，将重命名为 {os.path.basename(dest_label_path)}")

            try:
                shutil.move(src_label_path, dest_label_path)
                moved_count += 1
                print(f"✅ 已移动无对应图片的标签: {label_filename}")
            except Exception as e:
                print(f"❌ 移动标签 {label_filename} 失败: {str(e)}")

    print(f"📊 处理完成: 共检查 {total_labels} 个标签文件，移动 {moved_count} 个无对应图片的标签文件\n")
    return total_labels, moved_count


def main():
    # 配置基础路径和分组
    # root_path = r"D:\1_Python\datasets\switchgear"
    # root_path = r"D:\1_Python\datasets\knobs"
    root_path = r"D:\1_Python\datasets\liquids"
    # label_path = "labels"
    label_path = "labels_json"
    dataset_type = "train"
    # dataset_type = "val"
    # groups = [ ]  # 要处理的分组
    groups = ['01', ]  # 要处理的分组
    # groups = ['01', '02', '03', '04', '05']  # 要处理的分组
    suffix = ".txt" if label_path == "labels" else ".json"
    total_all = 0
    moved_all = 0

    print(f"===== 开始处理分组: {groups} =====")

    for group in groups:
        print(f"\n----- 处理分组 {group} -----")

        # 构建该分组的图片、标签目录和目标目录路径
        image_dir = os.path.join(root_path, "images", f"{dataset_type}{group}")
        # image_dir = os.path.join(root_path, "images", f"val{group}")
        label_dir = os.path.join(root_path, label_path, f"{dataset_type}{group}")
        # label_dir = os.path.join(root_path, "labels", f"val{group}")
        dest_dir = os.path.join(root_path, label_path, f"deleted{group}")

        # 处理当前分组
        total, moved = move_orphaned_label_files(image_dir, label_dir, dest_dir, label_suffix=suffix)
        total_all += total
        moved_all += moved

    print(f"===== 所有分组处理完成 =====")
    print(f"总统计: 共检查 {total_all} 个标签文件，移动 {moved_all} 个无对应图片的标签文件")


if __name__ == '__main__':
    main()

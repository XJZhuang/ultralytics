#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：rename_images_by_order.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/9 16:35 
@explain : 图片从头命名，参考12位数字的规范，批量重命名，需要支持其实数字，例如从000000245915编号开始。
'''

import os
import argparse

def batch_rename_images(folder_path, start_num, skip_standard=True):
    """
    批量重命名文件夹内的图片为COCO风格12位数字命名
    :param folder_path: 图片文件夹路径（如train05）
    :param start_num: 起始编号（整数，如245915）
    :param skip_existing: 是否跳过已为12位数字命名的图片
    """
    # 支持的图片后缀（可按需扩展，区分大小写但统一判断）
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP'}
    # 获取文件夹内所有文件（排除子文件夹）
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if not file_list:
        print(f"⚠️  文件夹 {folder_path} 内无任何文件")
        return

    # 筛选出图片文件
    img_files = []
    for f in file_list:
        ext = os.path.splitext(f)[1]
        if ext in img_extensions:
            img_files.append(f)

    if not img_files:
        print(f"⚠️  文件夹 {folder_path} 内未找到图片文件（支持格式：{img_extensions}）")
        return

    # 按文件修改时间排序（也可改为按名称排序，注释掉则按系统默认）
    img_files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

    rename_count = 0
    current_num = start_num

    for old_name in img_files:
        # 拆分文件名和后缀
        old_prefix, old_ext = os.path.splitext(old_name)

        # 严格判断是否符合COCO命名规范：前缀是12位纯数字 + 后缀是图片格式
        is_standard = False
        if skip_standard:
            if old_prefix.isdigit() and len(old_prefix) == 12 and old_ext in img_extensions:
                is_standard = True

        if is_standard:
            print(f"⏩ 跳过已符合COCO规范的文件：{old_name}")
            # 已规范文件不占用编号，编号继续沿用
            continue

        # 生成12位补零的新文件名
        new_prefix = str(current_num).zfill(12)
        new_name = f"{new_prefix}{old_ext}"
        old_path = os.path.join(folder_path, old_name)
        new_path = os.path.join(folder_path, new_name)

        # 避免文件名重复（若已存在则跳过并提示）
        if os.path.exists(new_path):
            print(f"❌ 新文件名 {new_name} 已存在，跳过原文件 {old_name}")
            current_num += 1
            continue

        # 执行重命名
        os.rename(old_path, new_path)
        print(f"✅ {old_name} → {new_name}")
        rename_count += 1
        current_num += 1

    print(f"\n📊 重命名统计：")
    print(f"- 待处理图片总数：{len(img_files)}")
    print(f"- 跳过规范文件数：{len(img_files) - rename_count - (current_num - start_num - rename_count)}")
    print(f"- 成功重命名数：{rename_count}")
    print(f"- 失败/跳过数：{current_num - start_num - rename_count}")
    print(f"- 最终编号截止：{current_num - 1}（若全部成功）")


if __name__ == "__main__":
    # 手动指定文件夹和起始编号
    folder_path = r"D:\1_Python\datasets\fire_security\images\train05"  # 你的图片文件夹路径
    start_num = 93         # 起始编号
    batch_rename_images(folder_path, start_num)
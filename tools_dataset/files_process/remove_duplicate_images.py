#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：remove_duplicate_images.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/21 14:28 
@explain : 
'''
import os
import hashlib
import shutil


def calculate_image_hash(image_path, block_size=65536):
    """
    计算图片文件的MD5哈希值，用于比较内容是否相同。
    为了效率，我们分块读取大文件。
    """
    hash_obj = hashlib.md5()
    try:
        with open(image_path, 'rb') as f:
            # 循环读取文件块并更新哈希值
            for block in iter(lambda: f.read(block_size), b''):
                hash_obj.update(block)
        return hash_obj.hexdigest()
    except FileNotFoundError:
        print(f"警告：文件未找到 {image_path}，已跳过。")
        return None
    except PermissionError:
        print(f"警告：没有权限访问 {image_path}，已跳过。")
        return None


def find_duplicate_images(directory):
    """
    遍历指定目录，查找内容重复的图片。
    返回一个字典，键是图片的哈希值，值是具有该哈希值的图片路径列表。
    """
    image_hashes = {}

    # 定义需要处理的图片文件扩展名
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    print(f"正在扫描目录: {directory}")

    # os.walk 会遍历目录下的所有文件和子目录
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # 检查文件扩展名是否为图片
            if filename.lower().endswith(image_extensions):
                file_path = os.path.abspath(os.path.join(root, filename))

                # 为了避免处理我们自己创建的分组文件夹里的图片，增加一个判断
                # 如果文件路径中已经包含了"重复组_"，就跳过
                if "重复组_" in file_path:
                    continue

                print(f"  正在处理文件: {os.path.relpath(file_path, directory)}")

                # 计算文件哈希
                file_hash = calculate_image_hash(file_path)

                if file_hash:
                    # 如果哈希值已存在于字典中，则添加路径；否则，创建新条目
                    if file_hash in image_hashes:
                        image_hashes[file_hash].append(file_path)
                    else:
                        image_hashes[file_hash] = [file_path]

    print("扫描完成。")
    return image_hashes

def group_duplicates(duplicate_dict, base_dir=r"D:\3_download\图片助手(ImageAssistant)_批量图片下载器\10.12.14.6_38068\2025-11-21_13-55-01\园区AI管家\重复"):
    """
    根据重复图片字典，为每一组重复图片创建一个专属文件夹，并将图片移动进去。
    """
    # 筛选出真正有重复的条目
    duplicates_to_group = {k: v for k, v in duplicate_dict.items() if len(v) > 1}

    if not duplicates_to_group:
        print("未找到任何重复的图片组。")
        return

    print(f"\n开始为重复图片创建专属文件夹...")

    group_counter = 1
    total_files_moved = 0

    for hash_value, file_paths in duplicates_to_group.items():
        # 为每组重复图片创建一个文件夹
        # 文件夹命名为 "重复组_1", "重复组_2", ...
        group_folder_name = f"重复组_{group_counter}"
        group_folder_path = os.path.join(base_dir, group_folder_name)

        # 如果文件夹已存在（小概率事件，比如两次运行），则创建一个新的
        while os.path.exists(group_folder_path):
            group_counter += 1
            group_folder_name = f"重复组_{group_counter}"
            group_folder_path = os.path.join(base_dir, group_folder_name)

        os.makedirs(group_folder_path)
        print(f"\n创建文件夹: {group_folder_path}")

        # 将该组的所有图片移动到新创建的文件夹中
        for path in file_paths:
            try:
                file_name = os.path.basename(path)
                dest_path = os.path.join(group_folder_path, file_name)

                # 如果目标文件夹中已有同名文件，为避免覆盖，添加后缀
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(file_name)
                    dest_path = os.path.join(group_folder_path, f"{name}_{counter}{ext}")
                    counter += 1

                shutil.move(path, dest_path)
                print(f"  - 已移动: {path} -> {dest_path}")
                total_files_moved += 1
            except Exception as e:
                print(f"  - 移动失败: {path}, 错误信息: {e}")

        group_counter += 1

    print(f"\n处理完成！总共创建了 {group_counter - 1} 个重复图片组，移动了 {total_files_moved} 张图片。")

def main():
    # --- 配置区域 ---
    # 设置你要扫描的文件夹路径
    # 使用 '.' 表示当前脚本所在的文件夹
    # 或者直接写路径，例如: "C:/Users/YourUser/Pictures" 或 "/home/user/photos"
    # scan_directory = r'D:\3_download\图片助手(ImageAssistant)_批量图片下载器\10.12.14.6_38068\2025-11-21_13-55-01\园区AI管家'
    scan_directory = r'D:\1_Python\datasets\fire_security\images\train03'
    # -----------------

    if not os.path.isdir(scan_directory):
        print(f"错误：目录 '{scan_directory}' 不存在或不是一个有效的目录。")
        return

    # 1. 查找重复图片
    duplicate_images = find_duplicate_images(scan_directory)

    # 2. 为每组重复图片创建文件夹并移动
    group_duplicates(duplicate_images, scan_directory)

if __name__ == "__main__":
    main()
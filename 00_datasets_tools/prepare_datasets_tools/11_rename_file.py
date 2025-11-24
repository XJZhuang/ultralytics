#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-main
@File    ：11_rename_file.py
@IDE     ：PyCharm
@Author  ：zhuangxujun
@Date    ：2024/12/26 10:58
@explain : 重命名文件，如1.txt重命名为prefix_1.txt，目的是防止多个train01-1/2/3文件夹合并时文件名重复
'''
import os


def batch_rename_txt(folder_path, prefix):
    """
    批量重命名文件夹中的txt文件

    参数:
        folder_path: 文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    # 获取文件夹中所有的txt文件
    for filename in os.listdir(folder_path):
        # 跳过classes.txt文件
        if filename.lower() == "classes.txt":
            print(f"跳过：{os.path.join(folder_path, filename)}")
            continue

        # 只处理txt文件
        if filename.endswith('.txt'):
            # 构建新的文件名
            new_filename = f"{prefix}_{filename}"

            # 构建完整的旧路径和新路径
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # 执行重命名
            try:
                os.rename(old_path, new_path)
                print(f"已重命名: {filename} -> {new_filename}")
            except Exception as e:
                print(f"重命名失败 {filename}: {e}")


def batch_rename_images(folder_path, prefix):
    """
    批量重命名文件夹中的图片文件

    参数:
        folder_path: 文件夹路径
        prefix: 要添加的前缀
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    # 定义支持的图片文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')

    # 获取文件夹中所有的图片文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图片（不区分大小写）
        if filename.lower().endswith(image_extensions):
            # 构建新的文件名
            new_filename = f"{prefix}_{filename}"

            # 构建完整的旧路径和新路径
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)

            # 避免文件名重复
            counter = 1
            while os.path.exists(new_path):
                # 如果新文件名已存在，添加数字后缀
                name, ext = os.path.splitext(filename)
                new_filename = f"{prefix}_{name}_{counter}{ext}"
                new_path = os.path.join(folder_path, new_filename)
                counter += 1
                print("error")

            # 执行重命名
            try:
                os.rename(old_path, new_path)
                print(f"已重命名: {filename} -> {new_filename}")
            except Exception as e:
                print(f"重命名失败 {filename}: {e}")

if __name__ == "__main__":
    prefix = "3"  # 可以根据需要修改前缀

    txt_folder = os.path.join(r"D:\1_Python\datasets\switchgear\labels", "train02-3")
    batch_rename_txt(txt_folder, prefix)

    # 文件夹路径
    images_folder = os.path.join(r"D:\1_Python\datasets\switchgear\images", "train02-3")  # 可以替换为你的图片文件夹路径
    batch_rename_images(images_folder, prefix)

    print("重命名操作完成")

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：bmp2jpg.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/12 9:12 
@explain : 
'''

import os
import shutil
from PIL import Image


def bmp_to_jpg(input_dir, delete_original=False):
    """
    批量将文件夹中的.bmp图片转为.jpg
    :param input_dir: 图片所在文件夹路径
    :param delete_original: 转换后是否删除原.bmp文件（默认False，保留原图）
    """
    # 创建备份文件夹（防止误操作）
    backup_dir = os.path.join(input_dir, "bmp_backup")
    os.makedirs(backup_dir, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_dir):
        # 只处理.bmp文件
        if filename.lower().endswith(".bmp"):
            bmp_path = os.path.join(input_dir, filename)
            # 构造.jpg文件名（去掉.bmp后缀）
            jpg_filename = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(input_dir, jpg_filename)

            try:
                # 打开bmp图片并保存为jpg（质量设为95，平衡体积和质量）
                with Image.open(bmp_path) as img:
                    # 如果图片有alpha通道（透明层），转为RGB避免报错
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    img.save(jpg_path, "JPEG", quality=95)
                print(f"转换成功：{filename} -> {jpg_filename}")

                # 备份原.bmp文件
                shutil.copy2(bmp_path, os.path.join(backup_dir, filename))
                # 如果需要删除原图
                if delete_original:
                    os.remove(bmp_path)
                    print(f"已删除原图：{filename}")

            except Exception as e:
                print(f"转换失败 {filename}：{str(e)}")


# --------------------------
# 使用说明：修改下面的路径后运行
# --------------------------
if __name__ == "__main__":
    input_folder = r"D:\1_Python\datasets\switchgear\images\train05"
    # 是否删除原.bmp文件（建议先保留，确认转换无误后再删除）
    delete_bmp = False
    bmp_to_jpg(input_folder, delete_bmp)
    print("转换完成！原.bmp文件已备份至 bmp_backup 文件夹")

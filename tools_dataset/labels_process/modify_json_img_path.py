#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import json
import os


def batch_modify_labelme_json(folder_path):
    """
    批量修改LabelMe生成的JSON文件
    逻辑：保留原始imagePath的文件名，在前面拼接 ../../images/train02/ 路径
    :param folder_path: JSON文件所在的文件夹路径
    """
    # 遍历文件夹下所有文件
    for filename in os.listdir(folder_path):
        # 只处理JSON文件
        if filename.endswith('.json'):
            json_path = os.path.join(folder_path, filename)

            try:
                # 读取JSON文件
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 读取原始imagePath（确保字段存在）
                original_image_path = data.get('imagePath')
                if not original_image_path:
                    print(f"警告：{filename} 中未找到 imagePath 字段，跳过处理")
                    continue

                # 提取原始路径中的文件名（不管原始路径是绝对路径还是相对路径）
                image_filename = os.path.basename(original_image_path)

                # 在文件名前拼接目标路径前缀（按要求格式）
                new_image_path = f'../../images/train02/{image_filename}'
                data['imagePath'] = new_image_path

                # 设置imageData为null
                data['imageData'] = None

                # 保存修改后的JSON文件（覆盖原文件）
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                print(f"成功处理：{filename}")
                print(f"  原始imagePath：{original_image_path}")
                print(f"  新imagePath：{new_image_path}\n")

            except Exception as e:
                print(f"处理失败：{filename}，错误信息：{str(e)}\n")

if __name__ == "__main__":
    target_folder = r"D:\1_Python\datasets\liquids\labels_json\train02"  # Windows示例

    # 验证文件夹是否存在
    if not os.path.exists(target_folder):
        print(f"错误：文件夹不存在 -> {target_folder}")
    else:
        batch_modify_labelme_json(target_folder)
        print("\n批量处理完成！")

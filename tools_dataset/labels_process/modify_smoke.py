#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：modify_smoke.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/10 9:08 
@explain : 只修改第一个head的smoke
'''
import os
import json
import argparse


def modify_head_smoke_flag(json_file_path):
    """
    修改单个JSON文件中head.flags.smoke为true，无flags则打印错误
    :param json_file_path: JSON文件路径
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        modified = False
        error_msg = ""

        # 遍历所有shapes，找到label=head的节点
        for shape in data.get('shapes', []):
            if shape.get('label') == 'head':
                # 检查是否有flags字段
                if 'flags' not in shape:
                    error_msg = f"❌ {os.path.basename(json_file_path)}: head节点无flags字段"
                    break

                # 修改smoke为true（无论原有值是什么）
                shape['flags']['smoke'] = True
                modified = True
                break  # 假设一个文件只有一个head节点，找到即退出

        # 打印错误或执行保存
        if error_msg:
            print(error_msg)
        elif modified:
            # 保存修改后的JSON（保持原格式）
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ {os.path.basename(json_file_path)}: head.flags.smoke已设为true")
        else:
            print(f"⚠️ {os.path.basename(json_file_path)}: 未找到label=head的节点")

    except Exception as e:
        print(f"❌ {os.path.basename(json_file_path)}: 读取/写入失败 - {str(e)}")


def batch_modify_json(folder_path):
    """
    批量处理文件夹内所有JSON文件
    :param folder_path: JSON文件所在文件夹路径
    """
    # 校验文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"❌ 文件夹 {folder_path} 不存在！")
        return

    # 获取所有JSON文件
    json_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.json') and os.path.isfile(os.path.join(folder_path, f))
    ]

    if not json_files:
        print(f"⚠️ 文件夹 {folder_path} 内未找到JSON文件")
        return

    # 批量处理每个JSON文件
    print(f"📁 开始处理文件夹：{folder_path}（共{len(json_files)}个JSON文件）\n")
    for json_file in json_files:
        modify_head_smoke_flag(json_file)

    print("\n📊 批量处理完成！")


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='批量修改labelme JSON文件中head.flags.smoke为true')
    parser.add_argument(
        '--folder', help='JSON文件所在文件夹路径（如./train05_annotations）',
        default=r"D:\1_Python\datasets\fire_security\labels_labelme\train05"
    )
    args = parser.parse_args()

    # 执行批量处理
    batch_modify_json(args.folder)
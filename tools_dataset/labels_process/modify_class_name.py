#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：modify_class_name.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/2 9:39 
@explain : 批量修改LabelMe JSON文件的标签名，将hat和person统一改为head
'''

import os
import json
import argparse
from typing import List


class LabelMeLabelRenamer:
    def __init__(self, json_dir: str, backup: bool = True):
        """
        初始化标签名修改器
        :param json_dir: LabelMe JSON文件所在目录（批量处理该目录下所有.json文件）
        :param backup: 是否备份原文件（备份文件后缀为.bak）
        """
        self.json_dir = os.path.abspath(json_dir)
        self.backup = backup
        # 定义标签映射：key为原标签名，value为目标标签名
        self.label_mapping = {
            "hat": "head",
            "person": "head"
        }

        # 记录处理状态
        self.total_count = 0
        self.success_count = 0
        self.modified_count = 0
        self.failed_files = []

    def rename_labels_in_json(self, json_path: str):
        """
        修改单个JSON文件中的标签名
        :param json_path: JSON文件路径
        """
        try:
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 标记是否修改过
            is_modified = False

            # 遍历所有标注形状，修改标签名
            for shape in data.get("shapes", []):
                original_label = shape.get("label", "").strip()
                # 如果原标签在映射表中，替换为目标标签
                if original_label in self.label_mapping:
                    shape["label"] = self.label_mapping[original_label]
                    is_modified = True
                    print(f"  替换标签: {original_label} -> {self.label_mapping[original_label]}")

            # 如果需要备份且文件被修改，先备份原文件
            if self.backup and is_modified:
                backup_path = f"{json_path}.bak"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  已备份原文件: {os.path.basename(backup_path)}")

            # 保存修改后的JSON文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.success_count += 1
            if is_modified:
                self.modified_count += 1

        except Exception as e:
            self.failed_files.append((os.path.basename(json_path), str(e)))
            print(f"  处理失败: 错误 - {str(e)}")

    def batch_rename(self):
        """
        批量处理目录下所有JSON文件
        """
        print(f"开始批量修改LabelMe标签名...")
        print(f"处理目录: {self.json_dir}")
        print(f"标签映射规则: {self.label_mapping}")
        print(f"是否备份原文件: {'是' if self.backup else '否'}")
        print("-" * 80)

        # 获取目录下所有JSON文件
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        self.total_count = len(json_files)

        if self.total_count == 0:
            print("警告: 未找到任何JSON文件！")
            return

        # 遍历处理每个JSON文件
        for idx, json_file in enumerate(json_files, 1):
            json_path = os.path.join(self.json_dir, json_file)
            print(f"\n[{idx}/{self.total_count}] 处理文件: {json_file}")
            self.rename_labels_in_json(json_path)

        # 输出处理统计
        print("\n" + "-" * 80)
        print(f"处理完成！")
        print(f"总文件数: {self.total_count}")
        print(f"成功处理: {self.success_count}")
        print(f"已修改标签的文件数: {self.modified_count}")
        print(f"处理失败: {len(self.failed_files)}")

        if self.failed_files:
            print("\n失败文件详情:")
            for filename, error in self.failed_files:
                print(f"  {filename}: {error}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量修改LabelMe JSON标签名（hat/person -> head）')
    parser.add_argument('--json-dir', help='LabelMe JSON文件所在目录', default=r"D:\1_Python\datasets\fire_security\labels_labelme\train03")

    args = parser.parse_args()

    # 创建修改器并执行批量修改
    renamer = LabelMeLabelRenamer(
        json_dir=args.json_dir,
        backup=False
    )
    renamer.batch_rename()


if __name__ == "__main__":
    main()

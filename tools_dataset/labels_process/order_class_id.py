#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：order_class_id.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/2 10:04 
@explain : 批量排序LabelMe JSON的标注顺序：先按类别优先级，再按坐标排序
           类别优先级：person → head → helmet → cigarette
           同类别排序：按边界框左上x坐标升序（x相同按y坐标升序）
'''


import os
import json
import argparse
from typing import List, Dict, Tuple


class LabelMeShapeSorter:
    def __init__(self, json_dir: str, backup: bool = True):
        """
        初始化标注排序器
        :param json_dir: LabelMe JSON文件所在目录（批量处理）
        :param backup: 是否备份原文件（备份后缀.bak，默认True）
        """
        self.json_dir = os.path.abspath(json_dir)
        self.backup = backup
        # 定义类别优先级：key=类别名，value=优先级（数字越小优先级越高）
        self.label_priority = {
            "person": 1,
            "head": 2,
            "helmet": 3,
            "cigarette": 4
        }

        # 记录处理状态
        self.total_count = 0
        self.success_count = 0
        self.failed_files = []

    def get_shape_sort_key(self, shape: Dict) -> Tuple[int, float, float]:
        """
        定义排序key：用于shapes数组排序
        :param shape: 单个标注对象（shape）
        :return: 排序关键字 tuple(类别优先级, 左上x坐标, 左上y坐标)
        """
        # 1. 类别优先级：不在优先级字典中的类别（如旧标签）优先级最低（设为99）
        label = shape.get("label", "").strip()
        priority = self.label_priority.get(label, 99)

        # 2. 边界框左上坐标：LabelMe矩形格式为[[x1,y1], [x2,y2]]，x1是左上x，y1是左上y
        points = shape.get("points", [])
        if len(points) >= 2 and all(isinstance(coord, (int, float)) for point in points for coord in point):
            x1 = points[0][0]  # 左上x坐标
            y1 = points[0][1]  # 左上y坐标
        else:
            # 异常坐标（如格式错误），设为极大值放到最后
            x1 = float('inf')
            y1 = float('inf')

        return (priority, x1, y1)

    def sort_single_json(self, json_path: str):
        """
        排序单个JSON文件的shapes数组
        :param json_path: JSON文件路径
        """
        try:
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # 获取原始shapes数组
            original_shapes = json_data.get("shapes", [])
            if not original_shapes:
                print(f"  无标注数据，跳过排序")
                self.success_count += 1
                return

            # 按自定义key排序
            sorted_shapes = sorted(original_shapes, key=self.get_shape_sort_key)

            # 备份原文件（如果需要）
            if self.backup:
                backup_path = f"{json_path}.bak"
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f"  已备份原文件: {os.path.basename(backup_path)}")

            # 更新shapes数组并保存
            json_data["shapes"] = sorted_shapes
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            # 输出排序信息
            label_count = {}
            for shape in sorted_shapes:
                label = shape["label"]
                label_count[label] = label_count.get(label, 0) + 1
            print(f"  排序完成：{label_count}")

            self.success_count += 1

        except Exception as e:
            self.failed_files.append((os.path.basename(json_path), str(e)))
            print(f"  处理失败: {str(e)}")

    def batch_sort(self):
        """
        批量处理目录下所有JSON文件
        """
        print(f"开始批量排序LabelMe标注...")
        print(f"处理目录: {self.json_dir}")
        print(f"类别优先级: person(1) → head(2) → helmet(3) → cigarette(4)")
        print(f"同类别排序: 左上x坐标升序 → 左上y坐标升序")
        print(f"是否备份原文件: {'是' if self.backup else '否'}")
        print("-" * 80)

        # 获取所有JSON文件
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        self.total_count = len(json_files)

        if self.total_count == 0:
            print("警告: 未找到任何JSON文件！")
            return

        # 遍历处理每个JSON
        for idx, json_file in enumerate(json_files, 1):
            json_path = os.path.join(self.json_dir, json_file)
            print(f"\n[{idx}/{self.total_count}] 处理文件: {json_file}")
            self.sort_single_json(json_path)

        # 输出统计信息
        print("\n" + "-" * 80)
        print(f"处理完成！")
        print(f"总文件数: {self.total_count}")
        print(f"成功处理: {self.success_count}")
        print(f"处理失败: {len(self.failed_files)}")

        if self.failed_files:
            print("\n失败文件详情:")
            for filename, error in self.failed_files:
                print(f"  {filename}: {error}")


def main():
    parser = argparse.ArgumentParser(description='LabelMe JSON标注排序工具（按类别+坐标排序）')
    parser.add_argument('--json-dir', help='LabelMe JSON文件所在目录（必填）', default=r"D:\1_Python\datasets\fire_security\labels_labelme\train06")

    args = parser.parse_args()

    # 创建排序器并执行批量排序
    sorter = LabelMeShapeSorter(
        json_dir=args.json_dir,
        backup=False
    )
    sorter.batch_sort()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：merge_txt_labelme_json.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/2 9:52 
@explain : 批量将YOLO格式的person标签，添加到已有head标签的LabelMe JSON中
'''


import os
import json
import argparse
from typing import List, Tuple


class PersonLabelMerger:
    def __init__(self, json_dir: str, yolo_txt_dir: str, person_class_id: int = 0):
        """
        初始化融合器
        :param json_dir: 原有LabelMe JSON文件所在目录（含head标签）
        :param yolo_txt_dir: YOLO格式txt文件所在目录（模型预测生成的person标签）
        :param person_class_id: person在YOLO txt中的类别ID（COCO默认0，需根据模型调整）
        """
        self.json_dir = os.path.abspath(json_dir)
        self.yolo_txt_dir = os.path.abspath(yolo_txt_dir)
        self.person_class_id = person_class_id  # person的YOLO类别ID
        self.target_label = "person"  # 要添加的标签名

        # 记录处理状态
        self.total_json = 0
        self.success_count = 0
        self.added_person_count = 0  # 总共添加的person标签数
        self.failed_files = []

    def yolo2labelme(self, yolo_line: str, img_w: int, img_h: int) -> List[List[float]]:
        """
        YOLO格式坐标转LabelMe矩形坐标（2点：左上+右下）
        :param yolo_line: YOLO格式的一行数据（class_id xc yc w h）
        :param img_w: 图片宽度
        :param img_h: 图片高度
        :return: LabelMe的points格式 [[x1,y1], [x2,y2]]
        """
        parts = yolo_line.strip().split()
        if len(parts) != 5:
            raise ValueError(f"YOLO格式错误：{yolo_line}")

        class_id = int(parts[0])
        xc = float(parts[1])  # 归一化x中心
        yc = float(parts[2])  # 归一化y中心
        w = float(parts[3])  # 归一化宽度
        h = float(parts[4])  # 归一化高度

        # 仅处理person类别
        if class_id != self.person_class_id:
            return None

        # 转换为像素坐标（左上+右下）
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h

        # 确保坐标非负（避免模型预测误差）
        x1 = max(0.0, x1)
        y1 = max(0.0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        return [[x1, y1], [x2, y2]]

    def merge_single_file(self, json_filename: str):
        """
        处理单个JSON文件：添加person标签
        :param json_filename: JSON文件名（与YOLO txt同名）
        """
        json_path = os.path.join(self.json_dir, json_filename)
        # YOLO txt文件名与JSON同名（如000480.json → 000480.txt）
        txt_filename = os.path.splitext(json_filename)[0] + '.txt'
        txt_path = os.path.join(self.yolo_txt_dir, txt_filename)

        try:
            # 读取原有LabelMe JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # 获取图片尺寸（用于坐标转换）
            img_w = json_data.get("imageWidth", 0)
            img_h = json_data.get("imageHeight", 0)
            if img_w == 0 or img_h == 0:
                raise ValueError("JSON中缺少图片尺寸信息（imageWidth/imageHeight）")

            # 读取YOLO txt中的person标签
            person_shapes = []
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # YOLO坐标转LabelMe格式
                        points = self.yolo2labelme(line, img_w, img_h)
                        if points is not None:
                            # 构建person标签的shape结构
                            person_shape = {
                                "label": self.target_label,
                                "points": points,
                                "group_id": None,
                                "shape_type": "rectangle",
                                "flags": {}
                            }
                            person_shapes.append(person_shape)

            # 若有person标签，添加到JSON的shapes数组
            if person_shapes:
                json_data["shapes"].extend(person_shapes)
                added_num = len(person_shapes)
                self.added_person_count += added_num
                print(f"  添加{added_num}个person标签")

            # 保存修改后的JSON文件（覆盖原有文件，建议先备份）
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

            self.success_count += 1

        except Exception as e:
            self.failed_files.append((json_filename, str(e)))
            print(f"  处理失败: {str(e)}")

    def batch_merge(self):
        """
        批量处理所有JSON文件
        """
        print(f"开始批量添加person标签到LabelMe JSON...")
        print(f"JSON目录（含head标签）: {self.json_dir}")
        print(f"YOLO txt目录（person标签）: {self.yolo_txt_dir}")
        print(f"person类别ID: {self.person_class_id}")
        print("-" * 80)

        # 获取所有LabelMe JSON文件
        json_files = [f for f in os.listdir(self.json_dir) if f.endswith('.json')]
        self.total_json = len(json_files)

        if self.total_json == 0:
            print("警告: 未找到任何JSON文件！")
            return

        # 遍历处理每个JSON
        for idx, json_file in enumerate(json_files, 1):
            print(f"\n[{idx}/{self.total_json}] 处理文件: {json_file}")
            self.merge_single_file(json_file)

        # 输出统计信息
        print("\n" + "-" * 80)
        print(f"处理完成！")
        print(f"总JSON文件数: {self.total_json}")
        print(f"成功处理: {self.success_count}")
        print(f"累计添加person标签数: {self.added_person_count}")
        print(f"处理失败: {len(self.failed_files)}")

        if self.failed_files:
            print("\n失败文件详情:")
            for filename, error in self.failed_files:
                print(f"  {filename}: {error}")


def main():
    parser = argparse.ArgumentParser(description='批量添加YOLO格式的person标签到LabelMe JSON')
    parser.add_argument('--json-dir', help='原有LabelMe JSON目录（含head标签）', default=r"D:\1_Python\datasets\fire_security\labels_labelme\train03")
    parser.add_argument('--yolo-txt-dir', help='YOLO格式txt目录（模型预测的person标签）', default=r"D:\1_Python\datasets\fire_security\images\train03\labels_txt\labels")
    parser.add_argument('--person-class-id', type=int, default=0, help='person在YOLO中的类别ID（COCO默认0）')

    args = parser.parse_args()

    # 执行批量融合
    merger = PersonLabelMerger(
        json_dir=args.json_dir,
        yolo_txt_dir=args.yolo_txt_dir,
        person_class_id=args.person_class_id
    )
    merger.batch_merge()


if __name__ == "__main__":
    main()
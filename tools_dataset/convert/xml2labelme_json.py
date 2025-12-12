#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：xml2labelme_json.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/1 15:39 
@explain : 将 VOC 格式（.xml） 的标签文件 转换为 labelme标注的JSON格式。
'''

import os
import json
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
import argparse


class VOC2LabelMeConverter:
    def __init__(self, voc_xml_dir: str, labelme_json_dir: str, image_dir: str = None):
        """
        初始化转换器
        :param voc_xml_dir: VOC格式XML文件所在目录
        :param labelme_json_dir: 输出LabelMe JSON文件的目录
        :param image_dir: 图片文件所在目录（如果与XML目录不同时指定）
        """
        self.voc_xml_dir = os.path.abspath(voc_xml_dir)
        self.labelme_json_dir = os.path.abspath(labelme_json_dir)
        self.image_dir = os.path.abspath(image_dir) if image_dir else self.voc_xml_dir

        # 创建输出目录
        os.makedirs(self.labelme_json_dir, exist_ok=True)

        # 记录转换状态
        self.success_count = 0
        self.fail_count = 0
        self.failed_files = []

    def get_image_info(self, xml_root: ET.Element) -> Tuple[str, int, int]:
        """
        从XML中获取图片信息
        :param xml_root: XML根节点
        :return: (图片文件名, 宽度, 高度)
        """
        filename = xml_root.find('filename').text
        size = xml_root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        return filename, width, height

    def get_image_size(self, xml_root: ET.Element) -> Tuple[int, int]:
        """
        仅从XML中获取图片尺寸（不再获取filename，改用XML文件名）
        :param xml_root: XML根节点（即解析XML后得到的root对象）
        :return: (宽度, 高度) -> 元组类型，两个值都是整数
        """
        size = xml_root.find('size')  # 从XML根节点找到<size>标签
        width = int(size.find('width').text)  # 从<size>中提取width值，转为整数
        height = int(size.find('height').text)  # 从<size>中提取height值，转为整数
        return width, height  # 返回宽度和高度（元组形式）

    def parse_xml_object(self, obj_elem: ET.Element) -> Dict:
        """
        解析XML中的单个目标对象
        :param obj_elem: 对象节点
        :return: LabelMe格式的标注信息
        """
        # 获取类别名称
        label = obj_elem.find('name').text

        # 获取边界框（VOC格式：xmin, ymin, xmax, ymax）
        bndbox = obj_elem.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # LabelMe格式：[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        points = [
            [xmin, ymin],
            # [xmax, ymin],
            [xmax, ymax],
            # [xmin, ymax]
        ]

        # 构建LabelMe标注对象
        labelme_obj = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "rectangle",  # 矩形类型
            "flags": {}
        }

        # 如果有difficult标签，添加到flags中
        # difficult = obj_elem.find('difficult')
        # if difficult is not None:
        #     labelme_obj["flags"]["difficult"] = bool(int(difficult.text))

        return labelme_obj

    def convert_single_file(self, xml_filename: str):
        """
        转换单个XML文件为LabelMe JSON
        :param xml_filename: XML文件名（带扩展名）
        """
        xml_path = os.path.join(self.voc_xml_dir, xml_filename)
        json_filename = os.path.splitext(xml_filename)[0] + '.json'
        json_path = os.path.join(self.labelme_json_dir, json_filename)

        try:
            # 解析XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图片信息
            # img_filename, width, height = self.get_image_info(root)
            # img_path = os.path.join(self.image_dir, img_filename)

            # 关键修改1：不再读取XML中的filename，改用XML文件名（去后缀）作为图片名
            xml_basename = os.path.splitext(xml_filename)[0]  # XML文件名（不含扩展名）
            img_filename = f"{xml_basename}.jpg"  # 图片名 = XML去后缀 + .jpg（可根据实际图片后缀修改）
            # 关键修改2：获取图片尺寸（宽度、高度）
            width, height = self.get_image_size(root)
            # 解析所有目标对象
            objects = root.findall('object')
            shapes = [self.parse_xml_object(obj) for obj in objects]

            # 构建LabelMe JSON结构
            labelme_data = {
                "version": "5.1.1",  # LabelMe默认版本
                "flags": {},
                "shapes": shapes,
                # "imagePath": "../../images/train03/" + os.path.basename(img_path).split(".")[0] + ".jpg",  # 图片相对路径
                "imagePath": "../../images/train03/" + img_filename,  # 图片相对路径
                "imageData": None,  # 不存储图片Base64数据（节省空间）
                "imageHeight": height,
                "imageWidth": width
            }

            # 保存JSON文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, ensure_ascii=False, indent=2)

            self.success_count += 1
            # print(f"成功转换: {xml_filename} -> {json_filename}")
            print(f"成功转换: {xml_filename} -> {json_filename} | 对应图片: {img_filename}")

        except Exception as e:
            self.fail_count += 1
            self.failed_files.append((xml_filename, str(e)))
            print(f"转换失败: {xml_filename} - 错误: {str(e)}")

    def convert_all(self):
        """
        批量转换目录下所有XML文件
        """
        print(f"开始转换...")
        print(f"XML目录: {self.voc_xml_dir}")
        print(f"JSON输出目录: {self.labelme_json_dir}")
        print(f"图片目录: {self.image_dir}")
        print("-" * 50)

        # 获取所有XML文件
        xml_files = [f for f in os.listdir(self.voc_xml_dir) if f.endswith('.xml')]

        if not xml_files:
            print("警告: 未找到任何XML文件！")
            return

        # 批量转换
        for xml_file in xml_files:
            self.convert_single_file(xml_file)

        # 输出转换统计
        print("-" * 50)
        print(f"转换完成！")
        print(f"总文件数: {len(xml_files)}")
        print(f"成功: {self.success_count}")
        print(f"失败: {self.fail_count}")

        if self.failed_files:
            print("\n失败文件详情:")
            for filename, error in self.failed_files:
                print(f"  {filename}: {error}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='VOC XML格式转LabelMe JSON格式工具')
    parser.add_argument('--xml-dir', help='VOC XML文件所在目录', default=r"D:\1_Python\datasets\fire_security\labels_xml\train03")
    parser.add_argument('--json-dir', help='输出LabelMe JSON文件的目录', default=r"D:\1_Python\datasets\fire_security\labels_labelme\train03")
    parser.add_argument('--image-dir', help='图片文件所在目录（默认与XML目录相同）', default=r"D:\1_Python\datasets\fire_security\images\train03")

    args = parser.parse_args()

    # 创建转换器并执行转换
    converter = VOC2LabelMeConverter(
        voc_xml_dir=args.xml_dir,
        labelme_json_dir=args.json_dir,
        image_dir=args.image_dir
    )
    converter.convert_all()


if __name__ == "__main__":
    main()
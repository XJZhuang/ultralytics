#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics-main 
@File    ：labelme_json2txt.py
@IDE     ：PyCharm 
@Author  ：zhuangxujun
@Date    ：2024/9/10 15:06 
@explain : 将labelme标注的 JSON文件（keypoint） 转化为 txt文件
'''

import os
import json
from typing import Dict, List, Tuple
from pathlib import Path  # 更现代的路径处理

# 目标检测类别映射（key：LabelMe标注名，value：YOLO类别ID）
BBOX_CLASS_MAP: Dict[str, int] = {
    'person': 0,
    'head': 1,
    'helmet': 2,
    'cigarette': 3
}

# 关键点类别顺序（需与LabelMe标注的"label"字段一致，决定TXT中关键点的输出顺序）
# KEYPOINT_CLASS_ORDER: List[str] = ['center', 'point']
KEYPOINT_CLASS_ORDER: List[str] = ['edge_point']

# 归一化坐标保留小数位数（YOLO常用5位，可调整）
DECIMAL_PLACES: int = 6

# 属性映射：bool值→整数（false=0，true=1）
ATTR_MAP: Dict[bool, int] = {False: 0, True: 1}


def convert_json_to_yolo_txt(
        json_file_path: str,
        output_txt_dir: str,
        bbox_class_map: Dict[str, int] = BBOX_CLASS_MAP,
        decimal_places: int = DECIMAL_PLACES
) -> Tuple[bool, str]:
    """
    将单个LabelMe JSON标注文件转换为YOLO（检测+关键点）格式TXT文件
    - 普通类别（person/helmet/cigarette）：class_id x_center y_center width height
    - head类别：class_id x_center y_center width height attr_smoke attr_no_helmet
    参数:
        json_file_path: LabelMe JSON文件的完整路径
        output_txt_dir: 输出TXT文件的目标目录
        bbox_class_map: 检测类别映射表
        keypoint_order: 关键点输出顺序
        decimal_places: 归一化坐标保留小数位数

    返回:
        Tuple[处理结果(bool), 消息(str)]: (True=成功, 成功信息) / (False=失败, 错误信息)
    """
    # 1. 路径预处理与目录创建
    output_txt_dir_path = Path(output_txt_dir)
    output_txt_dir_path.mkdir(parents=True, exist_ok=True)  # 递归创建目录（支持多级目录）

    # 提取JSON文件名（不含后缀），作为TXT文件名
    json_filename = Path(json_file_path).stem
    output_txt_path = output_txt_dir_path / f"{json_filename}.txt"

    try:
        # 2. 读取并解析JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            try:
                labelme_data = json.load(f)
            except json.JSONDecodeError as e:
                return False, f"JSON解析失败：{str(e)}"

        # 3. 提取图像基础信息（校验必要字段）
        required_fields = ['imageWidth', 'imageHeight', 'shapes']
        for field in required_fields:
            if field not in labelme_data:
                return False, f"JSON缺少必要字段：{field}"

        img_width = labelme_data['imageWidth']
        img_height = labelme_data['imageHeight']
        if img_width <= 0 or img_height <= 0:
            return False, f"图像尺寸异常（宽：{img_width}，高：{img_height}）"

        # 4. 处理矩形框标注，生成YOLO格式内容（移除关键点无关逻辑）
        yolo_content = []
        for bbox_ann in labelme_data['shapes']:
            if bbox_ann.get('shape_type') != 'rectangle':
                continue  # 只处理"rectangle"类型检测框

            # 校验检测框必要信息
            label = bbox_ann.get('label')
            if not label or label not in bbox_class_map:
                return False, f"边界框标签未定义：{label}（仅支持{list(bbox_class_map.keys())}）"
            if 'points' not in bbox_ann or len(bbox_ann['points']) != 2:
                return False, f"边界框坐标异常：{bbox_ann['points']}（需2个顶点）"

            # 4.1 计算检测框归一化坐标（YOLO格式：center_x, center_y, width, height）
            (x1, y1), (x2, y2) = bbox_ann['points']
            # 确保坐标顺序正确（左上角→右下角）
            bbox_left = min(x1, x2)
            bbox_right = max(x1, x2)
            bbox_top = min(y1, y2)
            bbox_bottom = max(y1, y2)

            # 归一化计算（保留decimal_places位小数）
            center_x = round((bbox_left + bbox_right) / 2 / img_width, decimal_places)
            center_y = round((bbox_top + bbox_bottom) / 2 / img_height, decimal_places)
            bbox_w = round((bbox_right - bbox_left) / img_width, decimal_places)
            bbox_h = round((bbox_bottom - bbox_top) / img_height, decimal_places)

            # 4.2 组装YOLO行（核心：区分head类和其他类）
            yolo_line = [str(bbox_class_map[label])]  # 先添加类别ID
            # 添加基础坐标字段
            yolo_line.extend([
                f"{center_x:.{decimal_places}f}",
                f"{center_y:.{decimal_places}f}",
                f"{bbox_w:.{decimal_places}f}",
                f"{bbox_h:.{decimal_places}f}"
            ])

            # 4.3 若为head类，添加属性字段（attr_smoke, attr_no_helmet）
            # if label == 'head':
            #     flags = bbox_ann.get('flags', {})
            #     # 提取属性（默认值：0=未吸烟，0=佩戴安全帽）
            #     attr_smoke = ATTR_MAP.get(flags.get('smoke'), 0)
            #     attr_no_helmet = ATTR_MAP.get(flags.get('no_helmet'), 0)
            #     yolo_line.extend([str(attr_smoke), str(attr_no_helmet)])

            # 添加当前行到内容中
            yolo_content.append(' '.join(yolo_line))

        # 5. 写入TXT文件（若有标注内容才写入，避免空文件）
        if yolo_content:
            with open(output_txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(yolo_content))
        else:
            return False, "JSON中无有效检测框标注（仅支持rectangle类型）"

        return True, f"成功：{Path(json_file_path).name} → {output_txt_path.name}"

    except Exception as e:
        # 捕获所有未预料的错误，返回具体信息
        return False, f"未知错误：{str(e)}"


def batch_process_groups(
        root_dir: str,
        group_index: List[str],
        json_subdir: str = "labels_labelme",
        txt_subdir: str = "labels"
) -> None:
    """
    批量处理多个分组的JSON文件，转换为YOLO TXT

    参数:
        root_dir: 数据集根目录
        group_index: 要处理的分组列表（如['dh', 'lt']）
        json_subdir: JSON文件所在的子目录（相对于root_dir）
        txt_subdir: TXT文件输出的子目录（相对于root_dir）
    """
    root_path = Path(root_dir)
    total_success = 0
    total_failed = 0
    failed_records = []  # 记录失败文件，便于后续排查

    print(f"===== 开始批量处理分组：{group_index} =====")
    print(f"数据集根目录：{root_path.resolve()}\n")

    for group in group_index:
        # 构建当前分组的JSON输入目录和TXT输出目录
        json_input_dir = root_path / json_subdir / f"train{group}"
        txt_output_dir = root_path / txt_subdir / f"train{group}"

        # 校验JSON目录是否存在
        if not json_input_dir.exists() or not json_input_dir.is_dir():
            print(f"⚠️ 分组[{group}]：JSON目录不存在 → {json_input_dir.resolve()}，跳过该分组")
            continue

        # 获取目录下所有JSON文件（优化：直接过滤后缀，避免split错误）
        json_files = [f for f in json_input_dir.iterdir() if f.suffix.lower() == '.json']
        if not json_files:
            print(f"⚠️ 分组[{group}]：JSON目录为空 → {json_input_dir.resolve()}，跳过该分组")
            continue

        print(f"----- 处理分组[{group}] -----")
        print(f"JSON目录：{json_input_dir.resolve()}")  # D:\1_Python\datasets\switchgear\knobs\labels_json\train1
        print(f"待处理JSON文件数：{len(json_files)}")

        # 遍历处理当前分组的每个JSON文件
        for json_file in json_files:
            success, msg = convert_json_to_yolo_txt(
                json_file_path=str(json_file.resolve()),
                output_txt_dir=str(txt_output_dir.resolve())
            )
            if success:
                total_success += 1
                print(f"✅ {msg}")
            else:
                total_failed += 1
                failed_records.append(f"分组[{group}] {json_file.name}：{msg}")
                print(f"❌ {json_file.name} → {msg}")

        # 统计当前分组失败数
        group_failed = len([r for r in failed_records if f'分组[{group}]' in r])
        print(f"分组[{group}]处理完成：成功{len(json_files) - group_failed}个，失败{group_failed}个\n")

    # 输出最终统计结果
    print("===== 批量处理结束 =====")
    print(f"总统计：成功{total_success}个，失败{total_failed}个")
    if failed_records:
        print("\n❌ 失败文件详情：")
        for idx, record in enumerate(failed_records, 1):
            print(f"  {idx}. {record}")


if __name__ == '__main__':
    # -------------------------- 用户配置区 --------------------------
    DATASET_ROOT = r'D:\1_Python\datasets\fire_security'  # 数据集根目录
    TARGET_GROUPS = ['01', '03', '04']  # 需处理的分组

    # 执行批量处理
    batch_process_groups(
        root_dir=DATASET_ROOT,
        group_index=TARGET_GROUPS
    )

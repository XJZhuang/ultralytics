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
    # 'knob': 0,
    'liquid': 0,
}

# 关键点类别顺序（需与LabelMe标注的"label"字段一致，决定TXT中关键点的输出顺序）
# KEYPOINT_CLASS_ORDER: List[str] = ['center', 'point']
KEYPOINT_CLASS_ORDER: List[str] = ['edge_point']

# 归一化坐标保留小数位数（YOLO常用5位，可调整）
DECIMAL_PLACES: int = 6

# 关键点可见性映射（LabelMe的"description"字段值 → YOLO关键点可见性标签）0: 不可见, 1: 遮挡, 2: 可见（遵循YOLO keypoint格式规范）
VISIBILITY_MAP: Dict[str, int] = {
    '': 2,  # 空备注视为"可见"
    '0': 0,  # 自定义"不可见"标记
    '1': 1,  # 自定义"遮挡"标记
    '2': 2  # 自定义"可见"标记
}


def convert_json_to_yolo_txt(
        json_file_path: str,
        output_txt_dir: str,
        bbox_class_map: Dict[str, int] = BBOX_CLASS_MAP,
        keypoint_order: List[str] = KEYPOINT_CLASS_ORDER,
        decimal_places: int = DECIMAL_PLACES
) -> Tuple[bool, str]:
    """
    将单个LabelMe JSON标注文件转换为YOLO（检测+关键点）格式TXT文件

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

        # 4. 预处理：提取所有关键点标注（避免重复遍历shapes，提升效率）
        # 结构：List[Dict] → 每个元素包含关键点的坐标、标签、可见性
        keypoints_list: List[Dict] = []
        for ann in labelme_data['shapes']:
            if ann.get('shape_type') != 'point':
                continue  # 只保留"point"类型标注

            # 校验关键点必要信息
            if 'points' not in ann or len(ann['points']) != 1:
                print("跳过无效关键点（如多点标注）")
                continue  #
            if 'label' not in ann:
                print("跳过无标签的关键点")
                continue  #

            kp_x, kp_y = ann['points'][0]
            keypoints_list.append({
                'x': float(kp_x),  # 保留float，避免整数截断误差
                'y': float(kp_y),
                'label': ann['label'],
                'visibility': VISIBILITY_MAP.get(ann.get('description', ''), 0)  # 默认为"不可见"
            })

        # 5. 处理矩形框标注，生成YOLO格式内容
        yolo_content = []
        for bbox_ann in labelme_data['shapes']:
            if bbox_ann.get('shape_type') != 'rectangle':
                continue  # 只处理"rectangle"类型检测框

            # 校验检测框必要信息
            if 'label' not in bbox_ann or bbox_ann['label'] not in bbox_class_map:
                return False, f"边界框标签未定义：{bbox_ann.get('label')}（仅支持{list(bbox_class_map.keys())}）"
            if 'points' not in bbox_ann or len(bbox_ann['points']) != 2:
                return False, f"边界框坐标异常：{bbox_ann['points']}（需2个顶点）"

            # 5.1 计算检测框归一化坐标（YOLO格式：center_x, center_y, width, height）
            # 提取矩形框两个顶点坐标（无需转int，避免精度丢失）
            (x1, y1), (x2, y2) = bbox_ann['points']
            # 确保坐标顺序正确（左上角→右下角）
            bbox_left = min(x1, x2)
            bbox_right = max(x1, x2)
            bbox_top = min(y1, y2)
            bbox_bottom = max(y1, y2)

            # 计算边界框的 中心坐标 和 宽高 （归一化）
            center_x = (bbox_left + bbox_right) / 2 / img_width
            center_y = (bbox_top + bbox_bottom) / 2 / img_height
            bbox_w = (bbox_right - bbox_left) / img_width
            bbox_h = (bbox_bottom - bbox_top) / img_height

            # 5.2 匹配当前检测框内的关键点
            bbox_keypoints = {}  # 存储当前框内的关键点（key：关键点标签，value：(x_norm, y_norm, visibility)）
            for kp in keypoints_list:
                # 判断关键点是否在检测框内（边缘不算，避免跨框误匹配）
                if (bbox_left < kp['x'] < bbox_right) and (bbox_top < kp['y'] < bbox_bottom):
                    if kp['label'] in keypoint_order:
                        # 关键点坐标归一化
                        kp_x_norm = kp['x'] / img_width
                        kp_y_norm = kp['y'] / img_height
                        bbox_keypoints[kp['label']] = (kp_x_norm, kp_y_norm, kp['visibility'])
                    else:
                        print(f"⚠️ 关键点标签不匹配 {kp['label']}")

            # 5.3 组装YOLO行（检测框 + 按顺序排列的关键点）
            # 检测框部分：类别ID + 中心坐标 + 宽高
            yolo_line = [str(bbox_class_map[bbox_ann['label']])]
            yolo_line.extend([
                f"{center_x:.{decimal_places}f}",
                f"{center_y:.{decimal_places}f}",
                f"{bbox_w:.{decimal_places}f}",
                f"{bbox_h:.{decimal_places}f}"
            ])
            # 关键点部分：按预设顺序添加（无关键点则补0 0 0）
            for kp_label in keypoint_order:     # 遍历预先定义的关键点类别
                if kp_label in bbox_keypoints:  # 遍历实际标注的关键点类别
                    kp_x, kp_y, kp_vis = bbox_keypoints[kp_label]
                    yolo_line.extend([
                        f"{kp_x:.{decimal_places}f}",
                        f"{kp_y:.{decimal_places}f}",
                        str(kp_vis)
                    ])
                else:
                    # 无该关键点：坐标0 + 可见性0
                    yolo_line.extend(["0.0", "0.0", "0"])

            # 添加当前行到内容中
            yolo_content.append(' '.join(yolo_line))

        # 6. 写入TXT文件（若有标注内容才写入，避免空文件）
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
        json_subdir: str = "labels_json",
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

        print(
            f"分组[{group}]处理完成：成功{len(json_files) - len([r for r in failed_records if f'分组[{group}]' in r])}个，失败{len([r for r in failed_records if f'分组[{group}]' in r])}个\n")

    # 输出最终统计结果
    print("===== 批量处理结束 =====")
    print(f"总统计：成功{total_success}个，失败{total_failed}个")
    if failed_records:
        print("\n❌ 失败文件详情：")
        for idx, record in enumerate(failed_records, 1):
            print(f"  {idx}. {record}")


if __name__ == '__main__':
    # -------------------------- 用户配置区 --------------------------
    DATASET_ROOT = r'D:\1_Python\datasets\liquids'  # 数据集根目录
    # TARGET_GROUPS = ['01', '04']  # 需处理的分组
    TARGET_GROUPS = ['02']  # 需处理的分组

    # 执行批量处理
    batch_process_groups(
        root_dir=DATASET_ROOT,
        group_index=TARGET_GROUPS
    )

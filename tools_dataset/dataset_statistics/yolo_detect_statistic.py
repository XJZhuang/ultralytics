#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：81_datasets_statistic.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/10 14:40 
@explain : 统计 smoke helmet 数据集的标签数量（区分训练集和验证集）
'''
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
from ultralytics.utils import LOGGER

'''
@Project ：Dataset Analysis
@File    ：dataset_statistic_dynamic.py
@Date    ：2025/10/10
@explain : 统计 YOLO 数据集标签分布（基于动态相对尺度，不依赖绝对像素阈值）
'''

import os
import glob
import statistics
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import matplotlib

# 防止在无图形界面的服务器上报错
matplotlib.use('TkAgg')


class DatasetAnalyzer:
    def __init__(self, class_names):
        self.class_names = class_names
        self._reset_stats()

    def _reset_stats(self):
        self.stats = {
            'total_images': 0,
            'empty_images': 0,
            'total_instances': 0,
            'img_sizes': [],
            'instances_per_img': [],    # 记录每张图的目标数，用于密度分析
            'class_stats': defaultdict(lambda: {
                'count': 0,
                'images': 0,
                'areas': [],  # 绝对像素面积
                'rel_areas': [],  # 相对面积比例 (Sqrt(ObjArea) / Sqrt(ImgArea))
                'ratios': [],  # 宽高比
                'center_xs': [],    # 记录中心点坐标用于偏见分析
                'center_ys': [],
                'parent_img_w': [], # 记录每个实例所属图片的分辨率
                'parent_img_h': []
            }),
            # 使用相对概念统计尺度分布
            'scale_dist': {
                'tiny': 0,  # 相对边长 < 3% (极小)
                'small': 0,  # 3% ~ 10% (小)
                'medium': 0,  # 10% ~ 30% (中)
                'large': 0  # > 30% (大)
            }
        }

    def process_folder(self, images_root, labels_root, folder_name):
        img_folder = os.path.join(images_root, folder_name)
        label_folder = os.path.join(labels_root, folder_name)

        if not os.path.exists(img_folder):
            print(f"警告: 文件夹不存在 {img_folder}")
            return

        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(img_folder, ext)))

        self.stats['total_images'] += len(image_files)

        for img_path in tqdm(image_files, desc=f"统计 {folder_name}"):
            # 1. 读取图片真实尺寸
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
                    self.stats['img_sizes'].append((img_w, img_h))

                    # 计算当前图片的基准尺度 (几何平均值 Sqrt(Area)) 比如 1920x1080 -> 约 1440
                    img_scale_base = math.sqrt(img_w * img_h)
            except Exception as e:
                print(f"无法读取图片: {img_path}, 错误: {e}")
                continue

            # 2. 读取标签
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_folder, f"{img_name}.txt")

            has_label = False
            classes_in_this_image = set()
            count_in_this_image = 0  # 当前图目标计数

            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

                if lines:
                    has_label = True
                    for line in lines:
                        parts = line.split()
                        try:
                            cls_id = int(parts[0])
                            # 兼容不同格式，防止越界
                            if len(parts) >= 5:
                                cx, cy, w_norm, h_norm = map(float, parts[1:5])
                            else:
                                LOGGER.error(f"标签格式错误: {label_path}")
                                cx, cy, w_norm, h_norm = 0, 0, 0, 0

                            if 0 <= cls_id < len(self.class_names):
                                cls_name = self.class_names[cls_id]

                                self.stats['total_instances'] += 1
                                count_in_this_image += 1
                                self.stats['class_stats'][cls_name]['count'] += 1
                                classes_in_this_image.add(cls_name)

                                if w_norm > 0 and h_norm > 0:
                                    # A. 绝对像素计算
                                    real_w_px = w_norm * img_w
                                    real_h_px = h_norm * img_h
                                    pixel_area = real_w_px * real_h_px

                                    # 计算宽高比 (Width / Height)
                                    # > 1: 扁 (宽大于高)
                                    # < 1: 瘦 (高大于宽)
                                    wh_ratio = real_w_px / real_h_px

                                    self.stats['class_stats'][cls_name]['areas'].append(pixel_area)
                                    self.stats['class_stats'][cls_name]['ratios'].append(wh_ratio)
                                    self.stats['class_stats'][cls_name]['center_xs'].append(cx)
                                    self.stats['class_stats'][cls_name]['center_ys'].append(cy)

                                    # 记录分辨率来源 ---
                                    self.stats['class_stats'][cls_name]['parent_img_w'].append(img_w)
                                    self.stats['class_stats'][cls_name]['parent_img_h'].append(img_h)

                                    obj_scale = math.sqrt(pixel_area)
                                    # 目标相对于整张图的比例 (0.0 ~ 1.0)
                                    relative_ratio = obj_scale / img_scale_base
                                    self.stats['class_stats'][cls_name]['rel_areas'].append(relative_ratio)

                                    # C. 动态归类
                                    # < 3% : 极小 (Tiny) - 在1080P里对应 < 43像素宽
                                    # 3% ~ 10% : 小 (Small) - 43 ~ 144像素
                                    # 10% ~ 30% : 中 (Medium) - 144 ~ 430像素
                                    # > 30% : 大 (Large) - > 430像素
                                    if relative_ratio < 0.03:
                                        self.stats['scale_dist']['tiny'] += 1
                                    elif relative_ratio < 0.10:
                                        self.stats['scale_dist']['small'] += 1
                                    elif relative_ratio < 0.30:
                                        self.stats['scale_dist']['medium'] += 1
                                    else:
                                        self.stats['scale_dist']['large'] += 1
                            else:
                                LOGGER.error(f"标签格式错误: {label_path}")
                        except Exception:
                            pass

            self.stats['instances_per_img'].append(count_in_this_image)

            if not has_label:
                self.stats['empty_images'] += 1
            else:
                for cls_name in classes_in_this_image:
                    self.stats['class_stats'][cls_name]['images'] += 1

    def merge(self, other):
        """合并多个数据集的统计结果"""
        self.stats['total_images'] += other.stats['total_images']
        self.stats['empty_images'] += other.stats['empty_images']
        self.stats['total_instances'] += other.stats['total_instances']
        self.stats['img_sizes'].extend(other.stats['img_sizes'])
        self.stats['instances_per_img'].extend(other.stats['instances_per_img'])  # 合并密度

        for k in self.stats['scale_dist']:
            self.stats['scale_dist'][k] += other.stats['scale_dist'][k]

        for cls, data in other.stats['class_stats'].items():
            self.stats['class_stats'][cls]['count'] += data['count']
            self.stats['class_stats'][cls]['images'] += data['images']
            self.stats['class_stats'][cls]['areas'].extend(data['areas'])
            self.stats['class_stats'][cls]['rel_areas'].extend(data['rel_areas'])
            self.stats['class_stats'][cls]['ratios'].extend(data['ratios'])
            self.stats['class_stats'][cls]['center_xs'].extend(data['center_xs'])
            self.stats['class_stats'][cls]['center_ys'].extend(data['center_ys'])
            # 合并新增字段
            self.stats['class_stats'][cls]['parent_img_w'].extend(data['parent_img_w'])
            self.stats['class_stats'][cls]['parent_img_h'].extend(data['parent_img_h'])

    def export_report(self, output_dir, prefix="Dataset"):
        if self.stats['total_images'] == 0: return
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        excel_path = os.path.join(output_dir, f"{prefix}_动态尺度报告_{timestamp}.xlsx")

        # --- Sheet 1: 简报 (保持原样) ---
        img_sizes = self.stats['img_sizes']
        if img_sizes:
            widths = [w for w, h in img_sizes]
            heights = [h for w, h in img_sizes]

            # 宽统计
            max_w, min_w = max(widths), min(widths)
            avg_w = statistics.mean(widths)

            # 高统计
            max_h, min_h = max(heights), min(heights)
            avg_h = statistics.mean(heights)

            # 分辨率统计 (按面积找最大/最小)
            img_pairs = list(zip(widths, heights))
            max_res_pair = max(img_pairs, key=lambda x: x[0] * x[1])
            min_res_pair = min(img_pairs, key=lambda x: x[0] * x[1])
            max_res_str = f"{max_res_pair[0]}x{max_res_pair[1]}"
            min_res_str = f"{min_res_pair[0]}x{min_res_pair[1]}"

            # 主流分辨率 (Top 5)
            res_summary_str = "; ".join(
                [f"{k}: {v}" for k, v in Counter([f"{w}x{h}" for w, h in img_sizes]).most_common(5)])
        else:
            max_w, min_w, avg_w = 0, 0, 0
            max_h, min_h, avg_h = 0, 0, 0
            max_res_str, min_res_str, res_summary_str = "N/A", "N/A", "无数据"

        # 密度统计
        inst_per_img = self.stats['instances_per_img']
        if inst_per_img:
            max_density = max(inst_per_img)
            min_density = min(inst_per_img)
            avg_density = statistics.mean(inst_per_img)
        else:
            max_density, min_density, avg_density = 0, 0, 0

        # 背景图计算
        total_imgs = max(1, self.stats['total_images'])
        empty_imgs = self.stats['empty_images']
        empty_ratio = empty_imgs / total_imgs

        summary_data = {
            '维度': [
                '图片总量', '标注框总量', '背景图数量（占比%)',
                '最大分辨率', '最小分辨率', '主流分辨率',
                '最大宽', '最小宽', '平均宽',
                '最大高', '最小高', '平均高',
                '单图最大目标数', '单图最小目标数', '平均每图目标数',
            ],
            '数值': [
                self.stats['total_images'],
                self.stats['total_instances'],
                f"{empty_imgs} ({empty_ratio:.2%})",
                max_res_str, min_res_str, res_summary_str,
                f"{max_w}", f"{min_w}", f"{int(avg_w)}",
                f"{max_h}", f"{min_h}", f"{int(avg_h)}",

                max_density, min_density, f"{avg_density:.1f}"
            ]
        }

        # --- Sheet 2: 详情分析 (含整体汇总) ---
        class_rows = []
        total_inst = max(1, self.stats['total_instances'])

        # 1. 准备全量数据 (ALL)
        all_ratios = []
        all_rel_areas = []
        all_cxs, all_cys = [], []
        all_p_ws, all_p_hs = [], []  # 所有实例对应的图片宽和高

        for data in self.stats['class_stats'].values():
            all_ratios.extend(data['ratios'])
            all_rel_areas.extend(data['rel_areas'])
            all_cxs.extend(data['center_xs'])
            all_cys.extend(data['center_ys'])
            all_p_ws.extend(data['parent_img_w'])
            all_p_hs.extend(data['parent_img_h'])

        # 2. 定义通用计算函数
        def calculate_metrics(name, count, img_count, ratios, rel_areas, cxs, cys, p_ws, p_hs):
            if count == 0: return None

            # 分辨率统计
            avg_img_w = statistics.mean(p_ws)
            std_img_w = statistics.stdev(p_ws) if len(p_ws) > 1 else 0
            avg_img_h = statistics.mean(p_hs)
            std_img_h = statistics.stdev(p_hs) if len(p_hs) > 1 else 0
            cv_img_w = std_img_w / avg_img_w if avg_img_w > 0 else 0
            cv_img_h = std_img_h / avg_img_h if avg_img_h > 0 else 0

            # 寻找最小/最大分辨率 (按面积排序)
            img_pairs = list(zip(p_ws, p_hs))
            min_res = min(img_pairs, key=lambda x: x[0] * x[1])
            max_res = max(img_pairs, key=lambda x: x[0] * x[1])

            if cv_img_w > 0.2 or cv_img_h > 0.2:
                src_eval = "分辨率非常离散"
            elif cv_img_w > 0.1:
                src_eval = "分辨率较为离散"
            else:
                src_eval = "分辨率较为统一"

            # 尺度统计
            avg_rel = statistics.mean(rel_areas)
            min_rel, max_rel = min(rel_areas), max(rel_areas)  # 极值
            std_rel = statistics.stdev(rel_areas) if len(rel_areas) > 1 else 0
            cv_rel = std_rel / avg_rel if avg_rel > 0 else 0

            # 动态计算该类别下的分布详情
            c_tiny = sum(1 for r in rel_areas if r < 0.03)
            c_small = sum(1 for r in rel_areas if 0.03 <= r < 0.10)
            c_medium = sum(1 for r in rel_areas if 0.10 <= r < 0.30)
            c_large = sum(1 for r in rel_areas if r >= 0.30)

            # 给出主要评价
            if avg_rel < 0.03:
                scale_desc = "极小"
            elif avg_rel < 0.10:
                scale_desc = "小"
            elif avg_rel < 0.30:
                scale_desc = "中"
            else:
                scale_desc = "大"

            # --- 形状统计 ---
            avg_ratio = statistics.mean(ratios)
            std_ratio = statistics.stdev(ratios) if len(ratios) > 1 else 0
            min_ratio, max_ratio = min(ratios), max(ratios)  # 极值

            # 位置统计
            avg_cx, std_cx = statistics.mean(cxs), statistics.stdev(cxs) if len(cxs) > 1 else 0
            avg_cy, std_cy = statistics.mean(cys), statistics.stdev(cys) if len(cys) > 1 else 0
            min_cx, max_cx = min(cxs), max(cxs)  # 极值
            min_cy, max_cy = min(cys), max(cys)  # 极值

            return {
                '类别名称': name,
                '实例数量': count,
                '数量占比': f"{count / total_inst:.2%}",
                '出现图片数': img_count,
                '图片覆盖率': f"{img_count / total_imgs:.2%}",

                '---图像特征---': '---',
                '平均分辨率': f"{int(avg_img_w)}x{int(avg_img_h)}",
                '最小分辨率 (Area Min)': f"{int(min_res[0])}x{int(min_res[1])}",  # 新增
                '最大分辨率 (Area Max)': f"{int(max_res[0])}x{int(max_res[1])}",  # 新增
                # 范围说明：0是完美统一，>0.2就很杂了
                '分辨率离散度(CV) [范围:0~1.0, 越大分辨率越离散]': f"{max(cv_img_w, cv_img_h):.2%}",
                # '数据源评价': src_eval,
                '宽度波动(Std)': f"{std_img_w:.1f} px",
                '高度波动(Std)': f"{std_img_h:.1f} px",

                '---尺度特征---': '---',
                '平均占比': f"{avg_rel:.2%}",
                '最小占比 (Min)': f"{min_rel:.2%}",  # 新增
                '最大占比 (Max)': f"{max_rel:.2%}",  # 新增
                '面积波动(Std)': f"{std_rel:.2%}",
                # 范围说明：0.1很稳，1.0差异极大
                '尺度离散度(CV) [范围:0~1.5, 越大面积大小差异越剧烈]': f"{cv_rel:.2f}",
                '极小目标占比 (<3%)': f"{c_tiny / count:.2%}",
                '小目标占比 (3~10%)': f"{c_small / count:.2%}",
                '中目标占比 (10~30%)': f"{c_medium / count:.2%}",
                '大目标占比 (>30%)': f"{c_large / count:.2%}",
                # '评价': scale_desc,

                '---形状特征---': '---',
                '平均宽高比': f"{avg_ratio:.2f}",
                '最小宽高比 (Min W/H)': f"{min_ratio:.2f}",  # 新增
                '最大宽高比 (Max W/H)': f"{max_ratio:.2f}",  # 新增
                # 范围说明：0是固定形状，0.5说明形状千奇百怪
                '形状波动(Std) [范围:0~0.5, 越大说明形状越乱]': f"{std_ratio:.2f}",

                '---位置分布---': '---',
                'X轴中心均值': f"{avg_cx:.2f}",
                # 范围说明：0.29是完全随机分布，<0.05是位置锁死
                'X轴偏差(Std)': f"{std_cx:.2f}",
                'X轴极值 (Min/Max)': f"{min_cx:.2f} / {max_cx:.2f}",  # 新增
                'Y轴中心均值': f"{avg_cy:.2f}",
                'Y轴偏差(Std)': f"{std_cy:.2f}",
                'Y轴极值 (Min/Max)': f"{min_cy:.2f} / {max_cy:.2f}"  # 新增
            }

        # 添加“整体”行
        # 3. 生成“整体”行 (Total)
        # 整体出现的图片数 = 总图数 - 空图数 (即只要有任意目标的图就算)
        imgs_with_objects = self.stats['total_images'] - self.stats['empty_images']

        total_row = calculate_metrics(
            "汇总", total_inst, imgs_with_objects, all_ratios, all_rel_areas, all_cxs, all_cys, all_p_ws, all_p_hs
        )
        if total_row:
            class_rows.append(total_row)

        # 4. 生成“单类”行
        for cls in self.class_names:
            data = self.stats['class_stats'][cls]
            if data['count'] == 0: continue

            row = calculate_metrics(
                cls,
                data['count'],
                data['images'],  # 传入该类别出现的图片数
                data['ratios'], data['rel_areas'],
                data['center_xs'], data['center_ys'],
                data['parent_img_w'], data['parent_img_h']
            )
            class_rows.append(row)

        df_summary = pd.DataFrame(summary_data)

        # 排序并转置 (保证Total在第一列)
        if class_rows:
            # 先分离Total和其他
            total_rows = [r for r in class_rows if "TOTAL" in r['类别名称']]
            other_rows = [r for r in class_rows if "TOTAL" not in r['类别名称']]
            # 其他按数量排序
            other_rows.sort(key=lambda x: x['实例数量'], reverse=True)
            # 合并
            final_rows = total_rows + other_rows

            df_classes = pd.DataFrame(final_rows)
            df_transposed = df_classes.copy()
            # 2. 将 '类别名称' 设为索引，这样转置后它会变成列头
            df_transposed.set_index('类别名称', inplace=True)
            # 3. 转置 (T)
            df_transposed = df_transposed.T
            # 4. 重置索引，把原来的列名（如'实例数量'）变成第一列
            df_transposed.reset_index(inplace=True)
            df_transposed.rename(columns={'index': '统计维度'}, inplace=True)
        else:
            df_classes = pd.DataFrame()
            df_transposed = pd.DataFrame()

        # 保存 Excel
        with pd.ExcelWriter(excel_path) as writer:
            df_summary.to_excel(writer, sheet_name='简报', index=False)
            df_transposed.to_excel(writer, sheet_name='深度详情', index=False)

        print(f"Excel 报告已保存: {excel_path}")
        # 绘图忽略Total行
        if other_rows:
            self._plot_charts(output_dir, prefix, pd.DataFrame(other_rows))

    def _plot_charts(self, output_dir, prefix, df_classes):
        if df_classes.empty: return

        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'{prefix} - 统计图表', fontsize=20)

        # 1. 尺度分布饼图
        sizes = [self.stats['scale_dist'][k] for k in ['tiny', 'small', 'medium', 'large']]
        valid_data = [(l, s) for l, s in zip(['极小', '小', '中', '大'], sizes) if s > 0]
        if valid_data:
            axes[0].pie([x[1] for x in valid_data], labels=[x[0] for x in valid_data], autopct='%1.1f%%')
            axes[0].set_title("相对尺度分布")

        sns.barplot(data=df_classes, x='实例数量', y='类别名称', hue='类别名称', legend=False, ax=axes[1], palette="viridis")
        axes[1].set_title("各类别实例数量")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_分析图表.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    # 消防/安防数据集
    datasets_name = "fire_security"
    CLASS_NAMES = ["person", "head", "helmet", "cigarette"]
    TRAIN_FOLDERS = ["train01", "train03", "train04", "train05"]
    VAL_FOLDERS = ["val01", "val03", "val04", "val05"]

    # 开关柜数据集
    # datasets_name = "switchgear"
    # CLASS_NAMES = ["light", "knob", "pointer", "switch", "breaker", "liquid"]
    # TRAIN_FOLDERS = ["train01", "train02", "train03", "train04", "train05", "train06", ]
    # VAL_FOLDERS = ["val01", "val02", "val03", "val04", "val05", "val06", ]

    # ROI液位计数据集
    # datasets_name = "liquids"
    # CLASS_NAMES = ["liquid"]
    # TRAIN_FOLDERS = ["train01", "train02", ]
    # VAL_FOLDERS = ["val01", "val02", ]

    IMAGES_ROOT = os.path.join(r"D:\1_Python\datasets", datasets_name, "images")
    LABELS_ROOT = os.path.join(r"D:\1_Python\datasets", datasets_name, "labels")
    OUTPUT_DIR = os.path.join(r"D:\1_Python\datasets", datasets_name, "analysis_result")

    # 2. Process Train
    print(">>> 正在分析训练集 (Train Set)...")
    analyzer_train = DatasetAnalyzer(CLASS_NAMES)
    for folder in TRAIN_FOLDERS:
        analyzer_train.process_folder(IMAGES_ROOT, LABELS_ROOT, folder)

    # 3. Process Val
    print("\n>>> 正在分析验证集 (Val Set)...")
    analyzer_val = DatasetAnalyzer(CLASS_NAMES)
    for folder in VAL_FOLDERS:
        analyzer_val.process_folder(IMAGES_ROOT, LABELS_ROOT, folder)

    # 4. Process Total (Merge)
    print("\n>>> 正在合并生成全集统计 (Total Set)...")
    analyzer_total = DatasetAnalyzer(CLASS_NAMES)
    analyzer_total.merge(analyzer_train)
    analyzer_total.merge(analyzer_val)

    # 5. Export All
    print("\n>>> 正在导出 Excel 报告和图表...")
    # Train Report
    analyzer_train.export_report(OUTPUT_DIR, prefix="01_训练集_Train")
    # Val Report
    analyzer_val.export_report(OUTPUT_DIR, prefix="02_验证集_Val")
    # Total Report
    analyzer_total.export_report(OUTPUT_DIR, prefix="03_全数据集_Total")

    print(f"\n全部完成！所有结果已保存至: {OUTPUT_DIR}")

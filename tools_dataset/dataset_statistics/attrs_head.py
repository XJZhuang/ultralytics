#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/10 14:40 
@explain : 统计指定数据集的标签数量
'''

# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Dataset Analysis
@File    ：attr_dataset_statistic.py
@explain : 针对 Head 类别的多属性（吸烟、安全帽）进行全维深度统计
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

# 防止服务器无GUI报错
matplotlib.use('TkAgg')


class AttributeAnalyzer:
    def __init__(self, target_class_id=1, smoke_idx=-2, helmet_idx=-1):
        """
        :param target_class_id: 要统计的类别ID (Head=1)
        :param smoke_idx: txt中smoke属性的列索引
        :param helmet_idx: txt中no_helmet属性的列索引
        """
        self.target_id = target_class_id
        self.smoke_idx = smoke_idx
        self.helmet_idx = helmet_idx

        # 定义属性组合映射
        self.attr_map = {
            (0, 0): "正常 (无烟_戴帽)",
            (0, 1): "违规 (无烟_无帽)",
            (1, 0): "违规 (吸烟_戴帽)",
            (1, 1): "双重 (吸烟_无帽)"
        }
        self._reset_stats()

    def _reset_stats(self):
        self.stats = {
            'total_images': 0,
            'empty_images': 0,
            'total_instances': 0,
            'img_sizes': [],
            'instances_per_img': [],
            # key 是属性组合名称 (如 "正常 (无烟_戴帽)")
            'attr_stats': defaultdict(lambda: {
                'count': 0,
                'images': 0,
                'areas': [],
                'rel_areas': [],
                'ratios': [],
                'center_xs': [],
                'center_ys': [],
                'parent_img_w': [],
                'parent_img_h': []
            })
        }

    def process_folder(self, images_root, labels_root, folder_name):
        img_folder = os.path.join(images_root, folder_name)
        # 如果 labels 就在 labels_root 下 (如 labels/train01)，或者 labels_root 就是根目录
        # 这里假设结构与之前一致: root/labels/train01
        label_folder = os.path.join(labels_root, folder_name)

        # 兼容逻辑：如果 label_folder 不存在，尝试直接用 labels_root (如果传入的是子路径)
        if not os.path.exists(label_folder):
            if os.path.exists(os.path.join(labels_root, f"{folder_name}.txt")):
                # 这种情况很少见，通常是文件夹结构
                pass
            elif os.path.exists(labels_root):
                # 尝试直接读取 labels_root 下的 txt (非文件夹结构)
                pass

        if not os.path.exists(img_folder):
            print(f"⚠️ 警告: 图片文件夹不存在 {img_folder}")
            return

        # 获取图片
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(img_folder, ext)))

        self.stats['total_images'] += len(image_files)

        for img_path in tqdm(image_files, desc=f"统计 {folder_name}"):
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
                    self.stats['img_sizes'].append((img_w, img_h))
                    img_scale_base = math.sqrt(img_w * img_h)
            except Exception:
                continue

            img_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_folder, f"{img_name}.txt")

            has_target = False
            attrs_in_this_image = set()
            count_in_this_image = 0

            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]

                if lines:
                    for line in lines:
                        parts = line.split()
                        try:
                            # 1. 检查类别ID
                            cls_id = int(parts[0])
                            if cls_id != self.target_id:
                                continue

                            has_target = True
                            count_in_this_image += 1

                            # 2. 解析属性
                            # 兼容 1.0 或 1 写法
                            smoke = int(float(parts[self.smoke_idx]))
                            no_helmet = int(float(parts[self.helmet_idx]))

                            # 校验合法性
                            if smoke not in [0, 1] or no_helmet not in [0, 1]:
                                continue

                            attr_name = self.attr_map[(smoke, no_helmet)]

                            # 3. 解析坐标 (x, y, w, h)
                            cx, cy, w_norm, h_norm = map(float, parts[1:5])

                            if w_norm > 0 and h_norm > 0:
                                # 记录数据
                                self.stats['total_instances'] += 1
                                self.stats['attr_stats'][attr_name]['count'] += 1
                                attrs_in_this_image.add(attr_name)

                                pixel_area = (w_norm * img_w) * (h_norm * img_h)
                                wh_ratio = (w_norm * img_w) / (h_norm * img_h)
                                obj_scale = math.sqrt(pixel_area)
                                rel_area = obj_scale / img_scale_base

                                stats = self.stats['attr_stats'][attr_name]
                                stats['areas'].append(pixel_area)
                                stats['ratios'].append(wh_ratio)
                                stats['rel_areas'].append(rel_area)
                                stats['center_xs'].append(cx)
                                stats['center_ys'].append(cy)
                                stats['parent_img_w'].append(img_w)
                                stats['parent_img_h'].append(img_h)

                        except (IndexError, ValueError):
                            pass

            self.stats['instances_per_img'].append(count_in_this_image)

            if not has_target:
                self.stats['empty_images'] += 1
            else:
                for attr in attrs_in_this_image:
                    self.stats['attr_stats'][attr]['images'] += 1

    def merge(self, other):
        self.stats['total_images'] += other.stats['total_images']
        self.stats['empty_images'] += other.stats['empty_images']
        self.stats['total_instances'] += other.stats['total_instances']
        self.stats['img_sizes'].extend(other.stats['img_sizes'])
        self.stats['instances_per_img'].extend(other.stats['instances_per_img'])

        for attr, data in other.stats['attr_stats'].items():
            target = self.stats['attr_stats'][attr]
            target['count'] += data['count']
            target['images'] += data['images']
            target['areas'].extend(data['areas'])
            target['rel_areas'].extend(data['rel_areas'])
            target['ratios'].extend(data['ratios'])
            target['center_xs'].extend(data['center_xs'])
            target['center_ys'].extend(data['center_ys'])
            target['parent_img_w'].extend(data['parent_img_w'])
            target['parent_img_h'].extend(data['parent_img_h'])

    def export_report(self, output_dir, prefix="Attr_Stat"):
        if self.stats['total_images'] == 0: return
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        excel_path = os.path.join(output_dir, f"{prefix}_属性深度报告_{timestamp}.xlsx")

        # --- 1. 定义计算函数 ---
        def calculate_metrics(name, count, img_count, ratios, rel_areas, cxs, cys, p_ws, p_hs):
            if count == 0: return None

            # 分辨率统计
            avg_img_w = statistics.mean(p_ws)
            std_img_w = statistics.stdev(p_ws) if len(p_ws) > 1 else 0
            avg_img_h = statistics.mean(p_hs)
            std_img_h = statistics.stdev(p_hs) if len(p_hs) > 1 else 0
            cv_img_w = std_img_w / avg_img_w if avg_img_w > 0 else 0
            cv_img_h = std_img_h / avg_img_h if avg_img_h > 0 else 0

            if cv_img_w > 0.2 or cv_img_h > 0.2:
                src_eval = "极杂"
            elif cv_img_w > 0.1:
                src_eval = "波动"
            else:
                src_eval = "统一"

            # 尺度统计
            avg_rel = statistics.mean(rel_areas)
            std_rel = statistics.stdev(rel_areas) if len(rel_areas) > 1 else 0
            cv_rel = std_rel / avg_rel if avg_rel > 0 else 0

            # 分布占比
            c_tiny = sum(1 for r in rel_areas if r < 0.03)
            c_small = sum(1 for r in rel_areas if 0.03 <= r < 0.10)
            c_medium = sum(1 for r in rel_areas if 0.10 <= r < 0.30)
            c_large = sum(1 for r in rel_areas if r >= 0.30)

            if avg_rel < 0.03:
                scale_desc = "极小"
            elif avg_rel < 0.10:
                scale_desc = "小"
            elif avg_rel < 0.30:
                scale_desc = "中"
            else:
                scale_desc = "大"

            # 形状统计
            avg_ratio = statistics.mean(ratios)
            std_ratio = statistics.stdev(ratios) if len(ratios) > 1 else 0

            # 位置统计
            avg_cx = statistics.mean(cxs)
            std_cx = statistics.stdev(cxs) if len(cxs) > 1 else 0
            avg_cy = statistics.mean(cys)
            std_cy = statistics.stdev(cys) if len(cys) > 1 else 0

            total_inst = max(1, self.stats['total_instances'])
            total_imgs = max(1, self.stats['total_images'])

            # 如果 img_count 为 -1，说明是聚合数据，无法精确计算覆盖率，显示 N/A
            img_cover_str = f"{img_count / total_imgs:.2%}" if img_count >= 0 else "N/A"
            img_count_str = str(img_count) if img_count >= 0 else "N/A"

            return {
                '属性分组': name,
                '实例数量': count,
                '数量占比': f"{count / total_inst:.2%}",
                '出现图片数': img_count_str,
                '图片覆盖率': img_cover_str,

                '--- 来源图像特征 (Resolution) ---': '',
                '平均分辨率': f"{int(avg_img_w)}x{int(avg_img_h)}",
                '分辨率离散度(CV) [>0.2极杂]': f"{max(cv_img_w, cv_img_h):.2%}",
                '数据源评价': src_eval,

                '--- 尺度特征 (Size) ---': '',
                '平均占比': f"{avg_rel:.2%}",
                '尺度离散度(CV) [>1.0差异剧烈]': f"{cv_rel:.2f}",
                '极小目标占比 (<3%)': f"{c_tiny / count:.2%}",
                '小目标占比 (3~10%)': f"{c_small / count:.2%}",
                '评价': scale_desc,

                '--- 形状特征 (Shape) ---': '',
                '平均宽高比': f"{avg_ratio:.2f}",
                '形状波动(Std)': f"{std_ratio:.2f}",

                '--- 位置分布 (Position) ---': '',
                'X轴偏差(Std)': f"{std_cx:.2f}",
                'Y轴偏差(Std)': f"{std_cy:.2f}"
            }

        class_rows = []

        # 2. 初始化聚合容器 (修正 Key 名称)
        agg_data = {
            'total': defaultdict(list),
            'smoke_1': defaultdict(list),  # 所有吸烟
            'smoke_0': defaultdict(list),  # 所有不吸烟
            'no_helmet_1': defaultdict(list),  # 所有未戴帽 (违规)
            'no_helmet_0': defaultdict(list)  # 所有戴帽 (正常)
        }
        agg_counts = defaultdict(int)

        # 3. 遍历基础属性数据并聚合
        for attr_name, data in self.stats['attr_stats'].items():
            if data['count'] == 0: continue

            # A. 添加 4 个基础组合的行
            row = calculate_metrics(
                attr_name, data['count'], data['images'],
                data['ratios'], data['rel_areas'], data['center_xs'], data['center_ys'],
                data['parent_img_w'], data['parent_img_h']
            )
            class_rows.append(row)

            # B. 判断属性类别
            # attr_map 定义：
            # (1, x) -> 吸烟 (smoke_1)
            # (x, 1) -> 无帽 (no_helmet_1)

            # 通过字符串名称判断更直观
            is_smoke = "吸烟" in attr_name
            is_no_helmet = "无帽" in attr_name  # "违规 (无烟_无帽)" 或 "双重 (吸烟_无帽)"

            # C. 执行数据聚合
            def extend_agg(key, d):
                agg_data[key]['ratios'].extend(d['ratios'])
                agg_data[key]['rel_areas'].extend(d['rel_areas'])
                agg_data[key]['cxs'].extend(d['center_xs'])
                agg_data[key]['cys'].extend(d['center_ys'])
                agg_data[key]['p_ws'].extend(d['parent_img_w'])
                agg_data[key]['p_hs'].extend(d['parent_img_h'])
                agg_counts[key] += d['count']

            extend_agg('total', data)
            extend_agg('smoke_1' if is_smoke else 'smoke_0', data)
            # 修复了这里的KeyError：现在 agg_data 里有 no_helmet_1 了
            extend_agg('no_helmet_1' if is_no_helmet else 'no_helmet_0', data)

        # 4. 生成聚合行

        # Total
        d = agg_data['total']
        total_row = calculate_metrics(
            "★ 整体汇总 (TOTAL)", agg_counts['total'],
            self.stats['total_images'] - self.stats['empty_images'],
            d['ratios'], d['rel_areas'], d['cxs'], d['cys'], d['p_ws'], d['p_hs']
        )
        if total_row: class_rows.insert(0, total_row)

        # 单属性汇总 (Smoke, No Helmet)
        aggregates = [
            ('>>> 汇总：吸烟 (Smoke=1)', 'smoke_1'),
            ('>>> 汇总：未戴帽 (NoHelmet=1)', 'no_helmet_1')
        ]

        for label, key in aggregates:
            d = agg_data.get(key)
            if not d or agg_counts[key] == 0: continue
            row = calculate_metrics(
                label, agg_counts[key], -1,  # 聚合属性图片数难以去重，填-1表示N/A
                d['ratios'], d['rel_areas'], d['cxs'], d['cys'], d['p_ws'], d['p_hs']
            )
            class_rows.append(row)

        # 5. 导出 Excel
        if class_rows:
            df = pd.DataFrame(class_rows)

            # 转置 (Transpose)
            df_t = df.copy()
            df_t.set_index('属性分组', inplace=True)
            df_t = df_t.T
            df_t.reset_index(inplace=True)
            df_t.rename(columns={'index': '统计维度'}, inplace=True)

            with pd.ExcelWriter(excel_path) as writer:
                df_t.to_excel(writer, sheet_name='属性深度分析', index=False)
                df.to_excel(writer, sheet_name='原始列表', index=False)

            print(f"Excel 报告已保存: {excel_path}")
            self._plot_charts(output_dir, prefix, df)
    def _plot_charts(self, output_dir, prefix, df):
        # 过滤掉聚合行，只画 4 个基础组合
        plot_df = df[~df['属性分组'].str.contains("★|>>>")].copy()
        if plot_df.empty: return

        sns.set_theme(style="whitegrid")
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'{prefix} - 属性分布可视化', fontsize=20)

        # 饼图：数量占比
        axes[0].pie(plot_df['实例数量'], labels=plot_df['属性分组'], autopct='%1.1f%%')
        axes[0].set_title("各属性组合占比")

        # 条形图：平均尺度 (看看是不是吸烟的目标特别小)
        # 需要把 "5.2%" 这种字符串转回数字
        try:
            plot_df['avg_scale_num'] = plot_df['平均占比'].str.rstrip('%').astype(float)
            sns.barplot(data=plot_df, x='avg_scale_num', y='属性分组', ax=axes[1], palette="magma", hue='属性分组',
                        legend=False)
            axes[1].set_title("各属性目标的平均画面占比 (%)")
            axes[1].set_xlabel("占比 (%) - 越小越难检测")
        except Exception as e:
            print(f"绘图出错: {e}")
            pass

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_Charts.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    # --- 配置区域 ---
    # 图片根目录
    IMAGES_ROOT = r"D:\1_Python\datasets\fire_security\images"
    # 标签根目录 (含属性列的txt)
    LABELS_ROOT = r"D:\1_Python\datasets\fire_security\labels_attr"
    OUTPUT_DIR = r"D:\1_Python\datasets\fire_security\analysis_result2"

    # 填入包含属性标注的文件夹
    TRAIN_FOLDERS = ["train01", "train03", "train04", "train05"]
    VAL_FOLDERS = ["val01", "val03", "val04", "val05"]

    # --- 运行分析 ---
    print(">>> 开始分析属性数据...")
    analyzer = AttributeAnalyzer(target_class_id=1, smoke_idx=-2, helmet_idx=-1)

    # 合并训练集和验证集一起统计
    for f in TRAIN_FOLDERS + VAL_FOLDERS:
        analyzer.process_folder(IMAGES_ROOT, LABELS_ROOT, f)

    analyzer.export_report(OUTPUT_DIR, prefix="Head_Attribute")
    print(f"\n完成！")

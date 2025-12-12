#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：01_evaluate_model.py
@Author  ：zhuangxujun
@Date    ：2025-9-19 9:23 
@explain : 待测试
'''

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ultralytics import YOLO


def analyze_training_results(results_dir):
    """分析训练结果"""
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    # 加载训练日志
    results_path = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_path):
        print(f"未找到训练日志: {results_path}")
        return

    results = pd.read_csv(results_path)

    # 创建图表保存目录
    os.makedirs(os.path.join(results_dir, "analysis"), exist_ok=True)

    # 1. 绘制损失函数曲线
    plt.figure(figsize=(12, 8))
    plt.plot(results['epoch'], results['train/box_loss'], label='训练集边界框损失')
    plt.plot(results['epoch'], results['train/cls_loss'], label='训练集分类损失')
    plt.plot(results['epoch'], results['train/dfl_loss'], label='训练集分布焦点损失')
    plt.plot(results['epoch'], results['val/box_loss'], label='验证集边界框损失', linestyle='--')
    plt.plot(results['epoch'], results['val/cls_loss'], label='验证集分类损失', linestyle='--')
    plt.plot(results['epoch'], results['val/dfl_loss'], label='验证集分布焦点损失', linestyle='--')
    plt.xlabel('轮次')
    plt.ylabel('损失值')
    plt.title('训练与验证损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "analysis", "loss_curves.png"))
    plt.close()

    # 2. 绘制mAP曲线
    plt.figure(figsize=(12, 8))
    plt.plot(results['epoch'], results['metrics/mAP50-95(B)'], label='mAP@50-95')
    plt.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP@50')
    plt.xlabel('轮次')
    plt.ylabel('mAP值')
    plt.title('mAP指标曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "analysis", "map_curves.png"))
    plt.close()

    # 3. 绘制精确率和召回率曲线
    plt.figure(figsize=(12, 8))
    plt.plot(results['epoch'], results['metrics/precision(B)'], label='精确率')
    plt.plot(results['epoch'], results['metrics/recall(B)'], label='召回率')
    plt.xlabel('轮次')
    plt.ylabel('值')
    plt.title('精确率和召回率曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "analysis", "precision_recall_curves.png"))
    plt.close()

    print(f"分析结果已保存至: {os.path.join(results_dir, 'analysis')}")


def main():
    # 训练结果目录，根据实际情况修改
    results_dir = "runs/detect/my_yolov8n"

    # 分析训练结果
    analyze_training_results(results_dir)

    # 加载最佳模型并显示性能指标
    best_model_path = os.path.join(results_dir, "weights", "best.pt")
    if os.path.exists(best_model_path):
        model = YOLO(best_model_path)
        metrics = model.val()
        print("\n最佳模型性能指标:")
        print(f"mAP@50-95: {metrics.box.map:.4f}")
        print(f"mAP@50: {metrics.box.map50:.4f}")
        print(f"精确率: {metrics.box.mp:.4f}")
        print(f"召回率: {metrics.box.mr:.4f}")
    else:
        print(f"未找到最佳模型: {best_model_path}")


if __name__ == "__main__":
    main()
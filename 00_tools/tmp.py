#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：tmp.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/16 9:24 
@explain : 
'''


def calculate_liquid_score_v2(predicted: float, actual: float) -> float:
    """
    根据预测读数和真实读数计算液位计的得分（新版本）。
    得分规则：
    - AE ≤ 1% (0.01)         -> 优秀，得 100 分
    - 1% < AE ≤ 5% (0.05)    -> 良好，得分从 100 线性降至 80
    - 5% < AE ≤ 10% (0.10)   -> 及格，得分从 80 线性降至 60
    - AE > 10% (0.10)        -> 不合格，得 0 分

    Args:
        predicted (float): 模型预测的液位计读数 (范围: 0.0 到 1.0)。
        actual (float): 液位计的真实读数 (范围: 0.0 到 1.0)。

    Returns:
        float: 计算出的得分 (范围: 0.0 到 100.0)。

    Raises:
        ValueError: 如果输入的读数不在 0.0 到 1.0 的范围内。
    """
    # 1. 输入验证：确保读数在合理范围内
    if not (0.0 <= predicted <= 1.0 and 0.0 <= actual <= 1.0):
        raise ValueError("预测读数和真实读数都必须在 0.0 到 1.0 的范围内。")

    # 2. 计算绝对误差 (AE)
    ae = abs(predicted - actual)

    # 3. 根据 AE 的值计算得分
    if ae <= 0.01:
        # 优秀线
        score = 100.0
    elif ae <= 0.05:
        # 良好区间: 100 -> 80
        # 公式推导: 总扣分20分，区间长度0.04，斜率为 20 / 0.04 = 500
        score = 100.0 - 500 * (ae - 0.01)
    elif ae <= 0.1:
        # 及格区间: 80 -> 60
        # 公式推导: 总扣分20分，区间长度0.05，斜率为 20 / 0.05 = 400
        score = 80.0 - 400 * (ae - 0.05)
    else:
        # 不合格
        score = 0.0

    # 4. 确保得分不会因为浮点精度问题超出 0-100 的范围
    return max(0.0, min(100.0, score))

# --- 示例 1: 优秀 (AE = 0.5%) ---
pred_1, actual_1 = 0.505, 0.50
score_1 = calculate_liquid_score_v2(pred_1, actual_1)
print(f"示例 1: 预测值 = {pred_1:.3f}, 真实值 = {actual_1:.3f}")
print(f"绝对误差 (AE) = {abs(pred_1 - actual_1):.2%}")
print(f"得分 = {score_1:.2f}\n")

# --- 示例 2: 良好区间 (AE = 3%) ---
pred_2, actual_2 = 0.53, 0.50
score_2 = calculate_liquid_score_v2(pred_2, actual_2)
print(f"示例 2: 预测值 = {pred_2:.3f}, 真实值 = {actual_2:.3f}")
print(f"绝对误差 (AE) = {abs(pred_2 - actual_2):.2%}")
print(f"得分 = {score_2:.2f}\n")

# --- 示例 3: 良好区间边界 (AE = 5%) ---
pred_3, actual_3 = 0.55, 0.50
score_3 = calculate_liquid_score_v2(pred_3, actual_3)
print(f"示例 3: 预测值 = {pred_3:.3f}, 真实值 = {actual_3:.3f}")
print(f"绝对误差 (AE) = {abs(pred_3 - actual_3):.2%}")
print(f"得分 = {score_3:.2f}\n")

# --- 示例 4: 及格区间 (AE = 8%) ---
pred_4, actual_4 = 0.58, 0.50
score_4 = calculate_liquid_score_v2(pred_4, actual_4)
print(f"示例 4: 预测值 = {pred_4:.3f}, 真实值 = {actual_4:.3f}")
print(f"绝对误差 (AE) = {abs(pred_4 - actual_4):.2%}")
print(f"得分 = {score_4:.2f}\n")

# --- 示例 5: 及格区间边界 (AE = 10%) ---
pred_5, actual_5 = 0.60, 0.50
score_5 = calculate_liquid_score_v2(pred_5, actual_5)
print(f"示例 5: 预测值 = {pred_5:.3f}, 真实值 = {actual_5:.3f}")
print(f"绝对误差 (AE) = {abs(pred_5 - actual_5):.2%}")
print(f"得分 = {score_5:.2f}\n")

# --- 示例 6: 不合格 (AE = 11%) ---
pred_6, actual_6 = 0.61, 0.50
score_6 = calculate_liquid_score_v2(pred_6, actual_6)
print(f"示例 6: 预测值 = {pred_6:.3f}, 真实值 = {actual_6:.3f}")
print(f"绝对误差 (AE) = {abs(pred_6 - actual_6):.2%}")
print(f"得分 = {score_6:.2f}\n")

# --- 示例 7: 反向误差 (预测值低于真实值) ---
pred_7, actual_7 = 0.47, 0.50
score_7 = calculate_liquid_score_v2(pred_7, actual_7)
print(f"示例 7: 预测值 = {pred_7:.3f}, 真实值 = {actual_7:.3f}")
print(f"绝对误差 (AE) = {abs(pred_7 - actual_7):.2%}")
print(f"得分 = {score_7:.2f}\n")
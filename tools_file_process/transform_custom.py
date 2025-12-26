#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：transform.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/10 19:39 
@explain : 
'''
import cv2
import numpy as np
import torch

from ultralytics.data.augment import LetterBox


def preprocess(im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(im))
        if im.shape[-1] == 3:
            im = im[..., ::-1]  # BGR to RGB
        im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous 将内存转为连续（解决负步长问题）
        im = torch.from_numpy(im)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    im = im.to(device)
    im = im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im


def pre_transform(im: list[np.ndarray]) -> list[np.ndarray]:
    letterbox = LetterBox(
        new_shape=(640, 640),
        auto=True,
        stride=32,
    )
    return [letterbox(image=x) for x in im]  # 对一个 batch 的每一张图片进行 LetterBox 填充


def letterbox_image(source_path, img_size=640, auto=False, device='cuda'):
    """
    将图片路径转换为 YOLO 模型需要的 Tensor 输入
    """
    # 1. 读取图片 (BGR 格式)
    img = cv2.imread(source_path)
    if img is None:
        raise FileNotFoundError(f"找不到图片: {source_path}")

    # 2. Letterbox 处理 (核心步骤)
    # auto=False: 强制填充到 img_size (例如640x640)，不使用动态最小填充
    # stride=32: 确保长宽是 32 的倍数
    letterbox = LetterBox(new_shape=(img_size, img_size), auto=auto, stride=32)
    img_lb = letterbox(image=img)

    # 3. 格式转换
    # BGR -> RGB
    img_lb = img_lb[:, :, ::-1]
    # HWC (高,宽,通道) -> CHW (通道,高,宽)
    img_lb = img_lb.transpose(2, 0, 1)
    # 内存连续化 (PyTorch 要求)
    img_lb = np.ascontiguousarray(img_lb)

    # 4. 转 Tensor 并归一化
    img_tensor = torch.from_numpy(img_lb).to(device)
    img_tensor = img_tensor.float()  # uint8 -> float32
    img_tensor /= 255.0  # 0-255 -> 0.0-1.0

    # 5. 增加 Batch 维度: (3, 640, 640) -> (1, 3, 640, 640)
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, img # 同时返回原图以便可视化
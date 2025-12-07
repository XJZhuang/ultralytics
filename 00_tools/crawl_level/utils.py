#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：utils.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/21 9:14 
@explain : 
'''
import os, imagehash
from PIL import Image
from tqdm import tqdm

def dedup_and_resize(raw_dir, out_dir, size=1024):
    os.makedirs(out_dir, exist_ok=True)
    phashes = set()
    idx = 0
    for fname in tqdm(os.listdir(raw_dir), desc="Deduplicate+Resize"):
        path = os.path.join(raw_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
        except:
            continue
        # 1. 感知哈希去重
        ph = imagehash.average_hash(img, hash_size=8)
        if ph in phashes:
            continue
        phashes.add(ph)
        # 2. 等比缩放到最长边 = size，其余补白
        img.thumbnail((size, size), Image.LANCZOS)
        new_im = Image.new("RGB", (size, size), (255, 255, 255))
        w, h = img.size
        left = (size - w) // 2
        top  = (size - h) // 2
        new_im.paste(img, (left, top))
        # 3. 顺序命名
        idx += 1
        save_path = os.path.join(out_dir, f"level_gauge_{idx:05d}.jpg")
        new_im.save(save_path, quality=95)
    print(">>> 完成！共保留", idx, "张")
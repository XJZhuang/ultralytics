#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：bing_crawl.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/21 9:22 
@explain : 
'''

import os, requests, re, time
from PIL import Image
from io import BytesIO
from tqdm import tqdm

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Referer": "https://www.bing.com/images/search?q=level+gauge"
}
SAVE = "data/raw"
os.makedirs(SAVE, exist_ok=True)

def bing_fetch(keyword, pages=3):
    count = 0
    for first in range(0, pages*35, 35):
        # 1. 老接口依旧能返回 JSONP 混排
        api = "https://www.bing.com/images/async"
        params = {
            "q": keyword,
            "first": first,
            "count": 35,
            "qft": "+filterui:imagesize-large",   # 只要大图
            "FORM": "IRFLTR",
            "mmasync": 1
        }
        resp = requests.get(api, headers=HEADERS, params=params, timeout=20)
        html = resp.text
        # 2. 2024 年字段是 "murl"
        urls = re.findall(r'"murl":"(.*?)"', html)
        print(f"[DEBUG] 第 {first//35+1} 页抓到 {len(urls)} 条 murl")
        for u in tqdm(urls, desc=f"down page {first//35+1}"):
            u = u.encode().decode('unicode_escape')
            try:
                r = requests.get(u, headers=HEADERS, timeout=15, stream=True)
                img = Image.open(BytesIO(r.content)).convert("RGB")
                if min(img.size) < 500:
                    continue
                count += 1
                img.save(os.path.join(SAVE, f"bing_{count:04d}.jpg"), quality=95)
                time.sleep(0.8)
            except Exception as e:
                # print(e)
                continue
    print(">>> 本次共保存", count, "张")

if __name__ == "__main__":
    bing_fetch("level gauge OR magnetic level indicator", pages=4)
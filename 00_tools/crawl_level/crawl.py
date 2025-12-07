#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：crawl.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/21 9:13 
@explain : 
'''
import os, re, time, hashlib, requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from utils import dedup_and_resize

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}
SAVE_RAW  = "data/raw"
SAVE_1024 = "data/1024"
os.makedirs(SAVE_RAW, exist_ok=True)
os.makedirs(SAVE_1024, exist_ok=True)

# ----------- 1. 百度图片 -----------
def baidu_fetch(keyword, need=300):
    url = "https://image.baidu.com/search/index"
    params = {
        "tn": "baiduimage",
        "word": keyword,
        "ie": "utf-8",
        "oe": "utf-8",
        "cl": 2,
        "ic": 0,
        "fr": "ala",
        "width": "",
        "height": "",
        "lm": -1,
        "st": -1,
    }
    # 先拿第一页 html
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.encoding = "utf-8"
    print('百度返回前 500 字符：', resp.text[:500])   # <- 新增
    urls = re.findall(r'"objURL":"(.*?)"', resp.text)
    print('百度解析到 URL 数量：', len(urls))        # <- 新增
    urls = [u.replace("\\/", "/") for u in urls]
    downlist(urls, "baidu", need)

# ----------- 2. 1688 商品图 -----------
def alibaba1688_fetch(keyword, need=200):
    # 1688 搜索接口
    s_url = "https://s.1688.com/selloffer/offer_search.htm"
    payload = {
        "keywords": keyword,
        "n": "y",
        "netType": "1",
        "spm": "",
    }
    resp = requests.get(s_url, headers=HEADERS, params=payload, timeout=15)
    print('1688 返回前 500 字符：', resp.text[:500])  # <- 新增
    soup = BeautifulSoup(resp.text, "lxml")
    urls = []
    for img in soup.select("img[src*='.alicdn.com']"):
        src = img.get("src") or img.get("data-lazy-src")
        if src and src.startswith("//"):
            src = "https:" + src
        # 取高清：替换后缀
        src = re.sub(r"_[0-9]+x[0-9]+\.jpg", "", src).split(".jpg")[0] + ".jpg"
        urls.append(src)
    downlist(urls, "1688", need)

# ----------- 通用下载 -----------
def downlist(urls, prefix, need):
    exist = set()
    for path in os.listdir(SAVE_RAW):
        exist.add(path)
    cnt, idx = 0, len(exist)
    pbar = tqdm(urls, desc=f"Download {prefix}")
    for u in pbar:
        if cnt >= need:
            break
        try:
            rsp = requests.get(u, headers=HEADERS, timeout=12)
            if rsp.status_code != 200:
                continue
            img = Image.open(BytesIO(rsp.content)).convert("RGB")
            # 过滤小图
            if img.width < 500 or img.height < 500:
                continue
            idx += 1
            save_path = os.path.join(SAVE_RAW, f"{prefix}_{idx:05d}.jpg")
            img.save(save_path, quality=95)
            cnt += 1
            pbar.set_postfix(got=cnt)
            time.sleep(0.5)
        except Exception as e:
            # print(e)
            continue


if __name__ == "__main__":
    keyword = "液位计"
    baidu_fetch(keyword, 300)
    alibaba1688_fetch(keyword, 200)
    print(">>> 开始去重 & 尺寸统一 ...")
    dedup_and_resize(SAVE_RAW, SAVE_1024, size=1024)
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：opencv_test.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/28 8:49 
@explain : 
'''
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

bgr_img = cv2.imread('../ultralytics/assets/bus.jpg')
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
bgr_img2 = cv2.cvtColor(bgr_img, cv2.COLOR_RGB2BGR)

plt.subplot(1, 3, 1)
plt.imshow(bgr_img)  # 显示 BGR 图片（颜色失真）
plt.title("BGR (OpenCV)")

plt.subplot(1, 3, 2)
plt.imshow(bgr_img2)  #
plt.title("RGB (OpenCV)")

plt.subplot(1, 3, 3)
plt.imshow(rgb_img)  #
plt.title("RGB (OpenCV)")
plt.show()

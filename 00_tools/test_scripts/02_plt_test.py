#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：1_Python 
@File    ：02_plt_test.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/15 9:08 
@explain : 
'''

from ultralytics.utils import plt_settings
import matplotlib.pyplot as plt  # scope for faster 'import ultralytics'


@plt_settings({"font.size": 12})
def plot_function():
    plt.figure()
    plt.plot([1, 2, 3])
    plt.show()


if __name__ == '__main__':
    plot_function()
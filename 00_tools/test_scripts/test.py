#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：test.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/22 9:21 
@explain : 
'''

from ultralytics.utils import LOGGER
LOGGER.info("中文1")
print("中文2")


import sys
print("默认编码：", sys.getdefaultencoding())  # 应输出 utf-8
print("stdout 编码：", sys.stdout.encoding)    # 应输出 utf-8
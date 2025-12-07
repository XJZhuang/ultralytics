#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：cost_time.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/20 10:07 
@explain : 
'''

import time
from functools import wraps
from ultralytics.utils import LOGGER

def timed(message):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            LOGGER.info(f"{message}: {func.__name__} cost time: {end_time - start_time} seconds")
            return result

        return wrapper
    return decorator


@timed("first time example_function函数调用")
def example_function():
    time.sleep(1)  # 模拟耗时操作
    print(111)
    print(222)


if __name__ == '__main__':
    example_function()

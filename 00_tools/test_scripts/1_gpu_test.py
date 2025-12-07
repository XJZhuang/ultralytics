#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：1_gpu_test.py
@Author  ：zhuangxujun
@Date    ：2025-9-16 17:43 
@explain : 
'''
import torch

if __name__ == '__main__':
    print("gpu_test")
    print("torch版本: ", torch.__version__)

    print(f"torch的CUDA版本: {torch.version.cuda}")
    print(f"torch的cuDNN: {torch.backends.cudnn.version()}")

    print(f"cuda是否可用: {torch.cuda.is_available()}")


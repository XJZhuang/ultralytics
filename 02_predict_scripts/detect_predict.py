#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：detect_predict.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/11/11 17:15 
@explain : 参数
'''


from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO("yolov8s.pt")
    model = YOLO("yolov8s.yaml")
    # model = YOLO("yolov8n.yaml")

    source = r"../ultralytics/assets/bus.jpg"

    results = model(
        source,
        # save=True,
        # stream=True,
        project="yolo-detect",
        verbose=False,
    )

    # for result in results:
    #     box = result.boxes

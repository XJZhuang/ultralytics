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

    model = YOLO("yolov8s.pt")
    # model = YOLO("yolov8s.yaml")
    # model = YOLO("yolov8n.yaml")

    source = r"../ultralytics/assets/zidane.jpg"

    results = model(
        source,
        save=True,
        # stream=True,
        project="yolo-detect",
        verbose=False,
    )

    for result in results:
        box = result.boxes
        # print(box.xyxy)
        # print(box.conf)
        # print(box.cls)
        """
        tensor([[ 747.3132,   41.4733, 1140.3921,  712.9236],
                [ 144.8750,  200.0329, 1107.1973,  712.7000],
                [ 437.3798,  434.4805,  529.9606,  717.0511]])
        tensor([0.8891, 0.8845, 0.7178])
        tensor([ 0.,  0., 27.])
        """

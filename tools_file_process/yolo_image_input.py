#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ultralytics 
@File    ：yolo_image_input.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/12/11 17:39 
@explain : 验证NMS的结果的坐标是基于 640 的
'''


from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from transform_custom import letterbox_image


if __name__ == '__main__':

    model = YOLO("yolov8s.pt")
    # model = YOLO("yolov8s.yaml")
    # model = YOLO("yolov8n.yaml")

    # source = r"../ultralytics/assets/zidane.jpg"
    #
    # results = model(
    #     source,
    #     # save=True,
    #     # stream=True,
    #     project="yolo-detect",
    #     verbose=False,
    # )

    # for result in results:
    #     box = result.boxes

    source = r"../ultralytics/assets/zidane.jpg"
    input_tensor, original_img = letterbox_image(source, device='cpu', auto=False)
    print(f"原始图片形状 (H, W): {original_img.shape[:2]}")  # 通常是 (H, W)=(720, 1280)
    print(f"输入 Tensor 形状: {input_tensor.shape}")

    results = model.model(input_tensor)
    det = non_max_suppression(results)[0]  # 结果：基于 640x640 输入图
    print(det[:, :4].max())
    print(det)
    """
    auto=False时，填充为640*640正方形。720*1280->360*640->上下填充(640-360)/2=140
    tensor([
    [373.6075, 160.6647, 569.9683, 495.3574,   0.8913,   0.0000],
    [ 74.1536, 240.1378, 553.0082, 495.4836,   0.8809,   0.0000],
    [218.5876, 357.2149, 265.1495, 497.7216,   0.7450,  27.0000]
    ])
    
    # 高级接口的输出
    tensor([[ 747.3132,   41.4733, 1140.3921,  712.9236],
        [ 144.8750,  200.0329, 1107.1973,  712.7000],
        [ 437.3798,  434.4805,  529.9606,  717.0511]])
    tensor([0.8891, 0.8845, 0.7178])
    tensor([ 0.,  0., 27.])
    其中，x1 = (640坐标 - 上边填充) / 缩放比例 = (373.6 - 0) / 0.5 = 747.2 ≈ 747.3132
    y1 = (160.6 - 140) / 0.5 = 41.2 ≈  41.4733
    
    auto=True时，填充为360*640-->384*640(向上取整到32的倍数)。缩放比例=640/1280=0.5
    tensor([[373.6566,  32.7366, 570.1960, 368.4618,   0.8891,   0.0000],
        [ 72.4375, 112.0164, 553.5986, 368.3500,   0.8845,   0.0000],
        [218.6899, 229.2402, 264.9803, 370.5256,   0.7178,  27.0000]])
    其中，x1 = (640坐标 - 上边填充) / 缩放比例 = (373.6 - 0) / 0.5 = 747.2 ≈ 747.3132
    y1 = (32.7 - 12) / 0.5 = 41.4 ≈  41.4733
    """

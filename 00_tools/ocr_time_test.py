#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：ocr_time_test.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/20 14:24 
@explain : 
'''
from paddleocr import PaddleOCR

from paddle_ocr.image_ocr import OcrModelApp
import cv2
import time

if __name__ == '__main__':
    ocr_model = PaddleOCR(
        text_detection_model_dir=r'../predict_model/PP-OCRv5_mobile_det',
        text_detection_model_name=r'PP-OCRv5_mobile_det',

        text_recognition_model_dir=r'../predict_model/PP-OCRv5_mobile_rec',
        text_recognition_model_name=r'PP-OCRv5_mobile_rec',

        # text_detection_model_dir=r'../predict_model/PP-OCRv5_server_det',
        # text_recognition_model_dir=r'../predict_model/PP-OCRv5_server_rec',

        # ocr_version="PP-OCRv3",
        # ocr_version="PP-OCRv4",
        # ocr_version="PP-OCRv5",
        # ocr_version="PP-OCRv5_mobile",
        use_doc_orientation_classify=False,  # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
        use_doc_unwarping=False,  # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
        use_textline_orientation=False,  # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
    )

    image_paths = [
        "../images/liquid/VID_20250926_184811_0021.png",
        # "../images/3_1755326565328_43.jpeg",
        # "../images/output_3_3_224.png"
    ]

    t0 = time.perf_counter()  # 开始计时
    for image_path in image_paths:
        image = cv2.imread(image_path)
        results = ocr_model.predict(image)
        for result in results:
            print(result["rec_texts"])
            result.save_to_img("./ocr_res")
    elapsed = time.perf_counter() - t0
    print(f"耗时: {elapsed * 1000:.2f} ms")  # 秒转毫秒

    # 耗时: 83 s   # v5-server
    # 耗时: 12 s   # v5-mobile
    # 耗时: 9 s    # v4-mobile
    # 耗时: 2 s    # v3-mobile

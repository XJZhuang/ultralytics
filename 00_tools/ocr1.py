#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：switchgear-ai 
@File    ：ocr1.py
@Author  ：DELL(zhuangxujun)
@Date    ：2025/10/20 19:52 
@explain : 
'''

from paddleocr import PaddleOCR

ocr_model = PaddleOCR(
    text_detection_model_dir=r'../predict_model/PP-OCRv5_mobile_det',
    text_detection_model_name=r'PP-OCRv5_mobile_det',
    text_recognition_model_dir=r'../predict_model/PP-OCRv5_mobile_rec',
    text_recognition_model_name=r'PP-OCRv5_mobile_rec',
    return_word_box=True,
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
output = ocr_model.predict(input="../images/liquid/VID_20250926_184811_0021.png")
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")

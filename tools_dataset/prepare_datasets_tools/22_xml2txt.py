# -*- coding: utf-8 -*-
'''
@Project:   ultralytics-main
@File:      22_xml2txt.py
@Author:    123
@Date:      2025/1/9 11:21
@Explain:   将 xml 格式转换为 txt格式
'''
from xml.etree import ElementTree as ET
import os

xml_path = r"D:\1_Python\datasets\1_detect_safety\labels_xml\train"
label_path = r"D:\1_Python\datasets\1_detect_safety\labels\train_reflective"
# image_path = r"D:\1_Python\datasets\1_detect_safety\images\train"

class_name = []

for xml_file in os.listdir(xml_path):
    xml2txt_name = os.path.splitext(xml_file)[0] + '.txt'

    tree = ET.parse(os.path.join(xml_path, xml_file))
    root = tree.getroot()
    sign_object = root.findall('object')
    img_size = root.find("size")
    img_width = int(img_size.find("width").text)
    img_height = int(img_size.find("height").text)

    if sign_object:
        for sign in sign_object:
            name = sign.find('name')
            sign_name = name.text   # 获取类别名称
            if sign_name not in class_name:
                class_name.append(sign_name)

            # 自动转换
            # class_id = class_name.index(sign_name)

            # 自动定义类别序号
            if sign_name == 'reflective_clothes':
                class_id = 3
            elif sign_name == 'other_clothes':
                class_id = 4
            else:
                print(sign_name)
                raise Exception

            bndbox = sign.find('bndbox')
            xmin = float(bndbox[0].text)
            ymin = float(bndbox[1].text)
            xmax = float(bndbox[2].text)
            ymax = float(bndbox[3].text)

            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            x_center = (xmin + (xmax - xmin) / 2) / img_width
            y_center = (ymin + (ymax - ymin) / 2) / img_height

            with open(os.path.join(label_path, xml2txt_name), 'a') as f:
                f.write("{} {} {} {} {}\n".format(class_id, x_center, y_center, width, height))

    else:
        raise Exception
        # with open(os.path.join(label_path,xml2txt_name),'a') as f:
        #    f.write("")

        # remove_img_name = os.path.splitext(xml_file)[0] + '.jpg'
        # if os.path.exists(os.path.join(image_path, remove_img_name)):
        #     os.remove(os.path.join(image_path, remove_img_name))
        #     print("remove {}".format(remove_img_name))

print("class_name:", class_name)

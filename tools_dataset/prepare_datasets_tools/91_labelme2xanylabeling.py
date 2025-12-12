# -*- coding: utf-8 -*-
'''
@Project:   ultralytics-main
@File:      91_labelme2xanylabeling.py
@Author:    123
@Date:      2025/1/20 16:48
@Explain:   labelme（2个对角点）--> xanylabeling（4个顶点）
'''

import os
import json

def conver_json(labelme_dir, x_anylabeling_dir):

    # 确保X-AnyLabeling目录存在
    os.makedirs(x_anylabeling_dir, exist_ok=True)
    count = 0
    # 遍历Labelme目录中的所有JSON文件
    for labelme_filename in os.listdir(labelme_dir):
        if labelme_filename.endswith('.json'):
            # 构建Labelme JSON文件的完整路径
            labelme_json_path = os.path.join(labelme_dir, labelme_filename)

            # 构建X-AnyLabeling JSON文件的完整路径
            x_anylabeling_filename = labelme_filename  # 可以保持原名，也可以根据需要修改
            x_anylabeling_json_path = os.path.join(x_anylabeling_dir, x_anylabeling_filename)

            # 加载Labelme JSON文件
            with open(labelme_json_path, 'r', encoding='utf-8') as f:
                labelme_data = json.load(f)

            if len(labelme_data['shapes']) == 0:
                count = count + 1
                print(f"{labelme_filename} --> 空标注={count}")
            # 遍历并转换形状信息
            for shape in labelme_data['shapes']:
                if shape['shape_type'] == 'rectangle':  # 只处理矩形形状
                    if len(shape['points']) == 2:   # 需要转换
                        # Labelme使用两个对角点表示矩形
                        [x1, y1], [x2, y2] = shape['points']

                        # X-AnyLabeling使用四个点表示矩形框
                        shape['points'] = [
                            [x1, y1],  # 左上角
                            [x2, y1],  # 右上角
                            [x2, y2],  # 右下角
                            [x1, y2]  # 左下角
                        ]
                    else:   # 不需要转换，直接保存
                        print(shape['points'])
                    # 保存X-AnyLabeling的JSON文件
                    with open(x_anylabeling_json_path, 'w', encoding='utf-8') as f:
                        json.dump(labelme_data, f, ensure_ascii=False, indent=4)

                    print(f"Converted {labelme_filename} to X-AnyLabeling format and saved as {x_anylabeling_filename}")

    print(f"Batch conversion completed.-->空={count}")


if __name__ == '__main__':

    # Labelme JSON文件所在目录
    labelme_dir = r'D:\1_Python\datasets\4_detect_electricity_baoan\labels_json0\train_'

    # X-AnyLabeling JSON文件保存目录
    x_anylabeling_dir = r'D:\1_Python\datasets\4_detect_electricity_baoan\labels_json\train_'

    # group_index = ['dp', 'kc', 'kz', 'lt', 'ml', 'sj']   # 坪大
    group_index = ['fh', 'fy', 'gs', 'sg', 'sj', 'sy', 'xa', 'xq', 'xx', 'ya']   # 宝安

    for index in group_index:
        conver_json(labelme_dir + index, x_anylabeling_dir + index)


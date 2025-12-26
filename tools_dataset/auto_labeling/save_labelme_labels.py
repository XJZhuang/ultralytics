#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
YOLOv8 自动标注工具 - 生成 LabelMe 格式的 JSON 文件
"""

import os
import json
import base64
from pathlib import Path
import traceback

import cv2
from ultralytics import YOLO


class YOLOv8AutoLabeler:
    """使用 YOLOv8 模型自动生成 LabelMe 格式标注"""

    def __init__(self, model_path: str, conf_threshold: float = 0.25, imgsz: int = 640):
        """
        初始化自动标注器
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz=imgsz
        self.class_names = self.model.names  # 获取类别名称

        print(f"✓ 模型加载成功: {model_path}")
        print(f"✓ 类别列表: {self.class_names}")
        print(f"✓ 置信度阈值: {conf_threshold}")
        print(f"✓ 推理尺寸: {imgsz}")

    def create_shape(
            self,
            label: str,
            bbox: list,
            shape_type: str = "rectangle"
    ) -> dict:
        """
        创建单个标注形状 (LabelMe 格式)

        Args:
            label: 类别标签
            bbox: [x1, y1, x2, y2] 边界框
            shape_type: 形状类型，rectangle 只需要左上和右下两个点
        """
        x1, y1, x2, y2 = bbox

        shape = {
            "label": label,
            "points": [
                [float(x1), float(y1)],  # 左上角
                [float(x2), float(y2)]   # 右下角
            ],
            "group_id": None,
            "description": "",
            "shape_type": shape_type,
            "flags": {},
            "mask": None
        }
        return shape

    def create_labelme_json(
            self,
            image_path: str,
            output_folder: str,
            input_folder: str,
            shapes: list,
    ) -> dict:
        """
        创建 LabelMe 格式的 JSON 数据
        """
        # 读取图片获取尺寸
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")

        height, width = img.shape[:2]

        # 计算相对路径 (从 labels 到 images)
        # 例如: ..\..\images\train06\xxx.png
        output_path = Path(output_folder)
        input_path = Path(input_folder)
        image_name = Path(image_path).name

        # 计算相对路径
        try:
            relative_path = os.path.relpath(image_path, output_folder)
        except ValueError:
            # 如果在不同驱动器上，使用简单的相对路径
            relative_path = f"..\\..\\images\\{input_path.name}\\{image_name}"

        labelme_data = {
            "version": "5.8.3",
            "flags": {},
            "shapes": shapes,
            "imagePath": relative_path,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        return labelme_data

    def predict_and_convert(self, image_path: str, output_folder: str, input_folder: str) -> dict:
        """
        对单张图片进行预测并转换为 LabelMe 格式
        """
        # YOLOv8 推理
        results = self.model.predict(
            source=image_path,
            conf=self.conf_threshold,
            verbose=False,
            imgsz=self.imgsz,
        )[0]

        shapes = []

        # 处理检测结果
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes

            for i in range(len(boxes)):
                # 获取边界框坐标 (xyxy 格式)
                bbox = boxes.xyxy[i].cpu().numpy().tolist()

                # 获取类别
                cls_id = int(boxes.cls[i].cpu().numpy())
                label = self.class_names[cls_id]

                # 创建形状 (rectangle 格式，只需要两个点)
                shape = self.create_shape(
                    label=label,
                    bbox=bbox,
                    shape_type="rectangle"
                )
                shapes.append(shape)

        # 创建 LabelMe JSON
        labelme_data = self.create_labelme_json(
            image_path=image_path,
            output_folder=output_folder,
            input_folder=input_folder,
            shapes=shapes,
        )

        return labelme_data

    def process_folder(
            self,
            input_folder: str,
            output_folder: str,
            image_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    ) -> dict:
        """处理整个文件夹的图片"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)

        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)

        # 获取所有图片文件
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))

        # 去重并排序
        image_files = sorted(set(image_files))

        print(f"\n{'=' * 60}")
        print(f"开始自动标注")
        print(f"{'=' * 60}")
        print(f"输入文件夹: {input_folder}")
        print(f"输出文件夹: {output_folder}")
        print(f"图片数量: {len(image_files)}")

        if len(image_files) == 0:
            print("警告: 没有找到任何图片文件!")
            return {"error": "没有找到图片"}

        print(f"{'=' * 60}\n")

        stats = {
            "total_images": len(image_files),
            "processed": 0,
            "total_objects": 0,
            "class_counts": {},
            "errors": [],
        }

        for i, image_file in enumerate(image_files, 1):
            try:
                # 推理并转换
                labelme_data = self.predict_and_convert(
                    image_path=str(image_file),
                    output_folder=output_folder,
                    input_folder=input_folder
                )

                # 保存 JSON 文件
                json_filename = image_file.stem + ".json"
                json_path = output_path / json_filename

                with open(str(json_path), "w", encoding="utf-8") as f:
                    json.dump(labelme_data, f, indent=2, ensure_ascii=False)

                # 更新统计
                num_objects = len(labelme_data["shapes"])
                stats["processed"] += 1
                stats["total_objects"] += num_objects

                for shape in labelme_data["shapes"]:
                    label = shape["label"]
                    stats["class_counts"][label] = stats["class_counts"].get(label, 0) + 1

                print(f"[{i}/{len(image_files)}] {image_file.name} -> "
                      f"检测到 {num_objects} 个目标 ✓")

            except Exception as e:
                error_msg = f"{image_file.name}: {str(e)}"
                stats["errors"].append(error_msg)
                print(f"[{i}/{len(image_files)}] {image_file.name} -> 错误: {e} ✗")
                traceback.print_exc()

        # 打印统计信息
        print(f"\n{'=' * 60}")
        print(f"标注完成!")
        print(f"{'=' * 60}")
        print(f"处理图片: {stats['processed']}/{stats['total_images']}")
        print(f"检测目标总数: {stats['total_objects']}")
        print(f"\n各类别统计:")
        if stats["class_counts"]:
            for label, count in sorted(stats["class_counts"].items()):
                print(f"  - {label}: {count}")
        else:
            print("  (未检测到任何目标)")

        if stats["errors"]:
            print(f"\n错误数量: {len(stats['errors'])}")
            for err in stats["errors"][:5]:
                print(f"  - {err}")

        print(f"{'=' * 60}\n")

        return stats


def main():
    """主函数"""

    # ==================== 配置参数 ====================

    # YOLOv8 模型路径 (修改为你的模型路径)
    MODEL_PATH = r"D:\1_Python\fire-security-ai\src\main\model\yolo_detect\security\train_01\weights\best.pt"

    # 输入图片文件夹
    INPUT_FOLDER = r"D:\1_Python\datasets\fire_security\images\train06"

    # 输出 JSON 文件夹
    OUTPUT_FOLDER = r"D:\1_Python\datasets\fire_security\labels_labelme\train06"

    # 置信度阈值
    CONF_THRESHOLD = 0.7

    # ==================== 路径检查 ====================
    print("=" * 60)
    print("路径检查")
    print("=" * 60)

    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 - {MODEL_PATH}")
        print("请修改 MODEL_PATH 为正确的模型路径")
        return

    # 检查输入文件夹是否存在
    if not os.path.exists(INPUT_FOLDER):
        print(f"错误: 输入文件夹不存在 - {INPUT_FOLDER}")
        return

    # 创建自动标注器
    labeler = YOLOv8AutoLabeler(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        imgsz=1280
    )

    # 处理文件夹
    stats = labeler.process_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER
    )

    print("自动标注完成！")
    print(f"JSON 文件已保存到: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
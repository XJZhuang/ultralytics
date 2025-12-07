from ultralytics import YOLO

# 加载.pt模型
model = YOLO(r"D:\1_Python\switchgear-ai\src\resources\predict_model\light.pt")
# 获取类别名称字典
class_names = model.names

# 打印所有类别名称
for class_id, class_name in class_names.items():
    print(f"{class_id}: {class_name}")

import os
import json
from PIL import Image
import pandas as pd
from tqdm import tqdm
import hashlib

# 设置文件路径
# labels_dir = '../dataset/VisDrone/val/labels'
# images_dir = '../dataset/VisDrone/val/images'
# output_json_path = '../dataset/VisDrone/val/val.json'

# labels_dir = '../dataset/VisDrone/train/labels'
# images_dir = '../dataset/VisDrone/train/images'
# output_json_path = '../dataset/VisDrone/train/train.json'

labels_dir = '../dataset/VisDrone_part/init/train/labels'
images_dir = '../dataset/VisDrone_part/init/train/images'
output_json_path = '../dataset/VisDrone_part/init/train/train.json'



# 准备COCO格式的字典
coco_format = {
    "images": [],
    "annotations": [],
    "categories": [],
}

# 为了兼容COCO格式，创建一个简单的类别映射
class_mapping = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}

# class_mapping = {
#     0: "Human",
#     1: "Car",
#     2: "Truck",
#     3: "Van",
#     4: "Motorbike",
#     5: "Bicycle",
#     6: "Bus",
#     7: "Trailer"
# }

# 添加类别到COCO格式
for class_id, class_name in class_mapping.items():
    coco_format["categories"].append({
        "id": class_id,
        "name": class_name,
        "supercategory": "none"
    })

annotation_id = 1

# 读取每个标签文件并转换为COCO格式
for label_file in tqdm(os.listdir(labels_dir)):
    if not label_file.endswith('.txt'):
        continue

    image_file = label_file.replace('.txt', '.jpg')
    image_path = os.path.join(images_dir, image_file)

    # 打开图像以获取宽度和高度
    with Image.open(image_path) as img:
        width, height = img.size

    # 创建唯一的 image_id
    image_id = int(hashlib.sha256(image_file.encode('utf-8')).hexdigest(), 16) % 10**8
    coco_format["images"].append({
        "file_name": image_file,
        "height": height,
        "width": width,
        "id": image_id
    })

    # 读取YOLO标签文件
    label_path = os.path.join(labels_dir, label_file)
    with open(label_path, 'r') as file:
        for line in file.readlines():
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
            class_id = int(class_id) + 1
            # 转换坐标到COCO格式
            x_min = (x_center - bbox_width / 2) * width
            y_min = (y_center - bbox_height / 2) * height
            bbox_width *= width
            bbox_height *= height

            # 将标注添加到COCO格式
            coco_format["annotations"].append({
                "segmentation": [],
                "area": bbox_width * bbox_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "category_id": int(class_id),
                "id": annotation_id
            })
            annotation_id += 1

# 将结果保存为JSON文件
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, 'w') as json_file:
    json.dump(coco_format, json_file)

print(f"Conversion complete. COCO JSON saved to {output_json_path}")

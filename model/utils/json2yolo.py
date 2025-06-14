import json
import os

# 读取JSON文件
input_json_path = "../dataset/auair/annotations.json"  # 请将此路径替换为您的JSON文件路径
with open(input_json_path, "r") as file:
    data = json.load(file)

# 定义YOLO格式的保存路径
output_dir = "../dataset/auair/labels"
os.makedirs(output_dir, exist_ok=True)

# 定义YOLO类别列表
categories = data["categories"]

# 处理每个标注
for annotation in data["annotations"]:
    image_name = annotation["image_name"]
    image_width = annotation["image_width:"]
    image_height = annotation["image_height"]
    bboxes = annotation["bbox"]

    # 创建YOLO格式的标注内容
    yolo_format_lines = []
    for bbox in bboxes:
        class_id = bbox["class"]
        x_center = (bbox["left"] + bbox["width"] / 2) / image_width
        y_center = (bbox["top"] + bbox["height"] / 2) / image_height
        bbox_width = bbox["width"] / image_width
        bbox_height = bbox["height"] / image_height
        yolo_format_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}")

    # 将YOLO格式内容保存到文件
    output_file = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
    with open(output_file, "w") as file:
        file.write("\n".join(yolo_format_lines))

    print(f"Saved YOLO format for {image_name} to {output_file}")

print("转换完成!")

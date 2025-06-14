import os

# 类别名称字典
class_names = {
    0: 'pedestrian',
    1: 'people',
    2: 'bicycle',
    3: 'car',
    4: 'van',
    5: 'truck',
    6: 'tricycle',
    7: 'awning-tricycle',
    8: 'bus',
    9: 'motor'
}

def count_yolo_labels(label_dir):
    # 用于存储每个类别的计数
    class_count = {}

    # 遍历目录下所有的标签文件
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as file:
                for line in file:
                    # 类ID
                    class_id = int(line.split()[0])
                    if class_id in class_count:
                        class_count[class_id] += 1
                    else:
                        class_count[class_id] = 1

    return class_count

# 输入标签文件所在的目录
label_directory = 'dataset/VisDrone_part/init/labels'  # 请替换为你的标签文件夹路径
counts = count_yolo_labels(label_directory)

# 输出当前路径下有多少文件
total_files = len([f for f in os.listdir(label_directory) if f.endswith('.txt')])
print(f"Total number of label files: {total_files}")

# 按照class_names的顺序输出
for class_id in sorted(class_names.keys()):  # 按照类别ID的顺序排序
    class_name = class_names.get(class_id, f"Unknown Class {class_id}")
    count = counts.get(class_id, 0)  # 如果该类别没有出现，默认计数为0
    print(f"{class_name}: {count}")

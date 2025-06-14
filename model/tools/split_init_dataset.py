import os
import shutil
import random
from tqdm import tqdm

# 图片和标签原始存储路径
source_images_folder = "dataset/VisDrone/train/images"
source_labels_folder = "dataset/VisDrone/train/labels"

# 目标路径，用于存放随机选取的图片和标签
target_images_folder = "dataset/VisDrone_part/init/images"
target_labels_folder = "dataset/VisDrone_part/init/labels"

# 确保目标文件夹存在
os.makedirs(target_images_folder, exist_ok=True)
os.makedirs(target_labels_folder, exist_ok=True)

# 获取原始图片文件夹中所有图片的文件名
image_files = os.listdir(source_images_folder)
# 随机选取500张图片
selected_images = random.sample(image_files, 500)

for image_file in tqdm(selected_images):
    # 构造原始图片和标签的完整路径
    source_image_path = os.path.join(source_images_folder, image_file)
    source_label_path = os.path.join(
        source_labels_folder, image_file.replace(".jpg", ".txt")
    )

    # 构造目标图片和标签的完整路径
    target_image_path = os.path.join(target_images_folder, image_file)
    target_label_path = os.path.join(
        target_labels_folder, image_file.replace(".jpg", ".txt")
    )

    # 复制图片和标签到目标文件夹
    shutil.copy2(source_image_path, target_image_path)
    shutil.copy2(source_label_path, target_label_path)

print("Done")

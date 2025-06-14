'''
随机获得训练集中其余的100张图片
'''
import os
import shutil
import random
from tqdm import tqdm


def select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder, copy_images_folder, copy_labels_folder, n=100):
    # Read all files from source folders
    source_images = set(os.listdir(source_images_folder))
    source_labels = set(os.listdir(source_labels_folder))

    # 确保目标文件夹存在
    os.makedirs(copy_images_folder, exist_ok=True)
    os.makedirs(copy_labels_folder, exist_ok=True)

    # Read all files from target folders (to ensure we don't select these)
    existing_images = set(os.listdir(target_images_folder))
    existing_labels = set(os.listdir(target_labels_folder))

    # Filter out files that already exist in the target folders
    available_images = source_images - existing_images
    available_labels = source_labels - existing_labels

    # Ensure the filenames without extension match between images and labels
    available_files = set(file.split('.')[0] for file in available_images) & set(
        file.split('.')[0] for file in available_labels)

    # Randomly select n files, if there are enough
    if len(available_files) < n:
        raise ValueError("Not enough unique files to select from.")

    selected_files = random.sample(available_files, n)

    # Copy the selected images and labels to the target folders
    for file in tqdm(selected_files):
        image_file = next((img for img in available_images if img.startswith(file)), None)
        label_file = next((lbl for lbl in available_labels if lbl.startswith(file)), None)

        if image_file and label_file:
            shutil.copy(os.path.join(source_images_folder, image_file), os.path.join(copy_images_folder, image_file))
            shutil.copy(os.path.join(source_labels_folder, label_file), os.path.join(copy_labels_folder, label_file))


if __name__ == '__main__':
    source_images_folder = "../dataset/VisDrone/train/images"
    source_labels_folder = "../dataset/VisDrone/train/labels"
    target_images_folder = "../dataset/VisDrone_part/init/images"
    target_labels_folder = "../dataset/VisDrone_part/init/labels"
    copy_images_folder = "../dataset/VisDrone_part/random_select/1/images"
    copy_labels_folder = "../dataset/VisDrone_part/random_select/1/labels"
    pic_num = 100  # 图片数量
    select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder,target_labels_folder, copy_images_folder, copy_labels_folder, n=pic_num)

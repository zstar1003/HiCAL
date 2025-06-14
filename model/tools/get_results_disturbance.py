import os
import shutil
import torch
import numpy as np
import cv2
import sys
from tqdm import tqdm

# 确保 utils 模块在 sys.path 中
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from utils.augmentations import letterbox
from utils.general import non_max_suppression
from models.common import DetectMultiBackend


def add_noise_to_confidence(confidences, noise_level=0.1):
    noise = torch.randn_like(confidences) * noise_level
    return confidences + noise


def calculate_confidence_change(original_confidences, perturbed_confidences):
    original_class = torch.argmax(original_confidences, dim=0)  # 修改dim为0
    perturbed_class = torch.argmax(perturbed_confidences, dim=0)  # 修改dim为0
    change = (original_class != perturbed_class).float()
    return change


def select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder,
                          copy_images_folder, copy_labels_folder, model, device, n=100, noise_level=0.1):
    source_images = set(os.listdir(source_images_folder))
    source_labels = set(os.listdir(source_labels_folder))

    os.makedirs(copy_images_folder, exist_ok=True)
    os.makedirs(copy_labels_folder, exist_ok=True)

    existing_images = set(os.listdir(target_images_folder))
    existing_labels = set(os.listdir(target_labels_folder))

    available_images = source_images - existing_images
    available_labels = source_labels - existing_labels

    available_files = set(file.split('.')[0] for file in available_images) & set(
        file.split('.')[0] for file in available_labels)

    file_scores = []

    for file in tqdm(available_files, desc="Calculating confidence changes"):
        img_name = f"{file}.jpg"
        img_path = os.path.join(source_images_folder, img_name)
        img = cv2.imread(img_path)
        img = letterbox(img, 1280, stride=32, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device).float()
        im = im.unsqueeze(0) / 255.0

        with torch.no_grad():
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred[0], conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)[0]

        if pred is not None and len(pred):
            confidences = pred[:, 4]
            perturbed_confidences = add_noise_to_confidence(confidences, noise_level=noise_level)
            confidence_change = calculate_confidence_change(confidences, perturbed_confidences).item()
            score = confidence_change * 100  # 给予较大的筛选分数
            file_scores.append((img_name, score))

    selected_files = [file for file, _ in sorted(file_scores, key=lambda x: -x[1])[:n]]

    for file in tqdm(selected_files, desc="Copying selected files"):
        image_file = file
        label_file = file.replace('.jpg', '.txt')
        try:
            shutil.copy(os.path.join(source_images_folder, image_file), os.path.join(copy_images_folder, image_file))
            shutil.copy(os.path.join(source_labels_folder, label_file), os.path.join(copy_labels_folder, label_file))
        except:
            print(f"Error copying {image_file} and {label_file}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = "weights/visdrone_best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False, data='data/VisDrone.yaml', fp16=False)
    model.eval()

    source_images_folder = "dataset/VisDrone/train/images"
    source_labels_folder = "dataset/VisDrone/train/labels"
    target_images_folder = "dataset/VisDrone_part/init/images"
    target_labels_folder = "dataset/VisDrone_part/init/labels"
    copy_images_folder = "dataset/VisDrone_part/results_noise_select/5/images"
    copy_labels_folder = "dataset/VisDrone_part/results_noise_select/5/labels"
    pic_num = 500
    select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder,
                          copy_images_folder, copy_labels_folder, model, device, n=pic_num, noise_level=0.1)
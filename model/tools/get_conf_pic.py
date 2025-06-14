'''
根据置信度选择置信度最低的n张图片
'''
import os
import shutil
import torch
import numpy as np
import cv2
from tqdm import tqdm
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from models.common import DetectMultiBackend

# yolo prams
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False

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
    available_files = set(file.split('.')[0] for file in available_images) & set(file.split('.')[0] for file in available_labels)
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = "../weights/best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False, data='data/VisDrone.yaml', fp16=False)

    # Compute confidences for each file
    file_confidences = []
    for file in tqdm(available_files):
        img_name = f"{file}.jpg"
        img_path = os.path.join(source_images_folder, img_name)
        img0 = cv2.imread(img_path)
        img = letterbox(img0, 1280, stride=32, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255
        im = im.unsqueeze(0)
        pred = model(im, augment=False, visualize=False)
        pred = pred[0][1]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

        # Calculate average confidence
        if pred is not None and len(pred):
            confidences = pred[:, 4]
            avg_confidence = confidences.mean().item()
        else:
            avg_confidence = 100  # 若没有检测到目标，则置信度设成最大，防止被纳入
        file_confidences.append((file, avg_confidence))

    selected_files = [file for file, _ in sorted(file_confidences, key=lambda x: x[1])[:n]]
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
    copy_images_folder = "../dataset/VisDrone_part/conf_select/1/images"
    copy_labels_folder = "../dataset/VisDrone_part/conf_select/1/labels"
    pic_num = 100  # 图片数量
    select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder,target_labels_folder, copy_images_folder, copy_labels_folder, n=pic_num)

'''
根据源域目标类别分布进行目标域样本筛选
'''
import os
import shutil
import torch
import numpy as np
import cv2
from tqdm import tqdm
from models.backbone import DetectionModelBackboneFeature
from utils.augmentations import letterbox
from utils.downloads import attempt_download
from utils.general import intersect_dicts, non_max_suppression
from sklearn.cluster import MeanShift, estimate_bandwidth

class_num = 10  # 总类别数量


def parse_label(label_path):
    # 从标签文件中解析出类别
    with open(label_path) as f:
        classes = [int(x.split()[0]) for x in f.read().strip().splitlines()]
    return classes


def get_class_distribution(label_folder):
    # 计算类别分布
    class_counts = np.zeros(class_num, dtype=int)  # 假设总共有10个类别
    for label_file in os.listdir(label_folder):
        label_path = os.path.join(label_folder, label_file)
        classes = parse_label(label_path)
        for c in classes:
            class_counts[c] += 1
    return class_counts


def inference(model, device, img, conf_thres=0.25, iou_thres=0.45):
    img = letterbox(img, new_shape=1280)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR到RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # 图像归一化
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    return pred


def select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder,
                          copy_images_folder, copy_labels_folder, model, device, n=100):
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

    # 根据源域类别分布计算权重
    source_class_distribution = get_class_distribution(target_labels_folder)
    class_weights = 1.0 / (source_class_distribution + 1e-6)
    class_weights /= class_weights.sum()

    # 推理并为每个目标域图像计算得分
    scores = []
    for file in tqdm(available_files):
        img_name = f"{file}.jpg"
        img_path = os.path.join(source_images_folder, img_name)
        img = cv2.imread(img_path)  # 读取图像
        pred = inference(model, device, img)  # 对图像进行推理
        # 对推理结果进行处理，计算得分
        score = 0.0
        # 遍历每张图像的检测结果
        for i, det in enumerate(pred):
            if det is not None and len(det):
                for *xyxy, conf, cls in reversed(det):
                    # 如果int(cls)在[0, class_num-1]之间
                    if int(cls) < class_num:
                        # score += class_weights[int(cls)] * conf  # 根据类别权重和置信度计算得分
                        score += class_weights[int(cls)]  # 仅根据类别权重计算得分
        scores.append((file, score))
    # 根据得分选择图像
    selected_files = sorted(scores, key=lambda x: x[1], reverse=True)[:n]
    for file_score_tuple in tqdm(selected_files):
        file = file_score_tuple[0] + '.jpg'  # 提取文件名并添加扩展名
        image_file = file
        label_file = file.replace('.jpg', '.txt')
        shutil.copy(os.path.join(source_images_folder, image_file), os.path.join(copy_images_folder, image_file))
        shutil.copy(os.path.join(source_labels_folder, label_file), os.path.join(copy_labels_folder, label_file))


if __name__ == '__main__':
    # Setup device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = "../weights/best.pt"
    cfg = '../models/detect/yolov9-c.yaml'
    model = DetectionModelBackboneFeature(cfg).to(device)
    ckpt = torch.load(attempt_download(weights), map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])
    model.load_state_dict(csd, strict=False)
    model.eval()
    # Setup paths
    source_images_folder = "../dataset/VisDrone/train/images"
    source_labels_folder = "../dataset/VisDrone/train/labels"
    target_images_folder = "../dataset/VisDrone_part/init/images"
    target_labels_folder = "../dataset/VisDrone_part/init/labels"
    copy_images_folder = "../dataset/VisDrone_part/classbalence_select/1/images"
    copy_labels_folder = "../dataset/VisDrone_part/classbalence_select/1/labels"
    pic_num = 100  # 图片数量
    select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder,
                          copy_images_folder, copy_labels_folder, model, device, n=pic_num)

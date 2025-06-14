'''
根据目标域特征进行Mean Shift聚类
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
from utils.general import intersect_dicts
from sklearn.cluster import MeanShift, estimate_bandwidth

# Mean Shift聚类相关参数
bandwidth = None  # 带宽，None表示自动估计

def convert_features(features):
    features_2d = features.reshape(features.shape[0], -1)
    return features_2d

def extract_features(model, img, device):
    """
    提取图像特征
    """
    img = letterbox(img, 1280, stride=32, auto=False)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float()
    img = img.unsqueeze(0) / 255.0  # Add batch dimension and normalize
    with torch.no_grad():
        features = model(img)
    return features.squeeze().detach().cpu().numpy()

def mean_shift_clustering(features):
    """
    使用Mean Shift进行聚类
    """
    bandwidth = estimate_bandwidth(features, quantile=0.05)
    print(bandwidth)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(features)
    labels = ms.labels_
    return labels

def select_balanced_samples(labels, n):
    """
    从每个聚类中平衡选择样本
    """
    unique_labels = np.unique(labels)
    per_cluster = n // len(unique_labels)  # 每个聚类选择的样本数
    selected_indices = []
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        selected = np.random.choice(indices, size=min(per_cluster, len(indices)), replace=False)
        selected_indices.extend(selected)
    return selected_indices


def select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder, copy_images_folder, copy_labels_folder, model, device, n=100):
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
    available_files = list(available_files)
    # 提取目标域的特征
    features = []
    for file in tqdm(available_files):
        img_name = f"{file}.jpg"
        img_path = os.path.join(source_images_folder, img_name)
        img = cv2.imread(img_path)
        feature = extract_features(model, img, device)
        features.append(feature)
    features = np.array(features)
    features_2d = convert_features(features)
    # features_2d = features_2d.astype(np.int8)
    labels = mean_shift_clustering(features_2d)
    print(f"Number of clusters: {len(np.unique(labels))}")
    # 平衡选择样本
    selected_indices = select_balanced_samples(labels, n)
    selected_files = [available_files[i] for i in selected_indices]
    # Copy the selected images and labels to the target folders
    for file in tqdm(selected_files):
        file = f"{file}.jpg"
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
    copy_images_folder = "../dataset/VisDrone_part/cluster_select/1/images"
    copy_labels_folder = "../dataset/VisDrone_part/cluster_select/1/labels"
    pic_num = 100  # 图片数量
    select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder, copy_images_folder, copy_labels_folder, model, device, n=pic_num)
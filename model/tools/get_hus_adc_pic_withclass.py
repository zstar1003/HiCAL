'''
选择hus_adc最大的n张图片
'''
import os
import shutil
import torch
import numpy as np
import cv2
from tqdm import tqdm

from models.backbone import DetectionModelBackboneFeature
from utils.augmentations import letterbox, hist_equalize
from utils.downloads import attempt_download
from utils.general import non_max_suppression, intersect_dicts
from models.common import DetectMultiBackend

# yolo prams
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False
class_num = 20

def extract_features(model, img):
    with torch.no_grad():
        features = model(img)
    return features.squeeze()  # Remove unnecessary dimensions


def calculate_feature_distance(features1, features2):
    """
    Calculate the distance between two sets of features.
    """
    return torch.norm(features1 - features2, p=2, dim=1)  # Euclidean distances

# 归一化数组值
def normalize(values, total_values):
    return (values - np.min(total_values)) / (np.max(total_values) - np.min(total_values) + 1e-6)

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

def get_source_domain_centroid(existing_images, target_images_folder, model, device):
    """
    Calculate the centroid of the source domain based on the features extracted from the source images.
    """
    source_features = []
    for file in tqdm(existing_images, desc="Extracting features from source domain"):
        img_path = os.path.join(target_images_folder, file)
        img = cv2.imread(img_path)
        img = letterbox(img, 1280, stride=32, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device).float()
        im = im.unsqueeze(0) / 255.0  # Add batch dimension and normalize
        features = extract_features(model, im)
        source_features.append(features)
    source_features = torch.stack(source_features)
    return torch.mean(source_features, dim=0)  # Calculate the centroid


def entropy(confidences):
    """Calculate the entropy of confidence scores"""
    if len(confidences) == 0:
        return 0
    probabilities = confidences / confidences.sum()
    return -np.sum(probabilities * np.log(probabilities + 1e-9))

def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)  # Convert back to BGR
    return im

def calculate_inconsistency(*confidences_lists):
    """Calculate inconsistency as the variance of confidence scores across different augmentations"""
    if all(len(conf) == 0 for conf in confidences_lists):
        return 0  # 如果都是空的，返回0作为默认不一致值
    # 过滤掉空列表，只对非空列表进行操作
    non_empty_lists = [conf for conf in confidences_lists if len(conf) > 0]
    # 确保至少有一个非空列表存在
    if non_empty_lists:
        all_confidences = np.concatenate(non_empty_lists)
        return np.var(all_confidences)
    else:
        return 0  # 如果没有非空列表，返回0作为默认不一致值

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

def random_gaussian_blur(im):
    return cv2.GaussianBlur(im, (5, 5), 0)

def random_median_blur(im):
    return cv2.medianBlur(im, 5)

# 在图像处理流程中加入选择的数据增强方法
'''
HSV颜色空间调整 (augment_hsv)：
直方图均衡化 (hist_equalize)：可以改善图像的对比度，适用于图像太暗或太亮的情况。
高斯模糊 (cv2.GaussianBlur)：对图像应用轻微的模糊效果，可以帮助模型对抗噪声。
中值滤波 (cv2.medianBlur)：可以减少图像噪声，同时保持边缘信息。
'''
def augment_images(im):
    imgs = []
    imgs.append(augment_hsv(im))  # HSV颜色空间调整
    imgs.append(hist_equalize(im, clahe=True, bgr=True))  # 直方图均衡化
    imgs.append(random_gaussian_blur(im))  # 高斯模糊
    imgs.append(random_median_blur(im))  # 中值滤波
    imgs.append(im)
    return imgs


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
    weights = "../weights/visdrone_best.pt"
    cfg = '../models/detect/yolov9-c.yaml'
    model = DetectMultiBackend(weights, device=device, dnn=False, data='data/VOC.yaml', fp16=False)
    modelFeature = DetectionModelBackboneFeature(cfg).to(device)
    ckpt = torch.load(attempt_download(weights), map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    csd = intersect_dicts(csd, modelFeature.state_dict(), exclude=['anchor'])
    modelFeature.load_state_dict(csd, strict=False)
    modelFeature.eval()
    # Calculate the centroid of the source domain
    source_centroid = get_source_domain_centroid(existing_images, target_images_folder, modelFeature, device)
    # Compute confidences for each file
    file_inconsistencies = []

    # 根据源域类别分布计算权重
    source_class_distribution = get_class_distribution(target_labels_folder)
    class_weights = 1.0 / (source_class_distribution + 1e-6)
    class_weights /= class_weights.sum()

    total_distance = []
    total_file_inconsistency = []
    total_file_entropy = []
    total_score = []
    # 处理之前计算整体值
    for file in tqdm(available_files, desc="Calculate available files"):
        img_name = f"{file}.jpg"
        img_path = os.path.join(source_images_folder, img_name)
        img0 = cv2.imread(img_path)
        imgs = augment_images(img0)
        pred = inference(model, device, img0)  # 对图像进行推理
        # 对推理结果进行处理，计算得分
        score = 0
        file_entropy = 0
        # 遍历每张图像的检测结果
        for i, det in enumerate(pred):
            if det is not None and len(det):
                for *xyxy, conf, cls in reversed(det):
                    # 如果int(cls)在[0, class_num-1]之间
                    if int(cls) < class_num:
                        # score += class_weights[int(cls)] * conf  # 根据类别权重和置信度计算得分
                        score += class_weights[int(cls)]  # 仅根据类别权重计算得分
        confidences = []
        for img in imgs:
            img = letterbox(img, 1280, stride=32, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            im = im.unsqueeze(0)
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            if pred is not None and len(pred):
                conf = pred[:, 4].cpu().numpy()
                file_entropy = entropy(conf)
                confidences.append(pred[:, 4].cpu().numpy())
            else:
                file_entropy = 0
                confidences.append(np.array([]))
        file_inconsistency = calculate_inconsistency(*confidences)
        img = cv2.imread(img_path)
        img = letterbox(img, 1280, stride=32, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device).float()
        im = im.unsqueeze(0) / 255.0
        features = extract_features(modelFeature, im)
        distance = calculate_feature_distance(features.unsqueeze(0), source_centroid.unsqueeze(0)).mean().item()
        total_distance.append(distance)
        total_score.append(score)
        total_file_inconsistency.append(file_inconsistency)
        total_file_entropy.append(file_entropy)

    # 正式开始处理
    for file in tqdm(available_files, desc="Process available files"):
        img_name = f"{file}.jpg"
        img_path = os.path.join(source_images_folder, img_name)
        img0 = cv2.imread(img_path)
        imgs = augment_images(img0)
        pred = inference(model, device, img0)  # 对图像进行推理
        # 对推理结果进行处理，计算得分
        score = 0
        file_entropy = 0
        # 遍历每张图像的检测结果
        for i, det in enumerate(pred):
            if det is not None and len(det):
                for *xyxy, conf, cls in reversed(det):
                    # 如果int(cls)在[0, class_num-1]之间
                    if int(cls) < class_num:
                        # score += class_weights[int(cls)] * conf  # 根据类别权重和置信度计算得分
                        score += class_weights[int(cls)]  # 仅根据类别权重计算得分
        confidences = []
        for img in imgs:
            img = letterbox(img, 1280, stride=32, auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)
            im = torch.from_numpy(img).to(device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            im = im.unsqueeze(0)
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred[0], conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]
            if pred is not None and len(pred):
                conf = pred[:, 4].cpu().numpy()
                file_entropy = entropy(conf)
                confidences.append(pred[:, 4].cpu().numpy())
            else:
                file_entropy = 0
                confidences.append(np.array([]))
        file_inconsistency = calculate_inconsistency(*confidences)
        img = cv2.imread(img_path)
        img = letterbox(img, 1280, stride=32, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device).float()
        im = im.unsqueeze(0) / 255.0
        features = extract_features(modelFeature, im)
        distance = calculate_feature_distance(features.unsqueeze(0), source_centroid.unsqueeze(0)).mean().item()
        distance = normalize(distance, total_distance)
        file_inconsistency = normalize(file_inconsistency, total_file_inconsistency)
        file_entropy = normalize(file_entropy, total_file_entropy)
        score = normalize(score, total_score)
        # print(distance, file_inconsistency, file_entropy, score)
        # print("-------")
        file_inconsistency = 1 * distance + 2 * file_inconsistency * file_entropy + 1 * score
        file_inconsistencies.append((file, file_inconsistency))
    # Select files with the highest inconsistency
    selected_files = [file for file, _ in sorted(file_inconsistencies, key=lambda x: -x[1])[:n]]
    # Copy the selected images and labels to the target folders
    for file in tqdm(selected_files):
        file = f"{file}.jpg"
        image_file = file
        label_file = file.replace('.jpg', '.txt')
        shutil.copy(os.path.join(source_images_folder, image_file), os.path.join(copy_images_folder, image_file))
        shutil.copy(os.path.join(source_labels_folder, label_file), os.path.join(copy_labels_folder, label_file))


if __name__ == '__main__':
    source_images_folder = "../dataset/VOC2007/train/images"
    source_labels_folder = "../dataset/VOC2007/train/labels"
    target_images_folder = "../dataset/VOC2007_part/init/images"
    target_labels_folder = "../dataset/VOC2007_part/init/labels"
    copy_images_folder = "../dataset/VOC2007_part/husadc_select/1/images"
    copy_labels_folder = "../dataset/VOC2007_part/husadc_select/1/labels"
    pic_num = 100  # 图片数量
    select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder,target_labels_folder, copy_images_folder, copy_labels_folder, n=pic_num)
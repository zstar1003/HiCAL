'''
根据源域和目标域差异选取最大的n张图片
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


def extract_features(model, img):
    with torch.no_grad():
        # Assuming 'model' returns feature maps as its second output
        features = model(img)
    return features.squeeze()  # Remove unnecessary dimensions


def calculate_feature_distance(features1, features2):
    """
    Calculate the distance between two sets of features.
    """
    return torch.norm(features1 - features2, p=2, dim=1)  # Euclidean distance


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

    # Calculate the centroid of the source domain
    source_centroid = get_source_domain_centroid(existing_images, target_images_folder, model, device)

    # Initialize list to hold distance for each target image
    file_distances = []

    # Loop through target images and calculate distance from source centroid
    for file in tqdm(available_files, desc="Calculating distances for target domain"):
        img_name = f"{file}.jpg"
        img_path = os.path.join(source_images_folder, img_name)
        img = cv2.imread(img_path)
        img = letterbox(img, 1280, stride=32, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device).float()
        im = im.unsqueeze(0) / 255.0
        features = extract_features(model, im)
        distance = calculate_feature_distance(features.unsqueeze(0), source_centroid.unsqueeze(0)).mean().item()
        file_distances.append((img_name, distance))

    # Select files with the highest distance
    selected_files = [file for file, _ in sorted(file_distances, key=lambda x: -x[1])[:n]]
    # print(f"Selected {len(selected_files)} files from target domain.")
    # print(f"Files: {selected_files}")
    # Copy the selected images and labels to the target folders
    for file in tqdm(selected_files):
        image_file = file
        label_file = file.replace('.jpg', '.txt')
        try:
            shutil.copy(os.path.join(source_images_folder, image_file), os.path.join(copy_images_folder, image_file))
            shutil.copy(os.path.join(source_labels_folder, label_file), os.path.join(copy_labels_folder, label_file))
        except:
            pass

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
    copy_images_folder = "../dataset/VisDrone_part/defdomain_select/1/images"
    copy_labels_folder = "../dataset/VisDrone_part/defdomain_select/1/labels"
    pic_num = 100  # 图片数量
    select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder, copy_images_folder, copy_labels_folder, model, device, n=pic_num)
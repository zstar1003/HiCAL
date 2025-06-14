import os
import shutil
import torch
import numpy as np
import cv2
import sys
from tqdm import tqdm
# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录的路径
root_dir = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(root_dir)

from models.backbone import DetectionModelBackboneFeature
from utils.augmentations import letterbox
from utils.downloads import attempt_download
from utils.general import intersect_dicts


def extract_features(model, img):
    with torch.no_grad():
        # 假设 'model' 返回特征图作为其第二个输出
        features = model(img)
    return features.squeeze()  # 移除不必要的维度


def add_noise(features, noise_level=0.1):
    """
    给特征添加高斯噪声。
    """
    noise = torch.randn_like(features) * noise_level
    return features + noise


def spatial_dropout(features, p=0.2):
    mask = torch.ones_like(features)
    mask = torch.nn.functional.dropout2d(mask, p=p, training=True)
    return features * mask


def channel_wise_dropout(features, p=0.2):
    if features.dim() == 3:
        features = features.unsqueeze(0)
    batch_size, num_channels, _, _ = features.size()
    drop_mask = (torch.rand(batch_size, num_channels, 1, 1, device=features.device) > p).float()
    return features * drop_mask

def calculate_entropy(features):
    """
    计算特征的检测信息熵。
    """
    p = torch.softmax(features, dim=1)
    log_p = torch.log(p + 1e-10)
    entropy = -torch.sum(p * log_p, dim=1, keepdim=True)
    return entropy


def select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder,
                          copy_images_folder, copy_labels_folder, model, device, n=100, perturbation_method='noise'):
    # 读取源文件夹中的所有文件
    source_images = set(os.listdir(source_images_folder))
    source_labels = set(os.listdir(source_labels_folder))

    # 确保目标文件夹存在
    os.makedirs(copy_images_folder, exist_ok=True)
    os.makedirs(copy_labels_folder, exist_ok=True)

    # 读取目标文件夹中的所有文件（确保不选择这些文件）
    existing_images = set(os.listdir(target_images_folder))
    existing_labels = set(os.listdir(target_labels_folder))

    # 过滤出目标文件夹中不存在的文件
    available_images = source_images - existing_images
    available_labels = source_labels - existing_labels

    # 确保图像和标签的文件名（不包括扩展名）匹配
    available_files = set(file.split('.')[0] for file in available_images) & set(
        file.split('.')[0] for file in available_labels)

    # 初始化保存信息熵方差的列表
    file_entropies = []

    # 遍历目标文件夹中的图像，计算扰动前后的检测信息熵方差
    for file in tqdm(available_files, desc="Calculating entropy variances"):
        img_name = f"{file}.jpg"
        img_path = os.path.join(source_images_folder, img_name)
        img = cv2.imread(img_path)
        img = letterbox(img, 1280, stride=32, auto=False)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC 转 CHW，BGR 转 RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device).float()
        im = im.unsqueeze(0) / 255.0  # 添加批次维度并归一化

        # 提取原始特征
        features = extract_features(model, im)

        # 根据选择的扰动方法进行特征扰动
        if perturbation_method == 'noise':
            perturbed_features = add_noise(features)
        elif perturbation_method == 'spatial_dropout':
            perturbed_features = spatial_dropout(features)
        elif perturbation_method == 'channel_wise_dropout':
            perturbed_features = channel_wise_dropout(features)
        elif perturbation_method == 'noise spatial_dropout':
            perturbed_features = add_noise(features)
            perturbed_features = spatial_dropout(perturbed_features)
        elif perturbation_method == 'spatial_dropout channel_wise_dropout':
            perturbed_features = spatial_dropout(features)
            perturbed_features = channel_wise_dropout(perturbed_features)
        elif perturbation_method == 'noise channel_wise_dropout':
            perturbed_features = add_noise(features)
            perturbed_features = channel_wise_dropout(perturbed_features)
        elif perturbation_method == 'noise spatial_dropout channel_wise_dropout':
            perturbed_features = add_noise(features)
            perturbed_features = spatial_dropout(perturbed_features)
            perturbed_features = channel_wise_dropout(perturbed_features)
        else:
            raise ValueError("Unsupported perturbation method: {}".format(perturbation_method))

        # 计算扰动前后的检测信息熵
        original_entropy = calculate_entropy(features)
        perturbed_entropy = calculate_entropy(perturbed_features)
        # 计算信息熵方差
        entropy_variance = torch.var(perturbed_entropy - original_entropy).item()
        file_entropies.append((img_name, entropy_variance))

    # 选择信息熵方差最大的前 n 张图片
    selected_files = [file for file, _ in sorted(file_entropies, key=lambda x: -x[1])[:n]]

    # 复制选定的图像和标签到目标文件夹
    for file in tqdm(selected_files, desc="Copying selected files"):
        image_file = file
        label_file = file.replace('.jpg', '.txt')
        try:
            shutil.copy(os.path.join(source_images_folder, image_file), os.path.join(copy_images_folder, image_file))
            shutil.copy(os.path.join(source_labels_folder, label_file), os.path.join(copy_labels_folder, label_file))
        except:
            print(f"Error copying {image_file} and {label_file}")


if __name__ == '__main__':
    # 设置设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = "weights/auair_init_best.pt"
    cfg = 'models/detect/yolov9-c.yaml'
    model = DetectionModelBackboneFeature(cfg).to(device)
    ckpt = torch.load(attempt_download(weights), map_location='cpu')
    csd = ckpt['model'].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])
    model.load_state_dict(csd, strict=False)
    model.eval()
    # 设置路径
    source_images_folder = "dataset/VisDrone/train/images"
    source_labels_folder = "dataset/VisDrone/train/labels"
    target_images_folder = "dataset/VisDrone_part/init/images"
    target_labels_folder = "dataset/VisDrone_part/init/labels"
    copy_images_folder = "dataset/VisDrone_part/noise_spatial_dropout_select/15/images"
    copy_labels_folder = "dataset/VisDrone_part/noise_spatial_dropout_select/15/labels"
    pic_num = 5000 # 选择的图片数量
    select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder,
                          copy_images_folder, copy_labels_folder, model, device, n=pic_num,
                          perturbation_method='noise spatial_dropout')


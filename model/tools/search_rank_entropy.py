import os
import torch
import numpy as np
import cv2
import sys
from tqdm import tqdm
import argparse

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录的路径
root_dir = os.path.dirname(current_dir)
# 将项目根目录添加到 sys.path
sys.path.append(root_dir)

from utils.augmentations import letterbox
from utils.downloads import attempt_download
from utils.general import intersect_dicts, non_max_suppression
from models.backbone import DetectionModelBackboneFeature, DetectionModelCombined
from models.common import DetectMultiBackend

# yolo prams
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False


def calculate_entropy(features):
    """
    计算特征或置信度的检测信息熵。
    """
    p = torch.softmax(features, dim=1)
    log_p = torch.log(p + 1e-10)
    entropy = -torch.sum(p * log_p, dim=1, keepdim=True)
    return entropy


def entropy(confidences):
    """Calculate the entropy of confidence scores"""
    if len(confidences) == 0:
        return 0
    probabilities = confidences / confidences.sum()
    return -np.sum(probabilities * np.log(probabilities + 1e-9))


def select_files(
    source_images_folder,
    source_labels_folder,
    target_images_folder,
    target_labels_folder,
    model,
    device,
    n=100,
    perturbation_methods=None,
    spatial_dropout_probability=0.1,
    channel_dropout_probability=0.1,
    std=0.1,
    std_conf=0.1,
):
    if perturbation_methods is None:
        perturbation_methods = []
    # 读取源文件夹中的所有文件
    source_images = set(os.listdir(source_images_folder))
    source_labels = set(os.listdir(source_labels_folder))
    # 读取目标文件夹中的所有文件（确保不选择这些文件）
    existing_images = set(os.listdir(target_images_folder))
    existing_labels = set(os.listdir(target_labels_folder))
    # 过滤出目标文件夹中不存在的文件
    available_images = source_images - existing_images
    available_labels = source_labels - existing_labels
    # 确保图像和标签的文件名（不包括扩展名）匹配
    available_files = set(file.split(".")[0] for file in available_images) & set(
        file.split(".")[0] for file in available_labels
    )
    # 初始化保存信息熵差值的列表
    file_scores = []

    for file in tqdm(available_files):
        img_name = f"{file}.jpg"  # Assuming images are in JPG format
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
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
        )[0]

        # Calculate average confidence
        if pred is not None and len(pred):
            confidences = pred[:, 4].cpu().numpy()
            print(f"confidences shape: {confidences.shape}")
            print(f"confidences: {confidences}")
            file_entropy = entropy(confidences)
        else:
            file_entropy = 1e-9  # No detections lead to minimum entropy
        file_scores.append((file, file_entropy))

    return file_scores


def create_file_score_dict(file_scores):
    """
    创建一个字典，其中键为文件名，值为对应的分数，按分数从高到低排序。
    """
    sorted_scores = sorted(file_scores, key=lambda x: -x[1])
    return {file: score for file, score in sorted_scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply various perturbations to images, features, and model outputs."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="../weights/visdrone_init.pt",
        help="Path to the model weights file.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="../data/VisDrone.yaml",
        help="Path to the dataset configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="../models/detect/yolov9-c.yaml",
        help="Path to the model configuration file.",
    )
    parser.add_argument(
        "--source_images_folder",
        type=str,
        default="../dataset/VisDrone/train/images",
        help="Path to the source images folder.",
    )
    parser.add_argument(
        "--source_labels_folder",
        type=str,
        default="../dataset/VisDrone/train/labels",
        help="Path to the source labels folder.",
    )
    parser.add_argument(
        "--target_images_folder",
        type=str,
        default="../dataset/VisDrone_part/init/images",
        help="Path to the target images folder.",
    )
    parser.add_argument(
        "--target_labels_folder",
        type=str,
        default="../dataset/VisDrone_part/init/labels",
        help="Path to the target labels folder.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["flip", "hsv", "blur"],
        help="Perturbation methods to apply. Options: flip, hsv, blur, noise, spatial_dropout, channel_dropout, conf_noise",
    )
    parser.add_argument("--spatial_dropout_probability", default=0.3)
    parser.add_argument("--channel_dropout_probability", default=0.1)
    parser.add_argument("--std", default=0.1, help="std for feature")
    parser.add_argument("--std_conf", default=1, help="std for confidence")
    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        help="Number of images to select after perturbations.",
    )
    parser.add_argument(
        "--query_image", type=str, help="Path to the image to query its rank."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    model = DetectMultiBackend(
        args.weights, device=device, dnn=False, data=args.data, fp16=False
    )

    # Run select files
    file_scores = select_files(
        args.source_images_folder,
        args.source_labels_folder,
        args.target_images_folder,
        args.target_labels_folder,
        model,
        device,
        n=args.num,
        perturbation_methods=args.methods,
        spatial_dropout_probability=args.spatial_dropout_probability,
        channel_dropout_probability=args.channel_dropout_probability,
        std=args.std,
        std_conf=args.std_conf,
    )

    # Create file score dictionary
    file_score_dict = create_file_score_dict(file_scores)
    print("File scores dictionary (sorted by score):")
    for file, score in file_score_dict.items():
        print(f"{file}: {score}")

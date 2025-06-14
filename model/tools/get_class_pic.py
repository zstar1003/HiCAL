import os
import shutil
import torch
import numpy as np
import cv2
from tqdm import tqdm
from utils.augmentations import letterbox
from utils.general import non_max_suppression
from models.common import DetectMultiBackend
from collections import defaultdict

# yolo params
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False


def select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder,
                          copy_images_folder, copy_labels_folder, n=100):
    # 统计目标标签文件夹中的类别分布情况
    class_counts = defaultdict(int)
    target_labels = set(os.listdir(target_labels_folder))

    for label_file in target_labels:
        if label_file.endswith('.txt'):
            label_path = os.path.join(target_labels_folder, label_file)
            with open(label_path, 'r') as f:
                labels = [line.strip().split()[0] for line in f.readlines()]  # 假设标签在每行的第一个位置
                for label in labels:
                    class_counts[label] += 1

    # 统计类别的分布情况，找出稀少类别
    class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1]))  # 按照类别出现的数量升序排序
    print("类别分布：", class_counts)

    # 计算类别的阈值，用于优先选择稀缺类别
    min_class_count = min(class_counts.values())  # 最小类别数
    class_thresholds = {k: min_class_count * 1.5 for k in class_counts.keys()}  # 自定义阈值策略，挑选较少的类别

    # 创建文件夹
    os.makedirs(copy_images_folder, exist_ok=True)
    os.makedirs(copy_labels_folder, exist_ok=True)

    # 读取源文件夹中的图片和标签
    source_images = set(os.listdir(source_images_folder))
    source_labels = set(os.listdir(source_labels_folder))

    # 排除目标文件夹中已存在的文件
    existing_images = set(os.listdir(target_images_folder))
    existing_labels = set(os.listdir(target_labels_folder))

    available_images = source_images - existing_images
    available_labels = source_labels - existing_labels
    available_files = set(file.split('.')[0] for file in available_images) & set(
        file.split('.')[0] for file in available_labels)

    # 设置设备和模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = "../weights/best.pt"
    model = DetectMultiBackend(weights, device=device, dnn=False, data='data/VisDrone.yaml', fp16=False)

    selected_files = []

    # 计算每张图片的预测结果，并按照类别分配图片
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

        # 如果预测到的类别满足条件（稀缺类别），则选中该图片
        if pred is not None and len(pred):
            labels = pred[:, 5].cpu().numpy()  # 获取预测的类别标签
            for label in labels:
                label = str(int(label))
                if class_counts[label] < class_thresholds[label]:  # 如果类别数小于阈值，优先选择
                    selected_files.append(file)
                    break

    # 确保选择的图片数量满足要求
    selected_files = list(set(selected_files))  # 去重
    selected_files = selected_files[:n]  # 只取前n个文件

    # 将选中的图片和标签复制到目标文件夹
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
    copy_images_folder = "../dataset/VisDrone_part/new_select/images"
    copy_labels_folder = "../dataset/VisDrone_part/new_select/labels"
    pic_num = 1000  # 图片数量
    select_and_copy_files(source_images_folder, source_labels_folder, target_images_folder, target_labels_folder,
                          copy_images_folder, copy_labels_folder, n=pic_num)

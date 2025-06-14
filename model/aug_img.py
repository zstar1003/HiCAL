import os
import random

import cv2
import numpy as np


def image_flip(img, flip_code):
    """
    翻转图像（水平或垂直）。
    """
    return cv2.flip(img, flip_code)


def hsv_adjustment(img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    """
    调整图像的 HSV 颜色空间。
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(img_hsv)
    h = ((h + random.uniform(-1, 1) * h_gain * 180) % 180).astype(np.uint8)
    s = np.clip(s * (1 + random.uniform(-1, 1) * s_gain), 0, 255).astype(np.uint8)
    v = np.clip(v * (1 + random.uniform(-1, 1) * v_gain), 0, 255).astype(np.uint8)
    img_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def gaussian_blur(img, kernel_size=(7, 7), sigma=5.0):
    """
    对图像应用高斯模糊。
    """
    return cv2.GaussianBlur(img, kernel_size, sigma)


def save_images(input_folder, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图像文件
    for image_file in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, image_file)
        # 读取原始图像
        img = cv2.imread(input_image_path)
        if img is None:
            print(f"无法读取图像：{input_image_path}")
            continue

        # 获取图像文件名（不包含扩展名）
        image_name = os.path.splitext(os.path.basename(input_image_path))[0]

        # 应用所有图像增强并保存最终结果
        # enhanced_img = gaussian_blur(hsv_adjustment(image_flip(img, 1)))
        # enhanced_img = image_flip(img, 1)
        enhanced_img = gaussian_blur(img)
        # enhanced_img = gaussian_blur(hsv_adjustment(image_flip(img, 1)))
        enhanced_image_path = os.path.join(output_folder, f"{image_name}_enhanced3.jpg")
        cv2.imwrite(enhanced_image_path, enhanced_img)
        print(f"已保存增强图像到文件夹：{output_folder}")


if __name__ == "__main__":
    # 输入文件夹路径
    input_folder = "dataset/final_sample_img"  # 修改为您的文件夹路径
    # 输出文件夹路径
    output_folder = "dataset/final_sample_img5"  # 修改为您的输出文件夹路径
    # 保存图像
    save_images(input_folder, output_folder)

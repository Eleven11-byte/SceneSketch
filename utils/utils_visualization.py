import csv
import torch
import numpy as np
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import os
from time import sleep
from random import randint
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import nltk
import string
import warnings
from bresenham import bresenham
import scipy
from scipy import stats
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn.functional as F
from datetime import datetime
import cv2

def display_segmented_sketch(pixel_similarity_array, binary_sketch, classes, classes_colors, save_path=None,
                             live=False):
    # Find the class index with the highest similarity for each pixel
    class_indices = np.argmax(pixel_similarity_array, axis=0)
    # Create an HSV image placeholder
    hsv_image = np.zeros(class_indices.shape + (3,))  # Shape (512, 512, 3)
    hsv_image[..., 2] = 1  # Set Value to 1 for a white base

    # Set the hue and value channels
    for i, color in enumerate(classes_colors):
        rgb_color = np.array(color).reshape(1, 1, 3)
        hsv_color = rgb_to_hsv(rgb_color)
        mask = class_indices == i
        if i < len(classes):  # For the first N-2 classes, set color based on similarity
            hsv_image[..., 0][mask] = hsv_color[0, 0, 0]  # Hue
            hsv_image[..., 1][mask] = pixel_similarity_array[i][mask] > 0  # Saturation
            hsv_image[..., 2][mask] = pixel_similarity_array[i][mask]  # Value
        else:  # For the last two classes, set pixels to black
            hsv_image[..., 0][mask] = 0  # Hue doesn't matter for black
            hsv_image[..., 1][mask] = 0  # Saturation set to 0
            hsv_image[..., 2][mask] = 0  # Value set to 0, making it black

    mask_tensor_org = binary_sketch[:, :, 0] / 255
    hsv_image[mask_tensor_org == 1] = [0, 0, 1]

    # Convert the HSV image back to RGB to display and save
    rgb_image = hsv_to_rgb(hsv_image)

    # Display the image with class names
    plt.imshow(rgb_image)
    plt.axis('off')
    plt.tight_layout()

    if live:
        plt.savefig('output.png', bbox_inches='tight', pad_inches=0)

    else:
        save_dir = "/".join(save_path.split("/")[:-1])
        if save_dir != '':
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

        else:
            plt.show()

def visualize_feature_map(feature_map, save_path=None, title="Feature Map"):
    """
    可视化特征图（支持单通道或多通道）
    feature_map: 输入特征张量，形状为 [C, H, W] 或 [H, W]
    """
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.cpu().detach().numpy()

    # 处理单通道特征
    if len(feature_map.shape) == 2:
        plt.figure(figsize=(8, 8))
        plt.imshow(feature_map, cmap='viridis')
        plt.title(title)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        else:
            plt.show()
        plt.close()
        return

    # 处理多通道特征（取前16个通道可视化）
    C, H, W = feature_map.shape
    num_channels = min(C, 16)  # 最多显示16个通道
    rows = int(num_channels ** 0.5)
    cols = (num_channels + rows - 1) // rows

    plt.figure(figsize=(cols * 4, rows * 4))
    for i in range(num_channels):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_map[i], cmap='viridis')
        plt.title(f"Channel {i}")
        plt.axis('off')

    plt.suptitle(title, y=1.02)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.close()

def visualize_attention_weights(attention_weights, img_size, save_path=None, title="Attention Weights"):
    """可视化注意力权重（将注意力图上采样到原图大小）"""
    # attention_weights shape: [num_heads, H_patch, W_patch]
    num_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(min(num_heads, 8)):  # 可视化前8个注意力头
        attn_map = attention_weights[i].cpu().detach().numpy()
        # 上采样到原图大小
        attn_map = scipy.ndimage.zoom(attn_map, zoom=(img_size[0] / attn_map.shape[0], img_size[1] / attn_map.shape[1]),
                                      order=1)

        ax = axes[i]
        ax.imshow(attn_map, cmap="jet")
        ax.set_title(f"Head {i + 1}")
        ax.axis("off")

    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

def attn_to_heatmap(attn, num_tokens, img_size=224):
    """
    attn: (heads, N, N)
    num_prompts: 模型中添加的 prompt token 数量 P
    """
    # 平均 head
    attn = attn.mean(dim=0)  # (N, N)

    patch_start = 1 + num_tokens

    # 取 CLS 对 patch 的注意力
    cls_attn = attn[0, patch_start:]

    n = cls_attn.shape[0]
    grid = int(n ** 0.5)

    if grid * grid != n:
        print(f"[Warning] Patch token 数 {n} 非完美平方，裁剪为 {grid*grid}")
        cls_attn = cls_attn[:grid * grid]

    cls_attn = cls_attn.reshape(grid, grid)

    # 插值到原图大小
    cls_attn = torch.nn.functional.interpolate(
        cls_attn.unsqueeze(0).unsqueeze(0),
        size=(img_size, img_size),
        mode="bilinear"
    )[0, 0]

    cls_attn = cls_attn / (cls_attn.max() + 1e-6)
    return cls_attn

def visualize_cls_attention(attn, img, img_h, img_w, num_tokens=0):
    attn = attn.mean(0)
    start = 1 + num_tokens
    cls_attn = attn[0, start:]

    cls_attn = cls_attn.reshape(img_h, img_w)
    cls_attn = cls_attn.unsqueeze(0).unsqueeze(0)
    cls_attn = F.interpolate(
        cls_attn,
        size=img.shape[-2:],
        mode="bilinear",
        align_corners=False
    ).squeeze().cpu().numpy()

    img_np = img[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    plt.imshow(img_np)
    plt.imshow(cls_attn, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.show()

def visualize_segmentation_overlay(image, mask, palette, mask_no = None,groundtruth = None, show = False, result_file=None, class_names=None, alpha=0.5, title="Segmentation Overlay"):
    """
    可视化语义分割结果叠加在原图上，并显示类别与颜色映射。

    参数:
        image: np.ndarray 或 PIL.Image，原始输入图像 (H, W, 3)，RGB 格式。
        mask: torch.Tensor 或 np.ndarray，语义分割结果 (H, W)，表示每个像素的类别索引。
        palette: np.ndarray，形状 (num_classes, 3)，每个类别的 RGB 颜色值 (0-255)。
        class_names: list[str] 或 None，可选，类别名称列表（长度等于 num_classes）。
        alpha: float，可选，叠加透明度，0~1，越大颜色越明显。
        title: str，可选，主标题。
    """
    # --- 统一格式 ---
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    image = np.transpose(image, (1, 2, 0)) #TODO：还未改成通用，根据实际情况修改
    # --- 2️成分割彩色图 ---
    color_mask = palette[mask].astype(np.uint8)
    # color_mask_resized = cv2.resize(color_mask, (image.shape[1], image.shape[0]))

    # --- 3️叠加 ---
    overlay = cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)
    num_classes = len(class_names)

    if groundtruth is None:
        # --- 4️绘制结果 ---

        fig, axes = plt.subplots(1, 2, figsize=(6, 6))

        # 左图：叠加结果
        axes[0].imshow(overlay)
        axes[0].set_title(title)
        axes[0].axis("off")

        # 右图：类别与颜色映射
        # legend_img = np.zeros((num_classes * 30, 200, 3), dtype=np.uint8)
        legend_img = np.ones((num_classes * 30, 100, 3), dtype=np.uint8) * 255  # 白色背景
        for i in range(num_classes):
            legend_img[i * 30:(i + 1) * 30, :50, :] = palette[i]
            label = class_names[i] if class_names else f"Class {i}"
            axes[1].text(60, i * 30 + 20, label, fontsize=10, va='center', color='black')
        axes[1].imshow(legend_img)
        axes[1].set_title("Class - Color Mapping")
        axes[1].axis("off")

        plt.tight_layout()
    else:
        color_mask_gt = palette[groundtruth].astype(np.uint8)
        overlay_gt = cv2.addWeighted(image, 1 - alpha, color_mask_gt, alpha, 0)

        color_mask_no = palette[mask_no].astype(np.uint8)
        overlay_no = cv2.addWeighted(image, 1 - alpha, color_mask_no, alpha, 0)

        fig, axes = plt.subplots(2, 2, figsize=(6, 6))

        axes[0, 0].imshow(overlay_gt)
        axes[0, 0].set_title("groundtruth")
        axes[0, 0].axis("off")

        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title("result")
        axes[1, 0].axis("off")

        # 叠加结果
        axes[1, 1].imshow(overlay_no)
        axes[1, 1].set_title("directly_result")
        axes[1, 1].axis("off")
        # 右图：类别与颜色映射
        # legend_img = np.zeros((num_classes * 30, 200, 3), dtype=np.uint8)
        legend_img = np.ones((num_classes * 30, 100, 3), dtype=np.uint8) * 255  # 白色背景
        for i in range(num_classes):
            legend_img[i * 30:(i + 1) * 30, :50, :] = palette[i]
            label = class_names[i] if class_names else f"Class {i}"
            axes[0, 1].text(60, i * 30 + 20, label, fontsize=10, va='center', color='black')
        axes[0, 1].imshow(legend_img)
        axes[0, 1].set_title("Class - Color Mapping")
        axes[0, 1].axis("off")

        plt.tight_layout()
    if result_file is not None:
        plt.savefig(result_file)
    if show:
        plt.show()
    plt.close(fig)

def visualize_attn_map(attn_map,original_img,patch_size,alpha=0.5,agg='mean',save_path=None,save_name=None,dpi=200,show=True):
    """
    Visualize and optionally save attention heatmap overlay.

    Args:
        attn_map: Tensor [N, N] or [H, W, H, W] flattened beforehand
        original_img: Tensor [1, 3, H, W]
        patch_size: patch size of ViT
        alpha: heatmap blending factor
        agg: 'mean' or 'max'
        save_path: directory to save visualization
        save_name: filename (e.g. img_001_layer7_mean.png)
        dpi: save resolution
        show: whether to display the figure
    """

    # ---- 原图处理 ----
    img = original_img[0].permute(1, 2, 0).detach().cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    H, W = img.shape[:2]
    h, w = H // patch_size, W // patch_size

    # ---- 注意力聚合 ----
    attn = attn_map.detach().cpu().float()
    if agg == 'mean':
        attn_global = attn.mean(0)
    elif agg == 'max':
        attn_global = attn.max(0)[0]
    else:
        raise ValueError("agg must be 'mean' or 'max'")

    # ---- 归一化 + reshape ----
    attn_global = (attn_global - attn_global.min()) / (attn_global.max() - attn_global.min() + 1e-8)
    attn_global = attn_global.reshape(h, w)

    # ---- resize 到原图 ----
    attn_resized = cv2.resize(attn_global.numpy(), (W, H), interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    # ---- 叠加 ----
    overlay = alpha * heatmap + (1 - alpha) * img
    overlay = np.clip(overlay, 0, 1)

    # ---- 绘制 ----
    plt.figure(figsize=(8, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title(f"Attention Visualization ({agg})")

    # ---- 保存 ----
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if save_name is None:
            save_name = f"attn_{agg}.png"
        save_file = os.path.join(save_path, save_name)
        plt.savefig(save_file, bbox_inches="tight", pad_inches=0, dpi=dpi)
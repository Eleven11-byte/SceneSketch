from dataset import InkScene_test
import torch
from torchvision.transforms import InterpolationMode
from configs.config import Config
BICUBIC = InterpolationMode.BICUBIC
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from collections import OrderedDict
# from vpt.launch import default_argument_parser
import numpy as np
from PIL import Image
from dataset import fscoco_test
import json
# from models.pipeline import Pipeline
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torch.nn.functional as F
import csv
from datetime import datetime
import pandas as pd
from models.modified_model import ModifiedCLIP as CLIPer
import scipy.io as sio
from test_fscoco import PALETTE, visualize_segmentation_overlay
import matplotlib.pyplot as plt

def visualize_pred_mask(pred_mask,palette,save_path=None,show=True,title="Predicted Segmentation"):


    # -------- 转 numpy --------
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.detach().cpu().numpy()

    pred_mask = pred_mask.astype(np.int64)

    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    num_colors = palette.shape[0]

    # -------- 映射颜色 --------
    for cls_id in range(num_colors):
        color_mask[pred_mask == cls_id] = palette[cls_id]

    # -------- 显示 / 保存 --------
    plt.figure(figsize=(6, 6))
    plt.imshow(color_mask)
    plt.axis("off")
    plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()

    plt.close()

def compute_accuracy(gt_seg: np.ndarray,pred_seg: np.ndarray,ignore_index: int = 0):
    assert gt_seg.shape == pred_seg.shape, \
        f"Shape mismatch: {gt_seg.shape} vs {pred_seg.shape}"

    # valid pixels
    valid_mask = gt_seg != ignore_index

    if valid_mask.sum() == 0:
        return 0.0

    correct = (gt_seg[valid_mask] == pred_seg[valid_mask]).sum()
    total = valid_mask.sum()

    pixel_accuracy = correct / total
    return float(pixel_accuracy)

import numpy as np

def compute_miou(gt_seg: np.ndarray,pred_seg: np.ndarray,num_classes: int = None,ignore_index: int = 0):
    assert gt_seg.shape == pred_seg.shape, \
        f"Shape mismatch: {gt_seg.shape} vs {pred_seg.shape}"

    if num_classes is None:
        num_classes = int(max(gt_seg.max(), pred_seg.max())) + 1

    ious = []

    for cls in range(num_classes):
        if cls == ignore_index:
            continue

        gt_mask = (gt_seg == cls)
        pred_mask = (pred_seg == cls)

        # 跳过 GT 中完全不存在的类
        if gt_mask.sum() == 0:
            continue

        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()

        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union

        ious.append(iou)

    if len(ious) == 0:
        return 0.0

    return float(np.mean(ious))


def process_image(image,ori_img,fg_classes, bg_classes, segmentation_threshold: float,
                  class_threshold: float):

    # ori_img = torch.tensor(image.astype(float)).permute(2, 0, 1) / 255.
    # image = CLIPer.preprocess(Image.fromarray(image))[None]
    # TODO: text_features加模板
    # fg_text_features = CLIPer.classifier(fg_classes, cfg.semantic_templates)
    # bg_text_features = CLIPer.classifier(bg_classes, cfg.semantic_templates)
    """
    bg_text_features = torch.cat([bg_text_features,
                                  embedding[(fg_text_features @ embedding.T).sort().indices[:, ::1500]].flatten(0, 1)],
                                 dim=0)
    """
    # tokenized_classes = clip.tokenize(fg_classes).to(device)
    image = CLIPer.preprocess(Image.fromarray(image))[None]
    pred_mask, final_score = CLIPer(image.to(device), text_classes=fg_classes, bg_classes=bg_classes)
    pred_mask = F.interpolate(pred_mask[None], size=(ori_img.shape[-2], ori_img.shape[-1]), mode='bilinear')[0]
    pred_mask[final_score < class_threshold] = 0
    pred_mask = torch.cat([torch.ones_like(pred_mask[:1]) * segmentation_threshold, pred_mask]) #拼接背景通道
    mask = pred_mask.argmax(dim=0)
    return mask


def test_inkscene():
    preprocess = Compose([Resize((224, 224), interpolation=BICUBIC),ToTensor(),Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])

    dataset = InkScene_test(root="/home/xiaoyi/project/inkscene/clipasso-base_test",class_map_json="/home/xiaoyi/project/inkscene/clipasso-base_test/mapping.json")

    average_P_metric = []
    total_miou = 0

    for pil_img, gt_class_map, img_path in tqdm(dataset, desc="Testing InkScene"):
        # ------------------
        # 原图
        # ------------------
        ori_img = np.array(pil_img)
        ori_img_tensor = torch.tensor(ori_img).permute(2, 0, 1).float() / 255.

        device = "cpu"

        # CLIP 输入
        sketch_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # ------------------
        # GT → 类别文本
        # ------------------
        unique_ids = np.unique(gt_class_map)
        fg_class_names = [dataset.id2class[i] for i in unique_ids if i != 0]

        classes = ["blank_pixel"] + fg_class_names

        # ------------------
        # 模型预测（⚠️ 关键）
        # ------------------
        pred_mask = process_image(ori_img, ori_img_tensor,fg_class_names,bg_classes=[""],segmentation_threshold=segmentation_threshold,class_threshold=class_threshold)

        pred_mask = pred_mask.cpu().numpy()

        # visualize_pred_mask(pred_mask,PALETTE,save_path="pred_vis.png",show=True)

        img_name = img_path.split('/')[-1].split('.')[0]
        result_file = os.path.join(save_dir, f"{img_name}.jpg")
        # visualize_segmentation_overlay(ori_img, pred_sketch_seg, PALETTE, result_file, class_names=classes, alpha=0.5, title="Segmentation Overlay")


        # ------------------
        # GT: class_id → index
        # ------------------
        name2index = {name: i for i, name in enumerate(classes)}

        gt_index_map = np.zeros_like(gt_class_map)
        for class_id, class_name in dataset.id2class.items():
            if class_name in name2index:
                gt_index_map[gt_class_map == class_id] = name2index[class_name]

        # pred_mask 本身就是 index（对应 fg_class_names 顺序）
        pred_index_map = pred_mask.copy()

        # 只评估 GT 前景
        pred_index_map[gt_index_map == 0] = 0

        # ------------------
        # Metrics
        # ------------------
        P_metric = compute_accuracy(gt_index_map, pred_index_map)

        miou = compute_miou(gt_index_map,pred_index_map,num_classes=len(classes),ignore_index=0)

        average_P_metric.append(P_metric)
        total_miou += miou

        if visualized:
            mask = pred_mask.copy()
            mask[gt_index_map == 0] = 0
            visualize_segmentation_overlay(ori_img, mask, PALETTE, mask_no=pred_mask, groundtruth=gt_index_map,
                                           show=False, result_file=result_file, class_names=classes, alpha=0.5,
                                           title="Segmentation Overlay")



    print("\nInkScene Results")
    print(f"Pixel Accuracy: {sum(average_P_metric)/len(average_P_metric):.5f}")
    print(f"mIoU: {total_miou/len(dataset):.5f}")


if __name__ == '__main__':

    save_dir = "fscoco_test_result"
    cfg = Config("configs/test.yaml")
    cfg.update_from_cli()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.semantic_templates = [line.strip() for line in list(open(cfg.semantic_templates))]
    CLIPer = CLIPer(cfg, device=device)
    CLIPer = CLIPer.float()
    CLIPer.to(device)

    state_dict = torch.load(cfg.checkpoint_path)
    load_info = CLIPer.load_state_dict(state_dict, strict=False)
    print(load_info)

    # embedding = torch.load("embeddings_large14.pth", map_location=device)
    segmentation_threshold = cfg.segmentation_threshold

    class_threshold = cfg.class_threshold
    print(cfg)
    visualized = False

    test_inkscene()
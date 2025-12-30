import csv
import torch
import numpy as np
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
from PIL import Image
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

def compute_accuracy(strokes_seg, labels, classes, gt_seg, pred_seg):
    blank_pix_index = classes.index('blank_pixel')
    gt_pixels = gt_seg.flatten()[gt_seg.flatten() != blank_pix_index]
    pred_pixels = pred_seg.flatten()[pred_seg.flatten() != blank_pix_index]
    gt_labels_indices = [classes.index(label) for label in labels]
    # Compute pixel accuracy
    if len(gt_pixels) != len(pred_pixels):
        add_pix = len(gt_pixels) - len(pred_pixels)
        pred_pixels = np.append(pred_pixels, np.ones(add_pix))  # *unlabeled_pix_index

    P_metric = accuracy_score(gt_pixels, pred_pixels)
    # Compute stroke accuracy
    # For each stroke, determine the most frequent pixel label in prediction
    pred_strokes = []
    for stroke in strokes_seg:
        pred_stroke_pixels = pred_seg[stroke == 1]
        if len(pred_stroke_pixels) == 0:
            pred_strokes.append(blank_pix_index)
            continue
        # pred_strokes.append(stats.mode(pred_stroke_pixels)[0][0])
        pred_strokes.append(stats.mode(pred_stroke_pixels)[0].item())
    C_metric = accuracy_score(gt_labels_indices, pred_strokes)
    return P_metric, C_metric


def compute_miou(gt_seg, pred_seg, all_classes):
    num_classes = len(all_classes)
    # Initialize a matrix to hold the sum of IoUs for each class
    class_iou_sum = np.zeros(num_classes, dtype=np.float32)
    class_counts = np.zeros(num_classes, dtype=np.int32)

    for c in range(num_classes):
        if c == 0:
            continue
        # Compute the intersection and union for the current class
        intersection = np.sum((gt_seg == c) & (pred_seg == c))
        union = np.sum((gt_seg == c) | (pred_seg == c))

        # If the union is 0, this class is not present in either ground truth or prediction
        if union == 0:
            continue

        # Increment the class IoU sum and class counts
        class_iou_sum[c] += intersection / union
        class_counts[c] += 1

    # Compute mean IoU only for classes that are present (avoid division by 0)
    miou = np.sum(class_iou_sum[class_counts > 0]) / np.sum(class_counts[class_counts > 0])
    return miou

def process_image(model, device, image,ori_img,fg_classes, bg_classes, segmentation_threshold: float,
                  class_threshold: float, idx: int):
    image = model.preprocess(Image.fromarray(image))[None]
    pred_mask, final_score = model(image.to(device), text_classes=fg_classes, bg_classes=bg_classes, idx=idx)
    pred_mask = F.interpolate(pred_mask[None], size=(ori_img.shape[-2], ori_img.shape[-1]), mode='bilinear')[0]
    pred_mask[final_score < class_threshold] = 0
    pred_mask = torch.cat([torch.ones_like(pred_mask[:1]) * segmentation_threshold, pred_mask]) #TODO：拼接背景通道到第0维
    mask = pred_mask.argmax(dim=0)
    return mask
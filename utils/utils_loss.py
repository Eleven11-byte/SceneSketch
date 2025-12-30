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

def triplet_loss_func_L1(sketch_embeddings, class_embeddings, labels, margin=0.2):
    batch_size = sketch_embeddings.shape[0]

    # pairwise distance matrix using L1 distance
    sketch_embeddings = sketch_embeddings.contiguous()
    # dists = torch.cdist(sketch_embeddings, class_embeddings, p=1)
    dists = torch.cdist(
        sketch_embeddings.float(),  # 半精度→单精度
        class_embeddings.float(),
        p=1
    )
    # diagonal mask (for positive samples)
    diag_mask = torch.eye(batch_size, dtype=bool).to(sketch_embeddings.device)
    # Positive distances (diagonal elements)
    positive_dists = dists[diag_mask]
    # Class mask (for negative samples)
    class_mask = labels[:, None] == labels[None, :]  # Shape: (batch_size, batch_size)
    class_mask = class_mask.to(sketch_embeddings.device)
    # Negative distances
    negative_dists = dists.clone()
    negative_dists[diag_mask | class_mask] = float('inf')  # Set diagonal elements and same class elements to a large value
    min_negative_dists, _ = torch.min(negative_dists, dim=1)
    # Compute triplet loss
    loss = torch.clamp(margin + positive_dists - min_negative_dists, min=0.0)

    return loss.mean()

def img_sketch_align_loss(sketch_feat, img_feat):
    """
    Args:
        sketch_feat: 草图编码器输出的全局特征，shape [B, D]（B=批次大小，D=特征维度）
        img_feat: CLIP图像编码器输出的全局特征，shape [B, D]（已冻结，无梯度）
    Return:
        图像-草图对齐损失，标量
    """

    # 计算余弦相似度（值越近越好，损失取1-相似度）
    cos_sim = torch.nn.functional.cosine_similarity(sketch_feat, img_feat, dim=1)
    loss = 1 - torch.mean(cos_sim)  # 均值损失，使批次内所有样本的相似度接近1
    return loss

def cross_modal_distill_loss(f_s, f_t, f_txt=None, tau=0.2,
                             lambda_feat=1.0, lambda_con=1.0, lambda_sem=0.5):
    """
    跨模态蒸馏损失：草图编码器(学生) -> 图像编码器(教师, CLIP)
    Args:
        f_s: sketch encoder features [B, D]
        f_t: image encoder features [B, D] (detached)
        f_txt: optional text encoder features [B, D]
        tau: 温度参数
    """
    # L_feat: 特征对齐 (L2)
    l_feat = F.mse_loss(f_s, f_t.detach())

    # L_con: 对比式蒸馏 (InfoNCE)
    f_s_norm = F.normalize(f_s, dim=-1)
    f_t_norm = F.normalize(f_t.detach(), dim=-1)
    logits = torch.matmul(f_s_norm, f_t_norm.T) / tau
    labels = torch.arange(f_s.size(0), device=f_s.device)
    l_con = F.cross_entropy(logits, labels)

    # L_sem: 语义一致性 (可选)
    l_sem = 0.0
    if f_txt is not None:
        f_txt_norm = F.normalize(f_txt.detach(), dim=-1)
        l_sem = 1 - (f_s_norm * f_txt_norm).sum(dim=-1).mean()

    return lambda_feat * l_feat + lambda_con * l_con + lambda_sem * l_sem

def patch_distribution_distill_loss(f_s, f_t, tau=1.0):
    """
    Patch-level relational distillation.
    Args:
        f_s: sketch patch features [B, N, D] (no CLS)
        f_t: image patch features  [B, N, D] (detached)
        tau: temperature for similarity smoothing
    Returns:
        scalar loss
    """

    # normalize
    f_s = F.normalize(f_s, dim=-1)
    f_t = F.normalize(f_t.detach(), dim=-1)

    # patch-wise similarity matrices
    sim_s = torch.matmul(f_s, f_s.transpose(1, 2)) / tau  # [B, N, N]
    sim_t = torch.matmul(f_t, f_t.transpose(1, 2)) / tau

    return F.mse_loss(sim_s, sim_t)

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
import random

def save_results_to_csv(cfg,results: dict,csv_path="experiment_results.csv",include: list = None,exclude: list = None,flatten: bool = False,):
    """
    可配置的 config + result 写入 csv 函数。
    参数:
        cfg: Config 实例 / dict（cfg.config）
        results: dict 实验结果
        csv_path: 保存路径
        include: 仅写入这些 config 字段，例如 ["wandb.enable", "train.batch_size"]
        exclude: 从 config 中排除这些字段
        flatten: 若为 True，则将整个 config flatten 成 a.b.c: value 的平面结构
    """
    # 从 Config 对象中取 dict
    cfg_dict = cfg.config if hasattr(cfg, "config") else cfg

    # --- 处理 config flatten 为 a.b.c:value ---
    def flatten_dict(d, prefix=""):
        flat = {}
        for k, v in d.items():
            nk = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_dict(v, nk))
            else:
                flat[nk] = v
        return flat

    if flatten:
        flat_cfg = flatten_dict(cfg_dict)
        cfg_record = flat_cfg
    else:
        # 否则按 include/exclude 筛选
        flat_cfg = flatten_dict(cfg_dict)
        if include:
            cfg_record = {k: flat_cfg.get(k, None) for k in include}
        else:
            cfg_record = flat_cfg.copy()

        if exclude:
            for k in exclude:
                cfg_record.pop(k, None)

    # --- 合并最终要写入的一条记录 ---
    record = {
        **cfg_record,        # 指定的 config 字段
        **results,           # 实验指标
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # --- 写入 CSV ---
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    print(f"[Saved] 实验结果写入 {csv_path}")


def get_similarity_map(sm, shape):
    # sm: torch.Size([1, 196, 1])
    # min-max norm
    sm = (sm - sm.min(1, keepdim=True)[0]) / (
                sm.max(1, keepdim=True)[0] - sm.min(1, keepdim=True)[0])  # torch.Size([1, 196, 1])

    # reshape
    side = int(sm.shape[1] ** 0.5)  # square output, side = 14
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)

    # interpolate
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)

    return sm.squeeze(0)

def flatten(lst):
    result = []
    for i in lst:
        if isinstance(i, list):
            result.extend(flatten(i))
        else:
            result.append(i)
    return result

def get_noun_phrase(tokenized):
    # Taken from Su Nam Kim Paper...
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)

    chunked = chunker.parse(nltk.pos_tag(tokenized))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            current_chunk.append(' '.join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = ' '.join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk

def sketch_text_pairs(sketch_batch, captions, max_classes=3):
    all_sketches = []
    all_classes = []
    all_captions = []
    for (sketch, caption) in zip(sketch_batch, captions):
        caption = caption.replace('\n', ' ')
        translator = str.maketrans('', '', string.punctuation)
        caption = caption.translate(translator).lower()
        words = nltk.word_tokenize(caption)
        classes = get_noun_phrase(words)
        classes = list(dict.fromkeys(classes))
        if len(classes) > max_classes:
            classes = classes[:max_classes]
        if len(classes) == 0:
            classes = caption
            sketch = sketch.unsqueeze(0)
        else:
            sketch = sketch.repeat(len(classes), 1, 1, 1)
            caption = [caption] * len(classes)
        all_sketches.append(sketch)
        all_classes.append(classes)
        all_captions.append(caption)

    return torch.cat(all_sketches), flatten(all_classes), flatten(all_captions)

def sketch_text_image_pairs(sketch_batch,captions,image_batch, max_classes=3):
    all_sketches = []
    all_classes = []
    all_captions = []
    all_images = []
    for (sketch, caption, image) in zip(sketch_batch, captions, image_batch):
        caption = caption.replace('\n', ' ')
        translator = str.maketrans('', '', string.punctuation)
        caption = caption.translate(translator).lower()
        words = nltk.word_tokenize(caption)
        classes = get_noun_phrase(words)
        classes = list(dict.fromkeys(classes))
        if len(classes) > max_classes:
            classes = classes[:max_classes]
        if len(classes) == 0:
            classes = caption
            sketch = sketch.unsqueeze(0)
            image = image.unsqueeze(0)
        else:
            sketch = sketch.repeat(len(classes), 1, 1, 1)
            caption = [caption] * len(classes)
            image = image.repeat(len(classes), 1, 1, 1)
        all_sketches.append(sketch)
        all_classes.append(classes)
        all_captions.append(caption)
        all_images.append(image)

    return torch.cat(all_sketches), flatten(all_classes), flatten(all_captions), torch.cat(all_images)

def tensor_to_binary_img(tensor,device):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(device)
    images_unnormalized = tensor * std + mean
    images_gray = images_unnormalized.mean(dim=1)
    threshold = 0.5
    binary_images = (images_gray > threshold).float()
    binary_images = binary_images.unsqueeze(1)
    binary_images = binary_images.repeat(1,3,1,1)
    return binary_images

def zero_clapping(similarity_maps,threshold):
    batch_size= similarity_maps.shape[0]
    indices = torch.arange(batch_size).to(similarity_maps.device)
    diagonal_ps = similarity_maps[torch.arange(batch_size), :, :, indices]
    max_ps = torch.max(diagonal_ps, dim=1)[0]
    min_ps = torch.min(diagonal_ps, dim=1)[0]
    diagonal_ps = (diagonal_ps - min_ps[:, None]) / (max_ps[:, None] - min_ps[:, None])
    diagonal_ps[diagonal_ps < threshold] = 0
    diagonal_ps[diagonal_ps >= threshold] = 1
    weights = diagonal_ps
    return weights


def get_threshold(learnable_threshold):
    noise = torch.normal(mean=0, std=0.005, size=learnable_threshold.shape).to(learnable_threshold.device)
    learnable_threshold.data.add_(noise)
    threshold_value = 0.4 + 0.5 * torch.sigmoid(learnable_threshold)
    return threshold_value


def get_train_classes(dataset,max_classes=3):
    train_classes = []
    # for i, (_,caption,_) in enumerate(dataset):
    for _, caption in tqdm(dataset, desc="Processing captions", ncols=100):
        caption = caption.replace('\n',' ')
        translator = str.maketrans('', '', string.punctuation)
        caption = caption.translate(translator).lower()
        words = nltk.word_tokenize(caption)
        classes = get_noun_phrase(words)
        # remove synonyms in classes
        classes = list(dict.fromkeys(classes))
        if len(classes) >max_classes:
            classes = classes[:max_classes]
        if len(classes) ==0:
            classes = caption
        train_classes.append(classes)
    train_classes = np.unique(flatten(train_classes))
    return train_classes

def get_train_classes_with_image(dataset, max_classes=3):
    train_classes = []
    translator = str.maketrans('', '', string.punctuation)

    for _, caption, _ in tqdm(dataset, desc="Processing captions", ncols=100):
        caption = caption.replace('\n',' ').translate(translator).lower()
        words = nltk.word_tokenize(caption)
        classes = get_noun_phrase(words)
        # 去重但保持顺序
        classes = list(dict.fromkeys(classes))
        if len(classes) > max_classes:
            classes = classes[:max_classes]
        if len(classes) == 0:
            classes = [caption]
        train_classes.extend(classes)
    # 扁平化后唯一化，并排序保证可复现
    train_classes = sorted(list(set(train_classes)))
    return train_classes

def pen_state_to_strokes(sketches):
    strokes=[]
    i_prev=0
    for i in range(len(sketches)):
        if sketches[i,2]==1:
            strokes.append(sketches[i_prev:i+1])
            i_prev=i+1
    return strokes

def preprocess(sketch_points, side):
    sketch_points = sketch_points.astype(float)
    sketch_points[:, :2] = sketch_points[:, :2] / np.array([256, 256])
    sketch_points[:, :2] = sketch_points[:, :2] * side
    sketch_points = np.round(sketch_points)
    return sketch_points


def mydrawPNG(vector_image, Side):
    raster_image = np.zeros((int(Side), int(Side)), dtype=np.float32)
    initX, initY = int(vector_image[0, 0]), int(vector_image[0, 1])
    pixel_length = 0

    for i in range(0, len(vector_image)):
        if i > 0:
            if vector_image[i - 1, 2] == 1:
                initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

        cordList = list(bresenham(initX, initY, int(vector_image[i, 0]), int(vector_image[i, 1])))
        pixel_length += len(cordList)

        for cord in cordList:
            if (cord[0] > 0 and cord[1] > 0) and (cord[0] < Side and cord[1] < Side):
                raster_image[cord[1], cord[0]] = 255.0
        initX, initY = int(vector_image[i, 0]), int(vector_image[i, 1])

    raster_image = scipy.ndimage.binary_dilation(raster_image)
    return raster_image


def rasterize_Sketch(stroke_points, side):  # points that constitute one stroke [ # of points , (x,y,penup)]
    stroke_points = preprocess(stroke_points, side)
    raster_stroke = mydrawPNG(stroke_points, side)

    return raster_stroke

def print_trainable_params(model):
    """打印模型中可训练的参数（requires_grad=True）"""
    trainable_params = []
    total_trainable_params = 0  # 统计可训练参数总数

    print("=" * 50)
    print("可训练参数列表（需要更新的参数）：")
    print("=" * 50)

    # 遍历模型所有参数（named_parameters() 返回 (参数名, 参数张量)）
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 记录参数名称、形状、参数数量
            param_shape = param.shape
            param_num = param.numel()  # 计算参数元素总数
            trainable_params.append({
                "name": name,
                "shape": param_shape,
                "num_params": param_num
            })
            total_trainable_params += param_num

            # NOTE:打印单参数信息
            # print(f"参数名：{name:20s} | 形状：{str(param_shape):15s} | 参数数量：{param_num:,}")

    # 打印汇总信息
    print("=" * 50)
    print(f"可训练参数总数：{total_trainable_params:,}")
    print(f"可训练参数层数：{len(trainable_params)}")
    print("=" * 50)

def check_nan_inf(tensor, name):
    if torch.isnan(tensor).any():
        print(f"⚠️ {name} 包含NaN，数量：{torch.isnan(tensor).sum().item()}")
    if torch.isinf(tensor).any():
        print(f"⚠️ {name} 包含inf，数量：{torch.isinf(tensor).sum().item()}")


def check_param_update(model, pretrain_params, threshold=1e-6):
    # 检查可训练参数的更新幅度（阈值：1e-6，小于则视为未更新）
    no_update_params = []
    updated_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 计算参数绝对差异的均值
            diff_mean = torch.abs(param.data - pretrain_params["model." + name]).mean().item()
            if diff_mean < threshold:
                no_update_params.append((name, diff_mean))
            else:
                updated_params.append((name, diff_mean))

    print("=" * 50)
    print(f"可训练参数总数：{len(pretrain_params)}")
    print(f"未更新参数数：{len(no_update_params)} | 更新参数数：{len(updated_params)}")
    print("=" * 50)

    if no_update_params:
        print("未更新的参数（差异均值）：")
        for name, diff in no_update_params:  # 只打印前10个
            print(f"  {name}: {diff:.8f}")
    return no_update_params

def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

def safe_restore_rng_state(rng_state):
    """
    安全恢复随机状态，包括 torch, cuda, numpy, random。
    自动处理类型、设备和 contiguous 问题。
    """
    # ===== Torch CPU RNG =====
    rng_torch = rng_state["torch"]
    if not isinstance(rng_torch, torch.ByteTensor):
        rng_torch = torch.tensor(rng_torch, dtype=torch.uint8, device='cpu')
    else:
        rng_torch = rng_torch.cpu()
    rng_torch = rng_torch.contiguous()
    torch.set_rng_state(rng_torch)

    # ===== Torch CUDA RNG =====
    rng_cuda = rng_state["cuda"]
    if isinstance(rng_cuda, list):
        for i, state in enumerate(rng_cuda):
            if not isinstance(state, torch.ByteTensor):
                state = torch.tensor(state, dtype=torch.uint8, device='cpu').contiguous()
            else:
                state = state.cpu().contiguous()
            torch.cuda.set_rng_state(state, device=i)
    else:
        if not isinstance(rng_cuda, torch.ByteTensor):
            rng_cuda = torch.tensor(rng_cuda, dtype=torch.uint8, device='cpu').contiguous()
        else:
            rng_cuda = rng_cuda.cpu().contiguous()
        torch.cuda.set_rng_state(rng_cuda)

    # ===== Numpy RNG =====
    np.random.set_state(rng_state["numpy"])

    # ===== Python random RNG =====
    random.setstate(rng_state["random"])


def load_checkpoint(path, model, vit_optimizer, threshold_optimizer, device):
    """
    加载 checkpoint 并恢复训练状态，包括模型、优化器、随机状态、train_classes。
    """
    print(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])

    vit_optimizer.load_state_dict(ckpt["vit_optimizer"])
    threshold_optimizer.load_state_dict(ckpt["threshold_optimizer"])

    start_epoch = ckpt["epoch"] + 1
    global_step = ckpt.get("global_step", 0)

    if "rng_state" in ckpt:
        safe_restore_rng_state(ckpt["rng_state"])
    else:
        print("Warning: checkpoint does not contain rng_state. Randomness may differ.")

    if "train_classes" not in ckpt:
        raise RuntimeError(
            "Checkpoint does not contain train_classes. Resume is unsafe!"
        )
    train_classes = ckpt["train_classes"]
    print(f"Loaded train_classes from checkpoint ({len(train_classes)} classes)")

    return start_epoch, global_step, train_classes
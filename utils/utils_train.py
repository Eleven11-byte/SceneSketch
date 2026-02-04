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

import re
from typing import List, Optional, Iterable, Set

import nltk
from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

_LEM = WordNetLemmatizer()

# 冠词/指示词/一些弱量词（按需扩展）
_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those",
    "some", "any", "each", "every"
}

# 常见“泛词”，不建议当 class（按你的数据集可删减）
_GENERIC_NOUNS = {
    "thing", "things", "object", "objects", "item", "items", "stuff",
    "photo", "picture", "image", "scene", "drawing", "sketch", "figure"
}

# 数量词（避免 "two dogs" 变成 class="two dog"）
_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "many", "several", "few"
}

def _normalize_text(s: str) -> str:
    s = s.lower()
    # 去标点但保留连字符（fire-hydrant）
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_np(np: str) -> str:
    """
    NP 规范化：
    - 小写、去标点
    - 去开头冠词/数量词
    - 词形还原（主要处理复数名词）
    """
    np = _normalize_text(np)
    words = np.split()
    # 去掉开头的冠词/数量词
    while words and (words[0] in _DETERMINERS or words[0] in _NUMBER_WORDS):
        words = words[1:]

    if not words:
        return ""

    # 对每个词 lemmatize，名词优先（复数->单数）
    words = [_LEM.lemmatize(w, pos="n") for w in words]
    np = " ".join(words).strip()
    return np

def extract_noun_phrases(tokens: List[str]) -> List[str]:
    """
    使用 RegexpParser 抽取名词短语 NP，并直接遍历 NP 子树：
    - 不会漏掉句尾 NP
    - 不会把连续 NP 错拼在一起
    """
    grammar = r"""
        NBAR: {<JJ.*|NN.*>*<NN.*>}      # (adj|noun)* + noun
        NP:   {<NBAR>}
              {<NBAR><IN><NBAR>}       # e.g., "cup of tea"
              {<NBAR><POS><NBAR>}      # e.g., "dog 's tail"（所有格，效果视 pos_tag 而定）
    """
    chunker = nltk.RegexpParser(grammar)
    tagged = nltk.pos_tag(tokens)
    tree = chunker.parse(tagged)

    nps = []
    for st in tree.subtrees(lambda t: isinstance(t, nltk.Tree) and t.label() == "NP"):
        phrase = " ".join(w for w, _ in st.leaves())
        nps.append(phrase)
    return nps

def extract_classes_from_caption(
    caption: str,
    max_classes: int = 3,
    train_classes: Optional[Iterable[str]] = None,
    prefer_vocab_match: bool = True,
    keep_multiword: bool = True,
    drop_generic: bool = True,
) -> List[str]:
    """
    从 caption 提取 class（名词短语为主），用于塞进 template 再编码。
    返回的字符串可直接喂给 model.encode_text。

    - train_classes: 若提供，会优先返回词表中的类名（稳定 label）
    - keep_multiword: True 保留多词类（traffic light）；False 仅取 head noun（light）
    """
    cap = _normalize_text(caption.replace("\n", " "))
    tokens = nltk.word_tokenize(cap)

    # 1) 抽 NP 候选
    nps_raw = extract_noun_phrases(tokens)

    # 2) 规范化 + 去重（保序）
    cands: List[str] = []
    seen: Set[str] = set()
    for np in nps_raw:
        npn = _normalize_np(np)
        if not npn:
            continue
        if not keep_multiword:
            npn = npn.split()[-1]  # head noun
        if drop_generic and npn in _GENERIC_NOUNS:
            continue
        if npn not in seen:
            seen.add(npn)
            cands.append(npn)

    if not cands:
        return [caption]

    # 3) 若有 train_classes：优先对齐词表
    if train_classes is not None:
        # 词表规范化（一次性构建映射：norm -> orig）
        norm_to_orig = {}
        vocab_norm_set = set()
        for orig in train_classes:
            n = _normalize_np(orig)
            if not n:
                continue
            vocab_norm_set.add(n)
            # 同一个 norm 可能对应多个原词，保留更长/更具体的
            if n not in norm_to_orig or len(orig) > len(norm_to_orig[n]):
                norm_to_orig[n] = orig

        matched, rest = [], []
        for c in cands:
            if c in vocab_norm_set:
                matched.append(norm_to_orig.get(c, c))
            else:
                head = c.split()[-1]
                if head in vocab_norm_set:
                    matched.append(norm_to_orig.get(head, head))
                else:
                    rest.append(c)

        cands = matched + rest if prefer_vocab_match else cands

        # 再去重（matched 后可能重复）
        final = []
        seen2 = set()
        for x in cands:
            xn = _normalize_np(x)
            if drop_generic and xn in _GENERIC_NOUNS:
                continue
            if xn not in seen2:
                seen2.add(xn)
                final.append(x)

        return final[:max_classes]

    # 4) 无词表：优先更具体（多词/更长）但尽量不打乱顺序
    # 简单策略：按“词数多 → 更长 → 更早出现”排序
    final = sorted(
        cands,
        key=lambda x: (-(len(x.split())), -len(x))
    )
    return final[:max_classes]


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
        # classes = extract_classes_from_caption(caption, max_classes=max_classes,prefer_vocab_match=True,keep_multiword=True)
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
        # classes = extract_classes_from_caption(caption, max_classes=max_classes,prefer_vocab_match=True,keep_multiword=True)
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

def zero_clapping_attn(attn_map, threshold):
    """
    attn_map: [B, H, W] 或 [H, W]
    threshold: scalar
    """
    if attn_map.dim() == 2:
        attn_map = attn_map.unsqueeze(0)  # [1, H, W]

    # per-sample normalization（很重要）
    min_val = attn_map.view(attn_map.size(0), -1).min(dim=1)[0]
    max_val = attn_map.view(attn_map.size(0), -1).max(dim=1)[0]

    attn_norm = (attn_map - min_val[:, None, None]) / \
                (max_val[:, None, None] - min_val[:, None, None] + 1e-6)

    weights = (attn_norm >= threshold).float()
    return weights


def get_threshold(learnable_threshold):
    noise = torch.normal(mean=0, std=0.005, size=learnable_threshold.shape).to(learnable_threshold.device)
    learnable_threshold.data.add_(noise)
    threshold_value = 0.4 + 0.5 * torch.sigmoid(learnable_threshold)
    return threshold_value

def get_train_classes_modified(dataset):
    train_classes = []
    for _, caption,classes_lists, image_ids in tqdm(dataset, desc="Processing captions", ncols=100):
        classes = list(dict.fromkeys(classes_lists))
        train_classes.append(classes)
    train_classes = np.unique(flatten(train_classes))
    return train_classes

def get_train_classes(dataset,max_classes=3):
    train_classes = []
    # for i, (_,caption,_) in enumerate(dataset):
    for _, caption, in tqdm(dataset, desc="Processing captions", ncols=100):
        caption = caption.replace('\n',' ')
        translator = str.maketrans('', '', string.punctuation)
        caption = caption.translate(translator).lower()
        words = nltk.word_tokenize(caption)
        classes = get_noun_phrase(words)
        # classes = extract_classes_from_caption(caption, max_classes=max_classes, prefer_vocab_match=True, keep_multiword=True)
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
        # classes = extract_classes_from_caption(caption, max_classes=max_classes, prefer_vocab_match=True, keep_multiword=True)
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
            print(f"参数名：{name:20s} | 形状：{str(param_shape):15s} | 参数数量：{param_num:,}")

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

    return start_epoch, global_step

def get_attention_map(attn, image_size, patch_size, use_cls=True):
    """
    attn: [L*H, T, T]
    return: [H_img, W_img]
    """
    L = 12
    H = attn.shape[0] // L
    attn = attn.view(L, H, attn.shape[-2], attn.shape[-1])

    if use_cls:
        attn_map = attn[:, :, 0, 1:].mean(dim=(0, 1))  # [196]
    else:
        attn_map = attn.mean(dim=(0, 1, 2))[1:]        # fallback

    h = image_size[0] // patch_size
    w = image_size[1] // patch_size
    attn_map = attn_map.view(h, w)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).unsqueeze(0),
        size=image_size,
        mode="bilinear",
        align_corners=False
    ).squeeze()

    return attn_map

import torch
import torch.nn.functional as F

def get_attention_map_batch(
    attn: torch.Tensor,
    image_size,                 # (H_img, W_img)
    patch_size: int,
    B: int,
    num_heads: int,
    use_cls: bool = True,
    reduce_head: str = "mean",  # "mean" | "max"
    batch_mode: str = "identity" # "identity" | "broadcast" | "similarity"
):
    """
    attn: (num_heads*B, 197, 197)  where 197 = 1 (CLS) + 196 (patch tokens)
    return: attn_maps (B, H_img, W_img, B)
    """

    assert attn.dim() == 3 and attn.shape[-1] == attn.shape[-2], "attn must be (N, T, T)"
    T = attn.shape[-1]
    assert T == 197, f"Expected T=197, got {T}"
    assert attn.shape[0] == num_heads * B, f"Expected first dim = num_heads*B = {num_heads*B}, got {attn.shape[0]}"

    # 1) reshape -> (B, heads, T, T)
    attn = attn.view(B, num_heads, T, T)

    # 2) get patch attention vector of length 196 per sample
    if use_cls:
        # CLS -> patches
        patch_attn = attn[:, :, 0, 1:]          # (B, heads, 196)
    else:
        # fallback: average over query tokens, then take patches
        patch_attn = attn.mean(dim=2)[:, :, 1:] # (B, heads, 196)

    # 3) reduce heads -> (B, 196)
    if reduce_head == "mean":
        patch_attn = patch_attn.mean(dim=1)
    elif reduce_head == "max":
        patch_attn = patch_attn.max(dim=1)[0]
    else:
        raise ValueError("reduce_head must be 'mean' or 'max'")

    # 4) reshape 196 -> (B, h_p, w_p)
    h_p = image_size[0] // patch_size
    w_p = image_size[1] // patch_size
    assert h_p * w_p == 196, f"Expected (H/patch)*(W/patch)=196, got {h_p}*{w_p}={h_p*w_p}"
    attn_map = patch_attn.view(B, h_p, w_p)    # (B, 14, 14)

    # 5) upsample to image_size -> (B, H_img, W_img)
    attn_map = F.interpolate(
        attn_map.unsqueeze(1),
        size=image_size,
        mode="bilinear",
        align_corners=False
    ).squeeze(1)  # (B, H_img, W_img)

    # 6) construct (B, H_img, W_img, B)
    if batch_mode == "identity":
        # 每个样本只填到自己的 batch index（最稳，基本等价于原先(B,H,W)但多了最后一维）
        attn_maps = torch.zeros(B, image_size[0], image_size[1], B, device=attn.device, dtype=attn.dtype)
        idx = torch.arange(B, device=attn.device)
        attn_maps[idx, :, :, idx] = attn_map

    elif batch_mode == "broadcast":
        # 将每个样本的 attention 广播到最后一维B
        attn_maps = attn_map.unsqueeze(-1).expand(B, image_size[0], image_size[1], B)

    elif batch_mode == "similarity":
        # 用 batch 内 attention map 相似度做一个 cross-batch 权重
        x = attn_map.flatten(1)                           # (B, H*W)
        x = x / (x.norm(dim=1, keepdim=True) + 1e-6)
        sim = x @ x.T                                     # (B, B)
        attn_maps = attn_map.unsqueeze(-1) * sim[:, None, None, :]  # (B,H,W,B)

    else:
        raise ValueError("batch_mode must be 'identity', 'broadcast', or 'similarity'")

    return attn_maps


def expand_pairs(sketch_batch, captions, classes_lists, max_classes=3):
    all_sketches, all_classes, all_captions = [], [], []
    for sketch, caption, cls_list in zip(sketch_batch, captions, classes_lists):
        cls_list = cls_list[:max_classes] if cls_list else []
        cls_list = [c.strip() for c in cls_list if c and c.strip()]
        cls_list = list(dict.fromkeys(cls_list))[:max_classes]
        if len(cls_list) == 0:
            # fallback：不建议用整句当 class；更稳是用一个特殊 token
            # cls_list = ["background"]  # 或者 "unknown"
            cls_list = [caption]
        # 每个 class 一个 pair
        all_sketches.append(sketch.repeat(len(cls_list), 1, 1, 1))
        all_classes.extend(cls_list)
        all_captions.extend([caption.replace("\n", " ")] * len(cls_list))

    return torch.cat(all_sketches, dim=0), all_classes, all_captions

def _extract_self_map(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,H,W) or (B,H,W,B)
    return: (B,H,W)  (if batch-aware, take diagonal)
    """
    if x.dim() == 4:
        # (B,H,W,B) -> take diagonal self maps
        B = x.shape[0]
        idx = torch.arange(B, device=x.device)
        return x[idx, :, :, idx]
    elif x.dim() == 3:
        return x
    else:
        raise ValueError(f"Expected x dim 3 or 4, got {x.dim()}")

def _norm01_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (B,H,W)
    return: (B,H,W) scaled to [0,1] per sample
    """
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def threshold_mask(
    x: torch.Tensor,
    threshold: float,
    mode: str = "soft",          # "soft" | "hard"
    tau: float = 0.05,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    x: (B,H,W) or (B,H,W,B)
    threshold: scalar (float or 0-d/1-d tensor)
    return: mask (B,H,W) in [0,1] (soft) or {0,1} (hard)
    """
    x = _extract_self_map(x)  # -> (B,H,W)

    if normalize:
        x = _norm01_per_sample(x, eps=eps)

    if mode == "hard":
        return (x >= threshold).to(x.dtype)
    elif mode == "soft":
        # differentiable thresholding
        return torch.sigmoid((x - threshold) / tau)
    else:
        raise ValueError("mode must be 'soft' or 'hard'")

def fused_filter_mask(
    similarity_maps: torch.Tensor,
    attn_maps: torch.Tensor,
    threshold: float,
    attn_threshold: float,
    fuse: str = "product",        # "product" | "linear" | "cascade"
    alpha: float = 0.5,           # only for linear
    mode: str = "soft",           # "soft" | "hard"
    tau: float = 0.05,
    normalize: bool = True,
    eps: float = 1e-6,
      # only for cascade
) -> torch.Tensor:
    """
    similarity_maps: (B,H,W) or (B,H,W,B)
    attn_maps:       (B,H,W) or (B,H,W,B)  (if batch-aware, diagonal will be used)
    return: mask (B,H,W)
    """
    sim = _extract_self_map(similarity_maps)  # (B,H,W)
    att = _extract_self_map(attn_maps)        # (B,H,W)

    if normalize:
        sim = _norm01_per_sample(sim, eps=eps)
        att = _norm01_per_sample(att, eps=eps)

    if fuse == "product":
        fused = sim * att

        # 对 fused 再做一次归一化通常更稳（可选，但我建议开着）
        if normalize:
            fused = _norm01_per_sample(fused, eps=eps)

        return threshold_mask(fused, threshold, mode=mode, tau=tau, normalize=False, eps=eps)

    elif fuse == "linear":
        fused = alpha * sim + (1.0 - alpha) * att
        if normalize:
            fused = _norm01_per_sample(fused, eps=eps)
        return threshold_mask(fused, threshold, mode=mode, tau=tau, normalize=False, eps=eps)

    elif fuse == "cascade":
        # 先用 attention 粗筛，再用 similarity 精筛（更不容易全灭）
        if attn_threshold is None:
            attn_threshold = threshold

        att_mask = threshold_mask(att, attn_threshold, mode="hard", tau=tau, normalize=False, eps=eps)
        fused = sim * att_mask
        if normalize:
            fused = _norm01_per_sample(fused, eps=eps)
        return threshold_mask(fused, threshold, mode=mode, tau=tau, normalize=False, eps=eps)

    else:
        raise ValueError("fuse must be 'product', 'linear', or 'cascade'")

def generate_caption(classes, template="A scene sketch containing {}."):
    if len(classes) == 0:
        return "A scene sketch."
    if len(classes) == 1:
        class_str = classes[0]
    elif len(classes) == 2:
        class_str = f"{classes[0]} and {classes[1]}"
    else:
        class_str = ", ".join(classes[:-1]) + f", and {classes[-1]}"

    return template.format(class_str)


def sketch_text_pairs_inkscene(sketches, classes_list, max_classes=3):
    all_captions = []
    all_classes = []
    all_sketches = []
    for i in range(len(classes_list)):
        classes = list(dict.fromkeys(classes_list[i]))
        sketch = sketches[i]
        if len(classes) > max_classes:
            classes = classes[:max_classes]
        caption = generate_caption(classes)
        caption = [caption] * len(classes)
        sketch = sketch.repeat(len(classes), 1, 1, 1)
        all_sketches.append(sketch)
        all_classes.append(classes)
        all_captions.append(caption)

    return torch.cat(all_sketches), flatten(all_classes), flatten(all_captions)
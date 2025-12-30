"""
实现一些损失函数和辅助函数
"""
import csv
import torch
import numpy as np
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC
# from vpt.src.configs.config import get_cfg
import os
from time import sleep
from random import randint
# from vpt.src.utils.file_io import PathManager
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

warnings.filterwarnings("ignore")

"""
def setup(args):

    # Create configs and perform basic setups.
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR
    lr = cfg.SOLVER.BASE_LR
    wd = cfg.SOLVER.WEIGHT_DECAY
    output_folder = os.path.join(
        cfg.DATA.NAME, cfg.DATA.FEATURE, f"lr{lr}_wd{wd}")

    # train cfg.RUN_N_TIMES times
    count = 1
    while count <= cfg.RUN_N_TIMES:
        output_path = os.path.join(output_dir, output_folder, f"run{count}")
        # pause for a random time, so concurrent process with same setting won't interfere with each other. # noqa
        sleep(randint(3, 30))
        if not PathManager.exists(output_path):
            PathManager.mkdirs(output_path)
            cfg.OUTPUT_DIR = output_path
            break
        else:
            count += 1

    cfg.freeze()
    return cfg
"""


def _get_nested(cfg_dict, key):
    """支持 'a.b.c' key 读取嵌套字段"""
    parts = key.split('.')
    cur = cfg_dict
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def save_results_to_csv(
        cfg,
        results: dict,
        csv_path="experiment_results.csv",
        include: list = None,
        exclude: list = None,
        flatten: bool = False,
):
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
        **cfg_record,  # 指定的 config 字段
        **results,  # 实验指标
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

    # # Calculate centroids and render class names
    # for i, class_name in enumerate(classes):
    #     mask = class_indices == i
    #     if np.any(mask):
    #         y, x = np.nonzero(mask)
    #         centroid_x, centroid_y = np.mean(x), np.mean(y)
    #         plt.text(centroid_x, centroid_y, class_name, color=classes_colors[i], ha='center', va='center',fontsize=14,   # color=classes_colors[i]
    #         bbox=dict(facecolor='lightgrey', edgecolor='none', boxstyle='round,pad=0.2', alpha=0.8))

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
        classes = list(set(classes))
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


def sketch_text_pairs_addnone(sketch_batch, captions, max_classes=3):
    """
    根据 caption 提取名词短语并与 sketch 形成配对，
    固定每个样本输出 max_classes 个类别，保证 batch 维度一致。

    逻辑：
    - 若提取的 classes > max_classes：优先选取较短或句首的名词短语（语义更核心）
    - 若 classes < max_classes：用 "none" 填充，代表背景类（可起正则化作用）
    - 若 classes == 0：整句 caption 当作唯一类别
    """

    all_sketches = []
    all_classes = []
    all_captions = []

    for sketch, caption in zip(sketch_batch, captions):
        # --- 文本预处理 ---
        caption = caption.replace('\n', ' ')
        translator = str.maketrans('', '', string.punctuation)
        caption = caption.translate(translator).lower()
        words = nltk.word_tokenize(caption)

        # --- 提取名词短语 ---
        classes = get_noun_phrase(words)
        classes = list(dict.fromkeys(classes))  # 去重但保持顺序（句首优先）

        # --- 根据长度情况处理 ---
        if len(classes) == 0:
            # 如果完全没有名词短语，用 caption 自身
            classes = [caption] + ["none"] * (max_classes - 1)
        elif len(classes) < max_classes:
            # 类别不足，补背景类
            classes += ["none"] * (max_classes - len(classes))
        elif len(classes) > max_classes:
            # 优先选取较短的名词短语（往往语义更明确）
            classes = sorted(classes, key=len)[:max_classes]

        # --- 为每个 class 重复 sketch ---
        sketch = sketch.repeat(max_classes, 1, 1, 1)
        caption_list = [caption] * max_classes

        # --- 收集 ---
        all_sketches.append(sketch)
        all_classes.append(classes)
        all_captions.append(caption_list)

    # --- 拼接输出 ---
    return torch.cat(all_sketches), flatten(all_classes), flatten(all_captions)


def sketch_text_image_pairs(sketch_batch, captions, image_batch, max_classes=3):
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
        classes = list(set(classes))
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


def tensor_to_binary_img(tensor, device):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(device)
    images_unnormalized = tensor * std + mean
    images_gray = images_unnormalized.mean(dim=1)
    threshold = 0.5
    binary_images = (images_gray > threshold).float()
    binary_images = binary_images.unsqueeze(1)
    binary_images = binary_images.repeat(1, 3, 1, 1)
    return binary_images


def zero_clapping(similarity_maps, threshold):
    batch_size = similarity_maps.shape[0]
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


def get_train_classes(dataset, max_classes=3):
    train_classes = []
    # for i, (_,caption,_) in enumerate(dataset):
    for i, (_, caption) in enumerate(dataset):
        caption = caption.replace('\n', ' ')
        translator = str.maketrans('', '', string.punctuation)
        caption = caption.translate(translator).lower()
        words = nltk.word_tokenize(caption)
        classes = get_noun_phrase(words)
        # remove synonyms in classes
        classes = list(set(classes))
        if len(classes) > max_classes:
            classes = classes[:max_classes]
        if len(classes) == 0:
            classes = caption
        train_classes.append(classes)
    train_classes = np.unique(flatten(train_classes))
    return train_classes


def get_train_classes_with_image(dataset, max_classes=3):
    train_classes = []
    for i, (_, caption, _) in enumerate(tqdm(dataset, desc="Processing captions", ncols=100)):
        # for i, (_, caption) in enumerate(dataset):
        caption = caption.replace('\n', ' ')
        translator = str.maketrans('', '', string.punctuation)
        caption = caption.translate(translator).lower()
        words = nltk.word_tokenize(caption)
        classes = get_noun_phrase(words)
        # remove synonyms in classes
        classes = list(set(classes))
        if len(classes) > max_classes:
            classes = classes[:max_classes]
        if len(classes) == 0:
            classes = caption
        train_classes.append(classes)
    train_classes = np.unique(flatten(train_classes))
    return train_classes


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
    negative_dists[diag_mask | class_mask] = float(
        'inf')  # Set diagonal elements and same class elements to a large value
    min_negative_dists, _ = torch.min(negative_dists, dim=1)
    # Compute triplet loss
    loss = torch.clamp(margin + positive_dists - min_negative_dists, min=0.0)

    return loss.mean()


def visualize_attention_maps_with_tokens(pixel_similarity, tokens):
    # Convert the tensor to a numpy array and transpose it to match the dimensions required by imshow
    attention_maps = pixel_similarity.numpy().transpose(2, 0, 1)

    # Create a subplot for each attention map
    num_attention_maps = attention_maps.shape[0]
    fig, axes = plt.subplots(1, num_attention_maps, figsize=(15, 5))

    # Plot each attention map with corresponding text token
    for i in range(num_attention_maps):
        ax = axes[i]
        ax.imshow(attention_maps[i], cmap='gray', vmin=0, vmax=1)
        # ax.set_title(f'Attention Map {i+1}')
        ax.axis('off')

        # Add the corresponding text token as annotation below the attention map
        ax.annotate(tokens[i], xy=(0.5, -0.1), xycoords='axes fraction', ha='center')

    plt.tight_layout()
    # plt.savefig('attention_maps.png')
    plt.show()


## Sketch Preprocessing tools

def pen_state_to_strokes(sketches):
    strokes = []
    i_prev = 0
    for i in range(len(sketches)):
        if sketches[i, 2] == 1:
            strokes.append(sketches[i_prev:i + 1])
            i_prev = i + 1
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


def prerender_stroke(stroke_list, side=512):  # fig, xlim=[0,255], ylim=[0,255]):
    R = []
    for stroke in stroke_list:
        stroke = np.array([stroke, ])
        R.append(torch.tensor(rasterize_Sketch(stroke[0], side)).unsqueeze(0))
    return torch.stack(R, 0)


def pixel_level_segmentation(strokes_seg, labels, all_classes, size):
    num_classes = len(all_classes)
    sketch_seg = np.zeros((num_classes, size, size))  # Initialize segmentation array

    blank_pix_index = all_classes.index('blank_pixel')  # index of 'blank_pix'
    # Initially assign all pixels to 'blank_pix'
    sketch_seg[blank_pix_index] = np.ones((size, size))

    for stroke, label in zip(strokes_seg, labels):
        # Get the class index for the given label
        class_index = all_classes.index(label)

        # For each pixel in the stroke, assign its class in the sketch segmentation
        # Before assigning, remove the pixel from the 'blank_pix' class
        sketch_seg[class_index] += stroke  # This works because stroke values are either 0 or 1
        sketch_seg[blank_pix_index] -= stroke
        # sketch_seg[unlabeled_pix_index] -= stroke

    # Now each pixel in sketch_seg has a one-hot encoding across all classes.
    # We convert this to a single channel with the class labels as integers.
    pixel_level_seg = np.argmax(sketch_seg, axis=0)
    return pixel_level_seg


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


def visualize_feature_map(feature_map, save_path=None, title="Feature Map"):
    """可视化单通道特征图（取前8个通道示例）"""
    # feature_map shape: [B, H, W, C] 或 [B, C, H, W]
    if feature_map.dim() == 4:
        feature_map = feature_map[0]  # 取第一个样本
    if feature_map.shape[0] > 3:  # 若通道数过多，取前8个
        feature_map = feature_map[:8]

    num_channels = feature_map.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(num_channels):
        ax = axes[i]
        ax.imshow(feature_map[i].cpu().detach().numpy(), cmap="viridis")
        ax.set_title(f"Channel {i + 1}")
        ax.axis("off")

    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


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
        print(f"[Warning] Patch token 数 {n} 非完美平方，裁剪为 {grid * grid}")
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


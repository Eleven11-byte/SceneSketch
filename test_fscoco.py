
import torch
from torchvision.transforms import InterpolationMode
from configs.config import Config
BICUBIC = InterpolationMode.BICUBIC
from utils.utils import get_similarity_map, pen_state_to_strokes, prerender_stroke, \
    pixel_level_segmentation, compute_accuracy, compute_miou, save_results_to_csv
from utils.utils_compute import process_image
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
# from models import clip
import csv
from datetime import datetime
import pandas as pd
# from models.modified_clip.model import CLIPer
from models.modified_model import ModifiedCLIP
from utils.utils_visualization import visualize_segmentation_overlay

PALETTE = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]])


def main():
    preprocess = Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
                          Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    segmented_sketches = fscoco_test(root="/home/xiaoyi/project/OpenSketchSeg/DATA/fscoco-seg/test")

    # Read the classes from json file
    with open('/home/xiaoyi/project/OpenSketchSeg/DATA/fscoco-seg/test/all_classes.json', 'r') as f:
        all_classes = json.load(f)

    average_P_metric = []
    average_C_metric = []

    total_miou = 0
    # for idx, (pen_state, labels, caption, img_path) in enumerate(segmented_sketches):
    for idx, (pen_state, labels, caption, img_path) in enumerate(tqdm(segmented_sketches, desc="Processing sketches")):
        # Preprocess the vector sketch
        strokes = pen_state_to_strokes(pen_state)
        strokes = prerender_stroke(strokes).squeeze(1).float()
        strokes_seg = np.array(strokes)
        o_sketch = np.array(strokes).sum(0) * 255
        o_sketch = np.repeat(o_sketch[np.newaxis, :, :], 3, axis=0)
        o_sketch = np.transpose(o_sketch, (1, 2, 0)).astype('uint8')
        o_sketch = np.where(o_sketch > 0, 255, o_sketch)
        o_sketch = 255 - o_sketch
        pil_img = Image.fromarray(o_sketch)
        sketch_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        classes_per_sketch = np.unique(labels)
        fg_classes = classes_per_sketch.tolist()


        classes = ["blank_pixel"] + fg_classes

        gt_sketch_seg = pixel_level_segmentation(strokes_seg, labels, classes, size=strokes_seg.shape[-1]) #获取到分割结果的groundtruth

        bg_classes = ["blank_pixel"]
        ori_img = torch.tensor(o_sketch.astype(float)).permute(2, 0, 1) / 255.
        pred_mask = process_image(model, device, o_sketch, ori_img, fg_classes, bg_classes, segmentation_threshold, class_threshold, idx)
        pred_sketch_seg = pred_mask.cpu().numpy()
        img_name = img_path.split('/')[-1].split('.')[0]
        result_file = os.path.join(save_dir, f"{img_name}_1.jpg")
        # visualize_segmentation_overlay(ori_img, pred_sketch_seg, PALETTE, result_file, class_names=classes, alpha=0.5, title="Segmentation Overlay")

        if visualized:
            mask = pred_sketch_seg.copy()
            mask[gt_sketch_seg == 0] = 0
            visualize_segmentation_overlay(ori_img, mask, PALETTE, mask_no=pred_sketch_seg, groundtruth=gt_sketch_seg, show=False, result_file=result_file, class_names=classes, alpha=0.5,title="Segmentation Overlay")


        mapping_indices = {i: all_classes.index(j) for i, j in enumerate(classes)}
        pred_sketch_seg = np.vectorize(mapping_indices.get)(pred_sketch_seg)

        gt_sketch_seg = np.vectorize(mapping_indices.get)(gt_sketch_seg)
        pred_sketch_seg[gt_sketch_seg == 0] = 0



        P_metric, C_metric = compute_accuracy(strokes_seg, labels, all_classes, gt_sketch_seg, pred_sketch_seg)
        iou = compute_miou(gt_sketch_seg, pred_sketch_seg, all_classes)

        average_P_metric.append(P_metric)
        average_C_metric.append(C_metric)
        total_miou += iou

        # print(f"Processing {idx + 1}/{len(segmented_sketches)} sketches", end="\r")

    P_value = sum(average_P_metric) / len(average_P_metric)
    C_value = sum(average_C_metric) / len(average_C_metric)
    miou_value = total_miou / len(segmented_sketches)

    print("\n")
    print(f"Pixel Accuracy: {round(P_value, 5)}")
    print(f"Stroke Accuracy: {round(C_value, 5)}")
    print(f"mIoU: {round(miou_value, 5)}")

    results = {
        "PixelAccuracy": round(P_value, 5),
        "StrokeAccuracy": round(C_value, 5),
        "mIoU": round(miou_value, 5),
    }


    save_results_to_csv(
        cfg,
        results,
        exclude=["MODEL", "loss", "wandb"]
    )


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    # config = Config("./configs/test_config.yaml")
    # 从命令行更新配置
    # config.update_from_cli()
    save_dir = "fscoco_test_result_1"
    # cfg = load_yaml("configs/base_test.yaml")
    cfg = Config("configs/test.yaml")
    cfg.update_from_cli()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    cfg.semantic_templates = [line.strip() for line in list(open(cfg.semantic_templates))]
    model = ModifiedCLIP(cfg, device=device)
    model = model.float()

    # NOTE: 加载训练后的参数
    state_dict = torch.load(cfg.checkpoint_path)
    load_info = model.load_state_dict(state_dict, strict=False)
    print(load_info)

    model.to(device)

    # embedding = torch.load("embeddings_large14.pth", map_location=device)
    segmentation_threshold = cfg.segmentation_threshold

    class_threshold = cfg.class_threshold
    # print(cfg, CLIPer.attn_refine)
    visualized = False

    print(cfg)

    main()
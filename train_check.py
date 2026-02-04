import numpy as np
import torch
import json
torch.autograd.set_detect_anomaly(True)
from configs.config import Config
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC
from torch import nn
from torch.utils.data import DataLoader
from dataset import fscoco_train, SketchImageDataset, fscoco_train_modified, collate_fscoco_A
# from vpt.launch import default_argument_parser
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import random
"""
from utils.utils import get_similarity_map, zero_clapping, get_train_classes, \
    tensor_to_binary_img, sketch_text_image_pairs, get_threshold, triplet_loss_func_L1, get_train_classes_with_image, \
    img_sketch_align_loss, cross_modal_distill_loss, patch_distribution_distill_loss
"""
import wandb
import os
import torch.nn.functional as F
from tqdm import tqdm
from models.modified_model import ModifiedCLIP
from utils.utils_loss import img_sketch_align_loss, patch_distribution_distill_loss,triplet_loss_func_L1
from utils.utils_train import print_trainable_params, get_similarity_map, zero_clapping, tensor_to_binary_img, sketch_text_pairs, get_threshold,\
    get_train_classes_with_image, sketch_text_image_pairs, save_checkpoint, load_checkpoint, get_train_classes

# torch.autograd.set_detect_anomaly(True)  # NOTE：启用异常检测
from utils.utils_train import expand_pairs

def main(configs):
    cfg = configs
    preprocess_no_T = Compose([Resize((224, 224), interpolation=BICUBIC),  # ToTensor(),
                               Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    preprocess = Compose([Resize((224, 224), interpolation=BICUBIC),ToTensor(),
                          Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = fscoco_train_modified(root=cfg.dataset.root, transform=preprocess, augment=False)  # Load the training dataset #NOTE: 未进行数据增强
    # train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fscoco_A,
        pin_memory=True,
        drop_last=True
    )

    if os.path.exists("./train_classes.json"):
        with open("./train_classes.json", 'r', encoding='utf-8') as f:
            train_classes = np.array(json.load(f))
        print(f"读取类别文件成功, 共{train_classes.shape[0]}类")
    else:
        print("Extracting classes from training dataset.. This might take a minute.")
        train_classes = get_train_classes(train_dataset, max_classes=cfg.train.max_classes)
        # NOTE: 保存train_classes
        json_save_path = "./train_classes.json"  # json保存路径
        with open(json_save_path, 'w', encoding='utf-8') as f:
            json.dump(train_classes.tolist(), f, ensure_ascii=False, indent=4)
        print(f"类别信息已保存至: {json_save_path}")

    learnable_threshold = nn.Parameter(torch.tensor(cfg.train.threshold))  # Initialize with default threshold
    threshold_optimizer = torch.optim.AdamW([learnable_threshold], lr=1e-4)

    log_dir = cfg.wandb.name
    print(f"Checkpoints will be saved to: checkpoint/{log_dir}")
    os.makedirs(f"checkpoint/{log_dir}", exist_ok=True)

    # ===== Resume logic =====
    start_epoch = 0
    global_step = 0

    for epoch in range(1):
        total_loss = 0.0
        batch_count = 0

        pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{1}")

        for batch_idx, (sketches, captions, classes_lists, image_ids) in enumerate(pbar):
            # 数据预处理
            sketches = sketches.view(-1, 3, 224, 224).to(device)

            sketches_w, classes, captions_pair = expand_pairs(sketches, captions, classes_lists, max_classes=cfg.train.max_classes)

            sketches_w_binary = tensor_to_binary_img(sketches_w, device)
            sketches_b = 1 - sketches_w_binary

            class_to_idx = {name: i for i, name in enumerate(train_classes)}
            labels = torch.tensor([class_to_idx[name] for name in classes]).to(device)




            batch_count += 1


if __name__ == "__main__":
    config = Config("./configs/base_train_template.yaml")
    # 从命令行更新配置
    config.update_from_cli()

    # 打印配置
    print("最终配置:")
    print(config)

    config.semantic_templates = [line.strip() for line in list(open(config.semantic_templates))]

    # 保存最终配置（用于实验复现）
    experiment_name = config.wandb.name
    os.makedirs("./configs/experiments", exist_ok=True)
    config.save(f"./configs/experiments/{experiment_name}.yaml")

    main(config)
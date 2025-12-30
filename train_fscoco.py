import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)
from configs.config import Config
from torchvision.transforms import InterpolationMode

BICUBIC = InterpolationMode.BICUBIC
from torch import nn
from torch.utils.data import DataLoader
from dataset import fscoco_train, SketchImageDataset
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
    get_train_classes_with_image, sketch_text_image_pairs, save_checkpoint, load_checkpoint

# torch.autograd.set_detect_anomaly(True)  # NOTE：启用异常检测

def main(configs):
    cfg = configs
    preprocess_no_T = Compose([Resize((224, 224), interpolation=BICUBIC),  # ToTensor(),
                               Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

    preprocess = Compose([Resize((224, 224), interpolation=BICUBIC),ToTensor(),
                          Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711))])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModifiedCLIP(cfg=cfg, device=device)
    model = model.float()

    # print_trainable_params(model) #NOTE:用于训练过程中检查参数是否更新
    init_params = {name: param.clone().detach().to(device) for name, param in model.named_parameters() if
                   param.requires_grad}
    # print(init_params)

    model.to(device)
    print("Model loaded successfully")

    # NOTE：初始化add_weight参数
    if cfg.train.use_distill:
        print("Using distill")
        a1 = cfg.loss.add_weight

    train_dataset = SketchImageDataset(root=cfg.dataset.root, transform=preprocess, augment=False)  # Load the training dataset #NOTE: 未进行数据增强
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=8)

    print("Extracting classes from training dataset.. This might take a minute.")
    # FIXME：排序后再加载train_classes，保证获取到一样的class列表
    # train_classes = get_train_classes_with_image(train_dataset, max_classes=cfg.train.max_classes)

    learnable_threshold = nn.Parameter(torch.tensor(cfg.train.threshold))  # Initialize with default threshold
    threshold_optimizer = torch.optim.AdamW([learnable_threshold], lr=1e-4)


    vit_optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr))

    if cfg.wandb.enable:
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name, resume="allow")

    log_dir = cfg.wandb.name
    print(f"Checkpoints will be saved to: checkpoint/{log_dir}")
    os.makedirs(f"checkpoint/{log_dir}", exist_ok=True)

    # ===== Resume logic =====
    start_epoch = 0
    global_step = 0

    ckpt_dir = f"checkpoint/{log_dir}"
    last_ckpt_path = os.path.join(ckpt_dir, "last.pth")

    if os.path.exists(last_ckpt_path):
        start_epoch, global_step, train_classes = load_checkpoint(
            last_ckpt_path,
            model,
            vit_optimizer,
            threshold_optimizer,
            device
        )

    else:
        print("No checkpoint found, start training from scratch")
        print("Extracting classes from training dataset.. This might take a minute.")
        train_classes = get_train_classes_with_image(train_dataset, max_classes=cfg.train.max_classes)

    # 可学习令牌数量
    num_of_tokens = cfg.MODEL.PROMPT.NUM_TOKENS
    num_epochs = cfg.train.epochs

    print("Starting training")

    epoch_losses = []

    model.train()

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        batch_count = 0

        pbar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (sketches, captions, images) in enumerate(pbar):
            # for batch_idx, (sketches,captions,images) in enumerate(train_dataloader):

            # 数据预处理
            sketches = sketches.view(-1, 3, 224, 224).to(device)
            images = images.view(-1, 3, 224, 224).to(device)
            sketches_w, classes, captions_pair, images_pair = sketch_text_image_pairs(sketches, captions, images, max_classes=cfg.train.max_classes)

            sketches_w_binary = tensor_to_binary_img(sketches_w, device)
            sketches_b = 1 - sketches_w_binary

            caption_features = model.encode_text(captions_pair)
            class_features = model.encode_text(classes, use_template_embedding=True)

            scene_features_layers, attn, sketch_mid_feats_layers = model.encode_image(sketches_w, type="sketch")

            scene_features = scene_features_layers[-1].permute(1, 0, 2)  # FIXME: 改具体的层数，目前取最后一层对齐

            similarity = scene_features @ class_features.T  # 计算相似度矩阵
            patches_similarity = similarity[:, num_of_tokens + 1:, :]  # 去掉class token的相似度
            # patches_similarity = similarity[:, num_of_tokens + 1:, :] # TODO：有visual prompt时去掉visual_prompt，在无prompt时把num_tokens设置为0

            similarity_maps = get_similarity_map(patches_similarity, sketches_w.shape[2:])  # patch相似度映射到像素级

            # 阈值过滤与特征重提取
            threshold_value = get_threshold(learnable_threshold)  # 获取当前的可学习阈值
            # threshold_value = learnable_threshold
            weights_hard_sm = zero_clapping(similarity_maps, threshold_value)  # 低于阈值的像素置0

            weights_hard_sm = weights_hard_sm.unsqueeze(1).repeat(1, 3, 1, 1)  # 扩展权重的维度以匹配草图（重复3次以适配RGB通道）
            w_sketches = sketches_b * weights_hard_sm  # 应用权重到背景掩码，保留高置信度的背景区域（用于后续特征提取）
            w_sketches_white = w_sketches.max() - w_sketches  # 反转草图颜色
            w_sketches_white = preprocess_no_T(w_sketches_white)  # 对处理后的草图应用同样的预处理

            # w_sketch_features, caption_features, _ = model(w_sketches_white, tokenized_text=tokenized_captions)

            w_sketch_features, attn, _ = model.encode_image(w_sketches_white, type="sketch")

            w_sketch_features_l7 = w_sketch_features[7 - 1].permute(1, 0, 2)
            w_sketch_features_l10 = w_sketch_features[10 - 1].permute(1, 0, 2)
            w_sketch_features_l12 = w_sketch_features[12 - 1].permute(1, 0, 2)

            class_to_idx = {name: i for i, name in enumerate(train_classes)}
            labels = torch.tensor([class_to_idx[name] for name in classes]).to(device)

            triplet_loss_scene = triplet_loss_func_L1(scene_features[:, 0, :], caption_features, labels, margin=cfg.train.margin)
            triplet_loss_final_layer = triplet_loss_func_L1(w_sketch_features_l12[:, 0, :], class_features, labels, margin=cfg.train.margin)
            triplet_loss_l7 = triplet_loss_func_L1(w_sketch_features_l7[:, 0, :], class_features, labels, margin=cfg.train.margin)
            triplet_loss_l10 = triplet_loss_func_L1(w_sketch_features_l10[:, 0, :], class_features, labels, margin=cfg.train.margin)

            # loss = (1 -a1)*(triplet_loss_scene + triplet_loss_final_layer + triplet_loss_l7 + triplet_loss_l10) + a1 * sk_im_align_loss
            if cfg.train.use_distill:
                # NOTE:蒸馏损失计算部分, patch_distribution
                img_mid_feats = model.encode_image(images_pair, type="image", select_layers=cfg.loss.distill_layers)
                distill_layers = cfg.loss.distill_layers
                sketch_mid_feats = torch.stack([sketch_mid_feats_layers[l].permute(1, 0, 2) for l in distill_layers],
                                               dim=0).to(device)

                im = img_mid_feats[:, :, 1:, :]
                sk = sketch_mid_feats[:, :, 1:, :]  # [L, B, P, 768]
                sk_im_align_loss = 0

                for i in range(len(distill_layers)):
                    sk_im_align_loss_layer = patch_distribution_distill_loss(sk[i], im[i])
                    sk_im_align_loss = sk_im_align_loss + sk_im_align_loss_layer

                loss = (1 - a1) * (triplet_loss_scene + triplet_loss_final_layer + triplet_loss_l7 + triplet_loss_l10) + a1 * sk_im_align_loss
            else:
                loss = triplet_loss_scene + triplet_loss_final_layer + triplet_loss_l7 + triplet_loss_l10

            total_loss += loss.item()
            batch_count += 1

            vit_optimizer.zero_grad()
            threshold_optimizer.zero_grad()
            loss.backward()  # 反向传播

            # NOTE: 检查模型参数梯度
            has_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.norm(param.grad) > 1e-6:
                    # print(f"参数 {name} 有梯度，梯度范数：{torch.norm(param.grad):.6f}")
                    has_grad = True
            if not has_grad:
                print("⚠️ 所有模型参数梯度为None或接近0！")

            # 检查阈值参数梯度，已移除，后期再加

            vit_optimizer.step()
            threshold_optimizer.step()

            global_step += 1

            if cfg.wandb.enable:
                log_dict = {
                    "train/loss": loss.item(),
                    "train/threshold": threshold_value.item(),
                }

                if cfg.train.use_distill:
                    log_dict["train/distill_loss"] = sk_im_align_loss.item()
                wandb.log(log_dict, step=global_step)

            epoch_avg_loss = total_loss / batch_count
            epoch_losses.append(epoch_avg_loss)

        # 训练1个epoch后
        if cfg.wandb.enable:
            wandb.log({"epoch": epoch,
                       "Epoch Average Train Loss": epoch_avg_loss})

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "vit_optimizer": vit_optimizer.state_dict(),
            "threshold_optimizer": threshold_optimizer.state_dict(),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
            "train_classes": train_classes,
        }

        save_checkpoint(checkpoint, f"{ckpt_dir}/last.pth")

        if epoch == 0 or (epoch + 1) % int(cfg.train.save_every) == 0:
            if epoch==0 or epoch == num_epochs - 1:
                for name, param in model.named_parameters():
                    if param.requires_grad and name in init_params:
                        diff = torch.norm(param - init_params[name])
                        print(f"参数 {name} 训练后变化：{diff:.6f}")
                        if diff > 1e-6:
                            print(f"参数 {name} 已更新！")
                        else:
                            print(f"参数 {name} 无变化！")  # 仅验证第一个epoch
            torch.save(model.state_dict(), f"checkpoint/{log_dir}/model_{epoch + 1}.pth")
            print("saved", f" checkpoint/{log_dir}/model_{epoch + 1}.pth")
        # return scene_features, class_features

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
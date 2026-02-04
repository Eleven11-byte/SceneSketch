import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Resize, Normalize, InterpolationMode
from PIL import Image, ImageOps
import torchvision.transforms as transforms
# import nltk
import string
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# from vpt.launch import default_argument_parser
import models
from models import clip
from utils import ( #setup,
    get_similarity_map, zero_clapping, get_train_classes,
    tensor_to_binary_img, get_threshold, triplet_loss_func_L1, get_noun_phrase
)

os.environ["WANDB_DISABLE_SSL"] = "true"
os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
os.environ["WANDB_MODE"] = "offline"

# ==============================================================
# 🔥 支持 VSCode Debug 的关键参数（仅单机调试时启用）
# USE:
#   DEBUG_SINGLE_GPU=1 python train_ddp.py
# ==============================================================
DEBUG_MODE = os.environ.get("DEBUG_SINGLE_GPU", "0") == "1"


# ===============================================================
# Tensor 打印显示维度（debug）
# ===============================================================
def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


# ===============================================================
# 数据集
# ===============================================================
class fscoco_train(torch.utils.data.Dataset):
    def __init__(self, root="DATA/train", transform=None, augment=False, SKETCH_SIZE=512, max_classes=3):
        self.root = root
        self.transform = transform
        self.augment = augment
        self.max_classes = max_classes

        self.sketch_dir = os.path.join(root, "sketches")
        self.text_dir = os.path.join(root, "text")

        self.augmentation = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomCrop(450),
            transforms.Resize((SKETCH_SIZE, SKETCH_SIZE))
        ])

        self.sketch_subdirs = sorted(os.listdir(self.sketch_dir))
        self.txt_subdirs = sorted(os.listdir(self.text_dir))

        sketch_files, txt_files = [], []
        for sketch_subdir, txt_subdir in zip(self.sketch_subdirs, self.txt_subdirs):
            sketch_subdir_path = os.path.join(self.sketch_dir, sketch_subdir)
            txt_subdir_path = os.path.join(self.text_dir, txt_subdir)

            sketch_files_subdir = sorted(os.listdir(sketch_subdir_path))
            txt_files_subdir = sorted(os.listdir(txt_subdir_path))

            sketch_files += [os.path.join(sketch_subdir, f) for f in sketch_files_subdir]
            txt_files += [os.path.join(txt_subdir, f) for f in txt_files_subdir]

        self.samples = []
        for sketch_rel, txt_rel in zip(sketch_files, txt_files):
            text_path = os.path.join(self.text_dir, txt_rel)
            caption = open(text_path).read()

            caption_clean = caption.replace('\n', ' ')
            caption_clean = caption_clean.translate(str.maketrans('', '', string.punctuation)).lower()
            words = nltk.word_tokenize(caption_clean)

            classes = get_noun_phrase(words)
            classes = list(set(classes))
            if len(classes) > max_classes:
                classes = classes[:max_classes]
            if len(classes) == 0:
                classes = [caption_clean]

            for c in classes:
                self.samples.append((sketch_rel, c, caption_clean))

        print(f"📊 Expanded dataset: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sketch_rel, cls, caption = self.samples[index]

        sketch_path = os.path.join(self.sketch_dir, sketch_rel)
        sketch = Image.open(sketch_path).convert("RGB")

        sketch_aug = ImageOps.invert(sketch)
        aug_sketch = self.augmentation(sketch_aug)
        aug_sketch = ImageOps.invert(aug_sketch)

        if self.transform:
            sketch = self.transform(sketch)
            aug_sketch = self.transform(aug_sketch)

        if self.augment:
            sketch = torch.stack([sketch, aug_sketch])

        return sketch, cls, caption


# ===============================================================
# 训练入口
# ===============================================================
def main(args):
    # -------------------------------------
    # 🔥 VSCode DEBUG：单卡模式，不初始化 DDP
    # -------------------------------------
    if DEBUG_MODE:
        print("🔧 DEBUG MODE ENABLED (Single GPU, No DDP)")
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        # -----------------------------
        # 正常 DDP 初始化
        # -----------------------------
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

    is_main = (rank == 0)

    cfg = setup(args)

    if cfg.WANDB:
        wandb.init(project="sketch_segmentation_ddp", name=cfg.MODEL.PROMPT.LOG)

    # ===========================================================
    # 模型
    # ===========================================================
    BICUBIC = InterpolationMode.BICUBIC
    preprocess_no_T = Compose([
        Resize((224, 224), interpolation=BICUBIC),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711))
    ])

    model, preprocess = models.load("CS-ViT-B/16", device=device, cfg=cfg, train_bool=True)
    model = model.to(device)

    # ============================
    # 仅 DDP 模式才包装 DDP
    # ============================
    if not DEBUG_MODE:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # ===========================================================
    # 数据集
    # ===========================================================
    train_dataset = fscoco_train(transform=preprocess, augment=False, max_classes=cfg.max_classes)

    if DEBUG_MODE:
        # 单卡直接用普通 DataLoader
        train_sampler = None
    else:
        train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.bz,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4
    )

    train_classes = get_train_classes(train_dataset, max_classes=cfg.max_classes)
    class_to_idx = {name: i for i, name in enumerate(train_classes)}

    # ===========================================================
    # 优化器
    # ===========================================================
    learnable_threshold = nn.Parameter(torch.tensor(cfg.threshold, device=device))
    threshold_optimizer = torch.optim.AdamW([learnable_threshold], lr=1e-4)
    vit_optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    num_epochs = 20

    if is_main:
        print("🚀 Training started")

    # -----------------------------------------------------------
    # 训练循环
    # -----------------------------------------------------------
    for epoch in range(num_epochs):

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (sketches, classes, captions) in enumerate(train_dataloader):
            sketches = sketches.to(device)

            tokenized_classes = clip.tokenize(classes).to(device)
            tokenized_captions = clip.tokenize(captions).to(device)

            scene_features, class_features = model(sketches, tokenized_classes,
                                                   layer_num=[12], return_logits=False, mode="train")

            similarity = scene_features @ class_features.T
            patches_similarity = similarity[:, cfg.MODEL.PROMPT.NUM_TOKENS + 1:, :]
            similarity_maps = get_similarity_map(patches_similarity, sketches.shape[2:])

            threshold_value = get_threshold(learnable_threshold)
            weights_hard_sm = zero_clapping(similarity_maps, threshold_value)
            weights_hard_sm = weights_hard_sm.unsqueeze(1).repeat(1, 3, 1, 1)

            sketches_b = 1 - tensor_to_binary_img(sketches, device)
            w_sketches = sketches_b * weights_hard_sm
            w_sketches_white = w_sketches.max() - w_sketches
            w_sketches_white = preprocess_no_T(w_sketches_white)

            w_sketch_features, caption_features = model(
                w_sketches_white, tokenized_captions,
                layer_num=[7, 10, 12], return_logits=False, mode="train"
            )

            w_sketch_features_l7, w_sketch_features_l10, w_sketch_features_l12 = w_sketch_features

            labels = torch.tensor([class_to_idx[name] for name in classes]).to(device)

            triplet_loss_scene = triplet_loss_func_L1(scene_features[:, 0, :], caption_features, labels,
                                                      margin=cfg.margin)
            triplet_loss_final_layer = triplet_loss_func_L1(w_sketch_features_l12[:, 0, :], class_features, labels,
                                                            margin=cfg.margin)
            triplet_loss_l7 = triplet_loss_func_L1(w_sketch_features_l7[:, 0, :], class_features, labels,
                                                   margin=cfg.margin)
            triplet_loss_l10 = triplet_loss_func_L1(w_sketch_features_l10[:, 0, :], class_features, labels,
                                                    margin=cfg.margin)

            loss = triplet_loss_scene + triplet_loss_final_layer + triplet_loss_l7 + triplet_loss_l10

            vit_optimizer.zero_grad()
            threshold_optimizer.zero_grad()
            loss.backward()
            vit_optimizer.step()
            threshold_optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if is_main:
                print(f"[Train] Epoch {epoch}/{num_epochs} | Step {batch_idx} | Loss {loss:.4f}", end="\r")
                if cfg.WANDB:
                    wandb.log({"Train loss": loss.item()})
                    wandb.log({"Threshold value": threshold_value})

        avg_epoch_loss = epoch_loss / max(1, num_batches)
        if is_main:
            print(f"\n📉 Epoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}")
            if cfg.WANDB:
                wandb.log({"Train loss": avg_epoch_loss})

            if (epoch + 1) % cfg.save_every == 0:
                os.makedirs(f"checkpoint/{cfg.MODEL.PROMPT.LOG}", exist_ok=True)
                torch.save(model.state_dict(), f"checkpoint/{cfg.MODEL.PROMPT.LOG}/model_{epoch}.pth")

    if is_main:
        print("🎉 Training finished")

    if not DEBUG_MODE:
        dist.destroy_process_group()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)

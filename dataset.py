import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import random
import scipy.io as sio
from utils.utils_train import extract_classes_from_caption
from typing import List, Tuple, Any
import torch
import os
import json
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms
import torch

def dedup_trunc(cls_list, max_classes):
    # C 实现，保序去重
    cls_list = [c.strip() for c in cls_list if c and c.strip()]
    cls_list = list(dict.fromkeys(cls_list))  # ✅ 快
    return cls_list[:max_classes]

class fscoco_train(Dataset):
    def __init__(self, root="DATA/fscoco_seg/train", transform=None, augment=False, SKETCH_SIZE=512):
        self.root = root
        self.transform = transform
        self.augment = augment
        self.sketch_dir = os.path.join(root, "sketches")
        self.image_dir = os.path.join(root, "images")
        self.text_dir = os.path.join(root, "text")

        # 数据增强
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(20),  # Rotate by ±10°
            # transforms.RandomCrop(450),  # Crop of size 200x200 for example
            transforms.RandomCrop(450),  # FIXME：image大小不一，可能会报错
            transforms.Resize((SKETCH_SIZE, SKETCH_SIZE))  # Resize back to 224x224
        ])

        # Get list of subdirectories for sketches and captions
        self.sketch_subdirs = sorted(os.listdir(self.sketch_dir))
        self.txt_subdirs = sorted(os.listdir(self.text_dir))
        # self.image_subdirs = sorted(os.listdir(self.image_dir))

        # Get list of all sketch files and corresponding caption files
        self.sketch_files = []
        self.txt_files = []
        # self.image_files = []
        for sketch_subdir, txt_subdir in zip(self.sketch_subdirs, self.txt_subdirs):
            sketch_subdir_path = os.path.join(self.sketch_dir, sketch_subdir)
            txt_subdir_path = os.path.join(self.text_dir, txt_subdir)
            # image_subdir_path = os.path.join(self.image_dir, image_subdir)

            sketch_files_subdir = sorted(os.listdir(sketch_subdir_path))
            txt_files_subdir = sorted(os.listdir(txt_subdir_path))
            # image_files_subdir = sorted(os.listdir(image_subdir_path))

            self.sketch_files += [os.path.join(sketch_subdir, sketch_file) for sketch_file in sketch_files_subdir]
            self.txt_files += [os.path.join(txt_subdir, txt_file) for txt_file in txt_files_subdir]
            # self.image_files += [os.path.join(image_subdir, image_file) for image_file in image_files_subdir]

    def __len__(self):
        return len(self.sketch_files)

    def __getitem__(self, index):
        text_path = os.path.join(self.text_dir, self.txt_files[index])
        with open(text_path, "r") as f:
            caption = f.read()

        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[index])
        sketch = Image.open(sketch_path).convert("RGB")

        sketch_aug = ImageOps.invert(sketch)
        aug_sketch = self.augmentation(sketch_aug)
        aug_sketch = ImageOps.invert(aug_sketch)

        if self.transform:
            sketch = self.transform(sketch)
            aug_sketch = self.transform(aug_sketch)

        if self.augment:
            sketch = torch.stack([sketch, aug_sketch])

        return sketch, caption


class fscoco_test(Dataset):
    def __init__(self, root="DATA/fscoco-seg/test"):
        self.root = root
        self.img_dir = os.path.join(root, "images")
        self.text_dir = os.path.join(root, "captions")
        self.stroke_dir = os.path.join(root, "vector_sketches")
        self.label_dir = os.path.join(root, "classes")

        # Get list of subdirectories for images and captions
        self.img_files = sorted(os.listdir(self.img_dir))
        self.txt_files = sorted(os.listdir(self.text_dir))
        self.strokes_files = sorted(os.listdir(self.stroke_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

    def __len__(self):
        return len(self.strokes_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        strokes_path = os.path.join(self.stroke_dir, self.strokes_files[index])
        classes_path = os.path.join(self.label_dir, self.label_files[index])
        with open(classes_path, "r") as f:
            classes = json.load(f)
        pen_state = np.load(strokes_path, allow_pickle=True)  # (n,3) array, where n is the number of pen states
        text_path = os.path.join(self.text_dir, self.txt_files[index])

        with open(text_path, "r") as f:
            caption = f.read()

        return pen_state, classes, caption, img_path


class DynamicRandomCrop:
    """动态调整裁剪尺寸以适应不同大小的图像"""

    def __init__(self, min_crop_ratio=0.5, max_crop_ratio=0.9):
        self.min_crop_ratio = min_crop_ratio  # 最小裁剪比例（相对于图像最小边）
        self.max_crop_ratio = max_crop_ratio  # 最大裁剪比例（相对于图像最小边）

    def __call__(self, img):
        # 获取图像尺寸 (宽度, 高度)
        w, h = img.size
        min_side = min(w, h)

        # 计算最大和最小可裁剪尺寸
        max_crop_size = int(min_side * self.max_crop_ratio)
        min_crop_size = max(int(min_side * self.min_crop_ratio), 16)  # 确保至少16x16

        # 随机选择裁剪尺寸
        crop_size = random.randint(min_crop_size, max_crop_size)

        # 执行随机裁剪
        return transforms.RandomCrop(crop_size)(img)


class SketchImageDataset(Dataset):
    def __init__(self, root="DATA/fscoco-seg/train", transform=None, augment=False, SKETCH_SIZE=512):
        self.root = root
        self.transform = transform
        self.augment = augment
        self.sketch_dir = os.path.join(root, "sketches")
        self.image_dir = os.path.join(root, "images")
        self.text_dir = os.path.join(root, "text")

        self.augmentation = transforms.Compose([
            transforms.RandomRotation(20),  # 随机旋转±10°
            DynamicRandomCrop(min_crop_ratio=0.5, max_crop_ratio=0.9),  # 动态裁剪
            transforms.Resize((SKETCH_SIZE, SKETCH_SIZE)),  # 统一尺寸
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        ])

        # Get list of subdirectories for sketches and captions
        self.sketch_subdirs = sorted(os.listdir(self.sketch_dir))
        self.txt_subdirs = sorted(os.listdir(self.text_dir))
        self.image_subdirs = sorted(os.listdir(self.image_dir))

        # Get list of all sketch files and corresponding caption files
        self.sketch_files = []
        self.txt_files = []
        self.image_files = []
        for sketch_subdir, txt_subdir, image_subdir in zip(self.sketch_subdirs, self.txt_subdirs, self.image_subdirs):
            sketch_subdir_path = os.path.join(self.sketch_dir, sketch_subdir)
            txt_subdir_path = os.path.join(self.text_dir, txt_subdir)
            image_subdir_path = os.path.join(self.image_dir, image_subdir)

            sketch_files_subdir = sorted(os.listdir(sketch_subdir_path))
            txt_files_subdir = sorted(os.listdir(txt_subdir_path))
            image_files_subdir = sorted(os.listdir(image_subdir_path))

            self.sketch_files += [os.path.join(sketch_subdir, sketch_file) for sketch_file in sketch_files_subdir]
            self.txt_files += [os.path.join(txt_subdir, txt_file) for txt_file in txt_files_subdir]
            self.image_files += [os.path.join(image_subdir, image_file) for image_file in image_files_subdir]

    def __len__(self):
        return len(self.sketch_files)

    def __getitem__(self, index):
        text_path = os.path.join(self.text_dir, self.txt_files[index])
        with open(text_path, "r") as f:
            caption = f.read()

        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[index])
        sketch = Image.open(sketch_path).convert("RGB")

        image_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(image_path).convert("RGB")

        sketch_aug = ImageOps.invert(sketch)
        aug_sketch = self.augmentation(sketch_aug)
        aug_sketch = ImageOps.invert(aug_sketch)

        image_aug = ImageOps.invert(image)
        aug_image = self.augmentation(image_aug)
        aug_image = ImageOps.invert(aug_image)

        if self.transform:
            sketch = self.transform(sketch)
            aug_sketch = self.transform(aug_sketch)
            image = self.transform(image)
            aug_image = self.transform(aug_image)

        if self.augment:
            sketch = torch.stack([sketch, aug_sketch])
            image = torch.stack([image, aug_image])

        return sketch, caption, image


class InkScene_train(Dataset):
    def __init__(
            self,
            root="/home/xiaoyi/project/inkscene/clipasso-base_train",
            class_map_json="/home/xiaoyi/project/inkscene/clipasso-base_train/mapping.json",
            transform = None
    ):
        self.root = root
        self.img_dir = os.path.join(root, "DRAWING_GT")
        self.label_dir = os.path.join(root, "CLASS_GT_NEW")
        self.transform = transform

        self.img_files = sorted(os.listdir(self.img_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        assert len(self.img_files) == len(self.label_files)

        with open(class_map_json, "r") as f:
            self.id2class = {int(v): k for k, v in json.load(f).items()}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])

        img = Image.open(img_path).convert("RGB")
        mat = sio.loadmat(label_path)
        gt_class_map = mat["class_map"].astype(np.int64)

        unique_ids = np.unique(gt_class_map)
        fg_class_names = [self.id2class[i] for i in unique_ids if i != 0]

        if self.transform:
            img = self.transform(img)

        return img, fg_class_names

class InkScene_test(Dataset):
    def __init__(
            self,
            root="/home/xiaoyi/project/inkscene/clipasso-base_test",
            class_map_json="/home/xiaoyi/project/inkscene/clipasso-base_test/mapping.json"
    ):
        self.root = root
        self.img_dir = os.path.join(root, "DRAWING_GT")
        self.label_dir = os.path.join(root, "CLASS_GT_NEW")

        self.img_files = sorted(os.listdir(self.img_dir))
        self.label_files = sorted(os.listdir(self.label_dir))
        assert len(self.img_files) == len(self.label_files)

        with open(class_map_json, "r") as f:
            self.id2class = {int(v): k for k, v in json.load(f).items()}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_files[index])
        label_path = os.path.join(self.label_dir, self.label_files[index])

        img = Image.open(img_path).convert("RGB")
        mat = sio.loadmat(label_path)
        gt_class_map = mat["class_map"].astype(np.int64)

        return img, gt_class_map, img_path

class fscoco_train_modified(Dataset):
    def __init__(self, root="DATA/fscoco_seg/train", transform=None, augment=False,
                 SKETCH_SIZE=512, max_classes=3, train_classes=None,
                 cache_classes=True, cache_file="classes_cache.json"):
        self.root = root
        self.transform = transform
        self.augment = augment
        self.max_classes = max_classes
        self.train_classes = train_classes

        self.sketch_dir = os.path.join(root, "sketches")
        self.text_dir = os.path.join(root, "text")

        self.augmentation = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomCrop(450),
            transforms.Resize((SKETCH_SIZE, SKETCH_SIZE))
        ])

        self.sketch_subdirs = sorted(os.listdir(self.sketch_dir))
        self.txt_subdirs = sorted(os.listdir(self.text_dir))

        self.sketch_files = []
        self.txt_files = []
        for sketch_subdir, txt_subdir in zip(self.sketch_subdirs, self.txt_subdirs):
            sketch_subdir_path = os.path.join(self.sketch_dir, sketch_subdir)
            txt_subdir_path = os.path.join(self.text_dir, txt_subdir)

            sketch_files_subdir = sorted(os.listdir(sketch_subdir_path))
            txt_files_subdir = sorted(os.listdir(txt_subdir_path))

            self.sketch_files += [os.path.join(sketch_subdir, f) for f in sketch_files_subdir]
            self.txt_files += [os.path.join(txt_subdir, f) for f in txt_files_subdir]

        # ========= 预处理：缓存 classes_list =========
        self.classes_cache = [None] * len(self.txt_files)
        self.cache_path = os.path.join(root, cache_file)

        if cache_classes:
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                # obj: {index_str: [classes]}
                for k, v in obj.items():
                    self.classes_cache[int(k)] = v
            else:
                obj = {}
                for idx in range(len(self.txt_files)):
                    text_path = os.path.join(self.text_dir, self.txt_files[idx])
                    with open(text_path, "r", encoding="utf-8") as f:
                        cap = f.read()
                    cls = extract_classes_from_caption(cap, max_classes=self.max_classes, train_classes=self.train_classes)
                    cls = dedup_trunc(cls, self.max_classes)
                    self.classes_cache[idx] = cls
                    obj[str(idx)] = cls
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.sketch_files)

    def __getitem__(self, index):
        # caption
        text_path = os.path.join(self.text_dir, self.txt_files[index])
        with open(text_path, "r", encoding="utf-8") as f:
            caption = f.read()

        # sketch
        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[index])
        sketch = Image.open(sketch_path).convert("RGB")

        sketch_aug = ImageOps.invert(sketch)
        aug_sketch = self.augmentation(sketch_aug)
        aug_sketch = ImageOps.invert(aug_sketch)

        if self.transform:
            sketch = self.transform(sketch)
            aug_sketch = self.transform(aug_sketch)

        if self.augment:
            sketch = torch.stack([sketch, aug_sketch])

        classes_list = self.classes_cache[index] if self.classes_cache[index] is not None else []

        image_id = index  # 或者用文件名 hash，更稳定

        return sketch, caption, classes_list, image_id


def collate_fscoco_A(batch: List[Tuple[Any, str, List[str], int]]):
    """
    batch: [(sketch, caption, classes_list, image_id), ...] length=B
    sketch: Tensor [3,H,W] or [2,3,H,W] depending on augment
    caption: str
    classes_list: List[str] (variable length)
    image_id: int
    """
    sketches, captions, classes_lists, image_ids = zip(*batch)

    # sketches -> stack tensor
    sketches = torch.stack(sketches, dim=0)  # [B,...]
    captions = list(captions)                # List[str]
    classes_lists = list(classes_lists)      # List[List[str]] (variable length OK)
    image_ids = torch.tensor(image_ids, dtype=torch.long)

    return sketches, captions, classes_lists, image_ids

def collate_inkscene_A(batch: List[Tuple[Any, str, List[str], int]]):
    imgs, classes = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    classes_lists = list(classes)

    return imgs, classes_lists

"""
class SFSD_train(Dataset):
    def __init__(self, root="DATA/SFSD/train", transform=None, augment=False, SKETCH_SIZE=512):
        self.root = root
        self.transform = transform
        self.augment = augment
        self.sketch_dir = os.path.join(root, "sketches")
        self.image_dir = os.path.join(root, "images")
        self.text_dir = os.path.join(root, "text")

        # 数据增强
        self.augmentation = transforms.Compose([
            transforms.RandomRotation(20),  # Rotate by ±10°
            transforms.RandomCrop(450),  # Crop of size 200x200 for example
            transforms.Resize((SKETCH_SIZE, SKETCH_SIZE))  # Resize back to 224x224
        ])

        # Get list of subdirectories for sketches and captions
        self.sketch_subdirs = sorted(os.listdir(self.sketch_dir))
        self.txt_subdirs = sorted(os.listdir(self.text_dir))

        # Get list of all sketch files and corresponding caption files
        self.sketch_files = []
        self.txt_files = []
        for sketch_subdir, txt_subdir in zip(self.sketch_subdirs, self.txt_subdirs):
            sketch_subdir_path = os.path.join(self.sketch_dir, sketch_subdir)
            txt_subdir_path = os.path.join(self.text_dir, txt_subdir)

            sketch_files_subdir = sorted(os.listdir(sketch_subdir_path))
            txt_files_subdir = sorted(os.listdir(txt_subdir_path))

            self.sketch_files += [os.path.join(sketch_subdir, sketch_file) for sketch_file in sketch_files_subdir]
            self.txt_files += [os.path.join(txt_subdir, txt_file) for txt_file in txt_files_subdir]

    def __len__(self):
        return len(self.sketch_files)

    def __getitem__(self, index):
        text_path = os.path.join(self.text_dir, self.txt_files[index])
        with open(text_path, "r") as f:
            caption = f.read()

        sketch_path = os.path.join(self.sketch_dir, self.sketch_files[index])
        sketch = Image.open(sketch_path).convert("RGB")

        sketch_aug = ImageOps.invert(sketch)
        aug_sketch = self.augmentation(sketch_aug)
        aug_sketch = ImageOps.invert(aug_sketch)

        if self.transform:
            sketch =
"""
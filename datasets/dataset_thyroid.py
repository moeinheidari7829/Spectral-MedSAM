import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class ThyroidDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None, spectral_prompt=True):
        self.base_dir = base_dir
        self.split = split  # Not used yet, but for future train/val/test split
        self.transform = transform
        self.spectral_prompt = spectral_prompt

        self.images_dir = os.path.join(base_dir, "images")
        self.labels_dir = os.path.join(base_dir, "labels")
        self.segmaps_dir = os.path.join(base_dir, "semantic_segmentations", "laplacian", "crf_multi_region_new")

        self.image_files = sorted([f for f in os.listdir(self.images_dir) if f.endswith('.png')])
        # Implement train/val/test split (80/10/10)
        num_total = len(self.image_files)
        train_end = int(num_total * 0.8)
        val_end = int(num_total * 0.9)
        if self.split == "train":
            self.image_files = self.image_files[:train_end]
        elif self.split == "val":
            self.image_files = self.image_files[train_end:val_end]
        elif self.split == "test":
            self.image_files = self.image_files[val_end:]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name)
        if self.spectral_prompt:
            segmap_path = os.path.join(self.segmaps_dir, img_name)

        # Load image, label, segmap
        image = np.array(Image.open(img_path).convert('L'))  # grayscale
        # Load label as integer mask
        label = np.array(Image.open(label_path).convert('L'), dtype=np.uint8)
        # Remap grayscale values to class indices for 4-class segmentation
        # 0 (background), 63, 126, 189 -> 0, 1, 2, 3
        label[label == 63] = 1
        label[label == 126] = 2
        label[label == 189] = 3
        if self.spectral_prompt:
            segmap = np.array(Image.open(segmap_path))
        else:
            segmap = None

        sample = {'image': image, 'label': label}
        if self.spectral_prompt:
            sample['spectral_mask'] = segmap

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = img_name
        return sample

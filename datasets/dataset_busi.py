import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class BUSIDataset(Dataset):
    def __init__(self, base_dir, split="train", transform=None, spectral_prompt=True):
        self.base_dir = base_dir
        self.split = split  # Not used yet, but for future train/val/test split
        self.transform = transform
        self.spectral_prompt = spectral_prompt

        self.images_dir = os.path.join(base_dir, "images")
        self.labels_dir = os.path.join(base_dir, "labels")
        self.segmaps_dir = os.path.join(base_dir, "semantic_segmentations", "laplacian", "crf_multi_region")

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
        # else: use all if split is not specified

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name)
        segmap_path = os.path.join(self.segmaps_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        segmap = Image.open(segmap_path)

        image = np.array(image)
        label = np.array(label)
        segmap = np.array(segmap)

        if self.transform is not None:
            sample = self.transform({'image': image, 'label': label, 'segmap': segmap})
            image, label, segmap = sample['image'], sample['label'], sample.get('segmap', None)
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            label = torch.from_numpy(label).long()
            segmap = torch.from_numpy(segmap).long()

        if self.spectral_prompt:
            return image, label, segmap.float()
        else:
            return image, label

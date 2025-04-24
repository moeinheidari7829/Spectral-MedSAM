import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class BUSIRandomGenerator:
    def __init__(self, output_size=512):
        self.output_size = output_size
        self.to_tensor = transforms.ToTensor()
        self.resize_img = transforms.Resize((output_size, output_size), interpolation=Image.BILINEAR)
        self.resize_mask = transforms.Resize((output_size, output_size), interpolation=Image.NEAREST)

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']
        segmap = sample.get('segmap', None)
        spectral_prompt = sample.get('spectral_prompt', False)
        # Convert to PIL Images if they aren't already
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if not isinstance(label, Image.Image):
            label = Image.fromarray(label)
        if segmap is not None and not isinstance(segmap, Image.Image):
            segmap = Image.fromarray(segmap)
        image = self.resize_img(image)
        label = self.resize_mask(label)
        if segmap is not None:
            segmap = self.resize_mask(segmap)
        label_arr = np.array(label)
        label_arr[label_arr == 255] = 1
        if segmap is not None:
            segmap_arr = np.array(segmap)
            segmap_arr[segmap_arr == 255] = 1
        # Compose output
        out = {
            'image': self.to_tensor(image),
            'label': torch.from_numpy(label_arr).long()
        }
        if segmap is not None:
            out['segmap'] = torch.from_numpy(segmap_arr).long()
        return out

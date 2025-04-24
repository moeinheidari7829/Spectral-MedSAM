import torch
import numpy as np
from scipy.ndimage import zoom
from torchvision import transforms

class ThyroidRandomGenerator(object):
    def __init__(self, output_size, low_res=None, num_classes=2):
        self.output_size = output_size
        self.low_res = low_res
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        spectral_mask = sample.get('spectral_mask', None)

        # Ensure label is integer and remap only if binary
        if self.num_classes == 2:
            label[label == 255] = 1
        # For multi-class, assume labels are already 0,1,2,...

        # Resize spectral mask to match image size if needed
        if spectral_mask is not None and spectral_mask.shape != image.shape:
            sx, sy = image.shape[0] / spectral_mask.shape[0], image.shape[1] / spectral_mask.shape[1]
            spectral_mask = zoom(spectral_mask, (sx, sy), order=0)

        # Random rotation and flip
        if np.random.rand() > 0.5:
            k = np.random.randint(0, 4)
            image = np.rot90(image, k)
            label = np.rot90(label, k)
            if spectral_mask is not None:
                spectral_mask = np.rot90(spectral_mask, k)
        if np.random.rand() > 0.5:
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            if spectral_mask is not None:
                spectral_mask = np.flip(spectral_mask, axis=axis).copy()

        # Resize to output_size
        x, y = image.shape
        if (x, y) != tuple(self.output_size):
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        if spectral_mask is not None:
            spectral_mask = zoom(spectral_mask, (self.output_size[0] / spectral_mask.shape[0], self.output_size[1] / spectral_mask.shape[1]), order=0)
            sample['spectral_mask'] = spectral_mask  # Ensure later code uses the resized mask

        # Normalize image to [0, 1] and then to [-1, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)  # [0,1]
        image = image * 2 - 1  # [-1,1]

        # Convert to torch tensors
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)  # [1, H, W]
        label = torch.from_numpy(label.astype(np.float32))

        # Compose final image tensor
        if spectral_mask is not None:
            # Always ensure spectral_mask is [1, H, W]
            if isinstance(spectral_mask, np.ndarray):
                if spectral_mask.ndim == 3:
                    spectral_mask = spectral_mask.squeeze()
                spectral_mask = torch.from_numpy(spectral_mask.astype(np.float32))
            if spectral_mask.dim() == 2:
                spectral_mask = spectral_mask.unsqueeze(0)  # [1, H, W]
            # Repeat grayscale to 3 channels, then concat spectral as 4th
            image = torch.cat([image.repeat(3, 1, 1), spectral_mask], dim=0)  # [4, H, W]
            sample['spectral_mask'] = spectral_mask
        else:
            image = image.repeat(3, 1, 1)  # [3, H, W] if not using spectral

        sample['image'] = image
        sample['label'] = label.long()
        return sample

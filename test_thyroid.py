import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from importlib import import_module
import matplotlib.pyplot as plt
from datasets.dataset_thyroid import ThyroidDataset
from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry
# If you have a dice_coeff or similar metric function, import it here
# from utils import dice_coeff

def dice_coeff(pred, target):
    # Simple Dice for binary masks
    pred = pred.bool()
    target = target.bool()
    intersection = (pred & target).float().sum((1, 2))
    union = pred.float().sum((1, 2)) + target.float().sum((1, 2))
    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    return dice.mean().item()

def visualize(image, label, pred, save_path, idx):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).cpu(), cmap='gray')
    plt.title('Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(label.cpu(), cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(pred.cpu(), cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'vis_{idx:03d}.png'))
    plt.close()

def evaluate(model, dataloader, device, save_vis_dir=None):
    model.eval()
    dices = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            image = batch['image'].to(device)
            if image.dim() == 3:
                image = image.unsqueeze(0)
            label = batch['label'].to(device)
            spectral_mask = batch.get('spectral_mask', None)
            if spectral_mask is not None:
                spectral_mask = spectral_mask.to(device)
                if spectral_mask.dim() == 3:
                    spectral_mask = spectral_mask.unsqueeze(0)
                image_size = (512, 512)
                output = model(image, multimask_output=False, image_size=image_size, spectral_mask=spectral_mask)
            else:
                image_size = (512, 512)
                output = model(image, multimask_output=False, image_size=image_size)
            pred = output['masks']
            # Debug: Print output stats
            print(f"[DEBUG] idx={idx} pred shape: {pred.shape}, unique: {pred.unique()}, label unique: {label.unique()}, output raw min/max: {output['masks'].min()}/{output['masks'].max()}")
            # Visualization of raw prediction
            if idx < 3:
                plt.figure()
                plt.imshow(output['masks'][0,0].cpu(), cmap='gray')
                plt.title(f'Raw Model Output idx={idx}')
                plt.savefig(os.path.join(save_vis_dir, f'raw_output_{idx:03d}.png'))
                plt.close()
            # Thresholding/argmax
            if pred.shape[1] > 1:
                pred = pred.argmax(dim=1)
            else:
                # Map all negative values to 0, positives to 1 for visualization
                pred = (pred > 0).long().squeeze(1)
            # Debug: Print post-threshold stats
            print(f"[DEBUG] idx={idx} post-threshold pred unique: {pred.unique()}, label unique: {label.unique()}")
            dice = dice_coeff(pred, label)
            dices.append(dice)
            # Visualization
            if save_vis_dir:
                visualize(image[0], label[0], pred[0], save_vis_dir, idx)
    mean_dice = np.mean(dices)
    print(f'Mean Dice: {mean_dice:.4f}')
    return mean_dice

from datasets.thyroid_transforms import ThyroidRandomGenerator

if __name__ == '__main__':
    # ---- Config ----
    root_path = 'ThySegPreSeg/001_P1_1_left'
    split = 'test'  # or 'val'
    batch_size = 1
    num_classes = 2
    img_size = 512
    lora_ckpt = 'output/Thyroid_512_pretrain_vit_b_epo200_bs2_lr0.005/epoch_199.pth'  # Updated checkpoint path
    sam_ckpt = 'checkpoints/sam_vit_b_01ec64.pth'
    use_lora = True  # set False to evaluate original SAM
    save_vis_dir = './eval_vis'
    os.makedirs(save_vis_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Data ----
    eval_transform = ThyroidRandomGenerator(output_size=(512, 512))
    dataset = ThyroidDataset(base_dir=root_path, split=split, transform=eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # ---- Model ----
    sam, _ = sam_model_registry['vit_b'](
        image_size=img_size,
        num_classes=num_classes,
        checkpoint=sam_ckpt,
        pixel_mean=[0, 0, 0, 0],  # or [0,0,0] if not using spectral
        pixel_std=[1, 1, 1, 1]
    )
    model = LoRA_Sam(sam, r=4, use_spectral_prompt=True).to(device)

    print("\nModel parameters at evaluation:")
    for name, param in model.named_parameters():
        print(name)

    if use_lora:
        # Try LoRA-specific loading first
        try:
            result = model.load_lora_parameters(lora_ckpt)
            print(f"Loaded LoRA checkpoint: {lora_ckpt}")
        except KeyError:
            state_dict = torch.load(lora_ckpt)
            result = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded standard PyTorch checkpoint (partial weights): {lora_ckpt}")
            print("WARNING: LoRA weights NOT restored. Evaluation is for base model only!")
            print('Missing keys:', result.missing_keys)
            print('Unexpected keys:', result.unexpected_keys)
    else:
        print("Evaluating original SAM (no LoRA weights loaded).")

    # ---- Evaluation ----
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Model: {'SAM+LoRA (weights loaded)' if use_lora else 'Original SAM'}")
    print(f"Checkpoint: {lora_ckpt if use_lora else sam_ckpt}")
    print("==========================\n")
    evaluate(model, loader, device, save_vis_dir=save_vis_dir)

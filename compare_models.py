import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from datasets.dataset_thyroid import ThyroidDataset
from datasets.thyroid_transforms import ThyroidRandomGenerator
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random

# --- Utility functions ---
def dice_score(pred, target, num_classes):
    dice = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        if union == 0:
            dice.append(1.0)
        else:
            dice.append(2. * intersection / union)
    return dice

def iou_score(pred, target, num_classes):
    ious = []
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum()
        union = (pred_cls | target_cls).sum()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return ious

def colorize_mask(mask):
    # Convert mask (H, W) to color image for visualization
    palette = np.array([
        [0, 0, 0],      # background: black
        [255, 0, 0],    # class 1: red
        [0, 255, 0],    # class 2: green
        [0, 0, 255],    # class 3: blue
    ], dtype=np.uint8)
    color_mask = palette[mask]
    return color_mask.transpose(2, 0, 1)  # (C, H, W)

# --- Main comparison script ---
def main():
    root_path = '/home/moein/Desktop/BMEG591_Project/SAMed/ThySegPreSeg/001_P1_1_left'
    num_classes = 4
    img_size = 512
    batch_size = 1
    seed = 42
    test_ratio = 0.2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Prepare dataset and split ---
    full_dataset = ThyroidDataset(
        base_dir=root_path,
        split="all",
        transform=ThyroidRandomGenerator((img_size, img_size), num_classes=num_classes),
        spectral_prompt=False  # will override in loader
    )
    n_total = len(full_dataset)
    indices = list(range(n_total))
    random.seed(seed)
    random.shuffle(indices)
    n_test = int(n_total * test_ratio)
    test_indices = indices[-n_test:]

    # --- Prepare test datasets for both models ---
    test_dataset_no_prompt = Subset(ThyroidDataset(
        base_dir=root_path,
        split="all",
        transform=ThyroidRandomGenerator((img_size, img_size), num_classes=num_classes),
        spectral_prompt=False
    ), test_indices)
    test_dataset_prompt = Subset(ThyroidDataset(
        base_dir=root_path,
        split="all",
        transform=ThyroidRandomGenerator((img_size, img_size), num_classes=num_classes),
        spectral_prompt=True
    ), test_indices)

    test_loader_no_prompt = DataLoader(test_dataset_no_prompt, batch_size=batch_size, shuffle=False)
    test_loader_prompt = DataLoader(test_dataset_prompt, batch_size=batch_size, shuffle=False)

    # --- Load models ---
    from segment_anything.build_sam import build_sam_vit_b
    from sam_lora_image_encoder import LoRA_Sam

    # For model with spectral prompt (4 channels)
    base_sam_prompt, _ = build_sam_vit_b(
        image_size=img_size,
        num_classes=num_classes,
        checkpoint=None,
        in_chans=4,
        pixel_mean=[0, 0, 0, 0],
        pixel_std=[1, 1, 1, 1]
    )
    model_prompt = LoRA_Sam(base_sam_prompt, r=4, use_spectral_prompt=True)
    ckpt_prompt = torch.load('/home/moein/Desktop/BMEG591_Project/SAMed/output/Thyroid_512_pretrain_vit_b_epo100_bs2_lr0.0001/epoch_99.pth', map_location=device)
    if 'state_dict' in ckpt_prompt:
        state_dict = ckpt_prompt['state_dict']
    else:
        state_dict = ckpt_prompt
    model_prompt.load_state_dict(state_dict, strict=False)
    model_prompt = model_prompt.to(device)
    model_prompt.eval()

    # For model without spectral prompt (3 channels)
    base_sam_no_prompt, _ = build_sam_vit_b(
        image_size=img_size,
        num_classes=num_classes,
        checkpoint=None,
        in_chans=3,
        pixel_mean=[0, 0, 0],
        pixel_std=[1, 1, 1]
    )
    model_no_prompt = LoRA_Sam(base_sam_no_prompt, r=4, use_spectral_prompt=False)
    ckpt_no_prompt = torch.load('/home/moein/Desktop/BMEG591_Project/SAMed/output_no_prompt/Thyroid_512_pretrain_vit_b_epo100_bs2_lr0.0001/epoch_99.pth', map_location=device)
    if 'state_dict' in ckpt_no_prompt:
        state_dict = ckpt_no_prompt['state_dict']
    else:
        state_dict = ckpt_no_prompt
    model_no_prompt.load_state_dict(state_dict, strict=False)
    model_no_prompt = model_no_prompt.to(device)
    model_no_prompt.eval()

    # --- TensorBoard writer ---
    writer = SummaryWriter('runs/model_comparison')

    # --- Evaluation loop ---
    dice_scores_prompt, dice_scores_no_prompt = [], []
    iou_scores_prompt, iou_scores_no_prompt = [], []

    for idx, (sample_prompt, sample_no_prompt) in enumerate(zip(test_loader_prompt, test_loader_no_prompt)):
        image_prompt = sample_prompt['image'].to(device)
        label = sample_prompt['label'].to(device)
        if 'spectral_mask' in sample_prompt:
            spectral_mask = sample_prompt['spectral_mask'].to(device)
        else:
            spectral_mask = None

        image_no_prompt = sample_no_prompt['image'].to(device)

        # --- Predict with both models ---
        with torch.no_grad():
            if spectral_mask is not None:
                pred_prompt = model_prompt(image_prompt, multimask_output=True, image_size=(img_size, img_size), spectral_mask=spectral_mask)
            else:
                pred_prompt = model_prompt(image_prompt, multimask_output=True, image_size=(img_size, img_size))
            pred_no_prompt = model_no_prompt(image_no_prompt, multimask_output=True, image_size=(img_size, img_size))

            # Both outputs are dicts; extract 'masks' tensor
            pred_prompt = torch.argmax(pred_prompt['masks'], dim=1).cpu().numpy()[0]
            pred_no_prompt = torch.argmax(pred_no_prompt['masks'], dim=1).cpu().numpy()[0]
            gt = label.cpu().numpy()[0]

            # Clamp predictions to valid class range
            pred_prompt = np.clip(pred_prompt, 0, num_classes-1)
            pred_no_prompt = np.clip(pred_no_prompt, 0, num_classes-1)

            # Remap all non-[1,2,3] predictions to background (0)
            valid_classes = [1, 2, 3]
            pred_prompt = np.where(np.isin(pred_prompt, valid_classes), pred_prompt, 0)
            pred_no_prompt = np.where(np.isin(pred_no_prompt, valid_classes), pred_no_prompt, 0)

        # --- Metrics ---
        dice_p = dice_score(pred_prompt, gt, num_classes)
        dice_n = dice_score(pred_no_prompt, gt, num_classes)
        iou_p = iou_score(pred_prompt, gt, num_classes)
        iou_n = iou_score(pred_no_prompt, gt, num_classes)
        dice_scores_prompt.append(dice_p)
        dice_scores_no_prompt.append(dice_n)
        iou_scores_prompt.append(iou_p)
        iou_scores_no_prompt.append(iou_n)

        # --- Visualization ---
        img_vis_prompt = image_prompt.cpu().numpy()[0, 0]
        img_vis_prompt = np.stack([img_vis_prompt]*3, axis=0)  # grayscale to RGB
        img_vis_no_prompt = image_no_prompt.cpu().numpy()[0, 0]
        img_vis_no_prompt = np.stack([img_vis_no_prompt]*3, axis=0)  # grayscale to RGB
        gt_vis = colorize_mask(gt)
        pred_prompt_vis = colorize_mask(pred_prompt)
        pred_no_prompt_vis = colorize_mask(pred_no_prompt)
        grid = make_grid([
            torch.tensor(img_vis_prompt),
            torch.tensor(img_vis_no_prompt),
            torch.tensor(gt_vis),
            torch.tensor(pred_prompt_vis),
            torch.tensor(pred_no_prompt_vis)
        ], nrow=5, normalize=True)
        writer.add_image(f'comparison/{idx}', grid, 0)

    # --- Aggregate metrics ---
    dice_scores_prompt = np.array(dice_scores_prompt)
    dice_scores_no_prompt = np.array(dice_scores_no_prompt)
    iou_scores_prompt = np.array(iou_scores_prompt)
    iou_scores_no_prompt = np.array(iou_scores_no_prompt)

    for c in range(num_classes-1):
        writer.add_scalar(f'Dice/class_{c+1}_prompt', dice_scores_prompt[:, c].mean(), 0)
        writer.add_scalar(f'Dice/class_{c+1}_no_prompt', dice_scores_no_prompt[:, c].mean(), 0)
        writer.add_scalar(f'IoU/class_{c+1}_prompt', iou_scores_prompt[:, c].mean(), 0)
        writer.add_scalar(f'IoU/class_{c+1}_no_prompt', iou_scores_no_prompt[:, c].mean(), 0)
    writer.add_scalar('Dice/mean_prompt', dice_scores_prompt.mean(), 0)
    writer.add_scalar('Dice/mean_no_prompt', dice_scores_no_prompt.mean(), 0)
    writer.add_scalar('IoU/mean_prompt', iou_scores_prompt.mean(), 0)
    writer.add_scalar('IoU/mean_no_prompt', iou_scores_no_prompt.mean(), 0)

    writer.close()
    print('Comparison complete. View results in TensorBoard:')
    print('  tensorboard --logdir runs/model_comparison')

if __name__ == '__main__':
    main()

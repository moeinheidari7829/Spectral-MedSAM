"""
experiments_prompt_analysis.py

A notebook-style script for prompt ablation, visualization, and architectural experiments.

Goals:
- Visualize prompt and input channels.
- Run ablation (random prompt, prompt-only, no prompt).
- Compare prompt integration methods (concatenation, cross-attention, gating).
- Tune hyperparameters for prompt runs.
- Collect and plot quantitative and qualitative results.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from datasets.thyroid_transforms import ThyroidRandomGenerator
from torch.utils.data import DataLoader
from sam_lora_image_encoder import LoRA_Sam

# --- 1. Visualization: Prompt and Input Channels ---
def visualize_sample(sample, save_path=None):
    img = sample['image']
    segmap = sample.get('segmap', None)
    n_plots = img.shape[0] + (1 if segmap is not None else 0)
    fig, axs = plt.subplots(1, n_plots, figsize=(16, 4))
    if not isinstance(axs, np.ndarray):
        axs = [axs]
    for i in range(img.shape[0]):
        axs[i].imshow(img[i], cmap='gray')
        axs[i].set_title(f'Input Channel {i}')
        axs[i].axis('off')
    if segmap is not None:
        axs[-1].imshow(segmap[0], cmap='gray')
        axs[-1].set_title('Segmentation')
        axs[-1].axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# --- 2. Ablation: Random Prompt, Prompt-Only, No Prompt ---
def ablation_loader(base_loader, ablation_type='random_prompt'):
    for batch in base_loader:
        if ablation_type == 'random_prompt':
            # Replace prompt channel with random noise
            batch['image'][:, -1] = torch.randn_like(batch['image'][:, -1])
        elif ablation_type == 'prompt_only':
            # Keep only the prompt channel
            batch['image'] = batch['image'][:, -1:].clone()
        elif ablation_type == 'no_prompt':
            # Remove prompt channel
            batch['image'] = batch['image'][:, :3].clone()
        yield batch

# --- 3. Prompt Integration Methods ---
def build_model(method='concat', sam_model=None, r=4, **kwargs):
    # method: 'concat', 'cross_attention', 'gating'
    if method == 'concat':
        return LoRA_Sam(sam_model=sam_model, r=r, use_spectral_prompt=True, **kwargs)
    elif method == 'cross_attention':
        return LoRA_Sam(sam_model=sam_model, r=r, prompt_cross_attention=True, **kwargs)
    elif method == 'gating':
        # TODO: implement gating mechanism in LoRA_Sam
        raise NotImplementedError('Gating integration not implemented yet.')
    else:
        raise ValueError(f'Unknown method: {method}')

# --- SAM backbone loader ---
def get_sam_model():
    from segment_anything.build_sam import sam_model_registry
    sam_type = "vit_b"
    checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
    image_size = 512
    num_classes = 4
    model_tuple = sam_model_registry[sam_type](checkpoint=checkpoint, image_size=image_size, num_classes=num_classes)
    if isinstance(model_tuple, tuple):
        return model_tuple[0]
    return model_tuple

# --- 4. Hyperparameter Tuning (Skeleton) ---
def run_hparam_sweep(hparams, train_func):
    results = {}
    for hp_set in hparams:
        key = str(hp_set)
        results[key] = train_func(**hp_set)
    return results

# --- 5. Collect Results and Plot ---
def plot_metrics(metrics_dict):
    for label, metrics in metrics_dict.items():
        plt.plot(metrics['epochs'], metrics['dice'], label=f'{label} Dice')
        plt.plot(metrics['epochs'], metrics['iou'], label=f'{label} IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # --- 1. Visualize a sample ---
    from datasets.dataset_thyroid import ThyroidDataset
    from datasets.thyroid_transforms import ThyroidRandomGenerator
    from torch.utils.data import DataLoader

    print("\n[1] Visualizing input and prompt channels from first sample...")
    dataset = ThyroidDataset(
    '/home/moein/Desktop/BMEG591_Project/SAMed/ThySegPreSeg/001_P1_1_left',
    transform=ThyroidRandomGenerator(output_size=(512, 512), num_classes=4),
    spectral_prompt=True
)

    sample = dataset[0]
    visualize_sample(sample)

    # --- 2. Run ablation study ---
    print("\n[2] Visualizing ablation types (random_prompt, prompt_only, no_prompt)...")
    loader = DataLoader(dataset, batch_size=2)
    for ablation_type in ['random_prompt', 'prompt_only', 'no_prompt']:
        print(f"Ablation type: {ablation_type}")
        for batch in ablation_loader(loader, ablation_type):
            visualize_sample({'image': batch['image'][0].cpu(), 'segmap': batch.get('segmap', None)})
            break  # Only show first batch for each ablation

    # --- 3. Try different prompt integration methods ---
    print("\n[3] Building models for different prompt integration methods...")
    sam_model = get_sam_model()
    lora_rank = 4
    model_concat = build_model(method='concat', sam_model=sam_model, r=lora_rank, img_size=512, patch_size=16, embed_dim=768)
    print("Model (concatenation):", type(model_concat))
    try:
        model_cross = build_model(method='cross_attention', sam_model=sam_model, r=lora_rank, img_size=512, patch_size=16, embed_dim=768)
        print("Model (cross-attention):", type(model_cross))
    except Exception as e:
        print("Cross-attention model build failed:", e)
    try:
        model_gating = build_model(method='gating', sam_model=sam_model, r=lora_rank, img_size=512, patch_size=16, embed_dim=768)
        print("Model (gating):", type(model_gating))
    except Exception as e:
        print("Gating model build failed (expected):", e)

    # --- 4. Hyperparameter sweep (skeleton, not run by default) ---
    # Example:
    # hparams = [
    #     {'img_size': 512, 'patch_size': 16, 'embed_dim': 768, 'base_lr': 0.0001},
    #     {'img_size': 512, 'patch_size': 16, 'embed_dim': 768, 'base_lr': 0.00005},
    # ]
    # def dummy_train_func(**kwargs):
    #     return {'epochs': list(range(10)), 'dice': np.random.rand(10), 'iou': np.random.rand(10)}
    # results = run_hparam_sweep(hparams, dummy_train_func)
    # plot_metrics(results)

    print("\nAll prompt analysis steps complete.")

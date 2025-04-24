import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob
import numpy as np
from PIL import Image
import torch

# ========== CONFIG ==========
# Update these paths as needed
EXPERIMENT_ROOT = "final_output_busi_new/busi"
VARIANTS = {
    "no_prompt": "no_prompt",
    "spectral_prompt": "spectral_prompt",
    "spectral_prompt_cross_attention": "spectral_prompt_cross_attention"
}
# The logs are in subfolders of EXPERIMENT_ROOT/VARIANT/EXPERIMENT_NAME/
# We'll search for the latest experiment in each variant

# ========== 1. PLOT VALIDATION METRICS ==========
def find_latest_event_file(logdir):
    event_files = glob.glob(os.path.join(logdir, "events.out.tfevents.*"))
    if not event_files:
        return None
    return max(event_files, key=os.path.getmtime)

def plot_validation_metrics():
    plt.figure(figsize=(15, 4))
    for i, metric in enumerate(["val/loss_total", "val/dice", "val/iou"]):
        plt.subplot(1, 3, i+1)
        for variant, variant_dir in VARIANTS.items():
            exp_dirs = glob.glob(os.path.join(EXPERIMENT_ROOT, variant_dir, "*"))
            if not exp_dirs:
                continue
            latest_exp = max(exp_dirs, key=os.path.getmtime)
            event_file = find_latest_event_file(latest_exp)
            if not event_file:
                continue
            ea = event_accumulator.EventAccumulator(event_file)
            ea.Reload()
            if metric not in ea.Tags()['scalars']:
                continue
            events = ea.Scalars(metric)
            steps = [e.step for e in events]
            vals = [e.value for e in events]
            plt.plot(steps, vals, label=variant)
        plt.title(metric.replace('val/', '').capitalize())
        plt.xlabel('Epoch')
        plt.legend()
    plt.tight_layout()
    plt.savefig("busi_val_metrics.png")
    print("Saved validation metrics plot to busi_val_metrics.png")

# ========== 2. 5-COLUMN EXAMPLES ==========
# This part is a template. You must fill in model loading and inference logic.
def load_model(variant_dir):
    # TODO: Replace with your actual model loading code
    # Example: torch.load(os.path.join(variant_dir, 'best_model.pth'))
    return None

def get_val_dataset():
    # TODO: Replace with your actual dataset import and instantiation
    # from datasets.dataset_busi import BUSIDataset
    # return BUSIDataset(..., split="val", ...)
    return None

def plot_5_column_examples():
    val_dataset = get_val_dataset()
    idxs = [0, 1]  # Pick two examples
    models = {k: load_model(os.path.join(EXPERIMENT_ROOT, v)) for k, v in VARIANTS.items()}
    for idx in idxs:
        sample = val_dataset[idx]
        image = sample[0]
        label = sample[1]
        # TODO: Adjust this if your dataset returns more fields
        preds = {}
        for variant, model in models.items():
            # TODO: Replace with your actual inference code
            preds[variant] = np.zeros_like(label)  # Dummy prediction
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title('Image')
        axes[1].imshow(label.cpu().numpy())
        axes[1].set_title('Ground Truth')
        axes[2].imshow(preds['no_prompt'])
        axes[2].set_title('No Prompt')
        axes[3].imshow(preds['spectral_prompt'])
        axes[3].set_title('Spectral Prompt')
        axes[4].imshow(preds['spectral_prompt_cross_attention'])
        axes[4].set_title('Spectral Cross-Attn')
        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(f"busi_5col_example_{idx}.png")
        print(f"Saved 5-column example figure: busi_5col_example_{idx}.png")

if __name__ == "__main__":
    plot_validation_metrics()
    # plot_5_column_examples()  # Uncomment and complete the TODOs to enable this

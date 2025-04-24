#!/bin/bash

# Force all CUDA operations to use GPU 1
export CUDA_VISIBLE_DEVICES=0

# Create output directories if they don't exist
mkdir -p final_output_thyroid_new/thyroid/no_prompt
mkdir -p final_output_thyroid_new/thyroid/spectral_prompt
mkdir -p final_output_thyroid_new/thyroid/spectral_prompt_cross_attention

THYROID_ROOT="/home/moein/Desktop/BMEG591_Project/SAMed/ThySegPreSeg/001_P1_1_left"

# Run training for no_prompt
EXPERIMENT_NAME="Thyroid_512_vit_b_epo100_bs4_lr0.0001"
echo "Starting training with no prompt..."
python3 train.py \
  --dataset Thyroid \
  --root_path "$THYROID_ROOT" \
  --num_classes 4 \
  --batch_size 2 \
  --n_gpu 1 \
  --img_size 512 \
  --max_epochs 100 \
  --base_lr 0.0001 \
  --output final_output_thyroid_new/thyroid/no_prompt/${EXPERIMENT_NAME}

# Run training for spectral_prompt (concatenated to input)
EXPERIMENT_NAME="Thyroid_512_vit_b_spectral_epo100_bs4_lr0.0001"
echo "Starting training with spectral prompt (concatenated to input)..."
python3 train.py \
  --dataset Thyroid \
  --root_path "$THYROID_ROOT" \
  --num_classes 4 \
  --batch_size 2 \
  --n_gpu 1 \
  --img_size 512 \
  --max_epochs 100 \
  --base_lr 0.0001 \
  --spectral_prompt \
  --output final_output_thyroid_new/thyroid/spectral_prompt/${EXPERIMENT_NAME}

# Run training for spectral_prompt_cross_attention
EXPERIMENT_NAME="Thyroid_512_vit_b_spectral_cross_attention_epo100_bs4_lr0.0001"
echo "Starting training with spectral prompt cross attention..."
python3 train.py \
  --dataset Thyroid \
  --root_path "$THYROID_ROOT" \
  --num_classes 4 \
  --batch_size 2 \
  --n_gpu 1 \
  --img_size 512 \
  --max_epochs 100 \
  --base_lr 0.0001 \
  --prompt_multi_scale \
  --output final_output_thyroid_new/thyroid/spectral_prompt_cross_attention/${EXPERIMENT_NAME}

echo "All Thyroid training runs completed!"
# To visualize, run:
# tensorboard --logdir final_output_thyroid_new/thyroid --port 6006

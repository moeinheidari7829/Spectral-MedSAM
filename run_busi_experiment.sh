#!/bin/bash

# Create output directories if they don't exist
mkdir -p final_output_busi_new/busi/no_prompt
mkdir -p final_output_busi_new/busi/spectral_prompt
mkdir -p final_output_busi_new/busi/spectral_prompt_cross_attention

# Set your BUSI dataset base directory here:
BUSI_ROOT="/home/moein/Desktop/BMEG591_Project/SAMed/BUSI_Mixed"

# # Run training for no_prompt
EXPERIMENT_NAME="BUSI_512_vit_b_epo25_bs4_lr0.0001"
echo "Starting training with no prompt..."
python3 train.py \
  --dataset BUSI \
  --root_path "$BUSI_ROOT" \
  --output final_output_busi_new/busi/no_prompt/${EXPERIMENT_NAME} \
  --num_classes 2 \
  --batch_size 4 \
  --n_gpu 1 \
  --img_size 512 \
  --max_epochs 100 \
  --base_lr 0.0001

# Run training for spectral_prompt (concatenated to input)
echo "Starting training with spectral prompt (concatenated to input)..."
python3 train.py \
  --dataset BUSI \
  --root_path "$BUSI_ROOT" \
  --output final_output_busi_new/busi/spectral_prompt/${EXPERIMENT_NAME} \
  --num_classes 2 \
  --batch_size 4 \
  --n_gpu 1 \
  --img_size 512 \
  --max_epochs 100 \
  --base_lr 0.0001 \
  --spectral_prompt

# Run training for spectral_prompt_cross_attention
# echo "Starting training with spectral prompt cross attention..."
python3 train.py \
  --dataset BUSI \
  --root_path "$BUSI_ROOT" \
  --output final_output_busi_new/busi/spectral_prompt_cross_attention/${EXPERIMENT_NAME} \
  --num_classes 2 \
  --batch_size 2 \
  --n_gpu 1 \
  --img_size 512 \
  --max_epochs 100 \
  --base_lr 0.0001 \
  --prompt_multi_scale

echo "All BUSI training runs completed!"
# To visualize, run:
# tensorboard --logdir final_output_busi_new/busi --port 6006

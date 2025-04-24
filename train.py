import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from importlib import import_module

from sam_lora_image_encoder import LoRA_Sam
from segment_anything import sam_model_registry

from trainer import trainer_synapse
from trainer_thyroid import trainer_thyroid
from icecream import ic
from torch.utils.data import DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data/LarryXu/Synapse/preprocessed_data/train_npz', help='root dir for data')
parser.add_argument('--output', type=str, default='/output/sam/results')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=8, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=1, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=160, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_b', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--lora_ckpt', type=str, default=None, help='Finetuned lora checkpoint')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=250,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--module', type=str, default='sam_lora_image_encoder')
parser.add_argument('--dice_param', type=float, default=0.8)
parser.add_argument('--spectral_prompt', action='store_true', help='Use spectral prompt (concatenated to input)')
parser.add_argument('--spectral_prompt_cross_attention', action='store_true', help='Use spectral prompt with cross-attention')
parser.add_argument('--prompt_conditioning', action='store_true', help='Use spectral mask to modulate LoRA layers (not as input channel)')
parser.add_argument('--prompt_cross_attention', action='store_true', help='Use cross-attention with prompt tokens after patch embedding')
parser.add_argument('--prompt_multi_scale', action='store_true', help='Inject prompt cross-attention after every transformer block (multi-scale)')
args = parser.parse_args()

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        }
    }
    args.is_pretrain = True
    args.exp = dataset_name + '_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # Set normalization and model input channels according to prompt mode
    if args.spectral_prompt or args.spectral_prompt_cross_attention:
        pixel_mean = [123.675, 116.28, 103.53, 0.0]
        pixel_std = [58.395, 57.12, 57.375, 1.0]
    else:
        pixel_mean = [123.675, 116.28, 103.53]
        pixel_std = [58.395, 57.12, 57.375]

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=args.num_classes,
                                                                checkpoint=args.ckpt, pixel_mean=pixel_mean,
                                                                pixel_std=pixel_std)

    if args.ckpt is not None and os.path.isfile(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        if 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        else:
            model_state = checkpoint
        try:
            sam.load_state_dict(model_state, strict=False)
            print(f"Loaded weights from {args.ckpt}")
        except Exception as e:
            print(f"Failed to load weights from {args.ckpt}: {e}")
    else:
        print(f"No valid checkpoint found at {args.ckpt}, training from scratch.")

    if args.module == 'sam_lora_image_encoder':
        from sam_lora_image_encoder import LoRA_Sam
        # Patch model to accept 4 channels ONLY if prompt is enabled
        if args.spectral_prompt or args.spectral_prompt_cross_attention:
            original_proj = sam.image_encoder.patch_embed.proj
            if original_proj.in_channels == 3:
                import torch.nn as nn
                new_proj = nn.Conv2d(4, original_proj.out_channels, kernel_size=original_proj.kernel_size,
                                     stride=original_proj.stride, padding=original_proj.padding)
                with torch.no_grad():
                    new_proj.weight[:, :3, :, :] = original_proj.weight.clone()
                    new_proj.weight[:, 3:4, :, :] = original_proj.weight.mean(dim=1, keepdim=True)
                    if original_proj.bias is not None:
                        new_proj.bias = nn.Parameter(original_proj.bias.clone())
                sam.image_encoder.patch_embed.proj = new_proj
            # else: already 4 channels, do nothing
        net = LoRA_Sam(sam_model=sam, 
                      r=args.rank, 
                      use_spectral_prompt=args.spectral_prompt,
                      prompt_conditioning=args.prompt_conditioning,
                      prompt_cross_attention=args.prompt_cross_attention or args.spectral_prompt_cross_attention,
                      prompt_multi_scale=args.prompt_multi_scale,
                      img_size=args.img_size,
                      patch_size=16,
                      embed_dim=768,
                      num_heads=12,
                      num_prompt_tokens=8).cuda()

    # Patch mask decoder for BUSI to ensure correct output channels
    if args.dataset == "BUSI":
        mask_decoder = net.sam.mask_decoder
        num_mask_tokens = len(mask_decoder.output_hypernetworks_mlps)
        transformer_dim = mask_decoder.output_hypernetworks_mlps[0].layers[0].in_features
        hypernet_out_dim = mask_decoder.output_hypernetworks_mlps[0].layers[-1].out_features  # restore original output dim (usually 32)
        from segment_anything.modeling.mask_decoder import MLP
        mask_decoder.output_hypernetworks_mlps = torch.nn.ModuleList([
            MLP(transformer_dim, transformer_dim, hypernet_out_dim, 3)
            for _ in range(num_mask_tokens)
        ])
        mask_decoder.output_hypernetworks_mlps = mask_decoder.output_hypernetworks_mlps.to(next(net.parameters()).device)

    if args.lora_ckpt is not None:
        net.load_lora_parameters(args.lora_ckpt)

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False

    low_res = img_embedding_size * 4

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    # Set up transforms for each dataset
    if args.dataset == "Thyroid":
        from datasets.thyroid_transforms import ThyroidRandomGenerator
        use_spectral_prompt = args.spectral_prompt or args.spectral_prompt_cross_attention
        train_transform = ThyroidRandomGenerator(output_size=(args.img_size, args.img_size), num_classes=args.num_classes)
        val_transform = ThyroidRandomGenerator(output_size=(args.img_size, args.img_size), num_classes=args.num_classes)
        from datasets.dataset_thyroid import ThyroidDataset as DatasetClass
        train_dataset = DatasetClass(args.root_path, transform=train_transform, spectral_prompt=use_spectral_prompt)
        val_dataset = DatasetClass(args.root_path, split="val", transform=val_transform, spectral_prompt=use_spectral_prompt)
        test_dataset = DatasetClass(args.root_path, split="test", transform=val_transform, spectral_prompt=use_spectral_prompt)
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    elif args.dataset == "SegThy":
        from datasets.dataset_thyroid import ThyroidDataset as DatasetClass
        train_dataset = DatasetClass(args.root_path, split="train", transform=train_transform, spectral_prompt=args.spectral_prompt)
        val_dataset = DatasetClass(args.root_path, split="val", transform=val_transform, spectral_prompt=args.spectral_prompt)
        test_dataset = DatasetClass(args.root_path, split="test", transform=val_transform, spectral_prompt=args.spectral_prompt)
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    elif args.dataset == "BUSI":
        from datasets.busi_transforms import BUSIRandomGenerator
        train_transform = BUSIRandomGenerator()
        val_transform = BUSIRandomGenerator()
        from datasets.dataset_busi import BUSIDataset as DatasetClass
        train_dataset = DatasetClass(args.root_path, split="train", transform=train_transform, spectral_prompt=args.spectral_prompt)
        val_dataset = DatasetClass(args.root_path, split="val", transform=val_transform, spectral_prompt=args.spectral_prompt)
        test_dataset = DatasetClass(args.root_path, split="test", transform=val_transform, spectral_prompt=args.spectral_prompt)
        trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        valloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
        testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    else:
        from torchvision import transforms
        train_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    # Import trainers
    from trainer_thyroid import trainer_thyroid
    if args.dataset == "BUSI":
        from trainer_busi import trainer_busi

    # Define trainer functions for each dataset
    trainer = {
        "BUSI": trainer_busi if args.dataset == "BUSI" else None,
        "Thyroid": trainer_thyroid
    }

    # Determine prompt type
    prompt_type = "no_prompt"
    if args.spectral_prompt:
        prompt_type = "spectral_prompt"
    elif args.spectral_prompt_cross_attention:
        prompt_type = "spectral_prompt_cross_attention"
    
    # Call the appropriate trainer with the prompt type
    if dataset_name == "Thyroid":
        trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res, prompt_type, trainloader, valloader)
    elif dataset_name == "BUSI":
        trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res, trainloader, valloader, testloader)
    elif dataset_name == "SegThy":
        trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res, trainloader, valloader, testloader)
    else:
        trainer[dataset_name](args, net, snapshot_path, multimask_output, low_res, trainloader, valloader)

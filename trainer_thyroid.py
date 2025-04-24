import logging
import os
import sys
import torch
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from datasets.dataset_thyroid import ThyroidDataset
from datasets.thyroid_transforms import ThyroidRandomGenerator
from utils import DiceLoss, Focal_loss
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def dice_coefficient(pred, target, num_classes):
    # pred, target: (N, H, W)
    dice = 0.0
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        if union == 0:
            dice += 1  # perfect if no pixels for this class
        else:
            dice += (2. * intersection) / union
    return dice / (num_classes - 1)

def iou_coefficient(pred, target, num_classes):
    # pred, target: (N, H, W)
    iou = 0.0
    for cls in range(1, num_classes):  # skip background
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union == 0:
            iou += 1  # perfect if no pixels for this class
        else:
            iou += intersection / union
    return iou / (num_classes - 1)

def visualize_predictions(model, val_loader, writer, epoch, args, prompt_type="no_prompt"):
    """Visualize predictions for a fixed validation image (same across all runs)."""
    import matplotlib.pyplot as plt
    model.eval()
    # Always use the same validation image for all runs
    fixed_index = 0
    if hasattr(val_loader.dataset, '__getitem__'):
        sample = val_loader.dataset[fixed_index]
        if isinstance(sample, dict):
            image = sample['image'].unsqueeze(0).cuda()
            label = sample['label'].unsqueeze(0).cuda()
            segmap = sample.get('spectral_mask', None)
            if segmap is not None:
                segmap = segmap.unsqueeze(0).cuda()
        else:
            image = sample[0].unsqueeze(0).cuda()
            label = sample[1].unsqueeze(0).cuda()
            segmap = sample[2].unsqueeze(0).cuda() if len(sample) > 2 else None
    else:
        sampled_batch = next(iter(val_loader))
        if args.spectral_prompt or args.spectral_prompt_cross_attention:
            if isinstance(sampled_batch, dict):
                image = sampled_batch['image'].cuda()
                label = sampled_batch['label'].cuda()
                segmap = sampled_batch.get('spectral_mask', None)
                if segmap is not None:
                    segmap = segmap.cuda()
            else:
                if len(sampled_batch) > 2:
                    image, label, segmap = sampled_batch
                    image = image.cuda()
                    label = label.cuda()
                    segmap = segmap.cuda()
                else:
                    image, label = sampled_batch
                    image = image.cuda()
                    label = label.cuda()
                    segmap = None
        else:
            if isinstance(sampled_batch, dict):
                image = sampled_batch['image'].cuda()
                label = sampled_batch['label'].cuda()
            else:
                image, label = sampled_batch
                image = image.cuda()
                label = label.cuda()
            segmap = None
    if args.spectral_prompt or args.spectral_prompt_cross_attention:
        outputs = model(image, False, image.shape[-2:], spectral_mask=segmap)
    else:
        outputs = model(image, False, image.shape[-2:])
    if isinstance(outputs, dict):
        masks = outputs['masks'][:, :args.num_classes]
    elif isinstance(outputs, (tuple, list)):
        masks = outputs[0][:, :args.num_classes]
    else:
        masks = outputs[:, :args.num_classes]
    pred = torch.argmax(torch.softmax(masks, dim=1), dim=1)
    # Prepare images for logging
    if image.shape[1] == 1:
        image_np = image[0, 0].detach().cpu().numpy()
    elif image.shape[1] == 3:
        image_np = image[0].detach().cpu().numpy().transpose(1, 2, 0)
    else:
        image_np = image[0, 0].detach().cpu().numpy()  # fallback to first channel
    label_np = label[0].detach().cpu().numpy()
    pred_np = pred[0].detach().cpu().numpy()
    # Plot 3-column figure: image, ground truth (multi-class), prediction (multi-class)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image_np, cmap='gray' if image.shape[1] == 1 else None)
    ax[0].set_title('Input')
    ax[0].axis('off')
    ax[1].imshow(label_np, cmap='tab10', vmin=0, vmax=args.num_classes-1)
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')
    ax[2].imshow(pred_np, cmap='tab10', vmin=0, vmax=args.num_classes-1)
    ax[2].set_title('Prediction')
    ax[2].axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    fig_np = np.array(fig.canvas.renderer.buffer_rgba())
    writer.add_image(f'predictions/example', fig_np, epoch, dataformats='HWC')
    plt.close(fig)
    # Optionally, return metrics for logging (if used elsewhere)
    dice = dice_coefficient(torch.tensor(pred_np), torch.tensor(label_np), args.num_classes)
    iou = iou_coefficient(torch.tensor(pred_np), torch.tensor(label_np), args.num_classes)
    return dice, iou

def visualize_predictions_grid(models, val_loader, writer, epoch, args):
    """Visualize predictions for a fixed set of validation images using all prompt types in a single grid."""
    # models: dict with keys 'no_prompt', 'spectral_prompt', 'spectral_prompt_cross_attention'
    model_modes = ['no_prompt', 'spectral_prompt', 'spectral_prompt_cross_attention']
    preds = []
    image_np = None
    label_np = None
    for mode in model_modes:
        # Get a fresh batch for each mode to ensure correct channel count
        sampled_batch = next(iter(val_loader))
        if isinstance(sampled_batch, dict):
            image = sampled_batch['image'][0:1].cuda()
            label = sampled_batch['label'][0:1].cuda()
            segmap = sampled_batch.get('spectral_mask', None)
            if segmap is not None:
                segmap = segmap[0:1].cuda()
        else:
            if len(sampled_batch) > 2:
                image, label, segmap = sampled_batch
                image = image[0:1].cuda()
                label = label[0:1].cuda()
                segmap = segmap[0:1].cuda()
            else:
                image, label = sampled_batch
                image = image[0:1].cuda()
                label = label[0:1].cuda()
                segmap = None
        with torch.no_grad():
            model = models[mode]
            use_cross_attention = hasattr(model, "cross_attn") if mode == 'spectral_prompt_cross_attention' else False
            if mode == 'no_prompt':
                outputs = model(image, True, image.shape[-2:])
            elif mode == 'spectral_prompt':
                outputs = model(image, True, image.shape[-2:], spectral_mask=segmap)
            elif mode == 'spectral_prompt_cross_attention':
                outputs = model(image, True, image.shape[-2:], spectral_mask=segmap, use_cross_attention=True)
            # Extract masks
            if isinstance(outputs, dict):
                masks = outputs['masks'][:, :args.num_classes]
            elif isinstance(outputs, (tuple, list)):
                masks = outputs[0][:, :args.num_classes]
            else:
                masks = outputs[:, :args.num_classes]
            
            # Get prediction
            pred = torch.argmax(torch.softmax(masks, dim=1), dim=1)
            preds.append(pred[0].cpu().numpy())
        # Save the image and label from the first mode for visualization
        if image_np is None:
            # For grayscale input, take first channel
            image_np = image[0, 0].cpu().numpy()
            label_np = label[0].cpu().numpy()
    # Create 5-column figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    ax[0].imshow(image_np, cmap='gray'); ax[0].set_title('Input Image'); ax[0].axis('off')
    ax[1].imshow(label_np, cmap='viridis'); ax[1].set_title('Ground Truth'); ax[1].axis('off')
    ax[2].imshow(preds[0], cmap='viridis'); ax[2].set_title('No Prompt'); ax[2].axis('off')
    ax[3].imshow(preds[1], cmap='viridis'); ax[3].set_title('Spectral Prompt'); ax[3].axis('off')
    ax[4].imshow(preds[2], cmap='viridis'); ax[4].set_title('Cross Attention'); ax[4].axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    fig_np = np.array(fig.canvas.renderer.buffer_rgba())
    writer.add_image('predictions/grid', fig_np, epoch, dataformats='HWC')
    plt.close(fig)

def trainer_thyroid(args, model, snapshot_path, multimask_output, low_res, prompt_type="no_prompt", train_loader=None, val_loader=None, test_loader=None):
    logging.basicConfig(filename=os.path.join(args.output, 'train.log'), level=logging.INFO)
    writer = SummaryWriter(log_dir=os.path.join(args.output, 'tensorboard'))

    dice_loss = DiceLoss(args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
    # Add learning rate scheduler: drop LR by 10x after 30 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    max_epochs = 100
    best_val_dice = 0.0
    best_val_iou = 0.0
    best_model_path = os.path.join(snapshot_path, 'best_model.pth')
    for epoch_num in range(max_epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_dice_sum = 0.0
        epoch_iou_sum = 0.0
        epoch_batches = 0
        for i_batch, sampled_batch in enumerate(train_loader):
            # Robust batch extraction for both dict and tuple/list
            if isinstance(sampled_batch, dict):
                image_batch = sampled_batch['image'].cuda()
                label_batch = sampled_batch['label'].cuda()
                segmap_batch = sampled_batch.get('spectral_mask', None)
                if segmap_batch is not None:
                    segmap_batch = segmap_batch.cuda()
            else:
                if args.spectral_prompt or args.spectral_prompt_cross_attention:
                    image_batch = sampled_batch[0].cuda()
                    label_batch = sampled_batch[1].cuda()
                    segmap_batch = sampled_batch[2].cuda() if len(sampled_batch) > 2 else None
                else:
                    image_batch = sampled_batch[0].cuda()
                    label_batch = sampled_batch[1].cuda()
                    segmap_batch = None
            if args.spectral_prompt or args.spectral_prompt_cross_attention:
                outputs = model(image_batch, multimask_output, image_batch.shape[-2:], spectral_mask=segmap_batch)
            else:
                outputs = model(image_batch, multimask_output, image_batch.shape[-2:])
            if isinstance(outputs, dict):
                masks = outputs['masks'][:, :args.num_classes]
            elif isinstance(outputs, (tuple, list)):
                masks = outputs[0][:, :args.num_classes]
            else:
                masks = outputs[:, :args.num_classes]
            # Robust label handling (from BUSI)
            if label_batch.dim() == 4 and label_batch.size(1) == 1:
                label_batch = label_batch.squeeze(1)
            elif label_batch.dim() > 3:
                if label_batch.size(1) == args.num_classes:
                    label_batch = label_batch.argmax(dim=1)
                else:
                    label_batch = label_batch.view(label_batch.size(0), label_batch.size(-2), label_batch.size(-1))
            ce_loss = torch.nn.CrossEntropyLoss()
            loss = ce_loss(masks, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pred = torch.argmax(torch.softmax(masks, dim=1), dim=1)
                dice_score = dice_coefficient(pred, label_batch, args.num_classes)
                iou_score = iou_coefficient(pred, label_batch, args.num_classes)
            writer.add_scalar('train/loss', loss.item(), epoch_num * len(train_loader) + i_batch)
            writer.add_scalar('train/dice', dice_score, epoch_num * len(train_loader) + i_batch)
            writer.add_scalar('train/iou', iou_score, epoch_num * len(train_loader) + i_batch)
            epoch_loss_sum += loss.item()
            epoch_dice_sum += dice_score
            epoch_iou_sum += iou_score
            epoch_batches += 1
        epoch_loss = epoch_loss_sum / epoch_batches if epoch_batches > 0 else 0
        epoch_dice = epoch_dice_sum / epoch_batches if epoch_batches > 0 else 0
        epoch_iou = epoch_iou_sum / epoch_batches if epoch_batches > 0 else 0
        writer.add_scalar('train/epoch_loss', epoch_loss, epoch_num)
        writer.add_scalar('train/epoch_dice', epoch_dice, epoch_num)
        writer.add_scalar('train/epoch_iou', epoch_iou, epoch_num)
        logging.info(f"[Epoch {epoch_num}] Loss: {epoch_loss:.4f}, Dice: {epoch_dice:.4f}, IoU: {epoch_iou:.4f}")
        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in val_loader:
                if isinstance(val_batch, dict):
                    val_image = val_batch['image'].cuda()
                    val_label = val_batch['label'].cuda()
                    val_segmap = val_batch.get('spectral_mask', None)
                    if val_segmap is not None:
                        val_segmap = val_segmap.cuda()
                else:
                    if args.spectral_prompt or args.spectral_prompt_cross_attention:
                        val_image = val_batch[0].cuda()
                        val_label = val_batch[1].cuda()
                        val_segmap = val_batch[2].cuda() if len(val_batch) > 2 else None
                    else:
                        val_image = val_batch[0].cuda()
                        val_label = val_batch[1].cuda()
                        val_segmap = None
                if args.spectral_prompt or args.spectral_prompt_cross_attention:
                    val_outputs = model(val_image, multimask_output, val_image.shape[-2:], spectral_mask=val_segmap)
                else:
                    val_outputs = model(val_image, multimask_output, val_image.shape[-2:])
                if isinstance(val_outputs, dict):
                    val_masks = val_outputs['masks'][:, :args.num_classes]
                elif isinstance(val_outputs, (tuple, list)):
                    val_masks = val_outputs[0][:, :args.num_classes]
                else:
                    val_masks = val_outputs[:, :args.num_classes]
                if val_label.dim() == 4 and val_label.size(1) == 1:
                    val_label = val_label.squeeze(1)
                elif val_label.dim() > 3:
                    if val_label.size(1) == args.num_classes:
                        val_label = val_label.argmax(dim=1)
                    else:
                        val_label = val_label.view(val_label.size(0), val_label.size(-2), val_label.size(-1))
                val_loss = ce_loss(val_masks, val_label)
                val_pred = torch.argmax(torch.softmax(val_masks, dim=1), dim=1)
                val_dice = dice_coefficient(val_pred, val_label, args.num_classes)
                val_iou = iou_coefficient(val_pred, val_label, args.num_classes)
                val_loss_sum += val_loss.item()
                val_dice_sum += val_dice
                val_iou_sum += val_iou
                val_batches += 1
        avg_val_loss = val_loss_sum / val_batches if val_batches > 0 else 0
        avg_val_dice = val_dice_sum / val_batches if val_batches > 0 else 0
        avg_val_iou = val_iou_sum / val_batches if val_batches > 0 else 0
        writer.add_scalar('val/loss_total', avg_val_loss, epoch_num)
        writer.add_scalar('val/dice', avg_val_dice, epoch_num)
        writer.add_scalar('val/iou', avg_val_iou, epoch_num)
        logging.info(f"[Val | Epoch {epoch_num}] Loss: {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}, IoU: {avg_val_iou:.4f}")
        # Save best model based on validation dice
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            best_val_iou = avg_val_iou
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"[Epoch {epoch_num}] New best model saved with Dice: {best_val_dice:.4f}, IoU: {best_val_iou:.4f}")
        # Visualize predictions every 5 epochs
        if (epoch_num + 1) % 5 == 0 or epoch_num == 0:
            visualize_predictions(model, val_loader, writer, epoch_num, args, prompt_type=prompt_type)
        # Save model checkpoint every 20 epochs
        save_interval = 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save(save_mode_path)
            except:
                torch.save(model.state_dict(), save_mode_path)
        # Step the LR scheduler
        scheduler.step()
    # --- Test Loop at End ---
    if test_loader is not None:
        model.eval()
        test_loss_sum = 0.0
        test_dice_sum = 0.0
        test_iou_sum = 0.0
        test_batches = 0
        with torch.no_grad():
            for test_batch in test_loader:
                if isinstance(test_batch, dict):
                    test_image = test_batch['image'].cuda()
                    test_label = test_batch['label'].cuda()
                    test_segmap = test_batch.get('spectral_mask', None)
                    if test_segmap is not None:
                        test_segmap = test_segmap.cuda()
                else:
                    if args.spectral_prompt or args.spectral_prompt_cross_attention:
                        test_image = test_batch[0].cuda()
                        test_label = test_batch[1].cuda()
                        test_segmap = test_batch[2].cuda() if len(test_batch) > 2 else None
                    else:
                        test_image = test_batch[0].cuda()
                        test_label = test_batch[1].cuda()
                        test_segmap = None
                if args.spectral_prompt or args.spectral_prompt_cross_attention:
                    test_outputs = model(test_image, multimask_output, test_image.shape[-2:], spectral_mask=test_segmap)
                else:
                    test_outputs = model(test_image, multimask_output, test_image.shape[-2:])
                if isinstance(test_outputs, dict):
                    test_masks = test_outputs['masks'][:, :args.num_classes]
                elif isinstance(test_outputs, (tuple, list)):
                    test_masks = test_outputs[0][:, :args.num_classes]
                else:
                    test_masks = test_outputs[:, :args.num_classes]
                if test_label.dim() == 4 and test_label.size(1) == 1:
                    test_label = test_label.squeeze(1)
                elif test_label.dim() > 3:
                    if test_label.size(1) == args.num_classes:
                        test_label = test_label.argmax(dim=1)
                    else:
                        test_label = test_label.view(test_label.size(0), test_label.size(-2), test_label.size(-1))
                test_loss = ce_loss(test_masks, test_label)
                test_pred = torch.argmax(torch.softmax(test_masks, dim=1), dim=1)
                test_dice = dice_coefficient(test_pred, test_label, args.num_classes)
                test_iou = iou_coefficient(test_pred, test_label, args.num_classes)
                test_loss_sum += test_loss.item()
                test_dice_sum += test_dice
                test_iou_sum += test_iou
                test_batches += 1
        avg_test_loss = test_loss_sum / test_batches if test_batches > 0 else 0
        avg_test_dice = test_dice_sum / test_batches if test_batches > 0 else 0
        avg_test_iou = test_iou_sum / test_batches if test_batches > 0 else 0
        writer.add_scalar('test/loss_total', avg_test_loss, max_epochs)
        writer.add_scalar('test/dice', avg_test_dice, max_epochs)
        writer.add_scalar('test/iou', avg_test_iou, max_epochs)
        logging.info(f"[Test Results] Loss: {avg_test_loss:.4f}, Dice: {avg_test_dice:.4f}, IoU: {avg_test_iou:.4f}")
    writer.close()
    logging.info('Training finished for Thyroid dataset.')
    logging.info(f'Best validation Dice: {best_val_dice:.4f}, IoU: {best_val_iou:.4f}. Best model saved at: {best_model_path}')
    return model

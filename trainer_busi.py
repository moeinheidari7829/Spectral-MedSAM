import logging
import os
import torch
from tensorboardX import SummaryWriter
from utils import DiceLoss, Focal_loss
from torchvision import transforms
import numpy as np

def dice_coefficient(pred, target, num_classes):
    dice = 0.0
    for cls in range(1, num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        if union == 0:
            dice += 1
        else:
            dice += (2. * intersection) / union
    return dice / (num_classes - 1)

def iou_coefficient(pred, target, num_classes):
    iou = 0.0
    for cls in range(1, num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union == 0:
            iou += 1
        else:
            iou += intersection / union
    return iou / (num_classes - 1)

def visualize_predictions_busi(model, val_loader, writer, epoch, args):
    """Visualize predictions for a fixed set of validation images in BUSI."""
    model.eval()
    sampled_batch = next(iter(val_loader))
    if args.spectral_prompt:
        image = sampled_batch[0].cuda()
        label = sampled_batch[1].cuda()
        segmap = sampled_batch[2].cuda()
        outputs = model(image, False, image.shape[-2:], spectral_mask=segmap)
    else:
        image = sampled_batch[0].cuda() if isinstance(sampled_batch, (list, tuple)) else sampled_batch['image'].cuda()
        label = sampled_batch[1].cuda() if isinstance(sampled_batch, (list, tuple)) else sampled_batch['label'].cuda()
        outputs = model(image, False, image.shape[-2:])
    if isinstance(outputs, dict):
        masks = outputs['masks'][:, :args.num_classes]
    elif isinstance(outputs, (tuple, list)):
        masks = outputs[0][:, :args.num_classes]
    else:
        masks = outputs[:, :args.num_classes]
    pred = torch.argmax(torch.softmax(masks, dim=1), dim=1)
    image_np = image[0, 0].detach().cpu().numpy() if image.shape[1] == 1 else image[0].detach().cpu().numpy().transpose(1, 2, 0)
    label_np = label[0].detach().cpu().numpy()
    pred_np = pred[0].detach().cpu().numpy()
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(image_np, cmap='gray' if image.shape[1] == 1 else None)
    ax[0].set_title('Input')
    ax[0].axis('off')
    ax[1].imshow(label_np, cmap='viridis')
    ax[1].set_title('Ground Truth')
    ax[1].axis('off')
    ax[2].imshow(pred_np, cmap='viridis')
    ax[2].set_title('Prediction')
    ax[2].axis('off')
    fig.tight_layout()
    fig.canvas.draw()
    fig_np = np.array(fig.canvas.renderer.buffer_rgba())
    writer.add_image('predictions_busi/example', fig_np, epoch, dataformats='HWC')
    plt.close(fig)

def trainer_busi(args, model, snapshot_path, multimask_output, low_res, train_loader, val_loader, test_loader):
    logging.basicConfig(filename=os.path.join(args.output, 'train.log'), level=logging.INFO)
    writer = SummaryWriter(log_dir=os.path.join(args.output, 'tensorboard'))

    dice_loss = DiceLoss(args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)

    max_epochs = 100
    for epoch_num in range(max_epochs):
        model.train()
        epoch_dice_loss_sum = 0.0
        epoch_dice_score_sum = 0.0
        epoch_iou_score_sum = 0.0
        epoch_batches = 0
        for i_batch, sampled_batch in enumerate(train_loader):
            if args.spectral_prompt:
                image_batch, label_batch, segmap_batch = sampled_batch
            else:
                image_batch, label_batch = sampled_batch
            image_batch = image_batch.cuda()
            label_batch = label_batch.cuda()
            outputs = model(image_batch, multimask_output, image_batch.shape[-2:], spectral_mask=segmap_batch.cuda() if args.spectral_prompt else None)
            if isinstance(outputs, dict):
                masks = outputs['masks'][:, :args.num_classes]
            elif isinstance(outputs, (tuple, list)):
                masks = outputs[0][:, :args.num_classes]
            else:
                masks = outputs[:, :args.num_classes]
            if label_batch.dim() == 4 and label_batch.size(1) == 1:
                label_batch = label_batch.squeeze(1)
            elif label_batch.dim() > 3:
                if label_batch.size(1) == 2:
                    label_batch = label_batch.argmax(dim=1)
                else:
                    label_batch = label_batch.view(label_batch.size(0), label_batch.size(-2), label_batch.size(-1))
            loss_dice = dice_loss(masks, label_batch)
            loss = loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss_total', loss.item(), epoch_num * len(train_loader) + i_batch)
            writer.add_scalar('train/loss_dice', loss_dice.item(), epoch_num * len(train_loader) + i_batch)
            with torch.no_grad():
                pred = torch.argmax(torch.softmax(masks, dim=1), dim=1)
                dice_score = dice_coefficient(pred, label_batch, num_classes=args.num_classes)
                iou_score = iou_coefficient(pred, label_batch, num_classes=args.num_classes)
                epoch_dice_score_sum += dice_score
                epoch_iou_score_sum += iou_score
            epoch_dice_loss_sum += loss_dice.item()
            epoch_batches += 1
        avg_dice = epoch_dice_score_sum / epoch_batches if epoch_batches > 0 else 0
        avg_loss = epoch_dice_loss_sum / epoch_batches if epoch_batches > 0 else 0
        avg_iou = epoch_iou_score_sum / epoch_batches if epoch_batches > 0 else 0
        writer.add_scalar('train/epoch_dice', avg_dice, epoch_num)
        writer.add_scalar('train/epoch_loss', avg_loss, epoch_num)
        writer.add_scalar('train/epoch_iou', avg_iou, epoch_num)

        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_dice_sum = 0.0
            val_iou_sum = 0.0
            val_batches = 0
            val_loss_fn = DiceLoss(args.num_classes)
            with torch.no_grad():
                for val_batch in val_loader:
                    if args.spectral_prompt:
                        val_image, val_label, val_segmap = val_batch
                        val_image = val_image.cuda()
                        val_label = val_label.cuda()
                        val_segmap = val_segmap.cuda()
                        val_outputs = model(val_image, multimask_output, val_image.shape[-2:], spectral_mask=val_segmap)
                    else:
                        val_image, val_label = val_batch
                        val_image = val_image.cuda()
                        val_label = val_label.cuda()
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
                        if val_label.size(1) == 2:
                            val_label = val_label.argmax(dim=1)
                        else:
                            val_label = val_label.view(val_label.size(0), val_label.size(-2), val_label.size(-1))
                    val_loss = val_loss_fn(val_masks, val_label)
                    val_pred = torch.argmax(torch.softmax(val_masks, dim=1), dim=1)
                    val_dice = dice_coefficient(val_pred, val_label, num_classes=args.num_classes)
                    val_iou = iou_coefficient(val_pred, val_label, num_classes=args.num_classes)
                    val_loss_sum += val_loss.item()
                    val_dice_sum += val_dice
                    val_iou_sum += val_iou
                    val_batches += 1
            avg_val_loss = val_loss_sum / val_batches
            avg_val_dice = val_dice_sum / val_batches
            avg_val_iou = val_iou_sum / val_batches
            writer.add_scalar('val/loss_total', avg_val_loss, epoch_num)
            writer.add_scalar('val/dice', avg_val_dice, epoch_num)
            writer.add_scalar('val/iou', avg_val_iou, epoch_num)
            print(f"Epoch {epoch_num+1}/{max_epochs} - Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}, Val IoU: {avg_val_iou:.4f}")

        if (epoch_num + 1) % 5 == 0 or epoch_num == 0:
            visualize_predictions_busi(model, val_loader, writer, epoch_num, args)
        print(f'Epoch {epoch_num+1}/{max_epochs} completed. '
              f'Avg Dice: {avg_dice:.4f}, Avg IoU: {avg_iou:.4f}, Avg Loss: {avg_loss:.4f}')
        save_interval = 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save(save_mode_path)
            except:
                torch.save(model.state_dict(), save_mode_path)
    # --- Test Loop at End ---
    if test_loader is not None:
        model.eval()
        test_loss_sum = 0.0
        test_dice_sum = 0.0
        test_iou_sum = 0.0
        test_batches = 0
        test_loss_fn = DiceLoss(args.num_classes)
        with torch.no_grad():
            for test_batch in test_loader:
                if args.spectral_prompt:
                    test_image, test_label, test_segmap = test_batch
                    test_image = test_image.cuda()
                    test_label = test_label.cuda()
                    test_segmap = test_segmap.cuda()
                    test_outputs = model(test_image, multimask_output, test_image.shape[-2:], spectral_mask=test_segmap)
                else:
                    test_image, test_label = test_batch
                    test_image = test_image.cuda()
                    test_label = test_label.cuda()
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
                    if test_label.size(1) == 2:
                        test_label = test_label.argmax(dim=1)
                    else:
                        test_label = test_label.view(test_label.size(0), test_label.size(-2), test_label.size(-1))
                test_loss = test_loss_fn(test_masks, test_label)
                test_pred = torch.argmax(torch.softmax(test_masks, dim=1), dim=1)
                test_dice = dice_coefficient(test_pred, test_label, num_classes=args.num_classes)
                test_iou = iou_coefficient(test_pred, test_label, num_classes=args.num_classes)
                test_loss_sum += test_loss.item()
                test_dice_sum += test_dice
                test_iou_sum += test_iou
                test_batches += 1
        avg_test_loss = test_loss_sum / test_batches
        avg_test_dice = test_dice_sum / test_batches
        avg_test_iou = test_iou_sum / test_batches
        writer.add_scalar('test/loss_total', avg_test_loss, max_epochs)
        writer.add_scalar('test/dice', avg_test_dice, max_epochs)
        writer.add_scalar('test/iou', avg_test_iou, max_epochs)
        print(f"Test Results - Loss: {avg_test_loss:.4f}, Dice: {avg_test_dice:.4f}, IoU: {avg_test_iou:.4f}")
    writer.close()
    logging.info('Training finished for BUSI dataset.')

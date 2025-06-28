import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import cfg
from datasets.pose3d_dataset import Pose3DDataset
from models.hrnet_pose3d import HRNet3DPose
from utils.freeze_utils import freeze_hrnet_stages
import numpy as np
from torchvision.utils import make_grid

def mpjpe(preds, targets):
    """Mean Per Joint Position Error (MPJPE) in Euclidean distance"""
    return torch.mean(torch.norm(preds - targets, dim=2))

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    running_mpjpe = 0.0

    for i, (images, joints) in enumerate(loader):
        images = images.to(device)
        joints = joints.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, joints)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_mpjpe += mpjpe(outputs, joints).item()

        if i % 10 == 0:
            print(f"Epoch [{epoch}] Step [{i}/{len(loader)}] Loss: {loss.item():.4f} MPJPE: {running_mpjpe/(i+1):.4f}")

    avg_loss = running_loss / len(loader)
    avg_mpjpe = running_mpjpe / len(loader)

    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/MPJPE', avg_mpjpe, epoch)

    # Log a grid of input images (first batch)
    imgs_grid = make_grid(images.cpu(), normalize=True, scale_each=True)
    writer.add_image('Train/InputImages', imgs_grid, epoch)

    return avg_loss, avg_mpjpe

def validate(model, loader, criterion, device, epoch, writer):
    model.eval()
    val_loss = 0.0
    val_mpjpe = 0.0

    with torch.no_grad():
        for i, (images, joints) in enumerate(loader):
            images = images.to(device)
            joints = joints.to(device)

            outputs = model(images)
            loss = criterion(outputs, joints)

            val_loss += loss.item()
            val_mpjpe += mpjpe(outputs, joints).item()

    avg_loss = val_loss / len(loader)
    avg_mpjpe = val_mpjpe / len(loader)

    print(f"Validation Epoch [{epoch}] Loss: {avg_loss:.4f} MPJPE: {avg_mpjpe:.4f}")

    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/MPJPE', avg_mpjpe, epoch)

    return avg_loss, avg_mpjpe

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='path to config yaml')
    parser.add_argument('--logdir', type=str, default='experiments/logs', help='TensorBoard log directory')
    args = parser.parse_args()

    # Load config
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & Dataloader
    train_dataset = Pose3DDataset(os.path.join(cfg.DATASET.ROOT, 'train'))
    val_dataset = Pose3DDataset(os.path.join(cfg.DATASET.ROOT, 'val'))

    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.IMAGES_PER_GPU, shuffle=True, num_workers=cfg.WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TEST.IMAGES_PER_GPU, shuffle=False, num_workers=cfg.WORKERS)

    # Model
    model = HRNet3DPose(cfg, output_joints=21, feature_dim=2048).to(device)

    # Freeze HRNet backbone stages as per config; MLP always trainable
    freeze_hrnet_stages(model, train_stages=cfg.TRAIN.TRAINABLE_STAGES)

    # Optimizer (only trainable parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.TRAIN.LR,
        weight_decay=cfg.TRAIN.WD
    )

    criterion = nn.MSELoss()

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.logdir)

    best_val_mpjpe = float('inf')

    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        start_time = time.time()

        train_loss, train_mpjpe = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss, val_mpjpe = validate(model, val_loader, criterion, device, epoch, writer)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch} done in {elapsed:.1f}s")

        # Save best model
        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            save_path = os.path.join(args.logdir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with MPJPE {best_val_mpjpe:.4f}")

    writer.close()

if __name__ == '__main__':
    main()

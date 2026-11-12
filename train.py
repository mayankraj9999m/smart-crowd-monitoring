"""
train.py
Training pipeline for CSRNet on ShanghaiTech dataset.
Tracks Mean Absolute Error (MAE) and Mean Squared Error (MSE), saving the best checkpoint.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import CSRNet
from dataset import CrowdDataset


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    mae = 0.0
    mse = 0.0
    
    with torch.no_grad():
        for img, gt_density, _ in val_loader:
            img = img.to(device)
            gt_density = gt_density.to(device)
            
            pred_density = model(img)
            loss = criterion(pred_density, gt_density)
            val_loss += loss.item()
            
            # Count is the sum of density map values
            pred_count = pred_density.sum().item()
            gt_count = gt_density.sum().item()
            
            diff = abs(pred_count - gt_count)
            mae += diff
            mse += diff ** 2
            
    num_samples = len(val_loader)
    val_loss /= num_samples
    mae /= num_samples
    mse = np.sqrt(mse / num_samples)
    
    return val_loss, mae, mse


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training CSRNet on {args.dataset} ---")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    train_dir = os.path.join("ShanghaiTech", args.dataset, "train_data")
    val_dir = os.path.join("ShanghaiTech", args.dataset, "test_data")
    
    train_dataset = CrowdDataset(train_dir, is_train=True)
    val_dataset = CrowdDataset(val_dir, is_train=False)
    
    # Batch size 1 because images have variable resolutions
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    
    # Initialize CSRNet model
    model = CSRNet(pretrained_frontend=args.pretrained).to(device)
    
    criterion = nn.MSELoss(reduction="sum").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_mae = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for step, (img, gt_density, _) in enumerate(train_loader):
            img = img.to(device)
            gt_density = gt_density.to(device)
            
            optimizer.zero_grad()
            pred_density = model(img)
            loss = criterion(pred_density, gt_density)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        epoch_loss /= len(train_loader)
        scheduler.step()
        elapsed = time.time() - start_time
        
        # Validation every eval_freq epochs
        if epoch % args.eval_freq == 0:
            val_loss, val_mae, val_mse = validate(model, val_loader, criterion, device)
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] ({elapsed:.1f}s) | "
                  f"Train Loss: {epoch_loss:.2f} | Val Loss: {val_loss:.2f} | "
                  f"Val MAE: {val_mae:.2f} | Val MSE: {val_mse:.2f}")
            
            # Save best checkpoint
            if val_mae < best_mae:
                best_mae = val_mae
                best_path = os.path.join(args.save_dir, f"best_model_{args.dataset}.pth")
                torch.save(model.state_dict(), best_path)
                print(f"--> Saved New Best Model Checkpoint (MAE: {best_mae:.2f}) to {best_path}")
        else:
            print(f"Epoch [{epoch:03d}/{args.epochs:03d}] ({elapsed:.1f}s) | Train Loss: {epoch_loss:.2f}")

    print("\nTraining complete! Best Validation MAE:", best_mae)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSRNet on ShanghaiTech")
    parser.add_argument("--dataset", type=str, default="part_B", choices=["part_A", "part_B"],
                        help="ShanghaiTech subset (part_A: dense, part_B: surveillance)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--eval_freq", type=int, default=1, help="Validation frequency (epochs)")
    parser.add_argument("--pretrained", action="store_true", default=True, help="Use ImageNet pretrained VGG-16")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model weights")
    args = parser.parse_args()
    
    train(args)


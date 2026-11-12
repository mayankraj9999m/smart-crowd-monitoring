"""
evaluate.py
Evaluates trained CSRNet checkpoints on the ShanghaiTech test split.
Computes MAE, MSE, and saves qualitative density heatmap comparisons.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CSRNet
from dataset import CrowdDataset


def evaluate_model(checkpoint_path, dataset_type="part_B", output_dir="eval_results", num_vis=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating CSRNet on ShanghaiTech {dataset_type} ---")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    
    val_dir = os.path.join("ShanghaiTech", dataset_type, "test_data")
    val_dataset = CrowdDataset(val_dir, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    model = CSRNet(pretrained_frontend=False).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    mae = 0.0
    mse = 0.0
    records = []
    
    with torch.no_grad():
        for i, (img, gt_density, img_path) in enumerate(tqdm(val_loader, desc="Evaluating")):
            img = img.to(device)
            pred_density = model(img)
            
            pred_count = pred_density.sum().item()
            gt_count = gt_density.sum().item()
            diff = abs(pred_count - gt_count)
            
            mae += diff
            mse += diff ** 2
            
            records.append({
                "path": img_path[0],
                "gt": gt_count,
                "pred": pred_count,
                "diff": diff
            })
            
            # Save visual comparisons for first num_vis samples
            if i < num_vis:
                # Load original image for visualization
                raw_img = cv2.imread(img_path[0])
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                
                gt_map = gt_density.squeeze().cpu().numpy()
                pred_map = pred_density.squeeze().cpu().numpy()
                
                # Resize pred_map and gt_map to match original image size for display
                H, W = raw_img.shape[:2]
                pred_vis = cv2.resize(pred_map, (W, H))
                gt_vis = cv2.resize(gt_map, (W, H))
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                
                axes[0].imshow(raw_img)
                axes[0].set_title(f"Input: {os.path.basename(img_path[0])}\nActual Count: {gt_count:.1f}", fontsize=12)
                axes[0].axis("off")
                
                im1 = axes[1].imshow(gt_vis, cmap="jet")
                axes[1].set_title(f"Ground Truth Density Map\nCount: {gt_count:.1f}", fontsize=12)
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
                
                im2 = axes[2].imshow(pred_vis, cmap="jet")
                axes[2].set_title(f"CSRNet Predicted Density Map\nPredicted Count: {pred_count:.1f} (Err: {diff:.1f})", fontsize=12)
                axes[2].axis("off")
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
                
                save_img_path = os.path.join(output_dir, f"vis_sample_{i+1}.png")
                plt.tight_layout()
                plt.savefig(save_img_path, dpi=150)
                plt.close()

    total_samples = len(val_loader)
    mae /= total_samples
    mse = np.sqrt(mse / total_samples)
    
    print("\n" + "=" * 45)
    print(f"EVALUATION SUMMARY ({dataset_type.upper()})")
    print("=" * 45)
    print(f"Total Test Images : {total_samples}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error  (MSE): {mse:.2f}")
    print(f"Visualized results saved in: {output_dir}")
    print("=" * 45)
    
    return mae, mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
    parser.add_argument("--dataset", type=str, default="part_B", choices=["part_A", "part_B"])
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--num_vis", type=int, default=5)
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, args.dataset, args.output_dir, args.num_vis)


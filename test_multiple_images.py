"""
test_multiple_images.py
Batch Testing and Multi-Camera Public Safety Telemetry using CSRNet.
Evaluates a batch of images, saves visual density overlays, generates a CSV report,
and displays real-time crowd safety risk levels.
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from model import CSRNet


def load_model(checkpoint_path=None, device="cpu"):
    model = CSRNet(pretrained_frontend=False).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[Model] Successfully loaded weights from {checkpoint_path}")
    else:
        print("[Model] Initialized with random weights (test mode).")
    model.eval()
    return model


def predict_single_image(model, device, img_path):
    raw_img = cv2.imread(img_path)
    if raw_img is None:
        raise ValueError(f"Could not load image at {img_path}")
        
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = raw_img.shape[:2]
    
    # Preprocessing
    img_float = raw_img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    
    t0 = time.time()
    with torch.no_grad():
        pred_density = model(img_tensor)
    latency_ms = (time.time() - t0) * 1000.0
    
    density_raw = pred_density.squeeze().cpu().numpy()
    pred_count = max(0.0, float(np.sum(density_raw)))
    
    # Heatmap color rendering
    density_resized = cv2.resize(density_raw, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    density_norm = cv2.normalize(density_resized, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(np.uint8(np.clip(density_norm, 0, 255)), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(raw_img, 0.6, heatmap_rgb, 0.4, 0)
    
    # Look for ground truth in .h5 or .mat
    gt_count = None
    h5_path = img_path.replace(".jpg", ".h5")
    if os.path.exists(h5_path):
        import h5py
        with h5py.File(h5_path, "r") as hf:
            gt_count = float(np.sum(np.asarray(hf["density"])))
    else:
        mat_path = img_path.replace("images", "ground-truth").replace("IMG_", "GT_IMG_").replace(".jpg", ".mat")
        if os.path.exists(mat_path):
            import scipy.io as io
            mat = io.loadmat(mat_path)
            pts = mat['image_info'][0, 0][0, 0][0]
            gt_count = float(len(pts))
            
    return pred_count, gt_count, latency_ms, raw_img, heatmap_rgb, blended


def test_batch(image_paths, checkpoint_path=None, thresh_warn=100, thresh_crit=250, output_dir="batch_test_results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Batch Testing for {len(image_paths)} Images on {device} ---")
    
    model = load_model(checkpoint_path, device)
    os.makedirs(output_dir, exist_ok=True)
    
    records = []
    
    for i, img_path in enumerate(image_paths):
        filename = os.path.basename(img_path)
        pred_count, gt_count, latency_ms, raw_img, heatmap_rgb, blended = predict_single_image(model, device, img_path)
        
        # Risk Classification
        if pred_count < thresh_warn:
            risk_level = "SAFE"
            status_badge = "🟢 SAFE"
            action = "Normal monitoring. All turnstiles OPEN."
            border_color = (0, 255, 0)
        elif pred_count < thresh_crit:
            risk_level = "WARNING"
            status_badge = "🟡 WARNING"
            action = "Throttle ingress gates. Alert ground marshals."
            border_color = (255, 200, 0)
        else:
            risk_level = "CRITICAL"
            status_badge = "🔴 CRITICAL"
            action = "EMERGENCY: Sound Sirens & LOCK Ingress Gates!"
            border_color = (255, 0, 0)
            
        err = abs(pred_count - gt_count) if gt_count is not None else np.nan
        
        records.append({
            "Index": i + 1,
            "Image File": filename,
            "Predicted Count": round(pred_count, 1),
            "Ground Truth": round(gt_count, 1) if gt_count is not None else "N/A",
            "Absolute Error": round(err, 1) if not np.isnan(err) else "N/A",
            "Latency (ms)": round(latency_ms, 1),
            "Throughput (FPS)": round(1000.0 / latency_ms, 1) if latency_ms > 0 else 0,
            "Safety Level": status_badge,
            "Automated Actuator Action": action
        })
        
        # Save 3-panel visualization figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Border around input image
        bordered_img = raw_img.copy()
        cv2.rectangle(bordered_img, (0, 0), (bordered_img.shape[1]-1, bordered_img.shape[0]-1), border_color, 12)
        
        axes[0].imshow(bordered_img)
        axes[0].set_title(f"Input: {filename}\\nActual: {gt_count if gt_count is not None else 'N/A'}", fontsize=12)
        axes[0].axis("off")
        
        axes[1].imshow(heatmap_rgb)
        axes[1].set_title(f"CSRNet Density Heatmap\\nPredicted: {pred_count:.1f} People", fontsize=12)
        axes[1].axis("off")
        
        axes[2].imshow(blended)
        axes[2].set_title(f"Safety Status: {status_badge}\\n{action}", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        
        out_vis_path = os.path.join(output_dir, f"result_{i+1:03d}_{filename}.png")
        plt.tight_layout()
        plt.savefig(out_vis_path, dpi=120)
        plt.close()
        
    # Save Report to CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "crowd_safety_batch_report.csv")
    df.to_csv(csv_path, index=False)
    
    # Display ASCII Table
    print("\n" + "=" * 90)
    print("                     BATCH TESTING TELEMETRY SUMMARY REPORT")
    print("=" * 90)
    display_cols = ["Index", "Image File", "Predicted Count", "Ground Truth", "Absolute Error", "Latency (ms)", "Safety Level"]
    print(df[display_cols].to_string(index=False))
    print("=" * 90)
    
    # Venue-wide Aggregates
    total_crowd = sum(r["Predicted Count"] for r in records)
    avg_latency = np.mean([r["Latency (ms)"] for r in records])
    critical_hotspots = sum(1 for r in records if "CRITICAL" in r["Safety Level"])
    warning_hotspots = sum(1 for r in records if "WARNING" in r["Safety Level"])
    safe_zones = sum(1 for r in records if "SAFE" in r["Safety Level"])
    
    print(f"\n📊 VENUE-WIDE SAFETY TELEMETRY:")
    print(f"  • Total Crowd Count across Monitored Cameras : {total_crowd:,.0f} People")
    print(f"  • Safe Flow Zones (🟢)                       : {safe_zones} Locations")
    print(f"  • Congestion Warning Zones (🟡)             : {warning_hotspots} Locations")
    print(f"  • Critical Stampede Hazard Hotspots (🔴)    : {critical_hotspots} Locations")
    print(f"  • Average Inference Latency                  : {avg_latency:.1f} ms ({1000/avg_latency:.1f} FPS)")
    print(f"  • Visualizations & CSV Report Saved in       : {output_dir}")
    print("=" * 90 + "\n")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test CSRNet on multiple crowd images")
    parser.add_argument("--image_dir", type=str, default="ShanghaiTech/part_B/test_data/images",
                        help="Directory containing test images")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained .pth checkpoint")
    parser.add_argument("--num_images", type=int, default=6, help="Number of images to test (-1 for all)")
    parser.add_argument("--thresh_warn", type=int, default=100, help="Warning threshold")
    parser.add_argument("--thresh_crit", type=int, default=250, help="Critical threshold")
    parser.add_argument("--output_dir", type=str, default="batch_test_results", help="Directory to save figures")
    args = parser.parse_args()
    
    all_imgs = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if not all_imgs:
        print(f"Error: No .jpg images found in {args.image_dir}")
        exit(1)
        
    selected_imgs = all_imgs if args.num_images <= 0 else all_imgs[:args.num_images]
    test_batch(selected_imgs, args.checkpoint, args.thresh_warn, args.thresh_crit, args.output_dir)

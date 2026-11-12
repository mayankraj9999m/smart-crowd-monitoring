"""
make_density_maps.py
Generates ground truth density maps (.h5) for the ShanghaiTech dataset
following the CSRNet paper (CVPR 2018):
- Part A: Geometry-adaptive Gaussian kernels (k-nearest neighbors, k=3, beta=0.3)
- Part B: Fixed Gaussian kernel (sigma=15)
"""

import os
import glob
import h5py
import numpy as np
import scipy.io as io
from PIL import Image
from scipy.spatial import KDTree
from tqdm import tqdm


def generate_gaussian_density(img_path, mat_path, is_part_a=True):
    """
    Computes density map for a single image and corresponding .mat ground truth.
    Returns: density map (float32 numpy array)
    """
    img = Image.open(img_path)
    W, H = img.size
    
    mat = io.loadmat(mat_path)
    pts = mat['image_info'][0, 0][0, 0][0]
    num_pts = len(pts)
    
    density = np.zeros((H, W), dtype=np.float32)
    if num_pts == 0:
        return density

    if is_part_a and num_pts > 3:
        # Part A: Geometry-adaptive Gaussian
        tree = KDTree(pts.copy(), leafsize=2048)
        distances, _ = tree.query(pts, k=4)  # k=4 because index 0 is distance to self
        
        for i, pt in enumerate(pts):
            x, y = int(round(pt[0])), int(round(pt[1]))
            if x < 0 or x >= W or y < 0 or y >= H:
                continue
            
            # sigma = beta * mean(d1, d2, d3) with beta=0.3
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            sigma = min(max(sigma, 1.0), 30.0)  # Bound sigma to prevent extreme dispersion
            radius = int(np.ceil(3 * sigma))
            
            x1, x2 = max(0, x - radius), min(W, x + radius + 1)
            y1, y2 = max(0, y - radius), min(H, y + radius + 1)
            
            xs = np.arange(x1, x2)
            ys = np.arange(y1, y2)
            X, Y = np.meshgrid(xs, ys)
            kernel = np.exp(-((X - x)**2 + (Y - y)**2) / (2.0 * sigma**2))
            kernel /= (2.0 * np.pi * sigma**2)
            density[y1:y2, x1:x2] += kernel
    else:
        # Part B or sparse Part A: Fixed Gaussian (sigma=15)
        sigma = 15.0
        radius = int(np.ceil(3 * sigma))
        for pt in pts:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if x < 0 or x >= W or y < 0 or y >= H:
                continue
            x1, x2 = max(0, x - radius), min(W, x + radius + 1)
            y1, y2 = max(0, y - radius), min(H, y + radius + 1)
            xs = np.arange(x1, x2)
            ys = np.arange(y1, y2)
            X, Y = np.meshgrid(xs, ys)
            kernel = np.exp(-((X - x)**2 + (Y - y)**2) / (2.0 * sigma**2))
            kernel /= (2.0 * np.pi * sigma**2)
            density[y1:y2, x1:x2] += kernel

    return density


def process_dataset_split(dataset_dir, is_part_a=True):
    """
    Processes all images in dataset_dir and saves .h5 density maps.
    """
    img_paths = sorted(glob.glob(os.path.join(dataset_dir, "images", "*.jpg")))
    print(f"Processing {len(img_paths)} images in {dataset_dir}...")
    
    for img_path in tqdm(img_paths, desc=os.path.basename(os.path.dirname(dataset_dir))):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # Match with GT_IMG_X.mat
        mat_path = os.path.join(dataset_dir, "ground-truth", f"GT_{base_name}.mat")
        h5_path = img_path.replace(".jpg", ".h5")
        
        # Skip if already generated
        if os.path.exists(h5_path):
            continue
            
        if not os.path.exists(mat_path):
            print(f"Warning: Missing ground truth {mat_path}")
            continue
            
        density = generate_gaussian_density(img_path, mat_path, is_part_a=is_part_a)
        
        # Save to HDF5
        with h5py.File(h5_path, "w") as hf:
            hf["density"] = density


if __name__ == "__main__":
    base_dir = "ShanghaiTech"
    splits = [
        (os.path.join(base_dir, "part_A", "train_data"), True),
        (os.path.join(base_dir, "part_A", "test_data"), True),
        (os.path.join(base_dir, "part_B", "train_data"), False),
        (os.path.join(base_dir, "part_B", "test_data"), False),
    ]
    
    for split_dir, is_part_a in splits:
        if os.path.exists(split_dir):
            process_dataset_split(split_dir, is_part_a=is_part_a)
    print("\nAll density maps generated successfully!")


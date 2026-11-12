"""
dataset.py
PyTorch Dataset for ShanghaiTech crowd counting dataset.
Loads images (.jpg) and precomputed ground truth density maps (.h5).
"""

import os
import glob
import h5py
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CrowdDataset(Dataset):
    def __init__(self, data_dir, is_train=True, downsample_factor=8):
        """
        Args:
            data_dir: Path to train_data or test_data directory
            is_train: Boolean, applies data augmentations if True
            downsample_factor: Downsampling factor of CSRNet backend (default=8)
        """
        self.img_paths = sorted(glob.glob(os.path.join(data_dir, "images", "*.jpg")))
        self.is_train = is_train
        self.downsample_factor = downsample_factor
        
        # Standard ImageNet normalization for VGG-16 backbone
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        h5_path = img_path.replace(".jpg", ".h5")
        
        # Load image
        img = Image.open(img_path).convert("RGB")
        
        # Load ground truth density map
        with h5py.File(h5_path, "r") as hf:
            density = np.asarray(hf["density"], dtype=np.float32)

        # Random horizontal flip during training
        if self.is_train and np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            density = np.fliplr(density)

        # Apply ImageNet transforms to image
        img_tensor = self.transform(img)

        # Downsample density map by factor of 8 to match CSRNet output stride
        h, w = density.shape
        target_h = h // self.downsample_factor
        target_w = w // self.downsample_factor
        
        # Resize density and scale by factor squared (8x8 = 64) to conserve total count integral
        density_resized = cv2.resize(density, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        density_resized = density_resized * (self.downsample_factor ** 2)
        
        # Convert density to tensor (1, target_h, target_w)
        density_tensor = torch.from_numpy(density_resized.copy()).unsqueeze(0)
        
        return img_tensor, density_tensor, img_path


if __name__ == "__main__":
    test_dir = "ShanghaiTech/part_A/test_data"
    if os.path.exists(test_dir):
        ds = CrowdDataset(test_dir, is_train=False)
        img, den, path = ds[0]
        print("Sample image shape  :", img.shape)
        print("Density map shape   :", den.shape)
        print("Integrated crowd count:", den.sum().item())


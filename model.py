"""
model.py
CSRNet Architecture implementation in PyTorch.
Reference: "CSRNet: Dilated Convolutional Neural Networks for Understanding Highly Congested Scenes" (CVPR 2018)
"""

import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, pretrained_frontend=False):
        super(CSRNet, self).__init__()
        
        # Frontend: First 10 Conv layers of VGG-16 (up to layer 23, before 4th maxpool)
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained_frontend else None)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        
        # Backend: Dilated Convolutional Network (Configuration B: Dilation rate = 2)
        # Keeps spatial resolution while enlarging receptive field
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # Output layer: 1x1 Convolution producing 1-channel continuous density map
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        self._initialize_backend_weights()
        
    def forward(self, x):
        features = self.frontend(x)
        dilated_features = self.backend(features)
        density_map = self.output_layer(dilated_features)
        return density_map

    def _initialize_backend_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.output_layer.weight, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)


if __name__ == "__main__":
    model = CSRNet(pretrained_frontend=False)
    dummy_input = torch.randn(1, 3, 384, 512)
    output = model(dummy_input)
    print("CSRNet Input Shape :", dummy_input.shape)
    print("CSRNet Output Shape:", output.shape)
    print("Resolution Downsample Factor: 8x (Width/8, Height/8)")


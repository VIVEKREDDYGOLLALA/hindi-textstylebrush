"""
Content encoder for the Hindi TextStyleBrush model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.
    Same implementation as in style_encoder.py.
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ContentEncoder(nn.Module):
    """
    Content encoder network that extracts content features from a text image.
    Takes a grayscale image of text rendered with a standard font.
    """
    def __init__(self, content_dim=512):
        super(ContentEncoder, self).__init__()
        self.content_dim = content_dim
        
        # Initial convolutional layers for grayscale input
        self.conv0_1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv0_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Residual blocks
        self.resblock1 = self._make_layer(64, 128, blocks=1)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.resblock2 = self._make_layer(128, 256, blocks=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.resblock3 = self._make_layer(256, 512, blocks=5)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.resblock4 = self._make_layer(512, 512, blocks=3)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        # First block may change dimensions
        layers.append(ResidualBlock(in_channels, out_channels))
        # Subsequent blocks maintain dimensions
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass of the content encoder.
        
        Args:
            x: Grayscale text image tensor [B, 1, H, W]
        
        Returns:
            Content representation [B, C, H, W]
            where H and W are downsampled by a factor of 16 compared to input
        """
        # Initial convolutions
        x = F.relu(self.conv0_1(x))
        x = F.relu(self.conv0_2(x))
        x = self.pool1(x)
        
        # Residual blocks
        x = self.resblock1(x)
        x = F.relu(self.conv1(x))
        x = self.pool2(x)
        
        x = self.resblock2(x)
        x = F.relu(self.conv2(x))
        x = self.pool3(x)
        
        x = self.resblock3(x)
        x = F.relu(self.conv3(x))
        x = self.pool4(x)
        
        x = self.resblock4(x)
        content_features = F.relu(self.conv4_1(x))
        
        return content_features
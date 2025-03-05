"""
Discriminator for the Hindi TextStyleBrush model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block for the discriminator.
    """
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels or downsample:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Downsampling
        self.downsample_layer = nn.AvgPool2d(2) if downsample else nn.Identity()
    
    def forward(self, x):
        # Main path
        residual = self.skip(x)
        
        # Convolutions
        out = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        out = self.conv2(out)
        
        # Add skip connection
        out = out + residual
        
        # Downsample if needed
        if self.downsample:
            out = self.downsample_layer(out)
            residual = self.downsample_layer(residual)
        
        # Final activation
        out = F.leaky_relu(out, negative_slope=0.2)
        
        return out


class Discriminator(nn.Module):
    """
    Discriminator for the Hindi TextStyleBrush GAN.
    Uses a ResNet-based architecture with conditional input.
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=4):
        super(Discriminator, self).__init__()
        self.n_layers = n_layers
        
        # Initial convolutional layer
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Residual blocks with downsampling
        mult = 1
        self.blocks = nn.ModuleList()
        
        for i in range(n_layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            
            self.blocks.append(
                ResidualBlock(
                    ndf * mult_prev,
                    ndf * mult,
                    downsample=(i < n_layers - 1)  # No downsampling in the last block
                )
            )
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(ndf * mult, ndf * mult, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * mult, 1, kernel_size=4, padding=0)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass of the discriminator.
        
        Args:
            x: Input image [B, 3, H, W]
        
        Returns:
            Prediction [B, 1, H', W'] - Not averaged to a scalar
        """
        # Initial layer
        x = self.initial(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layers
        x = self.final(x)
        
        return x
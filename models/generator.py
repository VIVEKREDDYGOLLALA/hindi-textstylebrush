"""
StyleGAN2-based generator for the Hindi TextStyleBrush model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulatedConv2d(nn.Module):
    """
    Modulated Convolution layer that incorporates style information.
    Based on StyleGAN2 architecture.
    """
    def __init__(self, in_channels, out_channels, style_dim=512, kernel_size=3, padding=1, demodulate=True):
        super(ModulatedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.demodulate = demodulate
        
        # Weight for convolution
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
        # Scaling factor for weight normalization
        self.scale = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
        
        # Style modulation layer
        self.modulation = nn.Linear(style_dim, in_channels)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.modulation.bias)
        nn.init.ones_(self.modulation.weight)
    
    def forward(self, x, style):
        """
        Forward pass of the modulated convolution.
        
        Args:
            x: Input feature map [B, C_in, H, W]
            style: Style vector [B, C_style]
        
        Returns:
            Output feature map [B, C_out, H, W]
        """
        batch_size, in_channels, height, width = x.shape
        
        # Style modulation
        style = self.modulation(style)  # [B, C_in]
        style = style.view(batch_size, in_channels, 1, 1)  # [B, C_in, 1, 1]
        
        # Modulate weights
        weight = self.scale * self.weight  # [C_out, C_in, K, K]
        weight = weight.unsqueeze(0) * style  # [B, C_out, C_in, K, K]
        
        # Demodulation
        if self.demodulate:
            demod = torch.rsqrt(
                (weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8
            )  # [B, C_out, 1, 1, 1]
            weight = weight * demod  # [B, C_out, C_in, K, K]
        
        # Reshape for grouped convolution
        weight = weight.view(
            batch_size * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )
        
        # Reshape input for grouped convolution
        x = x.reshape(1, batch_size * in_channels, height, width)
        
        # Grouped convolution
        out = F.conv2d(
            x, weight, None, padding=self.padding, groups=batch_size
        )
        
        # Reshape output
        out = out.view(batch_size, self.out_channels, height, width)
        
        # Add bias
        out = out + self.bias
        
        return out


class StyleBlock(nn.Module):
    """
    A building block for the generator that applies style.
    """
    def __init__(self, in_channels, out_channels, style_dim=512, upsample=False):
        super(StyleBlock, self).__init__()
        self.upsample = upsample
        
        # Upsampling layer if needed
        self.upsampler = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else nn.Identity()
        
        # Modulated convolution layers
        self.conv1 = ModulatedConv2d(in_channels, out_channels, style_dim, kernel_size=3, padding=1)
        self.conv2 = ModulatedConv2d(out_channels, out_channels, style_dim, kernel_size=3, padding=1)
        
        # Bias terms
        self.bias1 = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
        # Layer for RGB output
        self.to_rgb = ModulatedConv2d(out_channels, 3, style_dim, kernel_size=1, padding=0, demodulate=False)
        self.to_rgb_bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        
        # Layer for Mask output
        self.to_mask = ModulatedConv2d(out_channels, 1, style_dim, kernel_size=1, padding=0, demodulate=False)
        self.to_mask_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
    
    def forward(self, x, rgb_prev, mask_prev, style1, style2, style_rgb, style_mask):
        """
        Forward pass of the style block.
        
        Args:
            x: Input feature map [B, C_in, H, W]
            rgb_prev: Previous RGB output [B, 3, H, W] or None
            mask_prev: Previous Mask output [B, 1, H, W] or None
            style1: Style vector for first conv [B, style_dim]
            style2: Style vector for second conv [B, style_dim]
            style_rgb: Style vector for RGB output [B, style_dim]
            style_mask: Style vector for Mask output [B, style_dim]
        
        Returns:
            x: Updated feature map [B, C_out, H', W']
            rgb: Updated RGB output [B, 3, H', W']
            mask: Updated Mask output [B, 1, H', W']
        """
        # Upsample if needed
        if self.upsample:
            x = self.upsampler(x)
            if rgb_prev is not None:
                rgb_prev = self.upsampler(rgb_prev)
            if mask_prev is not None:
                mask_prev = self.upsampler(mask_prev)
        
        # First conv with style
        x = self.conv1(x, style1)
        x = x + self.bias1
        x = F.leaky_relu(x, negative_slope=0.2)
        
        # Second conv with style
        x = self.conv2(x, style2)
        x = x + self.bias2
        x = F.leaky_relu(x, negative_slope=0.2)
        
        # Generate RGB output
        rgb = self.to_rgb(x, style_rgb)
        rgb = rgb + self.to_rgb_bias
        
        # Add skip connection from previous RGB if available
        if rgb_prev is not None:
            rgb = rgb + rgb_prev
        
        # Generate Mask output
        mask = self.to_mask(x, style_mask)
        mask = mask + self.to_mask_bias
        
        # Add skip connection from previous Mask if available
        if mask_prev is not None:
            mask = mask + mask_prev
        
        # Apply sigmoid to mask to get values in [0, 1]
        mask = torch.sigmoid(mask)
        
        return x, rgb, mask


class Generator(nn.Module):
    """
    StyleGAN2-based generator for text style transfer.
    Takes content features and style vectors to generate a stylized text image.
    """
    def __init__(self, content_dim=512, style_dim=512, num_channels=None):
        super(Generator, self).__init__()
        self.content_dim = content_dim
        self.style_dim = style_dim
        
        # Default channel configuration if not provided
        if num_channels is None:
            num_channels = [512, 512, 256, 128, 64]
            
        # Number of style blocks
        self.num_blocks = len(num_channels)
        
        # Input layer
        self.input_layer = nn.Conv2d(content_dim, num_channels[0], kernel_size=1)
        
        # Style blocks
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            in_channels = num_channels[max(0, i-1)]
            out_channels = num_channels[i]
            upsample = i > 0  # No upsampling for the first block
            self.blocks.append(StyleBlock(in_channels, out_channels, style_dim, upsample))
    
    def forward(self, content_features, style_vectors):
        """
        Forward pass of the generator.
        
        Args:
            content_features: Content feature map [B, content_dim, H, W]
            style_vectors: List of style vectors, each [B, style_dim]
        
        Returns:
            rgb: Generated RGB image [B, 3, H*2^(num_blocks-1), W*2^(num_blocks-1)]
            mask: Generated mask [B, 1, H*2^(num_blocks-1), W*2^(num_blocks-1)]
        """
        # Process input content features
        x = self.input_layer(content_features)
        
        rgb = None
        mask = None
        
        # Apply style blocks
        for i, block in enumerate(self.blocks):
            # Get style vectors for this block
            # Each block needs 4 style vectors: 2 for convs, 1 for RGB, 1 for mask
            idx = i * 4
            style1 = style_vectors[idx]
            style2 = style_vectors[idx + 1]
            style_rgb = style_vectors[idx + 2]
            style_mask = style_vectors[idx + 3]
            
            # Apply the block
            x, rgb, mask = block(x, rgb, mask, style1, style2, style_rgb, style_mask)
        
        return rgb, mask
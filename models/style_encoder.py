"""
Style encoder for the Hindi TextStyleBrush model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and a skip connection.
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


class ROIAlign(nn.Module):
    """
    Region of Interest Align module.
    Extracts features from a region of the input feature map.
    """
    def __init__(self, output_size=1):
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        
    def forward(self, features, normalized_bbox):
        """
        Args:
            features: Feature map [B, C, H, W]
            normalized_bbox: Normalized bounding box coordinates [B, 4] (x1, y1, x2, y2)
                             Values in range [0, 1]
        """
        batch_size, channels, height, width = features.shape
        
        # Convert normalized coordinates to pixel coordinates
        boxes = []
        for i in range(batch_size):
            x1, y1, x2, y2 = normalized_bbox[i]
            x1 = x1.item() * width
            y1 = y1.item() * height
            x2 = x2.item() * width
            y2 = y2.item() * height
            
            # Ensure box is within image boundaries
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            # Convert to RoI format [batch_idx, x1, y1, x2, y2]
            boxes.append(torch.tensor([i, x1, y1, x2, y2], device=features.device))
        
        # Stack all boxes
        boxes = torch.stack(boxes)
        
        # Use torchvision's ROI align if available, otherwise use a simple pooling approach
        try:
            from torchvision.ops import roi_align
            output = roi_align(features, boxes, output_size=[self.output_size, self.output_size])
        except ImportError:
            # Fallback to a simple approach
            output = []
            for i in range(batch_size):
                _, x1, y1, x2, y2 = boxes[i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                roi = features[i, :, y1:y2, x1:x2]
                # Adaptive pooling to get fixed size output
                roi = F.adaptive_avg_pool2d(roi, (self.output_size, self.output_size))
                output.append(roi)
            output = torch.stack(output)
        
        # Flatten to 1D vector per image
        return output.view(batch_size, -1)


class StyleEncoder(nn.Module):
    """
    Style encoder network that extracts style features from a text image.
    Takes a localized scene RGB image and a word bounding box.
    """
    def __init__(self, style_dim=512):
        super(StyleEncoder, self).__init__()
        self.style_dim = style_dim
        
        # Initial convolutional layers
        self.conv0_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
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
        
        # ROI align module
        self.roi_align = ROIAlign(output_size=1)
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        # First block may change dimensions
        layers.append(ResidualBlock(in_channels, out_channels))
        # Subsequent blocks maintain dimensions
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x, bbox):
        """
        Forward pass of the style encoder.
        
        Args:
            x: Input image tensor [B, 3, H, W]
            bbox: Normalized bounding box coordinates [B, 4] (x1, y1, x2, y2)
        
        Returns:
            Style vector [B, style_dim]
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
        x = F.relu(self.conv4_1(x))
        
        # Apply ROI align to extract features from the text region
        style_vector = self.roi_align(x, bbox)
        
        return style_vector
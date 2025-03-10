"""
Reconstruction losses for the Hindi TextStyleBrush model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss to ensure the generated images match the target style.
    Separately considers foreground (text) and background regions using the generated mask.
    """
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def forward(self, generated_images, target_images, masks):
        """
        Compute reconstruction loss with foreground/background separation.
        
        Args:
            generated_images: Generated images [B, 3, H, W]
            target_images: Target style images [B, 3, H, W]
            masks: Foreground masks from generator [B, 1, H, W]
        
        Returns:
            total_loss: Combined reconstruction loss
            losses: Dictionary of individual losses (foreground, background)
        """
        # Compute pixel-wise L1 loss
        pixel_loss = self.l1_loss(generated_images, target_images)  # [B, 3, H, W]
        
        # Average over channels
        pixel_loss = pixel_loss.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply masks to separate foreground and background
        foreground_loss = (pixel_loss * masks).sum() / (masks.sum() + 1e-8)
        background_loss = (pixel_loss * (1 - masks)).sum() / ((1 - masks).sum() + 1e-8)
        
        # Combined loss (with higher weight for foreground)
        total_loss = foreground_loss * 2.0 + background_loss
        
        losses = {
            'foreground': foreground_loss.item(),
            'background': background_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, losses


class CyclicReconstructionLoss(nn.Module):
    """
    Cyclic reconstruction loss to enforce style consistency.
    Re-encodes the generated image to obtain a new style vector, then regenerates the image
    and compares it with the original generated image.
    """
    def __init__(self):
        super(CyclicReconstructionLoss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
    
    def forward(self, original_images, cyclic_images, masks):
        """
        Compute cyclic reconstruction loss.
        
        Args:
            original_images: Original generated images [B, 3, H, W]
            cyclic_images: Cyclically reconstructed images [B, 3, H, W]
            masks: Foreground masks from generator [B, 1, H, W]
        
        Returns:
            total_loss: Combined cyclic reconstruction loss
            losses: Dictionary of individual losses (foreground, background)
        """
        # Compute pixel-wise L1 loss
        pixel_loss = self.l1_loss(cyclic_images, original_images)  # [B, 3, H, W]
        
        # Average over channels
        pixel_loss = pixel_loss.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Apply masks to separate foreground and background
        foreground_loss = (pixel_loss * masks).sum() / (masks.sum() + 1e-8)
        background_loss = (pixel_loss * (1 - masks)).sum() / ((1 - masks).sum() + 1e-8)
        
        # Combined loss (with higher weight for foreground)
        total_loss = foreground_loss * 2.0 + background_loss
        
        losses = {
            'cyclic_foreground': foreground_loss.item(),
            'cyclic_background': background_loss.item(),
            'cyclic_total': total_loss.item()
        }
        
        return total_loss, losses
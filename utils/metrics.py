"""
Evaluation metrics for the Hindi TextStyleBrush model.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score
import Levenshtein
import torch.nn.functional as F


def text_recognition_accuracy(predicted_texts, target_texts):
    """
    Calculate text recognition accuracy (exact match).
    
    Args:
        predicted_texts: List of predicted text strings
        target_texts: List of target text strings
    
    Returns:
        accuracy: Percentage of exact matches
    """
    correct = sum(p == t for p, t in zip(predicted_texts, target_texts))
    total = len(target_texts)
    
    return correct / total if total > 0 else 0.0


def character_error_rate(predicted_texts, target_texts):
    """
    Calculate character error rate (CER).
    
    Args:
        predicted_texts: List of predicted text strings
        target_texts: List of target text strings
    
    Returns:
        cer: Average character error rate
    """
    total_distance = 0
    total_length = 0
    
    for pred, target in zip(predicted_texts, target_texts):
        distance = Levenshtein.distance(pred, target)
        total_distance += distance
        total_length += len(target)
    
    return total_distance / total_length if total_length > 0 else 0.0


def word_error_rate(predicted_texts, target_texts):
    """
    Calculate word error rate (WER).
    
    Args:
        predicted_texts: List of predicted text strings
        target_texts: List of target text strings
    
    Returns:
        wer: Average word error rate
    """
    total_distance = 0
    total_words = 0
    
    for pred, target in zip(predicted_texts, target_texts):
        pred_words = pred.split()
        target_words = target.split()
        
        distance = Levenshtein.distance(pred_words, target_words)
        total_distance += distance
        total_words += len(target_words)
    
    return total_distance / total_words if total_words > 0 else 0.0


def mse(generated_images, target_images, masks=None):
    """
    Calculate Mean Squared Error between generated and target images.
    Optionally use masks to focus on foreground (text) regions.
    
    Args:
        generated_images: Generated images tensor [B, 3, H, W]
        target_images: Target images tensor [B, 3, H, W]
        masks: Optional foreground masks tensor [B, 1, H, W]
    
    Returns:
        mse: Mean squared error
    """
    squared_diff = (generated_images - target_images) ** 2
    
    if masks is not None:
        # Average over channels
        squared_diff = squared_diff.mean(dim=1, keepdim=True)
        
        # Apply masks
        foreground_error = (squared_diff * masks).sum() / (masks.sum() + 1e-8)
        background_error = (squared_diff * (1 - masks)).sum() / ((1 - masks).sum() + 1e-8)
        
        # Combined error
        return foreground_error.item(), background_error.item()
    else:
        # Average over all pixels and channels
        return squared_diff.mean().item()


def psnr(generated_images, target_images, masks=None, max_value=2.0):
    """
    Calculate Peak Signal-to-Noise Ratio between generated and target images.
    
    Args:
        generated_images: Generated images tensor [B, 3, H, W]
        target_images: Target images tensor [B, 3, H, W]
        masks: Optional foreground masks tensor [B, 1, H, W]
        max_value: Maximum possible pixel value (2.0 for range [-1, 1])
    
    Returns:
        psnr: Peak Signal-to-Noise Ratio
    """
    # Calculate MSE
    mse_value = mse(generated_images, target_images, masks)
    
    if masks is not None:
        # If masks are used, we get foreground and background MSE
        fg_mse, bg_mse = mse_value
        
        # Calculate PSNR for foreground and background
        fg_psnr = 20 * np.log10(max_value) - 10 * np.log10(fg_mse + 1e-8)
        bg_psnr = 20 * np.log10(max_value) - 10 * np.log10(bg_mse + 1e-8)
        
        return fg_psnr, bg_psnr
    else:
        # Calculate overall PSNR
        psnr_value = 20 * np.log10(max_value) - 10 * np.log10(mse_value + 1e-8)
        return psnr_value


def ssim(generated_images, target_images, window_size=11, masks=None):
    """
    Calculate Structural Similarity Index Measure (SSIM) between generated and target images.
    
    Args:
        generated_images: Generated images tensor [B, 3, H, W]
        target_images: Target images tensor [B, 3, H, W]
        window_size: Size of the Gaussian window
        masks: Optional foreground masks tensor [B, 1, H, W]
    
    Returns:
        ssim: Structural Similarity Index Measure
    """
    # Ensure images are in the right format
    if generated_images.size(1) != 3 or target_images.size(1) != 3:
        raise ValueError("Images must have 3 channels (RGB)")
    
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # Create a Gaussian window
    sigma = 1.5
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
    window = gauss.unsqueeze(0) * gauss.unsqueeze(1)
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1).to(generated_images.device)
    
    # Convert to [0, 1] range if in [-1, 1]
    if generated_images.min() < 0:
        gen_scaled = (generated_images + 1) / 2
        target_scaled = (target_images + 1) / 2
    else:
        gen_scaled = generated_images
        target_scaled = target_images
    
    # Calculate means
    mu1 = F.conv2d(gen_scaled, window, padding=window_size//2, groups=3)
    mu2 = F.conv2d(target_scaled, window, padding=window_size//2, groups=3)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calculate variances and covariance
    sigma1_sq = F.conv2d(gen_scaled ** 2, window, padding=window_size//2, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(target_scaled ** 2, window, padding=window_size//2, groups=3) - mu2_sq
    sigma12 = F.conv2d(gen_scaled * target_scaled, window, padding=window_size//2, groups=3) - mu1_mu2
    
    # Calculate SSIM
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    # Apply masks if provided
    if masks is not None:
        # Resize masks to match SSIM map dimensions
        masks_resized = F.interpolate(masks, size=ssim_map.shape[2:], mode='bilinear', align_corners=False)
        
        # Calculate SSIM for foreground and background separately
        fg_ssim = (ssim_map * masks_resized).sum() / (masks_resized.sum() + 1e-8)
        bg_ssim = (ssim_map * (1 - masks_resized)).sum() / ((1 - masks_resized).sum() + 1e-8)
        
        return fg_ssim.item(), bg_ssim.item()
    else:
        # Calculate overall SSIM
        return ssim_map.mean().item()
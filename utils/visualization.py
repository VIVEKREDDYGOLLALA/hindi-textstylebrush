"""
Visualization utilities for the Hindi TextStyleBrush model.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import cv2


def tensor_to_image(tensor):
    """
    Convert a normalized tensor to a numpy image.
    
    Args:
        tensor: Tensor of shape [C, H, W] with values in range [-1, 1]
    
    Returns:
        numpy array of shape [H, W, C] with values in range [0, 255]
    """
    # Ensure tensor is on CPU and detached from computation graph
    tensor = tensor.cpu().detach()
    
    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    
    # Clamp values to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose dimensions
    image = tensor.numpy()
    image = np.transpose(image, (1, 2, 0))
    
    # Handle grayscale images
    if image.shape[2] == 1:
        image = image[:, :, 0]
    
    # Convert to uint8
    image = (image * 255).astype(np.uint8)
    
    return image


def save_images(images, masks, original_images, target_images, filename, nrow=4):
    """
    Save a grid of images for visualization.
    
    Args:
        images: Generated images tensor [B, 3, H, W]
        masks: Generated masks tensor [B, 1, H, W]
        original_images: Original style images tensor [B, 3, H, W]
        target_images: Target content images tensor [B, 1, H, W]
        filename: Output filename
        nrow: Number of images per row in the grid
    """
    # Create a directory for images if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert masks to RGB
    masks_rgb = masks.repeat(1, 3, 1, 1)
    
    # Convert grayscale target images to RGB
    if target_images.size(1) == 1:
        target_images = target_images.repeat(1, 3, 1, 1)
    
    # Combine images into a grid
    combined = torch.cat([
        original_images,
        target_images,
        images,
        masks_rgb
    ], dim=0)
    
    # Create image grid
    grid = make_grid(combined, nrow=nrow, normalize=True, value_range=(-1, 1))
    
    # Convert to numpy image
    grid_image = tensor_to_image(grid)
    
    # Save the image
    if grid_image.ndim == 2:
        # Grayscale
        cv2.imwrite(filename, grid_image)
    else:
        # RGB to BGR for OpenCV
        cv2.imwrite(filename, cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))


def visualize_attention(image, attention_weights, filename):
    """
    Visualize attention weights over an image.
    
    Args:
        image: Image tensor [C, H, W]
        attention_weights: Attention weights tensor [H', W']
        filename: Output filename
    """
    # Convert tensors to numpy
    image = tensor_to_image(image)
    attention = attention_weights.cpu().detach().numpy()
    
    # Resize attention map to match image dimensions
    attention = cv2.resize(attention, (image.shape[1], image.shape[0]))
    
    # Normalize attention weights
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    # Apply colormap to attention
    heatmap = cv2.applyColorMap((attention * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Convert image to BGR if it's RGB
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # Convert grayscale to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Blend image and heatmap
    result = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
    
    # Save image
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(filename, result)


def plot_losses(losses, filename):
    """
    Plot training losses over time.
    
    Args:
        losses: Dictionary with loss names as keys and lists of values
        filename: Output filename
    """
    plt.figure(figsize=(12, 8))
    
    for name, values in losses.items():
        if len(values) > 0:
            plt.plot(values, label=name)
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save figure
    plt.savefig(filename)
    plt.close()
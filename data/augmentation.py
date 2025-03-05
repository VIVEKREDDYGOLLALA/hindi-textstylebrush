"""
Data augmentation utilities for Hindi text images.
"""

import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torchvision.transforms.functional as TF

class RandomRotation:
    """
    Randomly rotate the image between -degrees and +degrees.
    """
    def __init__(self, degrees=5):
        self.degrees = degrees
        
    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return TF.rotate(img, angle, fill=255)


class RandomShear:
    """
    Randomly shear the image along X and Y axis.
    """
    def __init__(self, shear_range=0.1):
        self.shear_range = shear_range
        
    def __call__(self, img):
        shear_x = random.uniform(-self.shear_range, self.shear_range)
        shear_y = random.uniform(-self.shear_range, self.shear_range)
        return TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=[shear_x, shear_y], fill=255)


class RandomTranslate:
    """
    Randomly translate the image.
    """
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range
        
    def __call__(self, img):
        max_dx = int(self.translate_range * img.size[0])
        max_dy = int(self.translate_range * img.size[1])
        translations = [random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy)]
        return TF.affine(img, angle=0, translate=translations, scale=1.0, shear=[0, 0], fill=255)


class RandomBrightness:
    """
    Randomly change the brightness of the image.
    """
    def __init__(self, brightness_range=(0.8, 1.2)):
        self.brightness_range = brightness_range
        
    def __call__(self, img):
        factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        enhancer = ImageEnhance.Brightness(img)
        return enhancer.enhance(factor)


class RandomContrast:
    """
    Randomly change the contrast of the image.
    """
    def __init__(self, contrast_range=(0.8, 1.2)):
        self.contrast_range = contrast_range
        
    def __call__(self, img):
        factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)


class RandomBackground:
    """
    Randomly change the background of the text image.
    This assumes the text is dark on a light background.
    """
    def __init__(self, bg_color_range=(200, 255)):
        self.bg_color_range = bg_color_range
        
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL image
            img = TF.to_pil_image(img)
            
        # Convert to grayscale to get text mask
        gray = img.convert('L')
        threshold = 127
        text_mask = gray.point(lambda p: p < threshold and 255)
        
        # Create new background
        bg_color = random.randint(self.bg_color_range[0], self.bg_color_range[1])
        bg = Image.new('RGB', img.size, (bg_color, bg_color, bg_color))
        
        # Composite original text onto new background
        inverted_mask = ImageOps.invert(text_mask)
        return Image.composite(img, bg, inverted_mask)


class RandomNoise:
    """
    Add random noise to the image.
    """
    def __init__(self, noise_factor=0.05):
        self.noise_factor = noise_factor
        
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Add noise to tensor
            noise = torch.randn_like(img) * self.noise_factor
            return torch.clamp(img + noise, 0, 1)
        else:
            # Convert PIL image to numpy array
            img_array = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.noise_factor, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 1)
            return Image.fromarray((img_array * 255).astype(np.uint8))


def get_augmentation_transforms():
    """
    Return a list of augmentation transforms for training.
    """
    augmentations = [
        RandomRotation(degrees=5),
        RandomShear(shear_range=0.1),
        RandomTranslate(translate_range=0.1),
        RandomBrightness(brightness_range=(0.8, 1.2)),
        RandomContrast(contrast_range=(0.8, 1.2)),
        RandomNoise(noise_factor=0.02)
    ]
    
    def apply_random_augmentations(img):
        # Apply 0-3 random augmentations
        n_augmentations = random.randint(0, 3)
        chosen_augmentations = random.sample(augmentations, n_augmentations)
        
        # Apply each augmentation
        for aug in chosen_augmentations:
            img = aug(img)
            
        return img
    
    return apply_random_augmentations
"""
Dataset classes for loading and processing Hindi text images.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFont, ImageDraw
import cv2
from torchvision import transforms
from config import Config

class HindiSynthTextDataset(Dataset):
    """
    Dataset for synthetic Hindi text generation.
    Used for training the typeface classifier.
    """
    def __init__(self, root_dir, num_samples=50000, transform=None):
        self.root_dir = root_dir
        self.num_samples = num_samples
        self.transform = transform
        self.font_paths = self._get_font_paths()
        self.num_fonts = len(self.font_paths)
        self.font_classes = {font_path: i for i, font_path in enumerate(self.font_paths)}
        self.chars = Config.ALL_CHARS
        
    def _get_font_paths(self):
        """Get paths to all .ttf files in the fonts directory."""
        font_dir = os.path.join(self.root_dir, 'fonts')
        if not os.path.exists(font_dir):
            os.makedirs(font_dir)
            print(f"Warning: Font directory {font_dir} does not exist. Please add Hindi fonts.")
            return []
        
        font_paths = []
        for file in os.listdir(font_dir):
            if file.endswith('.ttf') or file.endswith('.otf'):
                font_paths.append(os.path.join(font_dir, file))
        return font_paths
    
    def _create_text_image(self, text, font_path, font_size=32):
        """Create an image with the given text and font."""
        try:
            font = ImageFont.truetype(font_path, font_size)
            # Estimate text size for the image dimensions
            dummy_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
            text_width, text_height = dummy_draw.textsize(text, font=font)
            
            # Create image with white background
            img = Image.new('RGB', (text_width + 20, text_height + 20), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Draw text in black
            draw.text((10, 10), text, font=font, fill=(0, 0, 0))
            
            # Resize to fixed dimensions
            img = img.resize((Config.CONTENT_IMAGE_SIZE[1], Config.CONTENT_IMAGE_SIZE[0]))
            return img
        except Exception as e:
            print(f"Error creating text image with font {font_path}: {e}")
            # Return a blank image in case of an error
            return Image.new('RGB', (Config.CONTENT_IMAGE_SIZE[1], Config.CONTENT_IMAGE_SIZE[0]), color=(255, 255, 255))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Select a random font
        font_path = random.choice(self.font_paths)
        font_class = self.font_classes[font_path]
        
        # Generate random Hindi text (1-5 characters)
        text_length = random.randint(1, 5)
        text = ''.join(random.choice(self.chars) for _ in range(text_length))
        
        # Create the text image
        img = self._create_text_image(text, font_path)
        
        # Apply transformations if specified
        if self.transform:
            img = self.transform(img)
        
        return {'image': img, 'text': text, 'font_class': font_class, 'font_path': font_path}


class HindiTextDataset(Dataset):
    """
    Dataset for real Hindi text images for training the style transfer model.
    """
    def __init__(self, root_dir, transform=None, localized_transform=None, content_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.localized_transform = localized_transform
        self.content_transform = content_transform
        self.image_paths = []
        self.annotations = []
        
        # Load dataset annotations (assumes a format where each line contains: image_path, x1, y1, x2, y2, text)
        annotation_file = os.path.join(root_dir, 'annotations.txt')
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        img_path = parts[0]
                        bbox = [int(float(parts[1])), int(float(parts[2])), int(float(parts[3])), int(float(parts[4]))]
                        text = parts[5]
                        self.image_paths.append(os.path.join(root_dir, 'images', img_path))
                        self.annotations.append((bbox, text))
        else:
            print(f"Warning: Annotation file {annotation_file} not found.")
    
    def __len__(self):
        return len(self.image_paths)
    
    def _render_content_image(self, text):
        """Render the content text using a standard font."""
        img = Image.new('L', (Config.CONTENT_IMAGE_SIZE[1], Config.CONTENT_IMAGE_SIZE[0]), color=255)
        try:
            # Use a standard Hindi font
            font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_files', 'fonts', 'default_hindi.ttf')
            font = ImageFont.truetype(font_path, 32)
            draw = ImageDraw.Draw(img)
            
            # Calculate text position to center it
            dummy_draw = ImageDraw.Draw(Image.new('L', (1, 1)))
            text_width, text_height = dummy_draw.textsize(text, font=font)
            position = ((Config.CONTENT_IMAGE_SIZE[1] - text_width) // 2, (Config.CONTENT_IMAGE_SIZE[0] - text_height) // 2)
            
            # Draw text
            draw.text(position, text, font=font, fill=0)
        except Exception as e:
            print(f"Error rendering content image: {e}")
        
        if self.content_transform:
            img = self.content_transform(img)
        
        return img
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        bbox, text = self.annotations[idx]
        
        # Load the full image
        full_image = Image.open(img_path).convert('RGB')
        
        # Extract the localized image (word box with context)
        x1, y1, x2, y2 = bbox
        # Add context around the text box (20% padding)
        w, h = x2 - x1, y2 - y1
        context_x1 = max(0, x1 - int(w * 0.2))
        context_y1 = max(0, y1 - int(h * 0.2))
        context_x2 = min(full_image.width, x2 + int(w * 0.2))
        context_y2 = min(full_image.height, y2 + int(h * 0.2))
        
        localized_image = full_image.crop((context_x1, context_y1, context_x2, context_y2))
        
        # Extract the word image (text box only)
        word_image = full_image.crop((x1, y1, x2, y2))
        
        # Create the localized image with normalized text box coordinates
        normalized_bbox = [
            (x1 - context_x1) / (context_x2 - context_x1),
            (y1 - context_y1) / (context_y2 - context_y1),
            (x2 - context_x1) / (context_x2 - context_x1),
            (y2 - context_y1) / (context_y2 - context_y1),
        ]
        
        # Apply transforms if specified
        if self.transform:
            word_image = self.transform(word_image)
        
        if self.localized_transform:
            localized_image = self.localized_transform(localized_image)
        
        # Render the content image (text rendered with a standard font)
        content_image = self._render_content_image(text)
        
        # Create a different content text for the paired content
        content_text2 = text
        while content_text2 == text:
            # Generate a random Hindi text with similar length
            text_length = len(text)
            content_text2 = ''.join(random.choice(Config.ALL_CHARS) for _ in range(text_length))
        
        # Render the second content image
        content_image2 = self._render_content_image(content_text2)
        
        return {
            'localized_image': localized_image,
            'word_image': word_image,
            'content_image1': content_image,
            'content_image2': content_image2,
            'text': text,
            'content_text2': content_text2,
            'normalized_bbox': torch.tensor(normalized_bbox, dtype=torch.float32),
            'img_path': img_path
        }


def get_dataloaders(config):
    """
    Create and return dataloaders for training and validation.
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((config.CONTENT_IMAGE_SIZE[0], config.CONTENT_IMAGE_SIZE[1])),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    localized_transform = transforms.Compose([
        transforms.Resize((config.STYLE_IMAGE_SIZE, config.STYLE_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create datasets
    train_dataset = HindiTextDataset(
        os.path.join(config.DATA_DIR, 'train'),
        transform=transform,
        localized_transform=localized_transform,
        content_transform=content_transform
    )
    
    val_dataset = HindiTextDataset(
        os.path.join(config.DATA_DIR, 'val'),
        transform=transform,
        localized_transform=localized_transform,
        content_transform=content_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Create synthetic dataset for typeface classifier pretraining
    synth_dataset = HindiSynthTextDataset(
        config.DATA_DIR,
        num_samples=config.NUM_SYNTHETIC_FONTS * 100,
        transform=transform
    )
    
    synth_loader = DataLoader(
        synth_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    return train_loader, val_loader, synth_loader
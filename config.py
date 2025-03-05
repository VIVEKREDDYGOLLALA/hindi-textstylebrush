"""
Configuration settings for the Hindi TextStyleBrush model.
"""

import os
import torch

class Config:
    # General
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, 'data_files')
    OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Training settings
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    LEARNING_RATE = 0.002
    BETAS = (0.0, 0.99)
    NUM_EPOCHS = 200
    SAVE_INTERVAL = 10
    
    # Model settings
    STYLE_IMAGE_SIZE = 256
    CONTENT_IMAGE_SIZE = (64, 256)
    STYLE_DIM = 512
    CONTENT_DIM = 512
    
    # Loss weights
    LAMBDA_1 = 1.0      # Perceptual loss
    LAMBDA_2 = 500.0    # Texture loss
    LAMBDA_3 = 1.0      # Embedding loss
    LAMBDA_4 = 1.0      # Recognition loss
    LAMBDA_5 = 10.0     # Reconstruction loss
    LAMBDA_6 = 1.0      # Cyclic loss
    
    # Synthetic font dataset for typeface classifier
    NUM_SYNTHETIC_FONTS = 2000
    
    # Hindi specific
    HINDI_CHARS = "अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक्षत्रज्ञड़ढ़"
    HINDI_DIGITS = "०१२३४५६७८९"
    HINDI_VOWEL_SIGNS = "ािीुूृेैोौंःँ"
    HINDI_ADDITIONAL_CHARS = "।॥॰"
    
    # All characters
    ALL_CHARS = HINDI_CHARS + HINDI_DIGITS + HINDI_VOWEL_SIGNS + HINDI_ADDITIONAL_CHARS
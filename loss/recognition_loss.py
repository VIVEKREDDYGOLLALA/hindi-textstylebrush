"""
Recognition loss for the Hindi TextStyleBrush model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecognitionLoss(nn.Module):
    """
    Recognition loss to ensure the generated text images contain the correct content.
    Uses a pre-trained text recognizer.
    """
    def __init__(self, recognizer, char_to_idx, blank_idx=0):
        super(RecognitionLoss, self).__init__()
        self.recognizer = recognizer
        self.char_to_idx = char_to_idx
        self.blank_idx = blank_idx
        
        # Freeze the recognizer
        for param in self.recognizer.parameters():
            param.requires_grad = False
        
        # CTC loss
        self.ctc_loss = nn.CTCLoss(blank=blank_idx, reduction='mean', zero_infinity=True)
    
    def forward(self, images, texts):
        """
        Compute recognition loss using CTC loss.
        
        Args:
            images: Generated images [B, C, H, W]
            texts: Target text strings (list of strings)
        
        Returns:
            loss: Recognition loss
        """
        batch_size = images.size(0)
        
        # Convert images to grayscale if necessary
        if images.size(1) == 3:
            # Convert RGB to grayscale: 0.299R + 0.587G + 0.114B
            gray_images = 0.299 * images[:, 0, :, :] + 0.587 * images[:, 1, :, :] + 0.114 * images[:, 2, :, :]
            images = gray_images.unsqueeze(1)  # [B, 1, H, W]
        
        # Get predictions from recognizer
        predictions = self.recognizer(images)  # [B, T, num_classes]
        
        # Convert predictions to shape [T, B, C] for CTC loss
        log_probs = F.log_softmax(predictions, dim=2).permute(1, 0, 2)
        
        # Convert text strings to index tensors
        target_lengths = []
        targets = []
        
        for text in texts:
            # Convert each character to its index
            text_indices = [self.char_to_idx.get(char, self.char_to_idx.get(' ', 0)) for char in text]
            targets.extend(text_indices)
            target_lengths.append(len(text_indices))
        
        targets = torch.tensor(targets, dtype=torch.long, device=images.device)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=images.device)
        
        # Input lengths (assuming all have the same length)
        input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long, device=images.device)
        
        # Compute CTC loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        return loss


class AttentionRecognitionLoss(nn.Module):
    """
    Recognition loss using attention-based recognition model.
    """
    def __init__(self, recognizer, char_to_idx, eos_idx=None):
        super(AttentionRecognitionLoss, self).__init__()
        self.recognizer = recognizer
        self.char_to_idx = char_to_idx
        self.eos_idx = eos_idx if eos_idx is not None else len(char_to_idx)
        
        # Cross entropy loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Freeze the recognizer
        for param in self.recognizer.parameters():
            param.requires_grad = False
    
    def forward(self, images, texts):
        """
        Compute recognition loss using cross entropy.
        
        Args:
            images: Generated images [B, C, H, W]
            texts: Target text strings (list of strings)
        
        Returns:
            loss: Recognition loss
        """
        batch_size = images.size(0)
        
        # Convert images to grayscale if necessary
        if images.size(1) == 3:
            # Convert RGB to grayscale: 0.299R + 0.587G + 0.114B
            gray_images = 0.299 * images[:, 0, :, :] + 0.587 * images[:, 1, :, :] + 0.114 * images[:, 2, :, :]
            images = gray_images.unsqueeze(1)  # [B, 1, H, W]
        
        # Get predictions from recognizer
        predictions = self.recognizer(images)  # [B, max_len, num_classes]
        
        # Convert text strings to index tensors
        max_len = predictions.size(1)
        targets = torch.ones(batch_size, max_len, dtype=torch.long, device=images.device) * -1
        
        for i, text in enumerate(texts):
            # Convert each character to its index and add EOS token
            text_indices = [self.char_to_idx.get(char, self.char_to_idx.get(' ', 0)) for char in text]
            text_indices.append(self.eos_idx)  # Add EOS token
            
            # Pad if necessary
            length = min(len(text_indices), max_len)
            targets[i, :length] = torch.tensor(text_indices[:length], dtype=torch.long, device=images.device)
        
        # Reshape predictions for cross entropy
        pred_flat = predictions.view(-1, predictions.size(-1))
        targets_flat = targets.view(-1)
        
        # Compute cross entropy loss
        loss = self.criterion(pred_flat, targets_flat)
        
        return loss
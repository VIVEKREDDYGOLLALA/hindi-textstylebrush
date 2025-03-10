"""
Perceptual loss functions for the Hindi TextStyleBrush model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using features from the typeface classifier.
    Includes conventional perceptual loss, texture loss (Gram matrix), and embedding loss.
    """
    def __init__(self, typeface_classifier, layers=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']):
        super(PerceptualLoss, self).__init__()
        self.typeface_classifier = typeface_classifier
        self.layers = layers
        
        # Freeze the typeface classifier
        for param in self.typeface_classifier.parameters():
            param.requires_grad = False
    
    def forward(self, generated_images, target_images, lambda_per=1.0, lambda_tex=500.0, lambda_emb=1.0):
        """
        Compute perceptual, texture, and embedding losses.
        
        Args:
            generated_images: Generated images [B, 3, H, W]
            target_images: Target images [B, 3, H, W]
            lambda_per: Weight for perceptual loss
            lambda_tex: Weight for texture loss
            lambda_emb: Weight for embedding loss
        
        Returns:
            total_loss: Combined loss
            losses: Dictionary of individual losses
        """
        losses = {}
        
        # Perceptual loss using feature differences
        per_loss = 0.0
        for layer in self.layers:
            gen_features = self.typeface_classifier.extract_features(generated_images, layer)
            target_features = self.typeface_classifier.extract_features(target_images, layer)
            
            # L1 loss between features
            layer_loss = F.l1_loss(gen_features, target_features)
            per_loss += layer_loss
            losses[f'perceptual_{layer}'] = layer_loss.item()
        
        # Texture loss using Gram matrices
        tex_loss = 0.0
        for layer in self.layers:
            gen_features = self.typeface_classifier.extract_features(generated_images, layer)
            target_features = self.typeface_classifier.extract_features(target_images, layer)
            
            gen_gram = self.typeface_classifier.get_gram_matrix(gen_features)
            target_gram = self.typeface_classifier.get_gram_matrix(target_features)
            
            # L1 loss between Gram matrices
            layer_loss = F.l1_loss(gen_gram, target_gram)
            tex_loss += layer_loss
            losses[f'texture_{layer}'] = layer_loss.item()
        
        # Embedding loss using the penultimate layer of the classifier
        gen_embedding = self.typeface_classifier.extract_features(generated_images, 'classifier')
        target_embedding = self.typeface_classifier.extract_features(target_images, 'classifier')
        emb_loss = F.l1_loss(gen_embedding, target_embedding)
        losses['embedding'] = emb_loss.item()
        
        # Combine losses with weights
        total_loss = lambda_per * per_loss + lambda_tex * tex_loss + lambda_emb * emb_loss
        losses['total'] = total_loss.item()
        
        return total_loss, losses
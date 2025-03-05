"""
Typeface classifier for the Hindi TextStyleBrush model.
Pre-trained on synthetic Hindi fonts to provide perceptual loss for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TypefaceClassifier(nn.Module):
    """
    Font/Typeface classifier based on VGG19.
    Used for perceptual loss computation.
    """
    def __init__(self, num_classes=2000, pretrained=True):
        super(TypefaceClassifier, self).__init__()
        
        # Load VGG19 with batch normalization, pretrained on ImageNet
        vgg = models.vgg19_bn(pretrained=pretrained)
        
        # Extract features part (remove classifier)
        self.features = vgg.features
        
        # Intermediate feature layers for perceptual and texture losses
        self.relu1_1 = nn.Sequential(*list(self.features.children())[:2])
        self.relu2_1 = nn.Sequential(*list(self.features.children())[:7])
        self.relu3_1 = nn.Sequential(*list(self.features.children())[:14])
        self.relu4_1 = nn.Sequential(*list(self.features.children())[:27])
        self.relu5_1 = nn.Sequential(*list(self.features.children())[:40])
        
        # Modify classifier for our number of classes
        self.avgpool = vgg.avgpool
        
        classifier_layers = list(vgg.classifier.children())
        # Replace the last layer with our own
        classifier_layers[-1] = nn.Linear(4096, num_classes)
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Initialize the new layer
        nn.init.normal_(classifier_layers[-1].weight, 0, 0.01)
        nn.init.zeros_(classifier_layers[-1].bias)
    
    def forward(self, x):
        """
        Forward pass of the typeface classifier.
        
        Args:
            x: Input image [B, 3, H, W]
        
        Returns:
            class_output: Class logits [B, num_classes]
        """
        # Feed forward through the network
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        class_output = self.classifier(x)
        
        return class_output
    
    def extract_features(self, x, layer):
        """
        Extract features at a specific layer for perceptual loss computation.
        
        Args:
            x: Input image [B, 3, H, W]
            layer: Layer name: 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1', or 'classifier'
        
        Returns:
            Features at the specified layer
        """
        if layer == 'relu1_1':
            return self.relu1_1(x)
        elif layer == 'relu2_1':
            return self.relu2_1(x)
        elif layer == 'relu3_1':
            return self.relu3_1(x)
        elif layer == 'relu4_1':
            return self.relu4_1(x)
        elif layer == 'relu5_1':
            return self.relu5_1(x)
        elif layer == 'classifier':
            features = self.features(x)
            features = self.avgpool(features)
            features = torch.flatten(features, 1)
            features = self.classifier[:-1](features)  # All except the last layer
            return features
        else:
            raise ValueError(f"Unknown layer: {layer}")
    
    def get_gram_matrix(self, features):
        """
        Compute the Gram matrix for texture loss.
        
        Args:
            features: Feature map [B, C, H, W]
        
        Returns:
            Gram matrix [B, C, C]
        """
        batch_size, channels, height, width = features.size()
        features_reshaped = features.view(batch_size, channels, height * width)
        features_transposed = features_reshaped.permute(0, 2, 1)
        
        # Gram matrix computation
        gram = torch.bmm(features_reshaped, features_transposed)
        
        # Normalize by the number of elements
        return gram / (channels * height * width)
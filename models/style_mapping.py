"""
Style mapping network for the Hindi TextStyleBrush model.
Converts style vectors to layer-specific style parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleMappingNetwork(nn.Module):
    """
    Maps the style vector to multiple layer-specific style vectors
    for use in the generator's adaptive instance normalization.
    """
    def __init__(self, style_dim=512, num_layers=15):
        super(StyleMappingNetwork, self).__init__()
        self.style_dim = style_dim
        self.num_layers = num_layers
        
        # Normalization layer
        self.normalize = Normalize()
        
        # Fully connected layers
        self.fc1 = nn.Linear(style_dim, style_dim)
        self.fc2 = nn.Linear(style_dim, style_dim * num_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the network."""
        nn.init.kaiming_normal_(self.fc1.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        """
        Forward pass of the style mapping network.
        
        Args:
            x: Style vector [B, style_dim]
        
        Returns:
            List of layer-specific style vectors, each of shape [B, style_dim]
        """
        # Normalize input
        x = self.normalize(x)
        
        # First fully connected layer
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        
        # Second fully connected layer to get all style vectors at once
        x = self.fc2(x)
        
        # Reshape to get separate style vectors for each layer
        x = x.view(-1, self.num_layers, self.style_dim)
        
        # Create a list of style vectors
        style_vectors = [x[:, i, :] for i in range(self.num_layers)]
        
        return style_vectors


class Normalize(nn.Module):
    """
    Normalizes the input vector to unit length.
    """
    def __init__(self, eps=1e-8):
        super(Normalize, self).__init__()
        self.eps = eps
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, C]
        
        Returns:
            Normalized tensor [B, C]
        """
        norm = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + self.eps)
        return x / norm
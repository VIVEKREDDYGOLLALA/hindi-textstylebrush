"""
Text recognizer module for the Hindi TextStyleBrush model.
This component is pre-trained and used for the content loss during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence modeling in text recognition.
    """
    def __init__(self, input_size, hidden_size, output_size=None):
        super(BidirectionalLSTM, self).__init__()
        
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        
        if output_size is not None:
            self.embedding = nn.Linear(hidden_size * 2, output_size)
        else:
            self.embedding = None
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
        
        Returns:
            output: Output tensor [batch_size, seq_len, output_size or hidden_size*2]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(x)
        
        if self.embedding is not None:
            output = self.embedding(recurrent)
        else:
            output = recurrent
            
        return output


class TPS_SpatialTransformerNetwork(nn.Module):
    """
    Thin Plate Spline Spatial Transformer Network for rectifying text images.
    Simplified version for the implementation.
    """
    def __init__(self, input_size, output_size, num_control_points=20):
        super(TPS_SpatialTransformerNetwork, self).__init__()
        self.input_size = input_size  # (height, width)
        self.output_size = output_size  # (height, width)
        self.num_control_points = num_control_points
        
        # Localization network
        self.localization_network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # 1/2
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # 1/4
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # 1/8
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),  # 1/16
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, num_control_points * 2)
        )
        
        # Initialize the localization network to produce identity transformation
        self.localization_network[-1].weight.data.zero_()
        self.localization_network[-1].bias.data.copy_(
            torch.tensor([0.5, 0] * num_control_points, dtype=torch.float)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, C, H, W]
        
        Returns:
            warped_image: Rectified image [batch_size, C, output_height, output_width]
        """
        batch_size, _, h, w = x.size()
        
        # Convert to grayscale if not already
        if x.size(1) == 3:
            x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
            x_gray = x_gray.unsqueeze(1)
        else:
            x_gray = x
        
        # Predict control points
        points = self.localization_network(x_gray)
        points = points.view(batch_size, self.num_control_points, 2)
        
        # For simplicity, we'll use a built-in grid sampler instead of implementing TPS
        # In a full implementation, this would use the predicted control points to generate a sampling grid
        
        # Create a default grid (identity transform)
        grid = F.affine_grid(
            torch.eye(2, 3).unsqueeze(0).repeat(batch_size, 1, 1).to(x.device),
            size=torch.Size([batch_size, x.size(1), self.output_size[0], self.output_size[1]])
        )
        
        # Apply the grid to sample from the input
        warped_image = F.grid_sample(x, grid, align_corners=True)
        
        return warped_image


class ResidualBlock(nn.Module):
    """Residual block for the feature extraction network."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class FeatureExtraction(nn.Module):
    """Feature extraction network based on ResNet."""
    def __init__(self, input_channel=1, output_channel=512):
        super(FeatureExtraction, self).__init__()
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 1/2
            
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2),  # 1/4
            
            ResidualBlock(128, 256),
            nn.MaxPool2d(2, 2),  # 1/8
            
            ResidualBlock(256, output_channel),
            nn.MaxPool2d((2, 1), (2, 1)),  # 1/16 height, same width
        )
    
    def forward(self, x):
        return self.ConvNet(x)


class SequenceModeling(nn.Module):
    """
    Sequence modeling with Bidirectional LSTM.
    """
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(SequenceModeling, self).__init__()
        self.LSTM = nn.Sequential(
            BidirectionalLSTM(input_size, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
    
    def forward(self, x):
        # [batch_size, feature_size, seq_len] -> [batch_size, seq_len, feature_size]
        x = x.permute(0, 2, 1)
        x = self.LSTM(x)
        return x


class Prediction(nn.Module):
    """
    Prediction layer with attention mechanism.
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(Prediction, self).__init__()
        self.attention = nn.Linear(input_size, hidden_size)
        self.character_distribution = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Attention
        attention = F.softmax(self.attention(x), dim=1)
        # Apply attention
        context = torch.bmm(attention.permute(0, 2, 1), x)
        # Predict character distribution
        prediction = self.character_distribution(context)
        return prediction


class TextRecognizer(nn.Module):
    """
    Complete text recognition model with TPS transformation, feature extraction,
    sequence modeling, and prediction layers.
    """
    def __init__(self, num_classes, input_size=(32, 128)):
        super(TextRecognizer, self).__init__()
        
        # TPS transformation
        self.transformation = TPS_SpatialTransformerNetwork(
            input_size=input_size, output_size=input_size
        )
        
        # Feature extraction
        self.feature_extraction = FeatureExtraction(input_channel=1, output_channel=512)
        
        # Calculate output size of feature extraction
        feature_h = input_size[0] // 16  # After 4 max-pooling layers with factors (2,2), (2,2), (2,2), (2,1)
        feature_w = input_size[1] // 8   # After 3 max-pooling layers with factor 2 in width
        
        # Sequence modeling
        self.sequence_modeling = SequenceModeling(
            input_size=512, hidden_size=256
        )
        
        # Prediction
        self.prediction = Prediction(
            input_size=256 * 2,  # bidirectional
            hidden_size=256,
            num_classes=num_classes
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, C, H, W]
        
        Returns:
            prediction: Character probabilities at each position [batch_size, seq_len, num_classes]
        """
        # TPS transformation
        x = self.transformation(x)
        
        # Feature extraction
        visual_feature = self.feature_extraction(x)
        
        # Sequence modeling
        contextual_feature = self.sequence_modeling(visual_feature)
        
        # Prediction
        prediction = self.prediction(contextual_feature)
        
        return prediction
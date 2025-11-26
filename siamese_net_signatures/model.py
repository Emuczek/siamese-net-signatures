"""
Siamese Network Architecture for Signature Verification

This module contains the neural network architecture:
- CNN backbone for feature extraction
- Embedding network
- Siamese network wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBackbone(nn.Module):
    """
    Convolutional Neural Network backbone for feature extraction.
    Processes grayscale signature images and extracts feature representations.
    """
    
    def __init__(self, input_channels=1, conv_filters=[32, 64, 128, 256], 
                 conv_kernel_size=3, pool_size=2, dropout_rate=0.3):
        """
        Args:
            input_channels (int): Number of input channels (1 for grayscale)
            conv_filters (list): Number of filters in each convolutional layer
            conv_kernel_size (int): Kernel size for convolutions
            pool_size (int): Pool size for max pooling
            dropout_rate (float): Dropout probability
        """
        super(CNNBackbone, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # Build convolutional layers
        in_channels = input_channels
        for out_channels in conv_filters:
            # Convolutional layer
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                bias=False
            )
            self.conv_layers.append(conv)
            
            # Batch normalization
            bn = nn.BatchNorm2d(out_channels)
            self.bn_layers.append(bn)
            
            in_channels = out_channels
        
        self.num_conv_layers = len(conv_filters)
    
    def forward(self, x):
        """
        Forward pass through the CNN backbone.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, H, W)
            
        Returns:
            torch.Tensor: Feature maps of shape (batch_size, channels, h, w)
        """
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x)
            x = F.relu(x)
            x = self.pool(x)
            
            # Apply dropout after every second conv layer
            if i % 2 == 1:
                x = self.dropout(x)
        
        return x


class EmbeddingNetwork(nn.Module):
    """
    Complete embedding network: CNN backbone + fully connected layers.
    Outputs fixed-size embedding vectors for signature images.
    """
    
    def __init__(self, input_shape=(1, 224, 224), conv_filters=[32, 64, 128, 256],
                 fc_layers=[512, 256], embedding_dim=128, dropout_rate=0.3):
        """
        Args:
            input_shape (tuple): Shape of input images (channels, height, width)
            conv_filters (list): Number of filters in each convolutional layer
            fc_layers (list): Dimensions of fully connected layers
            embedding_dim (int): Output embedding dimension
            dropout_rate (float): Dropout probability
        """
        super(EmbeddingNetwork, self).__init__()
        
        # CNN backbone
        self.backbone = CNNBackbone(
            input_channels=input_shape[0],
            conv_filters=conv_filters,
            dropout_rate=dropout_rate
        )
        
        # Calculate the flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.backbone(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).shape[1]
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        self.fc_bn_layers = nn.ModuleList()
        
        in_features = self.flattened_size
        for out_features in fc_layers:
            fc = nn.Linear(in_features, out_features)
            self.fc_layers.append(fc)
            
            bn = nn.BatchNorm1d(out_features)
            self.fc_bn_layers.append(bn)
            
            in_features = out_features
        
        # Final embedding layer
        self.embedding_layer = nn.Linear(in_features, embedding_dim)
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.num_fc_layers = len(fc_layers)
    
    def forward(self, x):
        """
        Forward pass through the embedding network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, H, W)
            
        Returns:
            torch.Tensor: Embedding vector of shape (batch_size, embedding_dim)
        """
        # Extract features with CNN backbone
        x = self.backbone(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU and dropout
        for i in range(self.num_fc_layers):
            x = self.fc_layers[i](x)
            x = self.fc_bn_layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final embedding (no activation - let the loss function handle it)
        x = self.embedding_layer(x)
        
        # L2 normalize the embeddings
        x = F.normalize(x, p=2, dim=1)
        
        return x


class SiameseNetwork(nn.Module):
    """
    Siamese Network wrapper.
    Takes two images as input and outputs their embeddings.
    """
    
    def __init__(self, embedding_network):
        """
        Args:
            embedding_network (nn.Module): The embedding network to use
        """
        super(SiameseNetwork, self).__init__()
        self.embedding_network = embedding_network
    
    def forward_once(self, x):
        """
        Forward pass for a single image.
        
        Args:
            x (torch.Tensor): Input image
            
        Returns:
            torch.Tensor: Embedding vector
        """
        return self.embedding_network(x)
    
    def forward(self, x1, x2):
        """
        Forward pass for a pair of images.
        
        Args:
            x1 (torch.Tensor): First image
            x2 (torch.Tensor): Second image
            
        Returns:
            tuple: (embedding1, embedding2)
        """
        embedding1 = self.forward_once(x1)
        embedding2 = self.forward_once(x2)
        return embedding1, embedding2


class SiameseTripletNetwork(nn.Module):
    """
    Siamese Network for triplet loss.
    Takes three images (anchor, positive, negative) and outputs their embeddings.
    """
    
    def __init__(self, embedding_network):
        """
        Args:
            embedding_network (nn.Module): The embedding network to use
        """
        super(SiameseTripletNetwork, self).__init__()
        self.embedding_network = embedding_network
    
    def forward_once(self, x):
        """
        Forward pass for a single image.
        
        Args:
            x (torch.Tensor): Input image
            
        Returns:
            torch.Tensor: Embedding vector
        """
        return self.embedding_network(x)
    
    def forward(self, anchor, positive, negative):
        """
        Forward pass for a triplet of images.
        
        Args:
            anchor (torch.Tensor): Anchor image
            positive (torch.Tensor): Positive image (same person as anchor)
            negative (torch.Tensor): Negative image (different person)
            
        Returns:
            tuple: (anchor_embedding, positive_embedding, negative_embedding)
        """
        anchor_embedding = self.forward_once(anchor)
        positive_embedding = self.forward_once(positive)
        negative_embedding = self.forward_once(negative)
        return anchor_embedding, positive_embedding, negative_embedding


def create_siamese_model(config, loss_type='contrastive'):
    """
    Factory function to create a Siamese network model.
    
    Args:
        config: Configuration module with model parameters
        loss_type (str): Type of loss - 'contrastive' or 'triplet'
        
    Returns:
        nn.Module: Siamese network model
    """
    # Create embedding network
    embedding_net = EmbeddingNetwork(
        input_shape=(config.IMG_CHANNELS, config.IMG_HEIGHT, config.IMG_WIDTH),
        conv_filters=config.CONV_FILTERS,
        fc_layers=config.FC_LAYERS,
        embedding_dim=config.EMBEDDING_DIM,
        dropout_rate=config.DROPOUT_RATE
    )
    
    # Wrap in appropriate Siamese network
    if loss_type == 'triplet':
        model = SiameseTripletNetwork(embedding_net)
    else:
        model = SiameseNetwork(embedding_net)
    
    return model


if __name__ == '__main__':
    # Test the model architecture
    import sys
    sys.path.append('.')
    import config
    
    print("Testing Siamese Network Architecture...")
    print(f"Device: {config.DEVICE}")
    
    # Create model for contrastive loss
    model_contrastive = create_siamese_model(config, loss_type='contrastive')
    model_contrastive = model_contrastive.to(config.DEVICE)
    
    # Test with dummy data
    batch_size = 4
    dummy_img1 = torch.randn(batch_size, 1, 224, 224).to(config.DEVICE)
    dummy_img2 = torch.randn(batch_size, 1, 224, 224).to(config.DEVICE)
    
    # Forward pass
    emb1, emb2 = model_contrastive(dummy_img1, dummy_img2)
    
    print(f"\nContrastive Loss Model:")
    print(f"Input shape: {dummy_img1.shape}")
    print(f"Embedding 1 shape: {emb1.shape}")
    print(f"Embedding 2 shape: {emb2.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model_contrastive.parameters())
    trainable_params = sum(p.numel() for p in model_contrastive.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create model for triplet loss
    model_triplet = create_siamese_model(config, loss_type='triplet')
    model_triplet = model_triplet.to(config.DEVICE)
    
    dummy_img3 = torch.randn(batch_size, 1, 224, 224).to(config.DEVICE)
    emb_anchor, emb_pos, emb_neg = model_triplet(dummy_img1, dummy_img2, dummy_img3)
    
    print(f"\nTriplet Loss Model:")
    print(f"Anchor embedding shape: {emb_anchor.shape}")
    print(f"Positive embedding shape: {emb_pos.shape}")
    print(f"Negative embedding shape: {emb_neg.shape}")
    
    print("\n✓ Model architecture test passed!")

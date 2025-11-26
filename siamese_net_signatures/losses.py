"""
Loss Functions for Siamese Networks

This module implements:
- Contrastive Loss for pair-based training
- Triplet Loss for triplet-based training
- Hard mining strategies for triplet loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks.
    
    The loss encourages similar pairs to have small distances and dissimilar pairs
    to have distances larger than a margin.
    
    Loss = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    
    where Y=0 for genuine pairs and Y=1 for impostor pairs, D is Euclidean distance.
    """
    
    def __init__(self, margin=2.0):
        """
        Args:
            margin (float): Margin for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        """
        Forward pass of contrastive loss.
        
        Args:
            embedding1 (torch.Tensor): First embedding vectors (batch_size, embedding_dim)
            embedding2 (torch.Tensor): Second embedding vectors (batch_size, embedding_dim)
            label (torch.Tensor): Binary labels (batch_size,)
                                  0 = genuine pair (same person)
                                  1 = impostor pair (different people)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Calculate Euclidean distance
        euclidean_distance = F.pairwise_distance(embedding1, embedding2, p=2)
        
        # Contrastive loss formula
        loss_genuine = (1 - label) * torch.pow(euclidean_distance, 2)
        loss_impostor = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        
        loss = torch.mean(loss_genuine + loss_impostor) / 2.0
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss for Siamese Networks.
    
    The loss encourages the distance between anchor and positive to be smaller
    than the distance between anchor and negative by at least a margin.
    
    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin (float): Minimum distance difference between positive and negative
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        """
        Forward pass of triplet loss.
        
        Args:
            anchor (torch.Tensor): Anchor embeddings (batch_size, embedding_dim)
            positive (torch.Tensor): Positive embeddings (batch_size, embedding_dim)
            negative (torch.Tensor): Negative embeddings (batch_size, embedding_dim)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Calculate distances
        distance_positive = F.pairwise_distance(anchor, positive, p=2)
        distance_negative = F.pairwise_distance(anchor, negative, p=2)
        
        # Triplet loss
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean()


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss.
    
    For each anchor, select the hardest positive (furthest positive)
    and hardest negative (closest negative) within the batch.
    
    This is more efficient and often more effective than random triplet selection.
    """
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin (float): Minimum distance difference between positive and negative
        """
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        """
        Forward pass with batch hard mining.
        
        Args:
            embeddings (torch.Tensor): Embeddings (batch_size, embedding_dim)
            labels (torch.Tensor): Labels (batch_size,)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # Calculate pairwise distances
        pairwise_dist = self._pairwise_distances(embeddings)
        
        # For each anchor, get the hardest positive and hardest negative
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)
        
        # Hardest positive: maximum distance among valid positives
        anchor_positive_dist = mask_anchor_positive.float() * pairwise_dist
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1, keepdim=True)
        
        # Hardest negative: minimum distance among valid negatives
        # Add maximum value to invalid negatives so they are not picked
        max_anchor_negative_dist = pairwise_dist.max().detach()
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative.float())
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1, keepdim=True)
        
        # Combine to form triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return triplet_loss.mean()
    
    @staticmethod
    def _pairwise_distances(embeddings):
        """
        Compute pairwise distances between embeddings.
        
        Args:
            embeddings (torch.Tensor): (batch_size, embedding_dim)
        
        Returns:
            torch.Tensor: Distance matrix (batch_size, batch_size)
        """
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)  # Ensure non-negative
        
        # Add small epsilon to diagonal for numerical stability
        mask = torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
        distances = distances + mask.float() * 1e-16
        
        distances = torch.sqrt(distances)
        
        # Zero out diagonal
        distances = distances * (1.0 - mask.float())
        
        return distances
    
    @staticmethod
    def _get_anchor_positive_triplet_mask(labels):
        """
        Return a 2D mask where mask[a, p] is True if a and p have the same label.
        
        Args:
            labels (torch.Tensor): (batch_size,)
        
        Returns:
            torch.Tensor: Boolean mask (batch_size, batch_size)
        """
        # Check if labels[i] == labels[j]
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Exclude diagonal (an anchor cannot be its own positive)
        indices_not_equal = ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        
        return labels_equal & indices_not_equal
    
    @staticmethod
    def _get_anchor_negative_triplet_mask(labels):
        """
        Return a 2D mask where mask[a, n] is True if a and n have different labels.
        
        Args:
            labels (torch.Tensor): (batch_size,)
        
        Returns:
            torch.Tensor: Boolean mask (batch_size, batch_size)
        """
        # Check if labels[i] != labels[j]
        return labels.unsqueeze(0) != labels.unsqueeze(1)


class OnlineTripletLoss(nn.Module):
    """
    Online Triplet Loss with different mining strategies.
    
    Supports:
    - batch_hard: Hardest positive and hardest negative for each anchor
    - batch_all: All valid triplets in the batch
    """
    
    def __init__(self, margin=1.0, mining='batch_hard'):
        """
        Args:
            margin (float): Minimum distance difference
            mining (str): Mining strategy - 'batch_hard' or 'batch_all'
        """
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.mining = mining
    
    def forward(self, embeddings, labels):
        """
        Forward pass with online mining.
        
        Args:
            embeddings (torch.Tensor): Embeddings (batch_size, embedding_dim)
            labels (torch.Tensor): Labels (batch_size,)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        if self.mining == 'batch_hard':
            return self._batch_hard_triplet_loss(embeddings, labels)
        elif self.mining == 'batch_all':
            return self._batch_all_triplet_loss(embeddings, labels)
        else:
            raise ValueError(f"Unknown mining strategy: {self.mining}")
    
    def _batch_hard_triplet_loss(self, embeddings, labels):
        """Batch hard mining strategy."""
        pairwise_dist = self._pairwise_distances(embeddings)
        
        mask_anchor_positive = self._get_anchor_positive_mask(labels)
        mask_anchor_negative = self._get_anchor_negative_mask(labels)
        
        # Hardest positive
        anchor_positive_dist = mask_anchor_positive.float() * pairwise_dist
        hardest_positive_dist, _ = anchor_positive_dist.max(dim=1)
        
        # Hardest negative
        max_dist = pairwise_dist.max().detach()
        anchor_negative_dist = pairwise_dist + max_dist * (1.0 - mask_anchor_negative.float())
        hardest_negative_dist, _ = anchor_negative_dist.min(dim=1)
        
        # Triplet loss
        triplet_loss = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)
        
        return triplet_loss.mean()
    
    def _batch_all_triplet_loss(self, embeddings, labels):
        """Batch all mining strategy - uses all valid triplets."""
        pairwise_dist = self._pairwise_distances(embeddings)
        
        # Get valid triplet mask
        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)
        
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        
        # Get valid triplets mask
        mask = self._get_triplet_mask(labels)
        triplet_loss = mask.float() * triplet_loss
        
        # Remove negative losses
        triplet_loss = F.relu(triplet_loss)
        
        # Count valid triplets
        num_positive_triplets = (triplet_loss > 1e-16).float().sum()
        
        # Average over valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
        
        return triplet_loss
    
    @staticmethod
    def _pairwise_distances(embeddings):
        """Compute pairwise distances."""
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)
        
        mask = torch.eye(distances.size(0), dtype=torch.bool, device=distances.device)
        distances = distances + mask.float() * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask.float())
        
        return distances
    
    @staticmethod
    def _get_anchor_positive_mask(labels):
        """Get mask for valid anchor-positive pairs."""
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        indices_not_equal = ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        return labels_equal & indices_not_equal
    
    @staticmethod
    def _get_anchor_negative_mask(labels):
        """Get mask for valid anchor-negative pairs."""
        return labels.unsqueeze(0) != labels.unsqueeze(1)
    
    @staticmethod
    def _get_triplet_mask(labels):
        """Get mask for valid triplets (a, p, n)."""
        # Check that i, j, k are distinct
        indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)
        distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k
        
        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)
        valid_labels = i_equal_j & (~i_equal_k)
        
        return distinct_indices & valid_labels


def get_loss_function(loss_type, margin):
    """
    Factory function to get the appropriate loss function.
    
    Args:
        loss_type (str): Type of loss - 'contrastive', 'triplet', or 'online_triplet'
        margin (float): Margin parameter
    
    Returns:
        nn.Module: Loss function
    """
    if loss_type == 'contrastive':
        return ContrastiveLoss(margin=margin)
    elif loss_type == 'triplet':
        return TripletLoss(margin=margin)
    elif loss_type == 'batch_hard':
        return BatchHardTripletLoss(margin=margin)
    elif loss_type.startswith('online_'):
        mining_strategy = loss_type.split('_', 1)[1]
        return OnlineTripletLoss(margin=margin, mining=mining_strategy)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test loss functions
    print("Testing Loss Functions...")
    
    # Test Contrastive Loss
    print("\n1. Testing Contrastive Loss:")
    contrastive_loss = ContrastiveLoss(margin=2.0)
    
    emb1 = torch.randn(8, 128)
    emb2 = torch.randn(8, 128)
    labels = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])  # 0=genuine, 1=impostor
    
    loss = contrastive_loss(emb1, emb2, labels)
    print(f"Contrastive Loss: {loss.item():.4f}")
    
    # Test Triplet Loss
    print("\n2. Testing Triplet Loss:")
    triplet_loss = TripletLoss(margin=1.0)
    
    anchor = torch.randn(8, 128)
    positive = torch.randn(8, 128)
    negative = torch.randn(8, 128)
    
    loss = triplet_loss(anchor, positive, negative)
    print(f"Triplet Loss: {loss.item():.4f}")
    
    # Test Batch Hard Triplet Loss
    print("\n3. Testing Batch Hard Triplet Loss:")
    batch_hard_loss = BatchHardTripletLoss(margin=1.0)
    
    embeddings = torch.randn(16, 128)
    user_labels = torch.tensor([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6])
    
    loss = batch_hard_loss(embeddings, user_labels)
    print(f"Batch Hard Triplet Loss: {loss.item():.4f}")
    
    print("\n✓ All loss functions tested successfully!")

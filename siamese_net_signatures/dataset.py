"""
Dataset Module for Signature Verification

This module implements PyTorch datasets for:
- Pair-based training (Contrastive Loss)
- Triplet-based training (Triplet Loss)
- Data augmentation
"""

import os
import random
import glob
import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class SignaturePairDataset(Dataset):
    """
    Dataset for loading signature pairs for Contrastive Loss training.
    
    Generates both genuine pairs (same person) and impostor pairs (different people).
    """
    
    def __init__(self, data_dir, transform=None, pairs_per_user=10):
        """
        Args:
            data_dir (str): Path to directory containing user signature folders
            transform: Optional transform to apply to images
            pairs_per_user (int): Number of pairs to generate per user
        """
        self.data_dir = data_dir
        self.transform = transform
        self.pairs_per_user = pairs_per_user
        
        # Load all user data
        self.user_data = self._load_user_data()
        self.user_ids = list(self.user_data.keys())
        
        # Generate pairs
        self.pairs = self._generate_pairs()
        
        print(f"Loaded {len(self.user_ids)} users")
        print(f"Generated {len(self.pairs)} pairs")
    
    def _load_user_data(self):
        """Load all signature images organized by user."""
        user_data = {}
        
        # Get all user directories
        user_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for user_id in user_dirs:
            user_path = os.path.join(self.data_dir, user_id)
            
            # Load original signatures
            original_path = os.path.join(user_path, 'original')
            original_files = glob.glob(os.path.join(original_path, '*.png'))
            
            # Load forged signatures
            forged_path = os.path.join(user_path, 'forged')
            forged_files = glob.glob(os.path.join(forged_path, '*.png'))
            
            if original_files:  # Only add users with signatures
                user_data[user_id] = {
                    'original': original_files,
                    'forged': forged_files
                }
        
        return user_data
    
    def _generate_pairs(self):
        """Generate genuine and impostor pairs."""
        pairs = []
        
        for user_id in self.user_ids:
            user_originals = self.user_data[user_id]['original']
            user_forged = self.user_data[user_id]['forged']
            
            # Generate genuine pairs (same person - both original)
            if len(user_originals) >= 2:
                for _ in range(self.pairs_per_user // 2):
                    img1, img2 = random.sample(user_originals, 2)
                    pairs.append((img1, img2, 0))  # 0 = genuine pair
            
            # Generate impostor pairs (original vs forged)
            if user_forged:
                for _ in range(self.pairs_per_user // 2):
                    img1 = random.choice(user_originals)
                    img2 = random.choice(user_forged)
                    pairs.append((img1, img2, 1))  # 1 = impostor pair
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a pair of images and their label.
        
        Returns:
            tuple: (img1, img2, label)
                   label = 0 for genuine pair, 1 for impostor pair
        """
        img1_path, img2_path, label = self.pairs[idx]
        
        # Load images
        img1 = cv.imread(img1_path, cv.IMREAD_GRAYSCALE)
        img2 = cv.imread(img2_path, cv.IMREAD_GRAYSCALE)
        
        # Normalize to [0, 1]
        img1 = img1.astype('float32') / 255.0
        img2 = img2.astype('float32') / 255.0
        
        # Convert to torch tensors
        img1 = torch.from_numpy(img1).unsqueeze(0)  # Add channel dimension
        img2 = torch.from_numpy(img2).unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return img1, img2, label


class SignatureTripletDataset(Dataset):
    """
    Dataset for loading signature triplets for Triplet Loss training.
    
    Each sample consists of: anchor, positive (same person), negative (different person).
    """
    
    def __init__(self, data_dir, transform=None, triplets_per_user=10):
        """
        Args:
            data_dir (str): Path to directory containing user signature folders
            transform: Optional transform to apply to images
            triplets_per_user (int): Number of triplets to generate per user
        """
        self.data_dir = data_dir
        self.transform = transform
        self.triplets_per_user = triplets_per_user
        
        # Load all user data
        self.user_data = self._load_user_data()
        self.user_ids = list(self.user_data.keys())
        
        # Generate triplets
        self.triplets = self._generate_triplets()
        
        print(f"Loaded {len(self.user_ids)} users")
        print(f"Generated {len(self.triplets)} triplets")
    
    def _load_user_data(self):
        """Load all signature images organized by user."""
        user_data = {}
        
        user_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for user_id in user_dirs:
            user_path = os.path.join(self.data_dir, user_id)
            
            original_path = os.path.join(user_path, 'original')
            original_files = glob.glob(os.path.join(original_path, '*.png'))
            
            forged_path = os.path.join(user_path, 'forged')
            forged_files = glob.glob(os.path.join(forged_path, '*.png'))
            
            if original_files:
                user_data[user_id] = {
                    'original': original_files,
                    'forged': forged_files
                }
        
        return user_data
    
    def _generate_triplets(self):
        """Generate anchor, positive, negative triplets."""
        triplets = []
        
        for user_id in self.user_ids:
            user_originals = self.user_data[user_id]['original']
            
            # Skip users with too few signatures
            if len(user_originals) < 2:
                continue
            
            # Get other users for negative samples
            other_users = [uid for uid in self.user_ids if uid != user_id]
            
            if not other_users:
                continue
            
            for _ in range(self.triplets_per_user):
                # Anchor and positive from same user
                anchor, positive = random.sample(user_originals, 2)
                
                # Negative from different user
                negative_user = random.choice(other_users)
                negative_originals = self.user_data[negative_user]['original']
                negative = random.choice(negative_originals)
                
                triplets.append((anchor, positive, negative))
        
        # Shuffle triplets
        random.shuffle(triplets)
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """
        Get a triplet of images.
        
        Returns:
            tuple: (anchor, positive, negative)
        """
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        # Load images
        anchor = cv.imread(anchor_path, cv.IMREAD_GRAYSCALE)
        positive = cv.imread(positive_path, cv.IMREAD_GRAYSCALE)
        negative = cv.imread(negative_path, cv.IMREAD_GRAYSCALE)
        
        # Normalize
        anchor = anchor.astype('float32') / 255.0
        positive = positive.astype('float32') / 255.0
        negative = negative.astype('float32') / 255.0
        
        # Convert to tensors
        anchor = torch.from_numpy(anchor).unsqueeze(0)
        positive = torch.from_numpy(positive).unsqueeze(0)
        negative = torch.from_numpy(negative).unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
        
        return anchor, positive, negative


class SignatureBatchDataset(Dataset):
    """
    Dataset for batch-based training (e.g., Batch Hard Triplet Loss).
    
    Returns images with their user labels, allowing the loss function
    to mine hard triplets within the batch.
    """
    
    def __init__(self, data_dir, transform=None, samples_per_user=10):
        """
        Args:
            data_dir (str): Path to directory containing user signature folders
            transform: Optional transform to apply to images
            samples_per_user (int): Number of samples to use per user
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples_per_user = samples_per_user
        
        # Load all user data
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for batch training")
    
    def _load_samples(self):
        """Load samples with user labels."""
        samples = []
        
        user_dirs = [d for d in os.listdir(self.data_dir) 
                     if os.path.isdir(os.path.join(self.data_dir, d))]
        
        for user_id in user_dirs:
            user_path = os.path.join(self.data_dir, user_id)
            original_path = os.path.join(user_path, 'original')
            original_files = glob.glob(os.path.join(original_path, '*.png'))
            
            # Sample images for this user
            num_samples = min(self.samples_per_user, len(original_files))
            sampled_files = random.sample(original_files, num_samples)
            
            for img_path in sampled_files:
                samples.append((img_path, int(user_id)))
        
        # Shuffle samples
        random.shuffle(samples)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get an image and its user label.
        
        Returns:
            tuple: (image, user_label)
        """
        img_path, user_label = self.samples[idx]
        
        # Load image
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = img.astype('float32') / 255.0
        
        # Convert to tensor
        img = torch.from_numpy(img).unsqueeze(0)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        user_label = torch.tensor(user_label, dtype=torch.long)
        
        return img, user_label


class SignatureAugmentation:
    """
    Data augmentation for signature images.
    """
    
    def __init__(self, rotation_range=10, translation_range=0.1, 
                 scale_range=(0.9, 1.1), apply_prob=0.5):
        """
        Args:
            rotation_range (float): Maximum rotation in degrees
            translation_range (float): Maximum translation as fraction of image size
            scale_range (tuple): Range for random scaling
            apply_prob (float): Probability of applying augmentation
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.apply_prob = apply_prob
    
    def __call__(self, img):
        """
        Apply augmentation to an image tensor.
        
        Args:
            img (torch.Tensor): Image tensor of shape (1, H, W)
        
        Returns:
            torch.Tensor: Augmented image
        """
        if random.random() > self.apply_prob:
            return img
        
        # Convert to numpy for OpenCV operations
        img_np = img.squeeze(0).numpy()
        h, w = img_np.shape
        
        # Random rotation
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            center = (w // 2, h // 2)
            matrix = cv.getRotationMatrix2D(center, angle, 1.0)
            img_np = cv.warpAffine(img_np, matrix, (w, h), 
                                    borderMode=cv.BORDER_REPLICATE)
        
        # Random translation
        if random.random() < 0.5:
            tx = int(random.uniform(-self.translation_range, 
                                    self.translation_range) * w)
            ty = int(random.uniform(-self.translation_range, 
                                    self.translation_range) * h)
            matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            img_np = cv.warpAffine(img_np, matrix, (w, h), 
                                    borderMode=cv.BORDER_REPLICATE)
        
        # Random scaling
        if random.random() < 0.5:
            scale = random.uniform(*self.scale_range)
            new_w, new_h = int(w * scale), int(h * scale)
            img_np = cv.resize(img_np, (new_w, new_h))
            
            # Crop or pad to original size
            if scale > 1.0:
                start_x = (new_w - w) // 2
                start_y = (new_h - h) // 2
                img_np = img_np[start_y:start_y+h, start_x:start_x+w]
            else:
                pad_x = (w - new_w) // 2
                pad_y = (h - new_h) // 2
                img_np = cv.copyMakeBorder(img_np, pad_y, h-new_h-pad_y, 
                                           pad_x, w-new_w-pad_x, 
                                           cv.BORDER_REPLICATE)
        
        # Convert back to tensor
        img = torch.from_numpy(img_np).unsqueeze(0)
        
        return img


def get_data_loaders(config, loss_type='contrastive'):
    """
    Factory function to create data loaders based on loss type.
    
    Args:
        config: Configuration module
        loss_type (str): Type of loss - 'contrastive', 'triplet', or 'batch_hard'
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create augmentation transform if enabled
    transform = None
    if config.USE_AUGMENTATION:
        transform = SignatureAugmentation(
            rotation_range=config.ROTATION_RANGE,
            translation_range=config.TRANSLATION_RANGE,
            scale_range=config.SCALE_RANGE
        )
    
    # Create appropriate dataset based on loss type
    if loss_type == 'contrastive':
        train_dataset = SignaturePairDataset(
            config.TRAIN_DIR,
            transform=transform
        )
        test_dataset = SignaturePairDataset(
            config.TEST_DIR,
            transform=None  # No augmentation for test
        )
    elif loss_type == 'triplet':
        train_dataset = SignatureTripletDataset(
            config.TRAIN_DIR,
            transform=transform
        )
        test_dataset = SignatureTripletDataset(
            config.TEST_DIR,
            transform=None
        )
    elif loss_type in ['batch_hard', 'batch_all']:
        train_dataset = SignatureBatchDataset(
            config.TRAIN_DIR,
            transform=transform
        )
        test_dataset = SignatureBatchDataset(
            config.TEST_DIR,
            transform=None
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test datasets
    import sys
    sys.path.append('.')
    import config
    
    print("Testing Dataset Classes...")
    
    # Test if data directory exists
    if not os.path.exists(config.TRAIN_DIR):
        print(f"Warning: Training data directory not found at {config.TRAIN_DIR}")
        print("Please run dataset_prepare.py first!")
    else:
        # Test pair dataset
        print("\n1. Testing Pair Dataset:")
        pair_dataset = SignaturePairDataset(config.TRAIN_DIR, pairs_per_user=5)
        img1, img2, label = pair_dataset[0]
        print(f"Sample pair - Image 1: {img1.shape}, Image 2: {img2.shape}, Label: {label}")
        
        # Test triplet dataset
        print("\n2. Testing Triplet Dataset:")
        triplet_dataset = SignatureTripletDataset(config.TRAIN_DIR, triplets_per_user=5)
        anchor, pos, neg = triplet_dataset[0]
        print(f"Sample triplet - Anchor: {anchor.shape}, Positive: {pos.shape}, Negative: {neg.shape}")
        
        # Test batch dataset
        print("\n3. Testing Batch Dataset:")
        batch_dataset = SignatureBatchDataset(config.TRAIN_DIR, samples_per_user=5)
        img, label = batch_dataset[0]
        print(f"Sample - Image: {img.shape}, User Label: {label}")
        
        print("\n✓ All datasets tested successfully!")

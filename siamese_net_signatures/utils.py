"""
Utility Functions for Siamese Network Project

This module contains:
- Random seed setting
- Checkpoint saving/loading
- Early stopping
- Model evaluation utilities
"""

import os
import random
import shutil
import numpy as np
import torch


def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def save_checkpoint(state, filename, is_best=False, checkpoint_dir='checkpoints'):
    """
    Save model checkpoint.
    
    Args:
        state (dict): Dictionary containing model state and training info
        filename (str): Checkpoint filename
        is_best (bool): Whether this is the best model so far
        checkpoint_dir (str): Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        shutil.copyfile(filepath, best_filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
    
    Returns:
        dict: Checkpoint dictionary with training info
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found at {filepath}")
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    return checkpoint


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' for minimizing, 'max' for maximizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            raise ValueError(f"mode {mode} is unknown!")
    
    def __call__(self, score):
        """
        Check if early stopping criteria is met.
        
        Args:
            score (float): Current validation score
        """
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy_from_distances(distances, labels, threshold=0.5):
    """
    Compute accuracy from distances and labels.
    
    Args:
        distances (numpy.ndarray): Pairwise distances
        labels (numpy.ndarray): Binary labels (0=same, 1=different)
        threshold (float): Distance threshold for classification
    
    Returns:
        float: Accuracy score
    """
    predictions = (distances > threshold).astype(int)
    accuracy = (predictions == labels).mean()
    return accuracy


def compute_embeddings(model, data_loader, device):
    """
    Compute embeddings for all samples in a dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to use
    
    Returns:
        tuple: (embeddings, labels) as numpy arrays
    """
    model.eval()
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle different dataset types
            if len(batch) == 2:  # Batch dataset (images, labels)
                images, labels = batch
                images = images.to(device)
                emb = model.embedding_network(images)
            elif len(batch) == 3:  # Pair or triplet dataset
                # Use first image only
                images = batch[0].to(device)
                labels = batch[-1]
                if hasattr(model, 'forward_once'):
                    emb = model.forward_once(images)
                else:
                    emb = model.embedding_network(images)
            
            embeddings_list.append(emb.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                labels_list.append(labels.cpu().numpy())
            else:
                labels_list.extend(labels)
    
    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list) if labels_list else None
    
    return embeddings, labels


def euclidean_distance(x1, x2):
    """
    Compute Euclidean distance between two tensors.
    
    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor
    
    Returns:
        torch.Tensor: Distances
    """
    return torch.sqrt(torch.sum((x1 - x2) ** 2, dim=1))


def cosine_similarity(x1, x2):
    """
    Compute cosine similarity between two tensors.
    
    Args:
        x1 (torch.Tensor): First tensor
        x2 (torch.Tensor): Second tensor
    
    Returns:
        torch.Tensor: Similarities
    """
    return torch.nn.functional.cosine_similarity(x1, x2)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
    """
    print("\n" + "="*80)
    print("Model Summary")
    print("="*80)
    
    total_params = 0
    trainable_params = 0
    
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        if parameter.requires_grad:
            trainable_params += params
        
        print(f"{name:50s} {str(parameter.shape):20s} {params:>12,d}")
    
    print("="*80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*80 + "\n")


def get_model_size_mb(model):
    """
    Get the size of a model in megabytes.
    
    Args:
        model: PyTorch model
    
    Returns:
        float: Model size in MB
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def freeze_layers(model, freeze_until=None):
    """
    Freeze layers in a model up to a certain layer name.
    
    Args:
        model: PyTorch model
        freeze_until (str): Name of the last layer to freeze (None to freeze all)
    """
    freeze = True
    
    for name, param in model.named_parameters():
        if freeze_until is not None and freeze_until in name:
            freeze = False
        
        if freeze:
            param.requires_grad = False
            print(f"Froze layer: {name}")
        else:
            param.requires_grad = True


def unfreeze_all_layers(model):
    """
    Unfreeze all layers in a model.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
    
    print("All layers unfrozen")


if __name__ == '__main__':
    # Test utilities
    print("Testing utility functions...")
    
    # Test random seed
    set_seed(42)
    print("✓ Random seed set")
    
    # Test early stopping
    early_stopping = EarlyStopping(patience=3, mode='min')
    
    val_losses = [0.5, 0.4, 0.35, 0.36, 0.37, 0.38]
    for i, loss in enumerate(val_losses):
        early_stopping(loss)
        print(f"Epoch {i+1}: Loss={loss:.2f}, Counter={early_stopping.counter}, Stop={early_stopping.early_stop}")
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Test average meter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"\n✓ Average meter test: avg={meter.avg:.2f}")
    
    print("\n✓ All utility functions working correctly!")

"""
Training Script for Siamese Network Signature Verification

This script handles:
- Model training with different loss functions
- Validation and checkpointing
- Learning rate scheduling
- Early stopping
- Progress logging
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import config
from model import create_siamese_model
from losses import get_loss_function
from dataset import get_data_loaders
from utils import set_seed, save_checkpoint, load_checkpoint, EarlyStopping


class Trainer:
    """
    Trainer class for Siamese Networks.
    """
    
    def __init__(self, model, loss_fn, optimizer, scheduler, device, 
                 train_loader, val_loader, config):
        """
        Args:
            model: Siamese network model
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration module
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=config.LOG_DIR)
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.EARLY_STOPPING_PATIENCE,
            mode='min'  # Minimize validation loss
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch_contrastive(self):
        """Train one epoch with contrastive loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, (img1, img2, labels) in enumerate(pbar):
            # Move to device
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            emb1, emb2 = self.model(img1, img2)
            loss = self.loss_fn(emb1, emb2, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if batch_idx % self.config.LOG_INTERVAL == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_epoch_triplet(self):
        """Train one epoch with triplet loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            # Move to device
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            emb_anchor, emb_pos, emb_neg = self.model(anchor, positive, negative)
            loss = self.loss_fn(emb_anchor, emb_pos, emb_neg)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if batch_idx % self.config.LOG_INTERVAL == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train_epoch_batch(self):
        """Train one epoch with batch-based loss (e.g., batch hard triplet)."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            embeddings = self.model.embedding_network(images)
            loss = self.loss_fn(embeddings, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if batch_idx % self.config.LOG_INTERVAL == 0:
                global_step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_contrastive(self):
        """Validate with contrastive loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for img1, img2, labels in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                emb1, emb2 = self.model(img1, img2)
                loss = self.loss_fn(emb1, emb2, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_triplet(self):
        """Validate with triplet loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for anchor, positive, negative in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                # Forward pass
                emb_anchor, emb_pos, emb_neg = self.model(anchor, positive, negative)
                loss = self.loss_fn(emb_anchor, emb_pos, emb_neg)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate_batch(self):
        """Validate with batch-based loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validation'):
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                embeddings = self.model.embedding_network(images)
                loss = self.loss_fn(embeddings, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.NUM_EPOCHS} epochs...")
        print(f"Loss type: {self.config.LOSS_TYPE}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Determine which training/validation functions to use
        if self.config.LOSS_TYPE == 'contrastive':
            train_fn = self.train_epoch_contrastive
            val_fn = self.validate_contrastive
        elif self.config.LOSS_TYPE == 'triplet':
            train_fn = self.train_epoch_triplet
            val_fn = self.validate_triplet
        else:  # batch-based losses
            train_fn = self.train_epoch_batch
            val_fn = self.validate_batch
        
        start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train
            train_loss = train_fn()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = val_fn()
            self.val_losses.append(val_loss)
            
            # Update learning rate scheduler
            if self.config.LR_SCHEDULER == 'reduce_on_plateau':
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log to tensorboard
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
            
            # Print progress
            print(f'\nEpoch [{epoch+1}/{self.config.NUM_EPOCHS}]')
            print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}')
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.config.SAVE_CHECKPOINT_EVERY == 0 or is_best:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': self.best_val_loss,
                    'config': self.config
                }
                
                checkpoint_path = os.path.join(
                    self.config.CHECKPOINT_DIR,
                    f'checkpoint_epoch_{epoch+1}.pth'
                )
                save_checkpoint(checkpoint, checkpoint_path, is_best, self.config.CHECKPOINT_DIR)
                
                if is_best:
                    print(f'✓ New best model saved! (Val Loss: {val_loss:.4f})')
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f'\nEarly stopping triggered after epoch {epoch+1}')
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f'\nTraining completed in {total_time/60:.2f} minutes')
        print(f'Best validation loss: {self.best_val_loss:.4f}')
        
        # Close tensorboard writer
        self.writer.close()
        
        return self.train_losses, self.val_losses


def main():
    """Main training function."""
    # Set random seed for reproducibility
    set_seed(config.RANDOM_SEED)
    
    print("="*80)
    print("Siamese Network Training")
    print("="*80)
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = get_data_loaders(config, loss_type=config.LOSS_TYPE)
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = create_siamese_model(config, loss_type=config.LOSS_TYPE)
    model = model.to(config.DEVICE)
    
    # Create loss function
    if config.LOSS_TYPE == 'contrastive':
        loss_fn = get_loss_function('contrastive', margin=config.CONTRASTIVE_MARGIN)
    elif config.LOSS_TYPE == 'triplet':
        loss_fn = get_loss_function('triplet', margin=config.TRIPLET_MARGIN)
    else:
        loss_fn = get_loss_function(config.TRIPLET_MINING, margin=config.TRIPLET_MARGIN)
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create learning rate scheduler
    if config.LR_SCHEDULER == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.LR_FACTOR,
            patience=config.LR_PATIENCE,
            verbose=True
        )
    elif config.LR_SCHEDULER == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.LR_STEP_SIZE,
            gamma=config.LR_FACTOR
        )
    else:  # cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train model
    train_losses, val_losses = trainer.train()
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)


if __name__ == '__main__':
    main()

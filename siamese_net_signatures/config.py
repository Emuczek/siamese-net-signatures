"""
Configuration file for Siamese Network Signature Verification
Contains all hyperparameters, paths, and model settings
"""

import os
import torch

# ============================================================================
# Data Configuration
# ============================================================================
DATA_DIR = 'data/signatures_processsed'
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Image dimensions (from preprocessing)
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMG_CHANNELS = 1  # Grayscale

# ============================================================================
# Model Configuration
# ============================================================================
# Embedding size (output dimension of the Siamese network)
EMBEDDING_DIM = 128

# Convolutional architecture
CONV_FILTERS = [32, 64, 128, 256]  # Number of filters in each conv layer
CONV_KERNEL_SIZE = 3
POOL_SIZE = 2

# Fully connected layers
FC_LAYERS = [512, 256]  # Dimensions of FC layers before embedding

# Dropout rate
DROPOUT_RATE = 0.3

# ============================================================================
# Training Configuration
# ============================================================================
# Loss function selection: 'contrastive' or 'triplet'
LOSS_TYPE = 'contrastive'  # Can be changed to 'triplet'

# Contrastive Loss parameters
CONTRASTIVE_MARGIN = 2.0

# Triplet Loss parameters
TRIPLET_MARGIN = 1.0
TRIPLET_MINING = 'batch_hard'  # Options: 'batch_hard', 'batch_all', 'random'

# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-5

# Learning rate scheduler
LR_SCHEDULER = 'reduce_on_plateau'  # Options: 'reduce_on_plateau', 'step', 'cosine'
LR_PATIENCE = 5  # For ReduceLROnPlateau
LR_FACTOR = 0.5
LR_STEP_SIZE = 10  # For StepLR

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# ============================================================================
# Data Augmentation
# ============================================================================
USE_AUGMENTATION = True
ROTATION_RANGE = 10  # degrees
TRANSLATION_RANGE = 0.1  # fraction of image size
SCALE_RANGE = (0.9, 1.1)

# ============================================================================
# System Configuration
# ============================================================================
# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seed for reproducibility
RANDOM_SEED = 42

# Number of workers for data loading
NUM_WORKERS = 4

# ============================================================================
# Checkpointing and Logging
# ============================================================================
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
RESULTS_DIR = 'results'

# Save best model based on validation metric
SAVE_BEST_MODEL = True
METRIC_FOR_BEST_MODEL = 'val_accuracy'  # Options: 'val_loss', 'val_accuracy', 'val_f1'

# Logging frequency
LOG_INTERVAL = 10  # Log every N batches
SAVE_CHECKPOINT_EVERY = 5  # Save checkpoint every N epochs

# ============================================================================
# Evaluation Configuration
# ============================================================================
# Threshold for signature verification (distance threshold)
VERIFICATION_THRESHOLD = 0.5

# Number of pairs to evaluate
EVAL_PAIRS = 1000

# ============================================================================
# Create necessary directories
# ============================================================================
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

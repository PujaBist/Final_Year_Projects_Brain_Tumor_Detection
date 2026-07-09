# ============================================================
# Brain Tumor MRI Classification Project
# File: src/config.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Central configuration — all settings here.
#              Every other file imports from this file only.
# ============================================================

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 1. PROJECT ROOT
# ─────────────────────────────────────────────────────────────
# Automatically finds project root regardless of where
# you run the code — works on laptop AND Google Colab

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────
# 2. REPRODUCIBILITY SEED
# ─────────────────────────────────────────────────────────────
# Fixed seed = same results every run
# NEVER change this after first experiment

SEED = 42

# ─────────────────────────────────────────────────────────────
# 3. DATASET SETTINGS
# ─────────────────────────────────────────────────────────────
from pathlib import Path




# Four tumor classes — exact folder names in your dataset
CLASSES        = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES    = len(CLASSES)                          # 4

# Class index mapping
CLASS_TO_IDX   = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS   = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# Dataset sizes (for reference)
TRAIN_DATASET_SIZE    = 7023   # Masoudnickparvar dataset
EXTERNAL_TEST_SIZE    = 6000   # BRISC 2025 dataset

# Train / Validation / Test split ratio
TRAIN_RATIO = 0.70             # 70% → training
VAL_RATIO   = 0.15             # 15% → validation
TEST_RATIO  = 0.15             # 15% → internal test
# External test → BRISC 2025 (completely separate)

# ─────────────────────────────────────────────────────────────
# 4. DATASET PATHS
# ─────────────────────────────────────────────────────────────
# Laptop paths (VS Code)
BASE_DIR = Path("/content/drive/MyDrive/Final_Year_Projects_Brain_Tumor_Detection")

TRAIN_DATA_DIR = BASE_DIR / "Datasets" / "Training"
TEST_DATA_DIR = BASE_DIR / "Datasets" / "Testing"
# Colab paths (override in notebook if needed)
# TRAIN_DATA_DIR = Path('/content/drive/MyDrive/Brain_Tumor_Project/data/masoudnickparvar')
# EXTERNAL_TEST_DIR = Path('/content/drive/MyDrive/Brain_Tumor_Project/data/brisc2025')

# Data split file — saved ONCE, reused for ALL models
DATA_SPLIT_FILE = ROOT / 'results' / 'data_split.json'

# ─────────────────────────────────────────────────────────────
# 5. IMAGE PREPROCESSING SETTINGS
# ─────────────────────────────────────────────────────────────

IMAGE_SIZE   = 224             # Resize all images to 224×224
IMAGE_CHANNELS = 3             # RGB (3 channels)

# ImageNet normalization — used for pretrained models
# These are standard values for ResNet, DenseNet, EfficientNet
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# CLAHE settings (contrast enhancement)
CLAHE_CLIP_LIMIT    = 2.0
CLAHE_TILE_GRID     = (8, 8)

# ─────────────────────────────────────────────────────────────
# 6. DATA AUGMENTATION SETTINGS
# ─────────────────────────────────────────────────────────────
# Applied ONLY to training set — never to val or test

AUG_HORIZONTAL_FLIP  = True
AUG_VERTICAL_FLIP    = False   # MRI — vertical flip unnatural
AUG_ROTATION_LIMIT   = 15      # degrees — small rotation only
AUG_BRIGHTNESS_LIMIT = 0.2     # subtle brightness change
AUG_CONTRAST_LIMIT   = 0.2     # subtle contrast change
AUG_ZOOM_LIMIT       = 0.1     # subtle zoom

# ─────────────────────────────────────────────────────────────
# 7. TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────

BATCH_SIZE      = 32           # images per batch
NUM_EPOCHS      = 50           # maximum training epochs
LEARNING_RATE   = 1e-4         # initial learning rate
WEIGHT_DECAY    = 1e-4         # L2 regularization
MOMENTUM        = 0.9          # for SGD optimizer

# Learning rate scheduler
LR_SCHEDULER    = 'cosine'     # 'cosine' or 'step'
LR_STEP_SIZE    = 10           # for StepLR
LR_GAMMA        = 0.1          # for StepLR

# Early stopping
EARLY_STOP_PATIENCE = 10       # stop if no improvement
MIN_DELTA           = 1e-4     # minimum improvement threshold

# ─────────────────────────────────────────────────────────────
# 8. MODEL SETTINGS
# ─────────────────────────────────────────────────────────────

# Three baseline models
MODELS = ['resnet50', 'densenet121', 'efficientnetb0']

# Pretrained on ImageNet
PRETRAINED      = True

# Freeze backbone layers initially (fine-tuning strategy)
FREEZE_BACKBONE = False        # False = train all layers

# Dropout rate for uncertainty estimation (Phase 10)
DROPOUT_RATE    = 0.5
MC_DROPOUT_SAMPLES = 50        # Monte Carlo forward passes

# ─────────────────────────────────────────────────────────────
# 9. ENSEMBLE SETTINGS (Phase 6)
# ─────────────────────────────────────────────────────────────

ENSEMBLE_METHOD  = 'weighted_soft_voting'
# Weights assigned after individual model evaluation
# Updated after Phase 4 training
ENSEMBLE_WEIGHTS = {
    'resnet50'      : 1.0,     # updated after training
    'densenet121'   : 1.0,     # updated after training
    'efficientnetb0': 1.0,     # updated after training
}

# ─────────────────────────────────────────────────────────────
# 10. CALIBRATION SETTINGS (Phase 7)
# ─────────────────────────────────────────────────────────────

TEMPERATURE_INIT = 1.5         # initial temperature value
TEMPERATURE_LR   = 0.01        # learning rate for T optimization
TEMPERATURE_ITER = 1000        # optimization iterations
ECE_N_BINS       = 15          # bins for ECE calculation

# ─────────────────────────────────────────────────────────────
# 11. RESULTS AND OUTPUT PATHS
# ─────────────────────────────────────────────────────────────

RESULTS_DIR        = ROOT / 'results'
MODELS_DIR         = RESULTS_DIR / 'models'
FIGURES_DIR        = RESULTS_DIR / 'figures'
METRICS_DIR        = RESULTS_DIR / 'metrics'
LOGS_DIR           = RESULTS_DIR / 'logs'
CHECKPOINTS_DIR    = RESULTS_DIR / 'checkpoints'
DUPLICATE_DIR      = RESULTS_DIR / 'duplicate_report'

# Model save file names
MODEL_SAVE_NAMES = {
    'resnet50'      : MODELS_DIR / 'resnet50_best.pth',
    'densenet121'   : MODELS_DIR / 'densenet121_best.pth',
    'efficientnetb0': MODELS_DIR / 'efficientnetb0_best.pth',
    'ensemble'      : MODELS_DIR / 'ensemble.pth',
}

# ─────────────────────────────────────────────────────────────
# 12. DEVICE SETTINGS
# ─────────────────────────────────────────────────────────────

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY  = True if torch.cuda.is_available() else False

# ─────────────────────────────────────────────────────────────
# 13. LOGGING AND EXPERIMENT TRACKING
# ─────────────────────────────────────────────────────────────

USE_WANDB       = True          # set False if no internet
WANDB_PROJECT   = 'brain-tumor-classification'
WANDB_ENTITY    = 'puja-bist'   # your wandb username

LOG_INTERVAL    = 10            # log every N batches

# ─────────────────────────────────────────────────────────────
# 14. STATISTICAL TEST SETTINGS (Phase 12)
# ─────────────────────────────────────────────────────────────

ALPHA           = 0.05          # significance level
CONFIDENCE_CI   = 0.95          # 95% confidence interval

# ─────────────────────────────────────────────────────────────
# 15. VERIFICATION — print config when imported
# ─────────────────────────────────────────────────────────────

def print_config():
    """Print all key configuration settings."""
    print("=" * 55)
    print("  BRAIN TUMOR PROJECT — CONFIGURATION")
    print("=" * 55)
    print(f"  Project root  : {ROOT}")
    print(f"  Device        : {DEVICE}")
    print(f"  Seed          : {SEED}")
    print(f"  Classes       : {CLASSES}")
    print(f"  Num classes   : {NUM_CLASSES}")
    print(f"  Image size    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Epochs        : {NUM_EPOCHS}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Models        : {MODELS}")
    print(f"  Train split   : {TRAIN_RATIO}")
    print(f"  Val split     : {VAL_RATIO}")
    print(f"  Test split    : {TEST_RATIO}")
    print(f"  Pretrained    : {PRETRAINED}")
    print(f"  MC samples    : {MC_DROPOUT_SAMPLES}")
    print(f"  Use wandb     : {USE_WANDB}")
    print("=" * 55)


if __name__ == '__main__':
    print_config()
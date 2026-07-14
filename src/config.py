# ============================================================
# Brain Tumor MRI Classification Project
# File: src/config.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Central configuration — all settings here.
#              Every other file imports from this file only.
# ============================================================

import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 1. PROJECT ROOT
# ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────
# 2. REPRODUCIBILITY SEED
# ─────────────────────────────────────────────────────────────

SEED = 42

# ─────────────────────────────────────────────────────────────
# 3. DATASET CLASS SETTINGS
# ─────────────────────────────────────────────────────────────

# Must match exact folder names inside Training/ and Testing/
CLASSES     = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLASSES = len(CLASSES)  # 4

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# ─────────────────────────────────────────────────────────────
# 4. DATASET PATHS
# ─────────────────────────────────────────────────────────────

# Google Colab — Google Drive path
BASE_DIR    = Path('/content/drive/MyDrive/Final_Year_Projects_Brain_Tumor_Detection')

# Datasets/ contains Training/ and Testing/ subfolders
DATASET_DIR = BASE_DIR / 'Datasets'

# External test dataset (BRISC 2025) — Phase 8
EXTERNAL_TEST_DIR = BASE_DIR / 'Datasets' / 'BRISC2025'

# Split file — saved ONCE, reused for ALL 3 models
DATA_SPLIT_FILE = BASE_DIR / 'results' / 'data_split.json'

# ─────────────────────────────────────────────────────────────
# 5. DATASET SIZE REFERENCE
# ─────────────────────────────────────────────────────────────

TRAIN_DATASET_SIZE = 7200   # Training(5712) + Testing(1311)
EXTERNAL_TEST_SIZE = 6000   # BRISC 2025

# ─────────────────────────────────────────────────────────────
# 6. SPLIT RATIOS
# ─────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.70   # 70% → training
VAL_RATIO   = 0.15   # 15% → validation
TEST_RATIO  = 0.15   # 15% → internal test

# ─────────────────────────────────────────────────────────────
# 7. IMAGE PREPROCESSING SETTINGS
# ─────────────────────────────────────────────────────────────

IMAGE_SIZE     = 224
IMAGE_CHANNELS = 3

# ImageNet normalization — standard for pretrained models
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# CLAHE contrast enhancement
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)

# ─────────────────────────────────────────────────────────────
# 8. DATA AUGMENTATION SETTINGS
# ─────────────────────────────────────────────────────────────

AUG_HORIZONTAL_FLIP  = False
AUG_VERTICAL_FLIP    = False   # MRI — vertical flip unnatural
AUG_ROTATION_LIMIT   = 15      # degrees
AUG_BRIGHTNESS_LIMIT = 0.2
AUG_CONTRAST_LIMIT   = 0.2
AUG_ZOOM_LIMIT       = 0.1

# ─────────────────────────────────────────────────────────────
# 9. TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────

BATCH_SIZE          = 32
NUM_EPOCHS          = 50
LEARNING_RATE       = 1e-4
WEIGHT_DECAY        = 1e-4
MOMENTUM            = 0.9

LR_SCHEDULER        = 'cosine'
LR_STEP_SIZE        = 10
LR_GAMMA            = 0.1

EARLY_STOP_PATIENCE = 10
MIN_DELTA           = 1e-4

# ─────────────────────────────────────────────────────────────
# 10. MODEL SETTINGS
# ─────────────────────────────────────────────────────────────

MODELS          = ['resnet50', 'densenet121', 'efficientnetb0']
PRETRAINED      = True
FREEZE_BACKBONE = False
DROPOUT_RATE    = 0.5
MC_DROPOUT_SAMPLES = 50

# ─────────────────────────────────────────────────────────────
# 11. ENSEMBLE SETTINGS (Phase 6)
# ─────────────────────────────────────────────────────────────

ENSEMBLE_METHOD  = 'weighted_soft_voting'
ENSEMBLE_WEIGHTS = {
    'resnet50'      : 1.0,
    'densenet121'   : 1.0,
    'efficientnetb0': 1.0,
}

# ─────────────────────────────────────────────────────────────
# 12. CALIBRATION SETTINGS (Phase 7)
# ─────────────────────────────────────────────────────────────

TEMPERATURE_INIT = 1.5
TEMPERATURE_LR   = 0.01
TEMPERATURE_ITER = 1000
ECE_N_BINS       = 15

# ─────────────────────────────────────────────────────────────
# 13. RESULTS AND OUTPUT PATHS
# ─────────────────────────────────────────────────────────────

RESULTS_DIR     = BASE_DIR / 'results'
MODELS_DIR      = RESULTS_DIR / 'models'
FIGURES_DIR     = RESULTS_DIR / 'figures'
METRICS_DIR     = RESULTS_DIR / 'metrics'
LOGS_DIR        = RESULTS_DIR / 'logs'
CHECKPOINTS_DIR = RESULTS_DIR / 'checkpoints'
DUPLICATE_DIR   = RESULTS_DIR / 'duplicate_report'

MODEL_SAVE_NAMES = {
    'resnet50'      : MODELS_DIR / 'resnet50_best.pth',
    'densenet121'   : MODELS_DIR / 'densenet121_best.pth',
    'efficientnetb0': MODELS_DIR / 'efficientnetb0_best.pth',
    'ensemble'      : MODELS_DIR / 'ensemble.pth',
}

# ─────────────────────────────────────────────────────────────
# 14. DEVICE SETTINGS
# ─────────────────────────────────────────────────────────────

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY  = True if torch.cuda.is_available() else False

# ─────────────────────────────────────────────────────────────
# 15. LOGGING AND EXPERIMENT TRACKING
# ─────────────────────────────────────────────────────────────

USE_WANDB     = True
WANDB_PROJECT = 'brain-tumor-classification'
WANDB_ENTITY  = 'puja-bist'
LOG_INTERVAL  = 10

# ─────────────────────────────────────────────────────────────
# 16. STATISTICAL TEST SETTINGS (Phase 12)
# ─────────────────────────────────────────────────────────────

ALPHA         = 0.05
CONFIDENCE_CI = 0.95

# ─────────────────────────────────────────────────────────────
# 17. VERIFICATION
# ─────────────────────────────────────────────────────────────

def print_config():
    print("=" * 55)
    print("  BRAIN TUMOR PROJECT — CONFIGURATION")
    print("=" * 55)
    print(f"  Base dir      : {BASE_DIR}")
    print(f"  Dataset dir   : {DATASET_DIR}")
    print(f"  Dataset exists: {DATASET_DIR.exists()}")
    print(f"  Device        : {DEVICE}")
    print(f"  Seed          : {SEED}")
    print(f"  Classes       : {CLASSES}")
    print(f"  Num classes   : {NUM_CLASSES}")
    print(f"  Image size    : {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Epochs        : {NUM_EPOCHS}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Models        : {MODELS}")
    print(f"  Split         : {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  Data split    : {DATA_SPLIT_FILE}")
    print(f"  Results dir   : {RESULTS_DIR}")
    print("=" * 55)


if __name__ == '__main__':
    print_config()
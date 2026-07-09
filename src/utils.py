# ============================================================
# Brain Tumor MRI Classification Project
# File: src/utils.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Helper functions used across all other files.
#              Seed setting, logging, checkpoint save/load,
#              folder creation, timer, and plotting helpers.
# ============================================================

import os
import random
import time
import json
import logging
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — safe for Colab

from src.config import (
    SEED, DEVICE, RESULTS_DIR, MODELS_DIR,
    FIGURES_DIR, METRICS_DIR, LOGS_DIR,
    CHECKPOINTS_DIR, DUPLICATE_DIR, CLASSES
)

# ─────────────────────────────────────────────────────────────
# 1. REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    """
    Fix all random seeds for full reproducibility.
    Call this at the TOP of every notebook and script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['PYTHONHASHSEED']       = str(seed)
    print(f"✅ Seed set to {seed} — results are reproducible")


# ─────────────────────────────────────────────────────────────
# 2. FOLDER CREATION
# ─────────────────────────────────────────────────────────────

def create_directories() -> None:
    """
    Create all result folders if they do not exist.
    Safe to call multiple times — will not overwrite.
    """
    dirs = [
        RESULTS_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        LOGS_DIR,
        CHECKPOINTS_DIR,
        DUPLICATE_DIR,
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ All result directories ready")


# ─────────────────────────────────────────────────────────────
# 3. LOGGING SETUP
# ─────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Create a logger that writes to both console and file.

    Args:
        name:     logger name (usually model name)
        log_file: optional path to save log file

    Returns:
        configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


# ─────────────────────────────────────────────────────────────
# 4. DEVICE INFO
# ─────────────────────────────────────────────────────────────

def print_device_info() -> None:
    """Print GPU/CPU information."""
    print("=" * 45)
    print("  DEVICE INFORMATION")
    print("=" * 45)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU name        : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"  GPU memory      : {mem / 1e9:.1f} GB")
        print(f"  CUDA version    : {torch.version.cuda}")
    else:
        print("  Running on      : CPU (no GPU detected)")
        print("  Tip             : Use Google Colab for GPU training")
    print("=" * 45)


# ─────────────────────────────────────────────────────────────
# 5. CHECKPOINT SAVE AND LOAD
# ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    val_acc: float,
    model_name: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint every N epochs.
    If is_best=True, also saves as best model.

    Args:
        model:      PyTorch model
        optimizer:  optimizer state
        epoch:      current epoch number
        val_acc:    validation accuracy at this epoch
        model_name: one of 'resnet50', 'densenet121', 'efficientnetb0'
        is_best:    True if this is the best model so far
    """
    checkpoint = {
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc'             : val_acc,
        'model_name'          : model_name,
    }

    # Save periodic checkpoint
    ckpt_path = CHECKPOINTS_DIR / f'{model_name}_epoch{epoch:03d}.pth'
    torch.save(checkpoint, ckpt_path)

    # Save best model separately
    if is_best:
        best_path = MODELS_DIR / f'{model_name}_best.pth'
        torch.save(checkpoint, best_path)
        print(f"  ⭐ Best model saved → {best_path}")


def load_checkpoint(model_name: str, device=DEVICE):
    """
    Load the best saved model weights.

    Args:
        model_name: one of 'resnet50', 'densenet121', 'efficientnetb0'
        device:     torch device

    Returns:
        checkpoint dict with model state and metadata
    """
    best_path = MODELS_DIR / f'{model_name}_best.pth'

    if not best_path.exists():
        raise FileNotFoundError(
            f"No saved model found at {best_path}\n"
            f"Train the model first before loading."
        )

    checkpoint = torch.load(best_path, map_location=device)
    print(f"✅ Loaded {model_name} — epoch {checkpoint['epoch']}, "
          f"val_acc {checkpoint['val_acc']:.4f}")
    return checkpoint


# ─────────────────────────────────────────────────────────────
# 6. TRAINING TIMER
# ─────────────────────────────────────────────────────────────

class Timer:
    """Simple timer to measure training time."""

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()
        print("⏱  Timer started")

    def stop(self, label: str = "Elapsed") -> float:
        elapsed = time.time() - self.start_time
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        s = int(elapsed % 60)
        print(f"⏱  {label}: {h:02d}h {m:02d}m {s:02d}s")
        return elapsed


# ─────────────────────────────────────────────────────────────
# 7. METRICS SAVING
# ─────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, model_name: str) -> None:
    """
    Save evaluation metrics to JSON file.

    Args:
        metrics:    dict of metric name → value
        model_name: model identifier string
    """
    save_path = METRICS_DIR / f'{model_name}_metrics.json'
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✅ Metrics saved → {save_path}")


def load_metrics(model_name: str) -> dict:
    """Load previously saved metrics."""
    save_path = METRICS_DIR / f'{model_name}_metrics.json'
    with open(save_path, 'r') as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# 8. PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    model_name: str
) -> None:
    """
    Plot and save training loss and accuracy curves.

    Args:
        train_losses: list of training loss per epoch
        val_losses:   list of validation loss per epoch
        train_accs:   list of training accuracy per epoch
        val_accs:     list of validation accuracy per epoch
        model_name:   used for save filename
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss',  linewidth=2)
    ax1.plot(epochs, val_losses,   'r-', label='Val Loss',    linewidth=2)
    ax1.set_title(f'{model_name} — Loss Curve', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs,   'r-', label='Val Acc',   linewidth=2)
    ax2.set_title(f'{model_name} — Accuracy Curve', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = FIGURES_DIR / f'{model_name}_training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves saved → {save_path}")


def plot_class_distribution(
    class_counts: dict,
    title: str = "Class Distribution",
    save_name: str = "class_distribution"
) -> None:
    """
    Plot bar chart of class distribution.

    Args:
        class_counts: dict of class_name → count
        title:        plot title
        save_name:    filename without extension
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    bars = ax.bar(
        class_counts.keys(),
        class_counts.values(),
        color=colors,
        edgecolor='white',
        linewidth=1.2
    )

    # Add count labels on bars
    for bar, count in zip(bars, class_counts.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(count),
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Tumor Class', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(class_counts.values()) * 1.15)

    plt.tight_layout()
    save_path = FIGURES_DIR / f'{save_name}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Class distribution saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 9. PROGRESS DISPLAY
# ─────────────────────────────────────────────────────────────

def print_epoch_summary(
    epoch: int,
    num_epochs: int,
    train_loss: float,
    val_loss: float,
    train_acc: float,
    val_acc: float,
    lr: float
) -> None:
    """Print clean epoch summary during training."""
    print(
        f"Epoch [{epoch:03d}/{num_epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"LR: {lr:.6f}"
    )


def print_model_summary(model_name: str, model) -> None:
    """Print model parameter count."""
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*45}")
    print(f"  MODEL: {model_name.upper()}")
    print(f"{'='*45}")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {train:,}")
    print(f"  Frozen parameters   : {total - train:,}")
    print(f"{'='*45}\n")


# ─────────────────────────────────────────────────────────────
# 10. QUICK SANITY CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n Running utils.py sanity check...\n")

    # 1. Seed
    set_seed()

    # 2. Directories
    create_directories()

    # 3. Device info
    print_device_info()

    # 4. Timer test
    t = Timer()
    t.start()
    import time; time.sleep(1)
    t.stop("Test timer")

    # 5. Plot test
    plot_class_distribution(
        class_counts={
            'glioma'     : 1621,
            'meningioma' : 1645,
            'no_tumor'   : 2000,
            'pituitary'  : 1757,
        },
        title='Masoudnickparvar Dataset — Class Distribution',
        save_name='test_class_distribution'
    )

    print("\n✅ utils.py — all checks passed!")
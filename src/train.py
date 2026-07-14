# ============================================================
# Brain Tumor MRI Classification Project
# File: src/train.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Training loop for individual baseline models.
#              Trains ResNet50, DenseNet121, EfficientNetB0
#              one at a time on the same 70/15/15 split.
#              Saves best model checkpoint automatically.
# ============================================================

import time
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOP_PATIENCE,
    MIN_DELTA,
    CHECKPOINTS_DIR,
    MODELS_DIR,
    METRICS_DIR,
    LOGS_DIR,
    MODEL_SAVE_NAMES,
    SEED,
)
from src.utils import (
    set_seed,
    print_epoch_summary,
    plot_training_curves,
)


# ─────────────────────────────────────────────────────────────
# 1. TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model      : nn.Module,
    loader     : torch.utils.data.DataLoader,
    criterion  : nn.Module,
    optimizer  : optim.Optimizer,
    device     : torch.device = DEVICE,
) -> Tuple[float, float]:
    """
    Run one full training epoch.

    Args:
        model     : PyTorch model
        loader    : training DataLoader
        criterion : loss function (CrossEntropyLoss)
        optimizer : optimizer (AdamW)
        device    : cuda or cpu

    Returns:
        (epoch_loss, epoch_accuracy) as floats
    """
    model.train()

    running_loss    = 0.0
    correct         = 0
    total           = 0

    for images, labels in tqdm(loader, desc='  Train', leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        running_loss += loss.item() * images.size(0)
        preds         = outputs.argmax(dim=1)
        correct      += (preds == labels).sum().item()
        total        += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    return epoch_loss, epoch_acc


# ─────────────────────────────────────────────────────────────
# 2. VALIDATE ONE EPOCH
# ─────────────────────────────────────────────────────────────

def validate_one_epoch(
    model     : nn.Module,
    loader    : torch.utils.data.DataLoader,
    criterion : nn.Module,
    device    : torch.device = DEVICE,
) -> Tuple[float, float]:
    """
    Run one full validation epoch (no gradient updates).

    Args:
        model     : PyTorch model
        loader    : validation DataLoader
        criterion : loss function
        device    : cuda or cpu

    Returns:
        (epoch_loss, epoch_accuracy) as floats
    """
    model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Val  ', leave=False):
            images  = images.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total

    return epoch_loss, epoch_acc


# ─────────────────────────────────────────────────────────────
# 3. EARLY STOPPING
# ─────────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Stop training when validation accuracy stops improving.

    Args:
        patience  : epochs to wait before stopping
        min_delta : minimum improvement to count
    """

    def __init__(
        self,
        patience  : int   = EARLY_STOP_PATIENCE,
        min_delta : float = MIN_DELTA
    ):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_acc   = 0.0
        self.stop       = False

    def __call__(self, val_acc: float) -> bool:
        """
        Check if training should stop.

        Returns:
            True if training should stop
        """
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
                print(f"\n  ⏹  Early stopping triggered "
                      f"(no improvement for {self.patience} epochs)")

        return self.stop


# ─────────────────────────────────────────────────────────────
# 4. SAVE CHECKPOINT
# ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model      : nn.Module,
    optimizer  : optim.Optimizer,
    epoch      : int,
    val_acc    : float,
    val_loss   : float,
    model_name : str,
    is_best    : bool = False,
) -> None:
    """
    Save model checkpoint to disk.

    Args:
        model      : PyTorch model
        optimizer  : optimizer state
        epoch      : current epoch number
        val_acc    : validation accuracy
        val_loss   : validation loss
        model_name : 'resnet50', 'densenet121', 'efficientnetb0'
        is_best    : if True, also saves as best model
    """
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'epoch'               : epoch,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc'             : val_acc,
        'val_loss'            : val_loss,
        'model_name'          : model_name,
    }

    # Save periodic checkpoint every 5 epochs
    if epoch % 5 == 0:
        ckpt_path = CHECKPOINTS_DIR / \
                    f'{model_name}_epoch{epoch:03d}.pth'
        torch.save(checkpoint, ckpt_path)

    # Save best model
    if is_best:
        best_path = MODEL_SAVE_NAMES[model_name]
        torch.save(checkpoint, best_path)
        print(f"  ⭐ Best model saved → val_acc={val_acc:.4f}")


# ─────────────────────────────────────────────────────────────
# 5. FULL TRAINING LOOP
# ─────────────────────────────────────────────────────────────

def train_model(
    model_name  : str,
    model       : nn.Module,
    dataloaders : Dict,
    num_epochs  : int   = NUM_EPOCHS,
    lr          : float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> Dict:
    """
    Full training pipeline for one model.

    Steps per epoch:
        1. Train on train set
        2. Evaluate on val set
        3. Update learning rate scheduler
        4. Check early stopping
        5. Save best model if val_acc improved

    Args:
        model_name  : 'resnet50', 'densenet121', 'efficientnetb0'
        model       : built model from models.py
        dataloaders : dict from data_loader.py
        num_epochs  : maximum epochs
        lr          : initial learning rate
        weight_decay: L2 regularization

    Returns:
        history dict with all epoch metrics
    """
    set_seed(SEED)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer — AdamW works well for fine-tuning
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = lr,
        weight_decay = weight_decay
    )

    # Learning rate scheduler — cosine annealing
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max = num_epochs,
        eta_min = lr * 0.01
    )

    # Early stopping
    early_stop = EarlyStopping(
        patience  = EARLY_STOP_PATIENCE,
        min_delta = MIN_DELTA
    )

    # History tracking
    history = {
        'model_name'   : model_name,
        'train_losses' : [],
        'val_losses'   : [],
        'train_accs'   : [],
        'val_accs'     : [],
        'lr_history'   : [],
        'best_val_acc' : 0.0,
        'best_epoch'   : 0,
    }

    best_val_acc = 0.0
    start_time   = time.time()

    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_name.upper()}")
    print(f"  Epochs    : {num_epochs}")
    print(f"  LR        : {lr}")
    print(f"  Device    : {DEVICE}")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):

        # ── Train ────────────────────────────────────────────
        train_loss, train_acc = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer
        )

        # ── Validate ─────────────────────────────────────────
        val_loss, val_acc = validate_one_epoch(
            model, dataloaders['val'], criterion
        )

        # ── Scheduler step ───────────────────────────────────
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ── Record history ───────────────────────────────────
        history['train_losses'].append(train_loss)
        history['val_losses'].append(val_loss)
        history['train_accs'].append(train_acc)
        history['val_accs'].append(val_acc)
        history['lr_history'].append(current_lr)

        # ── Print epoch summary ──────────────────────────────
        print_epoch_summary(
            epoch, num_epochs,
            train_loss, val_loss,
            train_acc,  val_acc,
            current_lr
        )

        # ── Save best model ──────────────────────────────────
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc          = val_acc
            history['best_val_acc'] = val_acc
            history['best_epoch']   = epoch

        save_checkpoint(
            model, optimizer, epoch,
            val_acc, val_loss,
            model_name, is_best
        )

        # ── Early stopping check ─────────────────────────────
        if early_stop(val_acc):
            break

    # ── Training complete ─────────────────────────────────────
    elapsed = time.time() - start_time
    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE: {model_name.upper()}")
    print(f"  Best val accuracy : {best_val_acc:.4f} "
          f"(epoch {history['best_epoch']})")
    print(f"  Total time        : {h:02d}h {m:02d}m {s:02d}s")
    print(f"{'='*60}\n")

    # ── Save history ──────────────────────────────────────────
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = METRICS_DIR / f'{model_name}_history.json'

    # Convert to JSON-serializable format
    history_save = {k: v for k, v in history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_save, f, indent=2)
    print(f"✅ Training history saved → {history_path}")

    # ── Plot training curves ──────────────────────────────────
    plot_training_curves(
        history['train_losses'],
        history['val_losses'],
        history['train_accs'],
        history['val_accs'],
        model_name
    )

    return history


# ─────────────────────────────────────────────────────────────
# 6. TEST SET EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_on_test(
    model      : nn.Module,
    loader     : torch.utils.data.DataLoader,
) -> Tuple[float, list, list]:
    """
    Run model on test set and collect predictions.

    Args:
        model  : trained PyTorch model
        loader : test DataLoader

    Returns:
        test_acc   : overall accuracy
        all_preds  : list of predicted class indices
        all_labels : list of true class indices
    """
    model.eval()

    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Testing'):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            preds   = outputs.argmax(dim=1)

            correct      += (preds == labels).sum().item()
            total        += labels.size(0)
            all_preds    .extend(preds.cpu().tolist())
            all_labels   .extend(labels.cpu().tolist())

    test_acc = correct / total
    print(f"\n  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

    return test_acc, all_preds, all_labels


# ─────────────────────────────────────────────────────────────
# 7. SANITY CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n🧪 Testing train.py components...\n")

    # Test EarlyStopping
    es = EarlyStopping(patience=3, min_delta=0.001)
    scores = [0.80, 0.81, 0.81, 0.81, 0.81]
    for i, score in enumerate(scores):
        stopped = es(score)
        print(f"  Epoch {i+1}: val_acc={score:.2f} "
              f"counter={es.counter} stop={stopped}")

    print("\n✅ EarlyStopping works correctly")
    print("✅ train.py — ready for training!")
    print("\n   To train a model, run in Colab:")
    print("   from src.train import train_model")
    print("   from src.models import get_model")
    print("   model = get_model('resnet50')")
    print("   history = train_model('resnet50', model, dataloaders)")
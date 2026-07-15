# ============================================================
# Brain Tumor MRI Classification Project
# File: src/evaluate.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Full evaluation pipeline for trained models.
#              Computes all metrics needed for paper:
#              Accuracy, Precision, Recall, F1, Specificity,
#              AUC-ROC, MCC, Cohen's Kappa, Confusion Matrix
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

from src.config import (
    CLASSES,
    NUM_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    DEVICE,
    FIGURES_DIR,
    METRICS_DIR,
)


# ─────────────────────────────────────────────────────────────
# 1. GET PREDICTIONS
# ─────────────────────────────────────────────────────────────

def get_predictions(
    model  : nn.Module,
    loader : DataLoader,
) -> Tuple[List, List, np.ndarray]:
    """
    Run model on dataloader and collect predictions.

    Args:
        model  : trained PyTorch model
        loader : DataLoader (test or val)

    Returns:
        all_preds  : list of predicted class indices
        all_labels : list of true class indices
        all_probs  : numpy array of softmax probabilities
                     shape (N, num_classes)
    """
    model.eval()

    all_preds  = []
    all_labels = []
    all_probs  = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Evaluating'):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(images)
            probs   = softmax(outputs)
            preds   = outputs.argmax(dim=1)

            all_preds .extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs .append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    return all_preds, all_labels, all_probs


# ─────────────────────────────────────────────────────────────
# 2. COMPUTE ALL METRICS
# ─────────────────────────────────────────────────────────────

def compute_metrics(
    all_preds  : List[int],
    all_labels : List[int],
    all_probs  : np.ndarray,
    model_name : str,
) -> Dict:
    """
    Compute all evaluation metrics for paper.

    Metrics computed:
        - Accuracy
        - Precision (macro)
        - Recall / Sensitivity (macro)
        - F1-score (macro + per class)
        - Specificity (per class)
        - AUC-ROC (macro OvR)
        - Matthews Correlation Coefficient (MCC)
        - Cohen's Kappa
        - Per-class metrics

    Args:
        all_preds  : predicted class indices
        all_labels : true class indices
        all_probs  : softmax probabilities (N, 4)
        model_name : for saving results

    Returns:
        metrics dict with all computed values
    """
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # ── Overall metrics ──────────────────────────────────────
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,
                                average='macro', zero_division=0)
    recall    = recall_score(y_true, y_pred,
                             average='macro', zero_division=0)
    f1_macro  = f1_score(y_true, y_pred,
                         average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred,
                           average='weighted', zero_division=0)
    mcc       = matthews_corrcoef(y_true, y_pred)
    kappa     = cohen_kappa_score(y_true, y_pred)

    # ── AUC-ROC (one-vs-rest) ────────────────────────────────
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    auc_macro  = roc_auc_score(
        y_true_bin, all_probs,
        multi_class='ovr', average='macro'
    )
    auc_weighted = roc_auc_score(
        y_true_bin, all_probs,
        multi_class='ovr', average='weighted'
    )

    # ── Per-class metrics ────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)

    per_class = {}
    for i, class_name in enumerate(CLASSES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        sensitivity  = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec         = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_cls       = f1_score(y_true, y_pred,
                                labels=[i], average='macro',
                                zero_division=0)
        auc_cls      = roc_auc_score(
            y_true_bin[:, i], all_probs[:, i]
        )

        per_class[class_name] = {
            'precision'  : round(prec,        4),
            'recall'     : round(sensitivity,  4),
            'specificity': round(specificity,  4),
            'f1_score'   : round(f1_cls,       4),
            'auc'        : round(auc_cls,      4),
            'tp': int(tp), 'fp': int(fp),
            'fn': int(fn), 'tn': int(tn),
        }

    # ── Assemble metrics dict ────────────────────────────────
    metrics = {
        'model_name'   : model_name,
        'accuracy'     : round(accuracy,    4),
        'precision'    : round(precision,   4),
        'recall'       : round(recall,      4),
        'f1_macro'     : round(f1_macro,    4),
        'f1_weighted'  : round(f1_weighted, 4),
        'auc_macro'    : round(auc_macro,   4),
        'auc_weighted' : round(auc_weighted,4),
        'mcc'          : round(mcc,         4),
        'kappa'        : round(kappa,       4),
        'per_class'    : per_class,
        'confusion_matrix': cm.tolist(),
    }

    return metrics


# ─────────────────────────────────────────────────────────────
# 3. PRINT METRICS TABLE
# ─────────────────────────────────────────────────────────────

def print_metrics(metrics: Dict) -> None:
    """Print clean metrics table for paper."""

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS — {metrics['model_name'].upper()}")
    print(f"{'='*60}")
    print(f"  Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision       : {metrics['precision']*100:.2f}%")
    print(f"  Recall          : {metrics['recall']*100:.2f}%")
    print(f"  F1 (macro)      : {metrics['f1_macro']*100:.2f}%")
    print(f"  F1 (weighted)   : {metrics['f1_weighted']*100:.2f}%")
    print(f"  AUC-ROC (macro) : {metrics['auc_macro']*100:.2f}%")
    print(f"  MCC             : {metrics['mcc']:.4f}")
    print(f"  Cohen's Kappa   : {metrics['kappa']:.4f}")
    print(f"\n  Per-class results:")
    print(f"  {'Class':<15} {'Prec':>7} {'Recall':>8} "
          f"{'Spec':>7} {'F1':>7} {'AUC':>7}")
    print(f"  {'─'*55}")

    for cls, v in metrics['per_class'].items():
        print(f"  {cls:<15} "
              f"{v['precision']*100:>6.2f}% "
              f"{v['recall']*100:>7.2f}% "
              f"{v['specificity']*100:>6.2f}% "
              f"{v['f1_score']*100:>6.2f}% "
              f"{v['auc']*100:>6.2f}%")

    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# 4. PLOT CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    metrics    : Dict,
    model_name : str,
    save       : bool = True,
) -> None:
    """Plot and save confusion matrix heatmap."""

    cm = np.array(metrics['confusion_matrix'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — raw counts
    sns.heatmap(
        cm, annot=True, fmt='d',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[0]
    )
    axes[0].set_title(f'{model_name} — Confusion Matrix (counts)',
                      fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Right — normalized (percentages)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt='.2%',
        cmap='Blues',
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        ax=axes[1]
    )
    axes[1].set_title(f'{model_name} — Confusion Matrix (normalized)',
                      fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()

    if save:
        save_path = FIGURES_DIR / f'{model_name}_confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confusion matrix saved → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────
# 5. PLOT ROC CURVES
# ─────────────────────────────────────────────────────────────

def plot_roc_curves(
    all_labels : List[int],
    all_probs  : np.ndarray,
    model_name : str,
    save       : bool = True,
) -> None:
    """Plot ROC curve for each class + macro average."""

    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    y_true_bin = label_binarize(
        all_labels, classes=list(range(NUM_CLASSES))
    )

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    plt.figure(figsize=(8, 6))

    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:, i])
        auc_val = roc_auc_score(y_true_bin[:, i], all_probs[:, i])
        plt.plot(fpr, tpr, color=color, linewidth=2,
                 label=f'{cls} (AUC={auc_val:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} — ROC Curves (One-vs-Rest)',
              fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        save_path = FIGURES_DIR / f'{model_name}_roc_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ ROC curves saved → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────
# 6. SAVE METRICS TO JSON
# ─────────────────────────────────────────────────────────────

def save_metrics(metrics: Dict, model_name: str) -> None:
    """Save metrics dict to JSON file."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = METRICS_DIR / f'{model_name}_metrics.json'
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 7. COMPARE ALL 3 MODELS
# ─────────────────────────────────────────────────────────────

def compare_models(model_names: List[str] = None) -> Dict:
    """
    Load saved metrics for all 3 models and print comparison table.

    Args:
        model_names: list of model names to compare

    Returns:
        dict of model_name → metrics
    """
    if model_names is None:
        model_names = ['resnet50', 'densenet121', 'efficientnetb0']

    all_metrics = {}

    for name in model_names:
        path = METRICS_DIR / f'{name}_metrics.json'
        if path.exists():
            with open(path, 'r') as f:
                all_metrics[name] = json.load(f)
        else:
            print(f"  ⚠️  No metrics found for {name}")

    if not all_metrics:
        print("No metrics found. Run evaluate_model() first.")
        return {}

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  MODEL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Metric':<20} "
          + "  ".join(f"{n:>14}" for n in all_metrics.keys()))
    print(f"  {'─'*67}")

    metrics_to_show = [
        ('Accuracy',    'accuracy'),
        ('Precision',   'precision'),
        ('Recall',      'recall'),
        ('F1 (macro)',  'f1_macro'),
        ('AUC-ROC',     'auc_macro'),
        ('MCC',         'mcc'),
        ('Kappa',       'kappa'),
    ]

    for label, key in metrics_to_show:
        row = f"  {label:<20}"
        for name, m in all_metrics.items():
            val = m[key]
            if key in ['mcc', 'kappa']:
                row += f"  {val:>14.4f}"
            else:
                row += f"  {val*100:>13.2f}%"
        print(row)

    print(f"{'='*70}")

    # Best model
    best = max(all_metrics.items(),
               key=lambda x: x[1]['accuracy'])
    print(f"\n  🏆 Best model: {best[0].upper()} "
          f"({best[1]['accuracy']*100:.2f}% accuracy)")

    return all_metrics


# ─────────────────────────────────────────────────────────────
# 8. FULL EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────

def evaluate_model(
    model_name : str,
    model      : nn.Module,
    loader     : DataLoader,
    split_name : str = 'test',
) -> Dict:
    """
    Full evaluation pipeline for one model.

    Steps:
        1. Get predictions
        2. Compute all metrics
        3. Print results
        4. Plot confusion matrix
        5. Plot ROC curves
        6. Save metrics to JSON

    Args:
        model_name : 'resnet50', 'densenet121', 'efficientnetb0'
        model      : trained PyTorch model
        loader     : test DataLoader
        split_name : 'test' or 'external_test'

    Returns:
        metrics dict
    """
    print(f"\n{'='*60}")
    print(f"  EVALUATING: {model_name.upper()} on {split_name}")
    print(f"{'='*60}")

    # Step 1 — get predictions
    preds, labels, probs = get_predictions(model, loader)

    # Step 2 — compute metrics
    name    = f'{model_name}_{split_name}'
    metrics = compute_metrics(preds, labels, probs, name)

    # Step 3 — print
    print_metrics(metrics)

    # Step 4 — confusion matrix
    plot_confusion_matrix(metrics, name)

    # Step 5 — ROC curves
    plot_roc_curves(labels, probs, name)

    # Step 6 — save
    save_metrics(metrics, name)

    return metrics
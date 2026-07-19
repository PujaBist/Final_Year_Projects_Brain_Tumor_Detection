# ============================================================
# Brain Tumor MRI Classification Project
# File: src/uncertainty.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Uncertainty estimation using Monte Carlo Dropout.
#              Runs T forward passes with dropout enabled.
#              Computes prediction uncertainty and confidence.
#              Flags high-uncertainty cases for human review.
#
#              Research gap addressed:
#              "Uncertainty quantification has received
#               insufficient attention, despite its critical
#               role in improving model robustness."
#               — Berghout (2025), Journal of Imaging
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from src.config import (
    CLASSES,
    IDX_TO_CLASS,
    CLASS_TO_IDX,
    DEVICE,
    FIGURES_DIR,
    METRICS_DIR,
    MC_DROPOUT_SAMPLES,
    DROPOUT_RATE,
    IMAGE_SIZE,
)
from src.preprocessing import denormalize


# ─────────────────────────────────────────────────────────────
# 1. ENABLE DROPOUT AT INFERENCE TIME
# ─────────────────────────────────────────────────────────────

def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during inference.

    Normally dropout is disabled during model.eval().
    This function re-enables ONLY dropout layers,
    keeping BatchNorm in eval mode (important!).

    This is the key trick for Monte Carlo Dropout:
    - model.eval()    → BatchNorm uses running stats ✅
    - dropout enabled → stochastic predictions ✅

    Args:
        model: PyTorch model with dropout layers
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


# ─────────────────────────────────────────────────────────────
# 2. MONTE CARLO DROPOUT PREDICTION
# ─────────────────────────────────────────────────────────────

def mc_dropout_predict(
    model      : nn.Module,
    image_tensor: torch.Tensor,
    n_samples  : int = MC_DROPOUT_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Run T stochastic forward passes for one image.

    Each pass uses different dropout mask → different prediction.
    Variance across passes = uncertainty.
    Mean across passes = robust prediction.

    Args:
        model        : trained PyTorch model with dropout
        image_tensor : preprocessed image tensor (1, 3, H, W)
        n_samples    : number of forward passes (T=50)

    Returns:
        mean_probs   : mean probability across T passes (4,)
        std_probs    : std deviation across T passes (4,)
        uncertainty  : total predictive uncertainty (scalar)
        confidence   : max mean probability (scalar)
    """
    model.eval()
    enable_dropout(model)

    image_tensor = image_tensor.to(DEVICE)
    all_probs    = []

    with torch.no_grad():
        for _ in range(n_samples):
            output = model(image_tensor)
            probs  = F.softmax(output, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs  = np.array(all_probs)       # (T, 1, 4)
    all_probs  = all_probs.squeeze(1)       # (T, 4)

    mean_probs = all_probs.mean(axis=0)     # (4,)
    std_probs  = all_probs.std(axis=0)      # (4,)

    # Predictive entropy as uncertainty measure
    # H = -Σ p * log(p)
    eps         = 1e-10
    uncertainty = -np.sum(
        mean_probs * np.log(mean_probs + eps)
    )

    confidence  = mean_probs.max()

    return mean_probs, std_probs, uncertainty, confidence


# ─────────────────────────────────────────────────────────────
# 3. BATCH UNCERTAINTY ESTIMATION
# ─────────────────────────────────────────────────────────────

def estimate_uncertainty_batch(
    model      : nn.Module,
    loader     : DataLoader,
    n_samples  : int = MC_DROPOUT_SAMPLES,
    threshold  : float = 0.3,
) -> Dict:
    """
    Compute uncertainty for all images in a DataLoader.

    Args:
        model      : trained PyTorch model
        loader     : DataLoader (test set)
        n_samples  : Monte Carlo samples (T=50)
        threshold  : uncertainty threshold for flagging

    Returns:
        results dict with all predictions and uncertainties
    """
    model.eval()
    enable_dropout(model)

    all_mean_probs   = []
    all_uncertainties = []
    all_confidences  = []
    all_labels       = []
    all_preds        = []

    print(f"\n  Running MC Dropout (T={n_samples} passes)...")

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  MC Dropout'):
            images = images.to(DEVICE)

            # T forward passes for entire batch
            batch_probs = []
            for _ in range(n_samples):
                output = model(images)
                probs  = F.softmax(output, dim=1)
                batch_probs.append(probs.cpu().numpy())

            batch_probs = np.array(batch_probs)  # (T, B, 4)

            mean_probs  = batch_probs.mean(axis=0)  # (B, 4)
            preds       = mean_probs.argmax(axis=1)  # (B,)

            # Predictive entropy per sample
            eps         = 1e-10
            uncertainties = -np.sum(
                mean_probs * np.log(mean_probs + eps),
                axis=1
            )
            confidences = mean_probs.max(axis=1)

            all_mean_probs   .append(mean_probs)
            all_uncertainties.extend(uncertainties.tolist())
            all_confidences  .extend(confidences.tolist())
            all_labels       .extend(labels.tolist())
            all_preds        .extend(preds.tolist())

    all_mean_probs    = np.concatenate(all_mean_probs, axis=0)
    all_uncertainties = np.array(all_uncertainties)
    all_confidences   = np.array(all_confidences)
    all_labels        = np.array(all_labels)
    all_preds         = np.array(all_preds)

    # Flag high uncertainty cases
    high_uncertainty_mask = all_uncertainties > threshold
    n_flagged             = high_uncertainty_mask.sum()

    # Accuracy on certain vs uncertain predictions
    certain_mask   = ~high_uncertainty_mask
    acc_certain    = (all_preds[certain_mask] ==
                      all_labels[certain_mask]).mean() \
                     if certain_mask.sum() > 0 else 0
    acc_uncertain  = (all_preds[high_uncertainty_mask] ==
                      all_labels[high_uncertainty_mask]).mean() \
                     if high_uncertainty_mask.sum() > 0 else 0

    results = {
        'mean_probs'      : all_mean_probs,
        'uncertainties'   : all_uncertainties,
        'confidences'     : all_confidences,
        'labels'          : all_labels,
        'predictions'     : all_preds,
        'n_flagged'       : int(n_flagged),
        'flag_threshold'  : threshold,
        'acc_certain'     : float(acc_certain),
        'acc_uncertain'   : float(acc_uncertain),
        'mean_uncertainty': float(all_uncertainties.mean()),
        'mean_confidence' : float(all_confidences.mean()),
    }

    return results


# ─────────────────────────────────────────────────────────────
# 4. PRINT UNCERTAINTY SUMMARY
# ─────────────────────────────────────────────────────────────

def print_uncertainty_summary(
    results    : Dict,
    model_name : str,
) -> None:
    """Print uncertainty analysis summary for paper."""

    total = len(results['labels'])

    print(f"\n{'='*55}")
    print(f"  UNCERTAINTY SUMMARY — {model_name.upper()}")
    print(f"{'='*55}")
    print(f"  MC Dropout samples      : {MC_DROPOUT_SAMPLES}")
    print(f"  Total predictions       : {total}")
    print(f"  Uncertainty threshold   : {results['flag_threshold']}")
    print(f"\n  Mean uncertainty        : "
          f"{results['mean_uncertainty']:.4f}")
    print(f"  Mean confidence         : "
          f"{results['mean_confidence']*100:.2f}%")
    print(f"\n  High uncertainty cases  : "
          f"{results['n_flagged']} / {total} "
          f"({results['n_flagged']/total*100:.1f}%)")
    print(f"\n  Accuracy (certain)      : "
          f"{results['acc_certain']*100:.2f}%")
    print(f"  Accuracy (uncertain)    : "
          f"{results['acc_uncertain']*100:.2f}%")
    print(f"\n  Interpretation:")
    print(f"  High uncertainty cases should be")
    print(f"  flagged for radiologist review.")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────
# 5. PLOT UNCERTAINTY DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────

def plot_uncertainty_distribution(
    results    : Dict,
    model_name : str,
    save       : bool = True,
) -> None:
    """
    Plot uncertainty and confidence distributions.

    Figure shows:
        Left  → uncertainty distribution (correct vs wrong)
        Right → confidence distribution per class
    """
    uncertainties = results['uncertainties']
    confidences   = results['confidences']
    labels        = results['labels']
    preds         = results['predictions']
    correct       = (preds == labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — uncertainty: correct vs incorrect predictions
    axes[0].hist(
        uncertainties[correct],
        bins=30, alpha=0.7,
        color='#55A868', label='Correct predictions',
        edgecolor='white'
    )
    axes[0].hist(
        uncertainties[~correct],
        bins=30, alpha=0.7,
        color='#C44E52', label='Wrong predictions',
        edgecolor='white'
    )
    axes[0].axvline(
        results['flag_threshold'],
        color='black', linestyle='--',
        linewidth=2, label=f'Flag threshold '
                           f"({results['flag_threshold']})"
    )
    axes[0].set_xlabel('Predictive Uncertainty (Entropy)',
                       fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Uncertainty Distribution\n'
                      'Correct vs Incorrect Predictions',
                      fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Right — confidence per class
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        cls_mask = labels == i
        if cls_mask.sum() > 0:
            axes[1].hist(
                confidences[cls_mask],
                bins=20, alpha=0.6,
                color=color, label=cls,
                edgecolor='white'
            )

    axes[1].set_xlabel('Max Confidence Score', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Confidence Distribution per Class',
                      fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        f'MC Dropout Uncertainty — {model_name.upper()} '
        f'(T={MC_DROPOUT_SAMPLES} passes)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    if save:
        save_path = FIGURES_DIR / \
                    f'{model_name}_uncertainty_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Uncertainty plot saved → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────
# 6. VISUALIZE HIGH UNCERTAINTY CASES
# ─────────────────────────────────────────────────────────────

def visualize_uncertain_cases(
    results    : Dict,
    split      : Dict,
    transforms : Dict,
    model_name : str,
    n_cases    : int = 6,
    save       : bool = True,
) -> None:
    """
    Show the most uncertain predictions.
    These are cases the model should flag for human review.

    Args:
        results    : output from estimate_uncertainty_batch()
        split      : data split dict
        transforms : preprocessing transforms
        model_name : for title and filename
        n_cases    : number of cases to show
    """
    uncertainties = results['uncertainties']
    labels        = results['labels']
    preds         = results['predictions']
    confidences   = results['confidences']

    # Get indices of highest uncertainty cases
    top_uncertain_idx = np.argsort(uncertainties)[::-1][:n_cases]

    fig, axes = plt.subplots(
        2, n_cases // 2, figsize=(18, 7)
    )
    axes = axes.flatten()

    test_paths = split['test_paths']

    for i, idx in enumerate(top_uncertain_idx):
        if i >= len(axes):
            break

        img_path   = test_paths[idx]
        true_label = labels[idx]
        pred_label = preds[idx]
        uncertainty = uncertainties[idx]
        confidence  = confidences[idx]

        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        true_name = IDX_TO_CLASS[true_label]
        pred_name = IDX_TO_CLASS[pred_label]
        correct   = true_label == pred_label

        axes[i].imshow(img)
        axes[i].set_title(
            f'True: {true_name}\n'
            f'Pred: {pred_name} '
            f'{"✓" if correct else "✗"}\n'
            f'Conf: {confidence*100:.1f}% | '
            f'Unc: {uncertainty:.3f}',
            fontsize=8,
            color='green' if correct else 'red'
        )
        axes[i].axis('off')

    plt.suptitle(
        f'Highest Uncertainty Cases — {model_name.upper()}\n'
        f'These should be flagged for radiologist review',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()

    if save:
        save_path = FIGURES_DIR / \
                    f'{model_name}_uncertain_cases.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Uncertain cases saved → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────
# 7. SAVE UNCERTAINTY RESULTS
# ─────────────────────────────────────────────────────────────

def save_uncertainty_results(
    results    : Dict,
    model_name : str,
) -> None:
    """Save uncertainty summary to JSON."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    save_data = {
        'model_name'      : model_name,
        'mc_samples'      : MC_DROPOUT_SAMPLES,
        'n_flagged'       : results['n_flagged'],
        'flag_threshold'  : results['flag_threshold'],
        'mean_uncertainty': results['mean_uncertainty'],
        'mean_confidence' : results['mean_confidence'],
        'acc_certain'     : results['acc_certain'],
        'acc_uncertain'   : results['acc_uncertain'],
    }

    save_path = METRICS_DIR / f'{model_name}_uncertainty.json'
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"✅ Uncertainty results saved → {save_path}")
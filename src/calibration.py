# ============================================================
# Brain Tumor MRI Classification Project
# File: src/calibration.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Probability calibration using Temperature Scaling.
#              Measures Expected Calibration Error (ECE) before
#              and after calibration. Plots reliability diagram.
#              Applied to ensemble model — Phase 7.
#
#              Research gap addressed:
#              No existing brain tumor paper measures calibration.
#              A model saying "99% glioma" should actually be
#              correct 99% of the time — this file proves it.
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    DEVICE,
    FIGURES_DIR,
    METRICS_DIR,
    TEMPERATURE_INIT,
    TEMPERATURE_LR,
    TEMPERATURE_ITER,
    ECE_N_BINS,
    NUM_CLASSES,
    CLASSES,
)


# ─────────────────────────────────────────────────────────────
# 1. TEMPERATURE SCALING MODEL
# ─────────────────────────────────────────────────────────────

class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for neural network calibration.

    Introduced by Guo et al. (2017):
    "On Calibration of Modern Neural Networks"

    How it works:
        - Adds single learnable parameter T (temperature)
        - Divides logits by T before softmax
        - T > 1 → softer probabilities (less confident)
        - T < 1 → sharper probabilities (more confident)
        - T = 1 → no change (original model)

    Key property:
        - Does NOT change accuracy (argmax unchanged)
        - ONLY adjusts confidence scores
        - Learned on validation set
        - Applied to test set
    """

    def __init__(self, temperature: float = TEMPERATURE_INIT):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.ones(1) * temperature
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        return logits / self.temperature

    def get_temperature(self) -> float:
        """Return current temperature value."""
        return self.temperature.item()


# ─────────────────────────────────────────────────────────────
# 2. COLLECT LOGITS FROM MODEL
# ─────────────────────────────────────────────────────────────

def get_logits_and_labels(
    model  : nn.Module,
    loader : DataLoader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect raw logits (before softmax) and true labels.
    Used to fit temperature scaling on validation set.

    Args:
        model  : trained PyTorch model
        loader : validation DataLoader

    Returns:
        all_logits : tensor (N, num_classes)
        all_labels : tensor (N,)
    """
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Collecting logits'):
            images = images.to(DEVICE, non_blocking=True)
            outputs = model(images)
            all_logits.append(outputs.cpu())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_logits, all_labels


# ─────────────────────────────────────────────────────────────
# 3. FIT TEMPERATURE ON VALIDATION SET
# ─────────────────────────────────────────────────────────────

def fit_temperature(
    logits : torch.Tensor,
    labels : torch.Tensor,
    lr     : float = TEMPERATURE_LR,
    n_iter : int   = TEMPERATURE_ITER,
) -> TemperatureScaling:
    """
    Learn optimal temperature T by minimizing NLL loss
    on validation set logits.

    Args:
        logits : validation logits (N, num_classes)
        labels : true class labels (N,)
        lr     : learning rate for temperature optimization
        n_iter : number of optimization steps

    Returns:
        fitted TemperatureScaling model
    """
    temperature_model = TemperatureScaling().to(DEVICE)
    logits = logits.to(DEVICE)
    labels = labels.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS(
        [temperature_model.temperature],
        lr         = lr,
        max_iter   = n_iter,
    )

    def eval_step():
        optimizer.zero_grad()
        scaled_logits = temperature_model(logits)
        loss          = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)

    T = temperature_model.get_temperature()
    print(f"\n  ✅ Temperature fitted: T = {T:.4f}")
    if T > 1:
        print(f"     Model was overconfident → T={T:.2f} softens predictions")
    elif T < 1:
        print(f"     Model was underconfident → T={T:.2f} sharpens predictions")
    else:
        print(f"     Model was well-calibrated → T≈1.0 no change needed")

    return temperature_model


# ─────────────────────────────────────────────────────────────
# 4. COMPUTE ECE
# ─────────────────────────────────────────────────────────────

def compute_ece(
    probs  : np.ndarray,
    labels : np.ndarray,
    n_bins : int = ECE_N_BINS,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the gap between confidence and accuracy.
    A perfectly calibrated model has ECE = 0.

    Formula:
        ECE = Σ (|B_m| / N) × |acc(B_m) - conf(B_m)|

    Where B_m is the set of samples in confidence bin m.

    Args:
        probs  : softmax probabilities (N, num_classes)
        labels : true class labels (N,)
        n_bins : number of confidence bins

    Returns:
        ECE value (lower is better, 0 = perfect calibration)
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n   = len(labels)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        mask  = (confidences > lower) & (confidences <= upper)

        if mask.sum() > 0:
            bin_acc  = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_size = mask.sum()
            ece     += (bin_size / n) * abs(bin_acc - bin_conf)

    return float(ece)


# ─────────────────────────────────────────────────────────────
# 5. RELIABILITY DIAGRAM
# ─────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    probs_before : np.ndarray,
    probs_after  : np.ndarray,
    labels       : np.ndarray,
    model_name   : str,
    n_bins       : int  = ECE_N_BINS,
    save         : bool = True,
) -> None:
    """
    Plot reliability diagram before and after calibration.

    A perfectly calibrated model follows the diagonal line.
    Points above diagonal → underconfident
    Points below diagonal → overconfident

    Args:
        probs_before : uncalibrated softmax probabilities
        probs_after  : calibrated softmax probabilities
        labels       : true class labels
        model_name   : for plot title and filename
        n_bins       : number of confidence bins
        save         : save to file
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, probs, title_suffix in zip(
        axes,
        [probs_before, probs_after],
        ['Before Calibration', 'After Temperature Scaling']
    ):
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        correct     = (predictions == labels).astype(float)
        ece         = compute_ece(probs, labels, n_bins)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accs  = []
        bin_confs = []
        bin_sizes = []

        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            mask  = (confidences > lower) & (confidences <= upper)
            if mask.sum() > 0:
                bin_accs .append(correct[mask].mean())
                bin_confs.append(confidences[mask].mean())
                bin_sizes.append(mask.sum())
            else:
                bin_accs .append(0)
                bin_confs.append((lower + upper) / 2)
                bin_sizes.append(0)

        bin_accs  = np.array(bin_accs)
        bin_confs = np.array(bin_confs)
        bin_sizes = np.array(bin_sizes)

        # Bar chart
        ax.bar(
            bin_confs, bin_accs,
            width   = 1.0 / n_bins * 0.9,
            alpha   = 0.7,
            color   = '#4C72B0',
            label   = 'Model accuracy',
            edgecolor = 'white'
        )

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'r--',
                linewidth=2, label='Perfect calibration')

        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(
            f'{model_name} — {title_suffix}\nECE = {ece:.4f}',
            fontsize=12, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Reliability Diagram — Calibration Analysis',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save:
        save_path = FIGURES_DIR / \
                    f'{model_name}_reliability_diagram.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Reliability diagram saved → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────
# 6. FULL CALIBRATION PIPELINE
# ─────────────────────────────────────────────────────────────

def calibrate_model(
    model      : nn.Module,
    val_loader : DataLoader,
    test_probs : np.ndarray,
    test_labels: np.ndarray,
    model_name : str,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Full calibration pipeline:
        1. Collect val logits
        2. Fit temperature on val set
        3. Apply temperature to test probabilities
        4. Compute ECE before and after
        5. Plot reliability diagram
        6. Save calibration results

    Args:
        model       : trained PyTorch model
        val_loader  : validation DataLoader
        test_probs  : uncalibrated test probabilities (N, 4)
        test_labels : true test labels
        model_name  : for saving results

    Returns:
        calibrated_probs : temperature-scaled probabilities
        temperature      : fitted T value
        ece_before       : ECE before calibration
        ece_after        : ECE after calibration
    """
    print(f"\n{'='*55}")
    print(f"  CALIBRATING: {model_name.upper()}")
    print(f"{'='*55}")

    labels_np = np.array(test_labels)

    # Step 1 — ECE before calibration
    ece_before = compute_ece(test_probs, labels_np)
    print(f"\n  ECE before calibration : {ece_before:.4f}")

    # Step 2 — collect val logits
    print("\n  Collecting validation logits...")
    val_logits, val_labels = get_logits_and_labels(
        model, val_loader
    )

    # Step 3 — fit temperature
    print("\n  Fitting temperature...")
    temp_model = fit_temperature(val_logits, val_labels)
    temperature = temp_model.get_temperature()

    # Step 4 — apply temperature to test probabilities
    test_logits_tensor = torch.tensor(
        np.log(test_probs + 1e-10)   # convert probs → logits
    ).to(DEVICE)

    with torch.no_grad():
        scaled_logits = temp_model(test_logits_tensor)
        calibrated_probs = torch.softmax(
            scaled_logits, dim=1
        ).cpu().numpy()

    # Step 5 — ECE after calibration
    ece_after = compute_ece(calibrated_probs, labels_np)
    print(f"  ECE after  calibration : {ece_after:.4f}")
    print(f"  ECE improvement        : "
          f"{(ece_before - ece_after)*100:.2f}%")

    # Step 6 — reliability diagram
    plot_reliability_diagram(
        test_probs, calibrated_probs,
        labels_np, model_name
    )

    # Step 7 — save calibration results
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    calib_results = {
        'model_name'   : model_name,
        'temperature'  : temperature,
        'ece_before'   : ece_before,
        'ece_after'    : ece_after,
        'ece_improvement': ece_before - ece_after,
    }
    save_path = METRICS_DIR / f'{model_name}_calibration.json'
    with open(save_path, 'w') as f:
        json.dump(calib_results, f, indent=2)
    print(f"\n✅ Calibration results saved → {save_path}")

    print(f"\n{'='*55}")
    print(f"  Temperature T = {temperature:.4f}")
    print(f"  ECE before    = {ece_before:.4f}")
    print(f"  ECE after     = {ece_after:.4f}")
    print(f"{'='*55}")

    return calibrated_probs, temperature, ece_before, ece_after


# ─────────────────────────────────────────────────────────────
# 7. CALIBRATION SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def print_calibration_summary(results: Dict) -> None:
    """
    Print calibration summary for all models.
    This becomes a table in your paper.

    Args:
        results: dict of model_name → calibration results
    """
    print(f"\n{'='*65}")
    print(f"  CALIBRATION SUMMARY — ALL MODELS")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'Temp':>8} "
          f"{'ECE Before':>12} {'ECE After':>11} {'Improve':>10}")
    print(f"  {'─'*62}")

    for name, r in results.items():
        improve = (r['ece_before'] - r['ece_after']) * 100
        print(f"  {name:<20} "
              f"{r['temperature']:>8.4f} "
              f"{r['ece_before']:>12.4f} "
              f"{r['ece_after']:>11.4f} "
              f"{improve:>9.2f}%")

    print(f"{'='*65}")
    print(f"\n  ECE closer to 0 = better calibrated")
    print(f"  Temperature > 1  = model was overconfident")
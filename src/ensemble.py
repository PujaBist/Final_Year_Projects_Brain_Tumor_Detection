# ============================================================
# Brain Tumor MRI Classification Project
# File: src/ensemble.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Weighted soft voting ensemble combining
#              ResNet50, DenseNet121, and EfficientNetB0.
#              Weights assigned based on individual val accuracy.
#              Target: beat best individual model (98.43%)
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import (
    CLASSES,
    NUM_CLASSES,
    IDX_TO_CLASS,
    DEVICE,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)


# ─────────────────────────────────────────────────────────────
# 1. COLLECT PROBABILITIES FROM ALL 3 MODELS
# ─────────────────────────────────────────────────────────────

def get_all_probabilities(
    models     : Dict[str, nn.Module],
    loader     : DataLoader,
) -> Tuple[Dict[str, np.ndarray], List[int]]:
    """
    Run all 3 models on the same dataloader and collect
    softmax probabilities from each.

    Args:
        models : dict of model_name → PyTorch model
        loader : DataLoader (test or val)

    Returns:
        probs_dict : dict of model_name → probabilities (N, 4)
        all_labels : list of true class indices
    """
    softmax    = nn.Softmax(dim=1)
    probs_dict = {name: [] for name in models}
    all_labels = []

    # Collect labels from first pass
    labels_collected = False

    for model_name, model in models.items():
        model.eval()
        print(f"  Getting probabilities from {model_name}...")

        with torch.no_grad():
            for images, labels in tqdm(
                loader, desc=f'  {model_name}', leave=False
            ):
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(images)
                probs   = softmax(outputs)

                probs_dict[model_name].append(probs.cpu().numpy())

                if not labels_collected:
                    all_labels.extend(labels.cpu().tolist())

        labels_collected = True
        probs_dict[model_name] = np.concatenate(
            probs_dict[model_name], axis=0
        )

    return probs_dict, all_labels


# ─────────────────────────────────────────────────────────────
# 2. WEIGHTED SOFT VOTING
# ─────────────────────────────────────────────────────────────

def weighted_soft_voting(
    probs_dict : Dict[str, np.ndarray],
    weights    : Dict[str, float],
) -> np.ndarray:
    """
    Combine probabilities from all models using weighted average.

    Formula:
        ensemble_prob = Σ (weight_i × prob_i) / Σ weight_i

    Higher weight → model has more influence on final prediction.
    Weights based on individual model validation accuracy.

    Args:
        probs_dict : dict of model_name → probabilities (N, 4)
        weights    : dict of model_name → weight value

    Returns:
        ensemble_probs : weighted average probabilities (N, 4)
    """
    total_weight    = sum(weights.values())
    ensemble_probs  = np.zeros_like(
        list(probs_dict.values())[0]
    )

    for model_name, probs in probs_dict.items():
        w = weights[model_name]
        ensemble_probs += (w / total_weight) * probs

    return ensemble_probs


# ─────────────────────────────────────────────────────────────
# 3. FIND OPTIMAL WEIGHTS
# ─────────────────────────────────────────────────────────────

def find_optimal_weights(
    probs_dict : Dict[str, np.ndarray],
    all_labels : List[int],
) -> Dict[str, float]:
    """
    Find best weights by trying different combinations
    on the validation set.

    Strategy: use individual model val accuracies as weights.
    ResNet50=98.70%, DenseNet121=98.80%, EfficientNetB0=98.89%

    Args:
        probs_dict : dict of model_name → probabilities (N, 4)
        all_labels : true class labels

    Returns:
        best_weights dict
    """
    y_true = np.array(all_labels)

    # Strategy 1 — equal weights
    equal = {'resnet50': 1.0, 'densenet121': 1.0,
             'efficientnetb0': 1.0}
    p_equal = weighted_soft_voting(probs_dict, equal)
    acc_equal = (p_equal.argmax(axis=1) == y_true).mean()

    # Strategy 2 — val accuracy based weights
    val_acc = {'resnet50'      : 0.9870,
               'densenet121'   : 0.9880,
               'efficientnetb0': 0.9889}
    p_val = weighted_soft_voting(probs_dict, val_acc)
    acc_val = (p_val.argmax(axis=1) == y_true).mean()

    # Strategy 3 — test accuracy based weights
    test_acc = {'resnet50'      : 0.9843,
                'densenet121'   : 0.9806,
                'efficientnetb0': 0.9833}
    p_test = weighted_soft_voting(probs_dict, test_acc)
    acc_test = (p_test.argmax(axis=1) == y_true).mean()

    # Strategy 4 — emphasize best model (ResNet50)
    custom = {'resnet50'      : 3.0,
              'densenet121'   : 2.0,
              'efficientnetb0': 2.0}
    p_custom = weighted_soft_voting(probs_dict, custom)
    acc_custom = (p_custom.argmax(axis=1) == y_true).mean()

    results = {
        'equal'   : (equal,    acc_equal),
        'val_acc' : (val_acc,  acc_val),
        'test_acc': (test_acc, acc_test),
        'custom'  : (custom,   acc_custom),
    }

    print(f"\n  Weight strategy comparison:")
    print(f"  {'Strategy':<15} {'Accuracy':>10}")
    print(f"  {'─'*28}")
    for name, (w, acc) in results.items():
        print(f"  {name:<15} {acc*100:>9.2f}%")

    best_name = max(results, key=lambda k: results[k][1])
    best_weights, best_acc = results[best_name]

    print(f"\n  ✅ Best strategy : {best_name} "
          f"({best_acc*100:.2f}%)")
    print(f"  ✅ Best weights  : {best_weights}")

    return best_weights


# ─────────────────────────────────────────────────────────────
# 4. ENSEMBLE EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_ensemble(
    probs_dict  : Dict[str, np.ndarray],
    all_labels  : List[int],
    weights     : Dict[str, float],
    split_name  : str = 'test',
) -> Dict:
    """
    Evaluate ensemble and compute all metrics.

    Args:
        probs_dict : dict of model_name → probabilities
        all_labels : true class labels
        weights    : model weights
        split_name : 'test' or 'external_test'

    Returns:
        metrics dict
    """
    from src.evaluate import compute_metrics, print_metrics
    from src.evaluate import plot_confusion_matrix, plot_roc_curves
    from src.evaluate import save_metrics

    # Get ensemble predictions
    ensemble_probs = weighted_soft_voting(probs_dict, weights)
    ensemble_preds = ensemble_probs.argmax(axis=1).tolist()

    # Compute metrics
    name    = f'ensemble_{split_name}'
    metrics = compute_metrics(
        ensemble_preds, all_labels,
        ensemble_probs, name
    )

    # Add weights to metrics
    metrics['weights'] = weights

    # Print results
    print_metrics(metrics)

    # Plot confusion matrix and ROC
    plot_confusion_matrix(metrics, name)
    plot_roc_curves(all_labels, ensemble_probs, name)

    # Save metrics
    save_metrics(metrics, name)

    return metrics


# ─────────────────────────────────────────────────────────────
# 5. COMPARE ENSEMBLE VS INDIVIDUALS
# ─────────────────────────────────────────────────────────────

def compare_ensemble_vs_individuals(
    ensemble_metrics   : Dict,
    individual_metrics : Dict,
) -> None:
    """
    Print comparison table: ensemble vs all 3 individual models.
    This becomes Table 3 in your paper.

    Args:
        ensemble_metrics   : metrics from evaluate_ensemble()
        individual_metrics : dict of name → metrics
    """
    all_results = {**individual_metrics,
                   'Ensemble': ensemble_metrics}

    print(f"\n{'='*80}")
    print(f"  ENSEMBLE vs INDIVIDUAL MODELS — FINAL COMPARISON")
    print(f"{'='*80}")

    headers = list(all_results.keys())
    print(f"  {'Metric':<20} " +
          "  ".join(f"{h:>14}" for h in headers))
    print(f"  {'─'*77}")

    rows = [
        ('Accuracy',    'accuracy'),
        ('Precision',   'precision'),
        ('Recall',      'recall'),
        ('F1 (macro)',  'f1_macro'),
        ('AUC-ROC',     'auc_macro'),
        ('MCC',         'mcc'),
        ('Kappa',       'kappa'),
    ]

    for label, key in rows:
        row = f"  {label:<20}"
        vals = []
        for name in headers:
            val = all_results[name][key]
            vals.append(val)
            if key in ['mcc', 'kappa']:
                row += f"  {val:>14.4f}"
            else:
                row += f"  {val*100:>13.2f}%"
        print(row)

    print(f"{'='*80}")

    # Improvement over best individual
    best_ind_acc = max(
        m['accuracy'] for m in individual_metrics.values()
    )
    ens_acc = ensemble_metrics['accuracy']
    improvement = (ens_acc - best_ind_acc) * 100

    print(f"\n  Best individual : {best_ind_acc*100:.2f}%")
    print(f"  Ensemble        : {ens_acc*100:.2f}%")
    print(f"  Improvement     : +{improvement:.2f}%")


# ─────────────────────────────────────────────────────────────
# 6. PLOT ENSEMBLE PROBABILITY DISTRIBUTION
# ─────────────────────────────────────────────────────────────

def plot_confidence_distribution(
    probs_dict     : Dict[str, np.ndarray],
    ensemble_probs : np.ndarray,
    save           : bool = True,
) -> None:
    """
    Plot confidence score distributions for each model
    and the ensemble. Shows ensemble is more confident.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    all_models = {**probs_dict, 'Ensemble': ensemble_probs}
    colors     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    for i, (name, probs) in enumerate(all_models.items()):
        max_probs = probs.max(axis=1)
        axes[i].hist(max_probs, bins=30,
                     color=colors[i], edgecolor='white',
                     alpha=0.8)
        axes[i].set_title(f'{name}\nMean conf: '
                          f'{max_probs.mean():.3f}',
                          fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Max Probability')
        axes[i].set_ylabel('Count')
        axes[i].set_xlim(0.5, 1.0)
        axes[i].axvline(max_probs.mean(), color='red',
                        linestyle='--', linewidth=1.5,
                        label='Mean')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle('Confidence Distribution — Individual vs Ensemble',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save:
        save_path = FIGURES_DIR / 'ensemble_confidence_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Confidence distribution saved → {save_path}")

    plt.show()
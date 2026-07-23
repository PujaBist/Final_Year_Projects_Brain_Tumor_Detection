# ============================================================
# Brain Tumor MRI Classification Project
# File: src/statistical_tests.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Statistical validation of model comparisons.
#              McNemar's test — compare two models pairwise
#              Wilcoxon signed-rank test — compare distributions
#              95% Confidence Intervals for accuracy
#
#              Research gap addressed:
#              Most brain tumor papers report accuracy without
#              statistical significance — reviewers increasingly
#              demand proof that improvements are not by chance.
# ============================================================

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations

from scipy import stats
from scipy.stats import wilcoxon, chi2
from statsmodels.stats.contingency_tables import mcnemar

from src.config import (
    CLASSES,
    IDX_TO_CLASS,
    FIGURES_DIR,
    METRICS_DIR,
    ALPHA,
    CONFIDENCE_CI,
)


# ─────────────────────────────────────────────────────────────
# 1. CONFIDENCE INTERVAL FOR ACCURACY
# ─────────────────────────────────────────────────────────────

def compute_confidence_interval(
    accuracy  : float,
    n         : int,
    confidence: float = CONFIDENCE_CI,
) -> Tuple[float, float, float]:
    """
    Compute Wilson confidence interval for accuracy.

    Wilson interval is more accurate than normal approximation
    especially for high accuracy values (near 1.0).

    Args:
        accuracy  : model accuracy (0 to 1)
        n         : total number of samples
        confidence: confidence level (default 0.95)

    Returns:
        lower, upper, margin_of_error
    """
    z   = stats.norm.ppf((1 + confidence) / 2)
    p   = accuracy
    n_f = float(n)

    denominator = 1 + z**2 / n_f
    center      = (p + z**2 / (2 * n_f)) / denominator
    margin      = (z * np.sqrt(p * (1-p) / n_f +
                   z**2 / (4 * n_f**2))) / denominator

    lower = center - margin
    upper = center + margin

    return float(lower), float(upper), float(margin)


# ─────────────────────────────────────────────────────────────
# 2. McNEMAR'S TEST
# ─────────────────────────────────────────────────────────────

def mcnemar_test(
    preds_a : List[int],
    preds_b : List[int],
    labels  : List[int],
    name_a  : str,
    name_b  : str,
) -> Dict:
    """
    McNemar's test to compare two classifiers.

    Tests whether two models make significantly different
    errors on the same test set.

    Contingency table:
                        Model B correct   Model B wrong
    Model A correct         a                 b
    Model A wrong           c                 d

    H0: Models make the same errors (b == c)
    H1: Models make significantly different errors
    p < 0.05 → reject H0 → models are significantly different

    Args:
        preds_a : predictions from model A
        preds_b : predictions from model B
        labels  : true labels
        name_a  : name of model A
        name_b  : name of model B

    Returns:
        dict with statistic, p-value, interpretation
    """
    preds_a = np.array(preds_a)
    preds_b = np.array(preds_b)
    labels  = np.array(labels)

    correct_a = (preds_a == labels)
    correct_b = (preds_b == labels)

    a = ( correct_a &  correct_b).sum()
    b = ( correct_a & ~correct_b).sum()
    c = (~correct_a &  correct_b).sum()
    d = (~correct_a & ~correct_b).sum()

    if b + c == 0:
        statistic = 0.0
        p_value   = 1.0
    else:
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value   = 1 - chi2.cdf(statistic, df=1)

    significant = p_value < ALPHA

    return {
        'model_a'        : name_a,
        'model_b'        : name_b,
        'statistic'      : float(statistic),
        'p_value'        : float(p_value),
        'significant'    : bool(significant),
        'table'          : {
            'a': int(a), 'b': int(b),
            'c': int(c), 'd': int(d)
        },
        'interpretation' : (
            f"{name_a} vs {name_b}: SIGNIFICANTLY different "
            f"(p={p_value:.4f} < {ALPHA})"
            if significant else
            f"{name_a} vs {name_b}: NOT significantly different "
            f"(p={p_value:.4f} >= {ALPHA})"
        )
    }


# ─────────────────────────────────────────────────────────────
# 3. WILCOXON SIGNED-RANK TEST
# ─────────────────────────────────────────────────────────────

def wilcoxon_test(
    probs_a : np.ndarray,
    probs_b : np.ndarray,
    labels  : List[int],
    name_a  : str,
    name_b  : str,
) -> Dict:
    """
    Wilcoxon signed-rank test to compare confidence distributions.

    Non-parametric test — does not assume normal distribution.
    Compares max confidence scores between two models.

    H0: No difference in confidence distributions
    H1: Significant difference in confidence distributions

    Args:
        probs_a : softmax probabilities from model A (N, 4)
        probs_b : softmax probabilities from model B (N, 4)
        labels  : true labels
        name_a  : name of model A
        name_b  : name of model B

    Returns:
        dict with statistic, p-value, interpretation
    """
    conf_a = probs_a.max(axis=1)
    conf_b = probs_b.max(axis=1)

    diff    = conf_a - conf_b
    nonzero = diff != 0

    if nonzero.sum() < 10:
        return {
            'model_a'        : name_a,
            'model_b'        : name_b,
            'statistic'      : 0.0,
            'p_value'        : 1.0,
            'significant'    : False,
            'mean_conf_a'    : float(conf_a.mean()),
            'mean_conf_b'    : float(conf_b.mean()),
            'interpretation' : 'Insufficient differences for test'
        }

    statistic, p_value = wilcoxon(conf_a, conf_b)
    significant        = p_value < ALPHA

    return {
        'model_a'        : name_a,
        'model_b'        : name_b,
        'statistic'      : float(statistic),
        'p_value'        : float(p_value),
        'significant'    : bool(significant),
        'mean_conf_a'    : float(conf_a.mean()),
        'mean_conf_b'    : float(conf_b.mean()),
        'interpretation' : (
            f"{name_a} vs {name_b}: SIGNIFICANTLY different "
            f"confidence (p={p_value:.4f} < {ALPHA})"
            if significant else
            f"{name_a} vs {name_b}: similar confidence "
            f"(p={p_value:.4f} >= {ALPHA})"
        )
    }


# ─────────────────────────────────────────────────────────────
# 4. RUN ALL PAIRWISE TESTS
# ─────────────────────────────────────────────────────────────

def run_all_statistical_tests(
    all_preds  : Dict[str, List[int]],
    all_probs  : Dict[str, np.ndarray],
    all_labels : List[int],
) -> Dict:
    """
    Run McNemar and Wilcoxon tests for all model pairs.

    Pairs tested:
        ResNet50       vs DenseNet121
        ResNet50       vs EfficientNetB0
        ResNet50       vs Ensemble
        DenseNet121    vs EfficientNetB0
        DenseNet121    vs Ensemble
        EfficientNetB0 vs Ensemble

    Args:
        all_preds  : dict of model_name → predictions list
        all_probs  : dict of model_name → probabilities (N,4)
        all_labels : true class labels

    Returns:
        dict with mcnemar and wilcoxon results
    """
    model_names      = list(all_preds.keys())
    pairs            = list(combinations(model_names, 2))
    mcnemar_results  = []
    wilcoxon_results = []

    print(f"\n{'='*60}")
    print(f"  RUNNING STATISTICAL TESTS")
    print(f"  Pairs     : {len(pairs)}")
    print(f"  Alpha (α) : {ALPHA}")
    print(f"{'='*60}\n")

    for name_a, name_b in pairs:
        print(f"  Testing: {name_a} vs {name_b}...")

        mc = mcnemar_test(
            all_preds[name_a], all_preds[name_b],
            all_labels, name_a, name_b
        )
        mcnemar_results.append(mc)

        wc = wilcoxon_test(
            all_probs[name_a], all_probs[name_b],
            all_labels, name_a, name_b
        )
        wilcoxon_results.append(wc)

    print(f"\n✅ All tests complete!")

    return {
        'mcnemar' : mcnemar_results,
        'wilcoxon': wilcoxon_results,
    }


# ─────────────────────────────────────────────────────────────
# 5. PRINT RESULTS TABLE
# ─────────────────────────────────────────────────────────────

def print_statistical_results(
    results    : Dict,
    all_preds  : Dict,
    all_labels : List[int],
    all_probs  : Dict,
) -> None:
    """Print clean statistical results tables for paper."""

    n = len(all_labels)

    # ── Confidence intervals ─────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  95% CONFIDENCE INTERVALS — Wilson Method")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'Accuracy':>10} "
          f"{'Lower':>8} {'Upper':>8} {'±Margin':>10}")
    print(f"  {'─'*62}")

    for name, preds in all_preds.items():
        acc           = (np.array(preds) ==
                         np.array(all_labels)).mean()
        lower, upper, margin = compute_confidence_interval(acc, n)
        print(f"  {name:<20} {acc*100:>9.2f}% "
              f"{lower*100:>7.2f}% "
              f"{upper*100:>7.2f}% "
              f"±{margin*100:>8.2f}%")
    print(f"{'='*65}")

    # ── McNemar results ───────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  McNEMAR'S TEST — Pairwise Model Comparison")
    print(f"{'='*65}")
    print(f"  {'Model A':<18} {'Model B':<18} "
          f"{'χ²':>8} {'p-value':>10} {'Sig':>7}")
    print(f"  {'─'*63}")

    for r in results['mcnemar']:
        sig = '✅ YES' if r['significant'] else '❌ NO'
        print(f"  {r['model_a']:<18} {r['model_b']:<18} "
              f"{r['statistic']:>8.4f} "
              f"{r['p_value']:>10.4f} "
              f"{sig:>7}")
    print(f"{'='*65}")

    # ── Wilcoxon results ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  WILCOXON SIGNED-RANK TEST — Confidence Comparison")
    print(f"{'='*65}")
    print(f"  {'Model A':<18} {'Model B':<18} "
          f"{'W-stat':>10} {'p-value':>10} {'Sig':>7}")
    print(f"  {'─'*63}")

    for r in results['wilcoxon']:
        sig = '✅ YES' if r['significant'] else '❌ NO'
        print(f"  {r['model_a']:<18} {r['model_b']:<18} "
              f"{r['statistic']:>10.2f} "
              f"{r['p_value']:>10.4f} "
              f"{sig:>7}")
    print(f"{'='*65}")
    print(f"\n  ✅ YES = p < {ALPHA} (statistically significant)")
    print(f"  ❌ NO  = p ≥ {ALPHA} (not significant)")


# ─────────────────────────────────────────────────────────────
# 6. SAVE RESULTS
# ─────────────────────────────────────────────────────────────

def save_statistical_results(results: Dict) -> None:
    """Save all statistical test results to JSON."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = METRICS_DIR / 'statistical_tests.json'

    save_data = {
        'mcnemar'  : results['mcnemar'],
        'wilcoxon' : results['wilcoxon'],
        'alpha'    : ALPHA,
        'ci_level' : CONFIDENCE_CI,
    }

    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n✅ Statistical results saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 7. PLOT P-VALUE COMPARISON
# ─────────────────────────────────────────────────────────────

def plot_pvalue_comparison(
    results : Dict,
    save    : bool = True,
) -> None:
    """
    Bar chart of p-values for all model pairs.
    Red bars = significant. Blue bars = not significant.
    Red dashed line = α threshold.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, test_name, key in zip(
        axes,
        ['McNemar Test', 'Wilcoxon Signed-Rank Test'],
        ['mcnemar', 'wilcoxon']
    ):
        pairs   = [f"{r['model_a']}\nvs\n{r['model_b']}"
                   for r in results[key]]
        pvalues = [r['p_value'] for r in results[key]]
        colors  = ['#C44E52' if p < ALPHA else '#4C72B0'
                   for p in pvalues]

        bars = ax.bar(
            range(len(pairs)), pvalues,
            color=colors, edgecolor='white', linewidth=1.2
        )

        ax.axhline(
            y=ALPHA, color='red', linestyle='--',
            linewidth=2, label=f'α = {ALPHA}'
        )

        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pairs, fontsize=8)
        ax.set_ylabel('p-value', fontsize=11)
        ax.set_title(
            f'{test_name}\nRed = significant | Blue = not significant',
            fontsize=11, fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        max_p = max(pvalues) if pvalues else ALPHA
        ax.set_ylim(0, max(max_p * 1.3, ALPHA * 3))

        for bar, pv in zip(bars, pvalues):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f'p={pv:.3f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold'
            )

    plt.suptitle(
        'Statistical Significance — Model Comparisons\n'
        f'Significance level α = {ALPHA}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    if save:
        save_path = FIGURES_DIR / 'statistical_tests_pvalues.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ p-value plot saved → {save_path}")

    plt.show()
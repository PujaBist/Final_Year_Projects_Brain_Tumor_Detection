# ============================================================
# Brain Tumor MRI Classification Project
# File: src/duplicate_detection.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Detects duplicate and near-duplicate images
#              between Masoudnickparvar (train) and BRISC 2025
#              (external test) datasets using perceptual hashing.
#              MUST be run before any training begins.
# ============================================================

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import imagehash
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from tqdm import tqdm

from src.config import (
    TRAIN_DATA_DIR,
    EXTERNAL_TEST_DIR,
    DUPLICATE_DIR,
    FIGURES_DIR,
    CLASSES
)

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────

# Hash difference threshold
# 0  = exact duplicates only
# <5 = near-duplicates (recommended)
# <10 = very similar images
HASH_THRESHOLD = 5

# Supported image extensions
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


# ─────────────────────────────────────────────────────────────
# 1. COLLECT ALL IMAGE PATHS
# ─────────────────────────────────────────────────────────────

def get_image_paths(dataset_dir: Path, dataset_name: str) -> list:
    """
    Recursively collect all image paths from dataset directory.

    Args:
        dataset_dir:  root directory of the dataset
        dataset_name: name label for printing

    Returns:
        list of (image_path, class_name) tuples
    """
    if not dataset_dir.exists():
        print(f"⚠️  Directory not found: {dataset_dir}")
        print(f"   Please download the dataset first.")
        return []

    image_paths = []
    for class_name in CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            # Try uppercase or different folder names
            class_dir = dataset_dir / class_name.replace('_', ' ')
        if class_dir.exists():
            for img_file in class_dir.rglob('*'):
                if img_file.suffix.lower() in IMG_EXTENSIONS:
                    image_paths.append((img_file, class_name))

    print(f"✅ {dataset_name}: found {len(image_paths)} images")
    return image_paths


# ─────────────────────────────────────────────────────────────
# 2. COMPUTE PERCEPTUAL HASHES
# ─────────────────────────────────────────────────────────────

def compute_hashes(image_paths: list, dataset_name: str) -> dict:
    """
    Compute perceptual hash for every image.
    pHash is robust to small changes in brightness/contrast.

    Args:
        image_paths:  list of (path, class_name) tuples
        dataset_name: name for progress bar label

    Returns:
        dict of path_string → hash_value
    """
    hashes = {}
    failed = 0

    for img_path, class_name in tqdm(
        image_paths,
        desc=f"Hashing {dataset_name}",
        unit="img"
    ):
        try:
            img  = Image.open(img_path).convert('RGB')
            h    = imagehash.phash(img)
            hashes[str(img_path)] = {
                'hash'      : h,
                'class'     : class_name,
                'path'      : str(img_path),
            }
        except Exception as e:
            failed += 1

    if failed > 0:
        print(f"  ⚠️  Failed to hash {failed} images (corrupted files)")

    print(f"✅ {dataset_name}: computed {len(hashes)} hashes")
    return hashes


# ─────────────────────────────────────────────────────────────
# 3. FIND DUPLICATES
# ─────────────────────────────────────────────────────────────

def find_duplicates(
    train_hashes: dict,
    test_hashes: dict,
    threshold: int = HASH_THRESHOLD
) -> list:
    """
    Compare all test images against all train images.
    Flag any test image that is too similar to a train image.

    Args:
        train_hashes: hash dict for training set
        test_hashes:  hash dict for external test set
        threshold:    maximum hash difference to flag as duplicate

    Returns:
        list of duplicate pairs (train_path, test_path, difference)
    """
    print(f"\n🔍 Comparing datasets with threshold = {threshold}...")
    print(f"   Train images : {len(train_hashes)}")
    print(f"   Test images  : {len(test_hashes)}")
    print(f"   Total pairs  : {len(train_hashes) * len(test_hashes):,}")

    duplicates = []

    for test_path, test_info in tqdm(
        test_hashes.items(),
        desc="Detecting duplicates",
        unit="img"
    ):
        test_hash = test_info['hash']
        for train_path, train_info in train_hashes.items():
            train_hash = train_info['hash']
            diff = abs(test_hash - train_hash)
            if diff <= threshold:
                duplicates.append({
                    'train_path'      : train_path,
                    'train_class'     : train_info['class'],
                    'test_path'       : test_path,
                    'test_class'      : test_info['class'],
                    'hash_difference' : int(diff),
                })

    return duplicates


# ─────────────────────────────────────────────────────────────
# 4. SAVE DUPLICATE REPORT
# ─────────────────────────────────────────────────────────────

def save_duplicate_report(
    duplicates: list,
    train_total: int,
    test_total: int
) -> dict:
    """
    Save full duplicate report to JSON and print summary.

    Args:
        duplicates:  list of duplicate pairs from find_duplicates()
        train_total: total train images
        test_total:  total test images

    Returns:
        summary statistics dict
    """
    DUPLICATE_DIR.mkdir(parents=True, exist_ok=True)

    # Count unique duplicate test images
    duplicate_test_paths = set(d['test_path'] for d in duplicates)
    n_duplicates         = len(duplicate_test_paths)
    duplicate_pct        = (n_duplicates / test_total * 100) if test_total > 0 else 0

    # Class-wise breakdown
    class_counts = defaultdict(int)
    for d in duplicates:
        class_counts[d['test_class']] += 1

    summary = {
        'train_total'        : train_total,
        'test_total'         : test_total,
        'duplicates_found'   : n_duplicates,
        'duplicate_percent'  : round(duplicate_pct, 2),
        'threshold_used'     : HASH_THRESHOLD,
        'class_breakdown'    : dict(class_counts),
        'verdict'            : 'SAFE' if duplicate_pct < 10 else 'WARNING',
        'recommendation'     : (
            'Use BRISC 2025 as external test after removing duplicates'
            if duplicate_pct < 10 else
            'Use Mendeley dataset as external test instead'
        ),
    }

    # Save full duplicate list
    report = {
        'summary'    : summary,
        'duplicates' : duplicates,
    }
    report_path = DUPLICATE_DIR / 'duplicate_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    # Save list of test paths to remove
    remove_path = DUPLICATE_DIR / 'test_images_to_remove.txt'
    with open(remove_path, 'w') as f:
        for path in sorted(duplicate_test_paths):
            f.write(path + '\n')

    return summary


# ─────────────────────────────────────────────────────────────
# 5. PLOT DUPLICATE SUMMARY
# ─────────────────────────────────────────────────────────────

def plot_duplicate_summary(summary: dict) -> None:
    """
    Save a bar chart showing duplicate count per class.

    Args:
        summary: output from save_duplicate_report()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left — overall summary
    labels = ['Total Test\nImages', 'Duplicates\nFound', 'Clean\nImages']
    total  = summary['test_total']
    dups   = summary['duplicates_found']
    clean  = total - dups
    values = [total, dups, clean]
    colors = ['#4C72B0', '#C44E52', '#55A868']

    axes[0].bar(labels, values, color=colors, edgecolor='white', linewidth=1.2)
    for i, v in enumerate(values):
        axes[0].text(i, v + 10, str(v), ha='center',
                     fontweight='bold', fontsize=12)
    axes[0].set_title('Duplicate Detection — Overall', fontsize=13)
    axes[0].set_ylabel('Number of Images')
    axes[0].grid(axis='y', alpha=0.3)

    # Right — class breakdown
    class_data = summary['class_breakdown']
    if class_data:
        axes[1].bar(
            class_data.keys(),
            class_data.values(),
            color='#DD8452',
            edgecolor='white',
            linewidth=1.2
        )
        for i, (k, v) in enumerate(class_data.items()):
            axes[1].text(i, v + 0.5, str(v), ha='center',
                         fontweight='bold', fontsize=11)
        axes[1].set_title('Duplicates per Class', fontsize=13)
        axes[1].set_ylabel('Count')
        axes[1].grid(axis='y', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No duplicates found!',
                     ha='center', va='center',
                     fontsize=14, color='green',
                     transform=axes[1].transAxes)
        axes[1].set_title('Duplicates per Class', fontsize=13)

    plt.tight_layout()
    save_path = FIGURES_DIR / 'duplicate_detection_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Duplicate chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# 6. PRINT FINAL VERDICT
# ─────────────────────────────────────────────────────────────

def print_verdict(summary: dict) -> None:
    """Print final clean verdict with recommendation."""
    print("\n" + "=" * 55)
    print("  DUPLICATE DETECTION — FINAL REPORT")
    print("=" * 55)
    print(f"  Train images          : {summary['train_total']:,}")
    print(f"  Test images           : {summary['test_total']:,}")
    print(f"  Duplicates found      : {summary['duplicates_found']:,}")
    print(f"  Duplicate percentage  : {summary['duplicate_percent']:.2f}%")
    print(f"  Threshold used        : {summary['threshold_used']}")
    print(f"  Class breakdown       : {summary['class_breakdown']}")
    print("-" * 55)

    if summary['verdict'] == 'SAFE':
        print(f"  VERDICT  : ✅ SAFE")
        print(f"  ACTION   : Remove {summary['duplicates_found']} images")
        print(f"             from BRISC 2025 test set")
        print(f"             then proceed with training")
    else:
        print(f"  VERDICT  : ⚠️  WARNING — too many duplicates")
        print(f"  ACTION   : Switch to Mendeley 4-class dataset")
        print(f"             as your external test set instead")

    print(f"\n  Recommendation: {summary['recommendation']}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_duplicate_detection() -> dict:
    """
    Full duplicate detection pipeline.
    Run this ONCE before starting any training.

    Returns:
        summary statistics dict
    """
    print("\n" + "=" * 55)
    print("  DUPLICATE DETECTION PIPELINE")
    print("  Masoudnickparvar vs BRISC 2025")
    print("=" * 55 + "\n")

    # Step 1 — collect paths
    train_paths = get_image_paths(TRAIN_DATA_DIR, "Masoudnickparvar")
    test_paths  = get_image_paths(EXTERNAL_TEST_DIR, "BRISC 2025")

    if not train_paths or not test_paths:
        print("\n⚠️  One or both datasets not found.")
        print("   Please download datasets first:")
        print("   kaggle datasets download masoudnickparvar/brain-tumor-mri-dataset")
        print("   kaggle datasets download briscdataset/brisc2025")
        return {}

    # Step 2 — compute hashes
    print("\nComputing perceptual hashes...")
    train_hashes = compute_hashes(train_paths, "Masoudnickparvar")
    test_hashes  = compute_hashes(test_paths,  "BRISC 2025")

    # Step 3 — find duplicates
    duplicates = find_duplicates(train_hashes, test_hashes)

    # Step 4 — save report
    summary = save_duplicate_report(
        duplicates,
        train_total=len(train_paths),
        test_total=len(test_paths)
    )

    # Step 5 — plot
    plot_duplicate_summary(summary)

    # Step 6 — print verdict
    print_verdict(summary)

    return summary


# ─────────────────────────────────────────────────────────────
# 8. QUICK TEST (no dataset needed)
# ─────────────────────────────────────────────────────────────

def test_hashing() -> None:
    """
    Quick test to verify imagehash is working correctly.
    Uses synthetic images — no dataset needed.
    """
    print("\n🧪 Testing perceptual hashing...")

    # Create two identical images
    img1 = Image.new('RGB', (224, 224), color=(128, 64, 32))
    img2 = Image.new('RGB', (224, 224), color=(128, 64, 32))
    # Create one different image
    img3 = Image.new('RGB', (224, 224), color=(0, 0, 255))

    h1 = imagehash.phash(img1)
    h2 = imagehash.phash(img2)
    h3 = imagehash.phash(img3)

    print(f"  Same image difference   : {abs(h1 - h2)} (expect 0)")
    print(f"  Different image diff    : {abs(h1 - h3)} (expect >5)")

    assert abs(h1 - h2) == 0,  "FAIL: identical images should have diff=0"
    assert abs(h1 - h3) > 5,   "FAIL: different images should have diff>5"

    print("✅ Hashing test passed!\n")


if __name__ == '__main__':
    # Quick hash test first — no dataset needed
    test_hashing()

    # Full pipeline — only runs if datasets are downloaded
    summary = run_duplicate_detection()
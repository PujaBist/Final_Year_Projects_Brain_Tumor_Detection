# ============================================================
# Brain Tumor MRI Classification Project
# File: src/data_split.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Loads Masoudnickparvar dataset only.
#              Merges Training + Testing folders (7,023 images)
#              and creates stratified 70/15/15 split.
#              Split saved ONCE to JSON and reused for
#              ALL 3 models — ResNet50, DenseNet121,
#              EfficientNetB0 — for fair comparison.
#
#              BRISC 2025 (external test) is loaded separately
#              in data_loader.py — Phase 8 only.
# ============================================================

import json
from pathlib import Path
from typing import Tuple, List, Dict

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_DIR,
    DATA_SPLIT_FILE,
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    SEED,
)

# ─────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────

# Masoudnickparvar has exactly these two subfolders
DATASET_SUBFOLDERS = ['Training', 'Testing']

# Supported image formats
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Maps config class names → possible folder names in dataset
# Dataset uses 'notumor' (no underscore)
FOLDER_NAME_MAP = {
    'glioma'     : ['glioma',     'Glioma'],
    'meningioma' : ['meningioma', 'Meningioma'],
    'notumor'    : ['notumor',    'no_tumor',  'No Tumor'],
    'pituitary'  : ['pituitary',  'Pituitary'],
}


# ─────────────────────────────────────────────────────────────
# 1. COLLECT ALL IMAGE PATHS FROM MASOUDNICKPARVAR
# ─────────────────────────────────────────────────────────────

def collect_image_paths() -> Tuple[List[str], List[int]]:
    """
    Collect all images from Masoudnickparvar dataset.

    Merges both subfolders:
        Training/ → 5,712 images
        Testing/  → 1,311 images
        ─────────────────────────
        Total     → 7,023 images

    These are then split 70/15/15 in create_data_split().
    BRISC 2025 is NOT loaded here — see data_loader.py Phase 8.

    Returns:
        image_paths : list of absolute path strings
        labels      : list of integer class labels (parallel)
    """
    if not DATASET_DIR.exists():
        raise RuntimeError(
            f"\n❌ Dataset not found: {DATASET_DIR}\n"
            f"   Make sure Google Drive is mounted and\n"
            f"   Datasets/ folder exists at:\n"
            f"   {DATASET_DIR}"
        )

    image_paths  = []
    labels       = []
    class_totals = {cls: 0 for cls in CLASSES}

    print(f"\n📁 Loading Masoudnickparvar dataset")
    print(f"   Path     : {DATASET_DIR}")
    print(f"   Merging Training/ + Testing/ ...\n")

    for subfolder in DATASET_SUBFOLDERS:
        subfolder_dir   = DATASET_DIR / subfolder
        subfolder_count = 0

        if not subfolder_dir.exists():
            print(f"  ⚠️  {subfolder}/ not found — skipping")
            continue

        for class_name in CLASSES:
            # Try all folder name variations
            class_dir = None
            for possible_name in FOLDER_NAME_MAP[class_name]:
                candidate = subfolder_dir / possible_name
                if candidate.exists():
                    class_dir = candidate
                    break

            if class_dir is None:
                print(f"  ⚠️  {subfolder}/{class_name} not found")
                continue

            class_count = 0
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in IMG_EXTENSIONS:
                    image_paths.append(str(img_file))
                    labels.append(CLASS_TO_IDX[class_name])
                    class_count += 1

            class_totals[class_name] += class_count
            subfolder_count          += class_count

        print(f"  ✅ {subfolder:<12} → {subfolder_count:>5} images")

    # Per class summary
    print(f"\n  {'Class':<15} {'Images':>8}")
    print(f"  {'─'*25}")
    for cls, count in class_totals.items():
        print(f"  {cls:<15} {count:>8,}")
    print(f"  {'─'*25}")
    print(f"  {'TOTAL':<15} {len(image_paths):>8,}")

    return image_paths, labels


# ─────────────────────────────────────────────────────────────
# 2. CREATE AND SAVE 70/15/15 SPLIT
# ─────────────────────────────────────────────────────────────

def create_data_split(force_recreate: bool = False) -> Dict:
    """
    Create stratified 70/15/15 split from all 7,023 images.

    If split already exists → loads it directly.
    Use force_recreate=True to regenerate from scratch.

    CRITICAL: Split is created ONCE and reused for
    ResNet50, DenseNet121, AND EfficientNetB0 to ensure
    completely fair comparison between all 3 models.

    Args:
        force_recreate : True = regenerate even if exists

    Returns:
        dict with keys:
            train_paths, val_paths, test_paths
            train_labels, val_labels, test_labels
            total_images, seed
    """
    DATA_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Load existing split ──────────────────────────────────
    if DATA_SPLIT_FILE.exists() and not force_recreate:
        print(f"\n✅ Existing split found — loading:")
        print(f"   {DATA_SPLIT_FILE}\n")
        with open(DATA_SPLIT_FILE, 'r') as f:
            split = json.load(f)
        _print_split_summary(split)
        return split

    # ── Create new split ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("  CREATING DATA SPLIT — Masoudnickparvar")
    print(f"  Ratio : {int(TRAIN_RATIO*100)} / "
          f"{int(VAL_RATIO*100)} / "
          f"{int(TEST_RATIO*100)}  (train/val/test)")
    print(f"  Seed  : {SEED}")
    print("=" * 50)

    # Step 1 — collect all 7,023 images
    all_paths, all_labels = collect_image_paths()

    if len(all_paths) == 0:
        raise RuntimeError(
            "No images collected.\n"
            "Check DATASET_DIR in config.py."
        )

    # Step 2 — train 70% vs temp 30%
    train_paths, temp_paths, \
    train_labels, temp_labels = train_test_split(
        all_paths,
        all_labels,
        test_size    = round(VAL_RATIO + TEST_RATIO, 2),
        stratify     = all_labels,
        random_state = SEED
    )

    # Step 3 — val 15% vs test 15% from temp 30%
    val_paths, test_paths, \
    val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size    = 0.50,
        stratify     = temp_labels,
        random_state = SEED
    )

    split = {
        'train_paths'  : train_paths,
        'val_paths'    : val_paths,
        'test_paths'   : test_paths,
        'train_labels' : train_labels,
        'val_labels'   : val_labels,
        'test_labels'  : test_labels,
        'total_images' : len(all_paths),
        'seed'         : SEED,
        'dataset'      : 'Masoudnickparvar',
    }

    with open(DATA_SPLIT_FILE, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"\n✅ Split saved → {DATA_SPLIT_FILE}")
    _print_split_summary(split)

    return split


# ─────────────────────────────────────────────────────────────
# 3. PRINT SPLIT SUMMARY
# ─────────────────────────────────────────────────────────────

def _print_split_summary(split: Dict) -> None:
    """Print per-class image counts for each subset."""

    print("\n" + "=" * 62)
    print(f"  DATA SPLIT — {split.get('dataset','Masoudnickparvar')}"
          f"  (seed={split.get('seed', SEED)})")
    print("=" * 62)
    print(f"  {'Subset':<8} {'Total':>7}  "
          f"{'glioma':>9} {'menin':>8} "
          f"{'notumor':>9} {'pituit':>8}")
    print(f"  {'─'*59}")

    for subset in ['train', 'val', 'test']:
        lbl   = split[f'{subset}_labels']
        total = len(lbl)
        cnts  = [lbl.count(CLASS_TO_IDX[c]) for c in CLASSES]
        print(
            f"  {subset.upper():<8} {total:>7}  " +
            "  ".join(f"{c:>7}" for c in cnts)
        )

    total_all = sum(
        len(split[f'{s}_labels']) for s in ['train', 'val', 'test']
    )
    t = len(split['train_labels'])
    v = len(split['val_labels'])
    s = len(split['test_labels'])

    print(f"  {'─'*59}")
    print(f"  {'TOTAL':<8} {total_all:>7}")
    print(f"\n  Ratio : "
          f"{t/total_all*100:.1f}% / "
          f"{v/total_all*100:.1f}% / "
          f"{s/total_all*100:.1f}%  "
          f"(target 70/15/15)")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────
# 4. SANITY CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from src.utils import set_seed, create_directories

    set_seed()
    create_directories()

    print("\n🧪 Testing data_split.py...\n")

    try:
        split = create_data_split()

        total = (len(split['train_labels']) +
                 len(split['val_labels'])   +
                 len(split['test_labels']))

        assert total == split['total_images'], "Total mismatch!"
        assert split['seed']    == SEED,       "Seed mismatch!"

        print(f"\n  Dataset       : {split['dataset']}")
        print(f"  Total images  : {total:,}")
        print(f"  Train images  : {len(split['train_labels']):,}")
        print(f"  Val images    : {len(split['val_labels']):,}")
        print(f"  Test images   : {len(split['test_labels']):,}")
        print(f"  Split file    : {DATA_SPLIT_FILE}")

        print("\n✅ data_split.py — all checks passed!")

    except RuntimeError as e:
        print(f"\n⚠️  {e}")
        print("   Check config.py — DATASET_DIR must point to")
        print("   the folder containing Training/ and Testing/")
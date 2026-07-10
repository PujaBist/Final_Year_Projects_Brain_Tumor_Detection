# ============================================================
# Brain Tumor MRI Classification Project
# File: src/data_split.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Merges Training + Testing folders from
#              Masoudnickparvar dataset (7,023 images total)
#              and creates stratified 70/15/15 split.
#              Split saved ONCE to JSON and reused for
#              ALL 3 models — ResNet50, DenseNet121,
#              EfficientNetB0 — for fair comparison.
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

# Both subfolders in Masoudnickparvar dataset
DATASET_SUBFOLDERS = ['Training', 'Testing']

# Supported image formats
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Handle naming differences between dataset folders and config
# Dataset uses 'notumor' but config uses 'no_tumor'
FOLDER_NAME_MAP = {
    'glioma'     : ['glioma',    'Glioma'],
    'meningioma' : ['meningioma','Meningioma'],
    'no_tumor'   : ['notumor',   'no_tumor', 'No Tumor', 'noTumor'],
    'pituitary'  : ['pituitary', 'Pituitary'],
}


# ─────────────────────────────────────────────────────────────
# 1. COLLECT ALL IMAGE PATHS
# ─────────────────────────────────────────────────────────────

def collect_image_paths(
    dataset_dir: Path = DATASET_DIR
) -> Tuple[List[str], List[int]]:
    """
    Walk both Training/ and Testing/ subfolders and collect
    all image paths and their class labels.

    Merges:
        Training/ → 5,712 images
        Testing/  → 1,311 images
        Total     → 7,023 images

    Args:
        dataset_dir: root path of Masoudnickparvar dataset

    Returns:
        image_paths: list of absolute path strings
        labels:      list of integer class labels (parallel)
    """
    if not dataset_dir.exists():
        raise RuntimeError(
            f"\n❌ Dataset not found: {dataset_dir}\n"
            f"   Download with:\n"
            f"   kaggle datasets download "
            f"masoudnickparvar/brain-tumor-mri-dataset"
        )

    image_paths  = []
    labels       = []
    class_totals = {cls: 0 for cls in CLASSES}

    print(f"\n📁 Dataset root  : {dataset_dir}")
    print(f"   Subfolders     : {DATASET_SUBFOLDERS}")
    print(f"   Merging all images...\n")

    for subfolder in DATASET_SUBFOLDERS:
        subfolder_dir = dataset_dir / subfolder

        if not subfolder_dir.exists():
            print(f"  ⚠️  {subfolder}/ not found — skipping")
            continue

        subfolder_count = 0

        for class_name in CLASSES:
            # Try all possible folder name variations
            class_dir = None
            for possible_name in FOLDER_NAME_MAP[class_name]:
                candidate = subfolder_dir / possible_name
                if candidate.exists():
                    class_dir = candidate
                    break

            if class_dir is None:
                print(f"  ⚠️  {subfolder}/{class_name} not found")
                continue

            # Collect all images in this class folder
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
    Create stratified 70/15/15 train/val/test split from
    all 7,023 merged images. Saves to JSON file.

    If split already exists, loads it directly.
    Use force_recreate=True to regenerate.

    CRITICAL: This split is created ONCE and reused for
    ResNet50, DenseNet121, and EfficientNetB0 to ensure
    a completely fair comparison between all 3 models.

    Args:
        force_recreate: if True, regenerates even if exists

    Returns:
        dict with keys:
            train_paths, val_paths, test_paths  (lists of str)
            train_labels, val_labels, test_labels (lists of int)
            total_images, seed
    """
    DATA_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Load existing split ───────────────────────────────────
    if DATA_SPLIT_FILE.exists() and not force_recreate:
        print(f"\n✅ Existing split found — loading:")
        print(f"   {DATA_SPLIT_FILE}\n")
        with open(DATA_SPLIT_FILE, 'r') as f:
            split = json.load(f)
        _print_split_summary(split)
        return split

    # ── Create new split ──────────────────────────────────────
    print("\n" + "=" * 50)
    print("  CREATING DATA SPLIT")
    print(f"  Ratio  : {int(TRAIN_RATIO*100)} / "
          f"{int(VAL_RATIO*100)} / "
          f"{int(TEST_RATIO*100)}  (train/val/test)")
    print(f"  Seed   : {SEED}")
    print("=" * 50)

    # Step 1 — collect all images
    all_paths, all_labels = collect_image_paths()

    if len(all_paths) == 0:
        raise RuntimeError("No images found. Check dataset path.")

    # Step 2 — train (70%) vs temp (30%)
    train_paths, temp_paths, \
    train_labels, temp_labels = train_test_split(
        all_paths,
        all_labels,
        test_size    = round(VAL_RATIO + TEST_RATIO, 2),
        stratify     = all_labels,
        random_state = SEED
    )

    # Step 3 — val (15%) vs test (15%) from temp (30%)
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
    }

    # Save to disk
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

    print("\n" + "=" * 60)
    print(f"  DATA SPLIT SUMMARY  (seed = {split.get('seed', SEED)})")
    print("=" * 60)
    print(f"  {'Subset':<8} {'Total':>7}  "
          f"{'glioma':>8} {'menin':>7} "
          f"{'notumor':>9} {'pituit':>8}")
    print(f"  {'─'*57}")

    for subset in ['train', 'val', 'test']:
        lbl   = split[f'{subset}_labels']
        total = len(lbl)
        cnts  = [lbl.count(CLASS_TO_IDX[c]) for c in CLASSES]
        print(
            f"  {subset.upper():<8} {total:>7}  " +
            "  ".join(f"{c:>7}" for c in cnts)
        )

    total_all = sum(
        len(split[f'{s}_labels'])
        for s in ['train', 'val', 'test']
    )
    t = len(split['train_labels'])
    v = len(split['val_labels'])
    s = len(split['test_labels'])

    print(f"  {'─'*57}")
    print(f"  {'TOTAL':<8} {total_all:>7}")
    print(f"\n  Ratio  : "
          f"{t/total_all*100:.1f}% / "
          f"{v/total_all*100:.1f}% / "
          f"{s/total_all*100:.1f}%  "
          f"(target 70/15/15)")
    print("=" * 60)


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

        # Quick checks
        total = (len(split['train_labels']) +
                 len(split['val_labels'])   +
                 len(split['test_labels']))

        assert total == split['total_images'], \
            "Total mismatch after split!"

        assert split['seed'] == SEED, \
            "Seed mismatch!"

        print(f"\n  Total images  : {total:,}")
        print(f"  Train images  : {len(split['train_labels']):,}")
        print(f"  Val images    : {len(split['val_labels']):,}")
        print(f"  Test images   : {len(split['test_labels']):,}")
        print(f"  Split file    : {DATA_SPLIT_FILE}")

        print("\n✅ data_split.py — all checks passed!")

    except RuntimeError as e:
        print(f"\n⚠️  {e}")
        print("   Normal on laptop — datasets not downloaded yet.")
        print("   Run on Colab after downloading datasets.")
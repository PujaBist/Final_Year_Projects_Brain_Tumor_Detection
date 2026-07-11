# ============================================================
# Brain Tumor MRI Classification Project
# File: src/data_loader.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: PyTorch Dataset class and DataLoader creation.
#              Loads from pre-saved 70/15/15 split (data_split.py).
#              Also provides external test loader for BRISC 2025.
# ============================================================

from pathlib import Path
from typing import Tuple, Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.config import (
    EXTERNAL_TEST_DIR,
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    IMAGE_SIZE,
)

# Supported image formats
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

# Folder name variations for BRISC 2025
FOLDER_NAME_MAP = {
    'glioma'     : ['glioma',     'Glioma'],
    'meningioma' : ['meningioma', 'Meningioma'],
    'notumor'    : ['notumor',    'no_tumor', 'No Tumor'],
    'pituitary'  : ['pituitary',  'Pituitary'],
}


# ─────────────────────────────────────────────────────────────
# 1. PYTORCH DATASET CLASS
# ─────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    """
    PyTorch Dataset for brain tumor MRI images.

    Loads images from pre-saved split paths.
    Applies given transforms to each image.

    Args:
        image_paths : list of absolute image path strings
        labels      : list of integer class labels
        transform   : transforms to apply to each image
        subset      : 'train', 'val', or 'test' (for display)
    """

    def __init__(
        self,
        image_paths : List[str],
        labels      : List[int],
        transform   = None,
        subset      : str = 'train'
    ):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform
        self.subset      = subset

        assert len(image_paths) == len(labels), \
            "image_paths and labels must have equal length"

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load one image, apply transforms, return (image, label)."""
        img_path = self.image_paths[idx]
        label    = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Return black image if file is corrupted
            image = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_name(self, idx: int) -> str:
        """Return class name string for a given sample index."""
        return IDX_TO_CLASS[self.labels[idx]]

    def get_class_counts(self) -> Dict[str, int]:
        """Return count of images per class in this subset."""
        counts = {cls: 0 for cls in CLASSES}
        for label in self.labels:
            counts[IDX_TO_CLASS[label]] += 1
        return counts


# ─────────────────────────────────────────────────────────────
# 2. CREATE DATALOADERS FROM SPLIT
# ─────────────────────────────────────────────────────────────

def create_dataloaders(
    split      : Dict,
    transforms : Dict,
    batch_size : int = BATCH_SIZE
) -> Tuple[Dict, Dict]:
    """
    Create train / val / test DataLoaders from saved split.

    Imports split from data_split.py — create_data_split().
    Imports transforms from preprocessing.py — get_transforms().

    Args:
        split      : output of create_data_split()
        transforms : {'train': T, 'val': T, 'test': T}
        batch_size : images per batch

    Returns:
        dataloaders : dict with 'train', 'val', 'test' loaders
        datasets    : dict with 'train', 'val', 'test' datasets
    """
    datasets = {
        'train': BrainTumorDataset(
            split['train_paths'],
            split['train_labels'],
            transforms['train'],
            'train'
        ),
        'val': BrainTumorDataset(
            split['val_paths'],
            split['val_labels'],
            transforms['val'],
            'val'
        ),
        'test': BrainTumorDataset(
            split['test_paths'],
            split['test_labels'],
            transforms['test'],
            'test'
        ),
    }

    dataloaders = {
        'train': DataLoader(
            datasets['train'],
            batch_size  = batch_size,
            shuffle     = True,          # shuffle only train
            num_workers = NUM_WORKERS,
            pin_memory  = PIN_MEMORY,
            drop_last   = True,          # drop incomplete batch
        ),
        'val': DataLoader(
            datasets['val'],
            batch_size  = batch_size,
            shuffle     = False,         # never shuffle val/test
            num_workers = NUM_WORKERS,
            pin_memory  = PIN_MEMORY,
        ),
        'test': DataLoader(
            datasets['test'],
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = NUM_WORKERS,
            pin_memory  = PIN_MEMORY,
        ),
    }

    # Print summary
    print("\n" + "=" * 50)
    print("  DATALOADERS READY")
    print("=" * 50)
    print(f"  {'Subset':<8} {'Images':>8} {'Batches':>9}")
    print(f"  {'─'*28}")
    for name, dl in dataloaders.items():
        print(f"  {name:<8} "
              f"{len(dl.dataset):>8} "
              f"{len(dl):>9}")
    print(f"  {'─'*28}")
    print(f"  {'TOTAL':<8} "
          f"{sum(len(dl.dataset) for dl in dataloaders.values()):>8}")
    print("=" * 50)

    return dataloaders, datasets


# ─────────────────────────────────────────────────────────────
# 3. EXTERNAL TEST DATALOADER — BRISC 2025 (Phase 8)
# ─────────────────────────────────────────────────────────────

def create_external_test_loader(
    transform,
    batch_size : int = BATCH_SIZE
) -> Tuple[DataLoader, 'BrainTumorDataset']:
    """
    Load BRISC 2025 external test set.
    Used in Phase 8 — cross-dataset validation.

    Args:
        transform  : same test transform as internal test
        batch_size : images per batch

    Returns:
        ext_loader  : DataLoader for BRISC 2025
        ext_dataset : BrainTumorDataset for BRISC 2025
    """
    print(f"\n📁 Loading BRISC 2025: {EXTERNAL_TEST_DIR}")

    ext_paths  = []
    ext_labels = []

    for class_name in CLASSES:
        for possible_name in FOLDER_NAME_MAP[class_name]:
            class_dir = EXTERNAL_TEST_DIR / possible_name
            if class_dir.exists():
                for img_file in sorted(class_dir.iterdir()):
                    if img_file.suffix.lower() in IMG_EXTENSIONS:
                        ext_paths.append(str(img_file))
                        ext_labels.append(CLASS_TO_IDX[class_name])
                break

    if len(ext_paths) == 0:
        raise RuntimeError(
            f"\n❌ No images found in {EXTERNAL_TEST_DIR}\n"
            f"   Download BRISC 2025 dataset first."
        )

    ext_dataset = BrainTumorDataset(
        ext_paths, ext_labels, transform, 'external_test'
    )

    ext_loader = DataLoader(
        ext_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
    )

    print(f"✅ BRISC 2025 loaded: "
          f"{len(ext_dataset)} images | "
          f"{len(ext_loader)} batches")

    return ext_loader, ext_dataset


# ─────────────────────────────────────────────────────────────
# 4. SANITY CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import torchvision.transforms as T
    from src.utils import set_seed, create_directories
    from src.data_split import create_data_split

    set_seed()
    create_directories()

    print("\n🧪 Testing data_loader.py...\n")

    dummy_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    transforms = {
        'train': dummy_transform,
        'val'  : dummy_transform,
        'test' : dummy_transform,
    }

    try:
        # Load split from data_split.py
        split = create_data_split()

        # Create dataloaders
        dataloaders, datasets = create_dataloaders(
            split, transforms
        )

        # Check one batch
        images, labels = next(iter(dataloaders['train']))
        print(f"\n  Batch shape   : {images.shape}")
        print(f"  Label sample  : {labels[:8].tolist()}")
        print(f"  Class sample  : "
              f"{[IDX_TO_CLASS[l.item()] for l in labels[:8]]}")
        print(f"  Pixel range   : "
              f"[{images.min():.3f}, {images.max():.3f}]")

        # Class counts per split
        print("\n  Class distribution:")
        for subset, ds in datasets.items():
            counts = ds.get_class_counts()
            print(f"    {subset:<8}: {counts}")

        print("\n✅ data_loader.py — all checks passed!")

    except RuntimeError as e:
        print(f"\n⚠️  {e}")
        print("   Run on Colab after downloading datasets.")
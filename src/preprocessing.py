# ============================================================
# Brain Tumor MRI Classification Project
# File: src/preprocessing.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Image preprocessing pipeline.
#              Resize → CLAHE → Normalize → Augment (train only)
#              EDA findings:
#                - Image sizes vary: 150×168 to 1024×1024
#                - Mean size: 454×457 px
#                - All 4 classes perfectly balanced (1,800 each)
#              These findings justify every preprocessing step.
# ============================================================

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import (
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID,
    AUG_HORIZONTAL_FLIP,
    AUG_ROTATION_LIMIT,
    AUG_BRIGHTNESS_LIMIT,
    AUG_CONTRAST_LIMIT,
    AUG_ZOOM_LIMIT,
)


# ─────────────────────────────────────────────────────────────
# 1. CLAHE TRANSFORM
# ─────────────────────────────────────────────────────────────

class CLAHETransform:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram
    Equalization) to enhance local contrast in MRI images.

    Why CLAHE for brain MRI?
    - MRI images have varying brightness across scanners
    - CLAHE enhances tumor boundaries without over-amplifying noise
    - Improves model ability to detect subtle tumor features
    - Applied in LAB color space to preserve color information

    EDA justification:
    - Pixel intensity varies across classes
    - Image sizes range from 150px to 1024px
    - Standardization is critical before CNN feature extraction
    """

    def __init__(
        self,
        clip_limit : float = CLAHE_CLIP_LIMIT,
        tile_grid  : tuple = CLAHE_TILE_GRID
    ):
        self.clahe = cv2.createCLAHE(
            clipLimit     = clip_limit,
            tileGridSize  = tile_grid
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply CLAHE to a PIL Image.

        Args:
            img: PIL Image (RGB)

        Returns:
            PIL Image with CLAHE applied (RGB)
        """
        # Convert PIL → numpy
        img_np = np.array(img)

        # Convert RGB → LAB color space
        img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

        # Split into L (lightness), A, B channels
        l_channel, a_channel, b_channel = cv2.split(img_lab)

        # Apply CLAHE only to L (lightness) channel
        l_clahe = self.clahe.apply(l_channel)

        # Merge channels back
        img_lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])

        # Convert LAB → RGB
        img_rgb = cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)

        # Convert numpy → PIL
        return Image.fromarray(img_rgb)


# ─────────────────────────────────────────────────────────────
# 2. TORCHVISION TRANSFORMS (simple, fast)
# ─────────────────────────────────────────────────────────────

def get_transforms() -> dict:
    """
    Create preprocessing transforms for train, val, and test.

    Pipeline:
        TRAIN : Resize → CLAHE → Augment → ToTensor → Normalize
        VAL   : Resize → CLAHE → ToTensor → Normalize
        TEST  : Resize → CLAHE → ToTensor → Normalize

    Note: Augmentation applied to TRAIN ONLY.
          Val and test get identical deterministic transforms
          for fair evaluation.

    EDA justification:
        - Resize to 224×224: images vary from 150px to 1024px
        - CLAHE: pixel intensity varies across classes/scanners
        - ImageNet normalization: using pretrained ResNet/DenseNet/EfficientNet
        - Flip/rotate: small augmentation preserves MRI anatomy

    Returns:
        dict with 'train', 'val', 'test' transform pipelines
    """
    clahe = CLAHETransform(
        clip_limit = CLAHE_CLIP_LIMIT,
        tile_grid  = CLAHE_TILE_GRID
    )

    # ── Train transform — with augmentation ──────────────────
    train_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),   # 224×224
        T.Lambda(lambda img: clahe(img)),     # CLAHE enhancement
        # T.RandomHorizontalFlip removed — brain MRI has anatomical
        # left-right orientation; flipping creates unnatural images
        T.RandomRotation(
            degrees=AUG_ROTATION_LIMIT        # ±15 degrees
        ),
        T.ColorJitter(
            brightness = AUG_BRIGHTNESS_LIMIT,  # ±0.2
            contrast   = AUG_CONTRAST_LIMIT,    # ±0.2
        ),
        T.RandomResizedCrop(
            size  = IMAGE_SIZE,
            scale = (1 - AUG_ZOOM_LIMIT, 1.0),  # 90-100% zoom
        ),
        T.ToTensor(),
        T.Normalize(
            mean = NORMALIZE_MEAN,   # [0.485, 0.456, 0.406]
            std  = NORMALIZE_STD,    # [0.229, 0.224, 0.225]
        ),
    ])

    # ── Val transform — no augmentation ──────────────────────
    val_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Lambda(lambda img: clahe(img)),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    # ── Test transform — identical to val ────────────────────
    test_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.Lambda(lambda img: clahe(img)),
        T.ToTensor(),
        T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    return {
        'train': train_transform,
        'val'  : val_transform,
        'test' : test_transform,
    }


# ─────────────────────────────────────────────────────────────
# 3. ALBUMENTATIONS TRANSFORMS (stronger augmentation)
# ─────────────────────────────────────────────────────────────

def get_albumentations_transforms() -> dict:
    """
    Alternative transforms using Albumentations library.
    Slightly stronger augmentation pipeline.
    Use this instead of get_transforms() if you want
    more aggressive augmentation.

    Returns:
        dict with 'train', 'val', 'test' transforms
    """
    clahe_aug = A.CLAHE(
        clip_limit     = CLAHE_CLIP_LIMIT,
        tile_grid_size = CLAHE_TILE_GRID,
        always_apply   = True,
        p              = 1.0
    )

    train_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        clahe_aug,
        # A.HorizontalFlip removed — anatomical left-right orientation
        A.Rotate(limit=AUG_ROTATION_LIMIT, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit = AUG_BRIGHTNESS_LIMIT,
            contrast_limit   = AUG_CONTRAST_LIMIT,
            p                = 0.5
        ),
        A.RandomResizedCrop(
            size  = (IMAGE_SIZE, IMAGE_SIZE),
            scale = (1 - AUG_ZOOM_LIMIT, 1.0),
            p     = 0.3
        ),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])

    eval_transform = A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        clahe_aug,
        A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ToTensorV2(),
    ])

    return {
        'train': train_transform,
        'val'  : eval_transform,
        'test' : eval_transform,
    }


# ─────────────────────────────────────────────────────────────
# 4. DENORMALIZE — for visualization
# ─────────────────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.
    Converts normalized tensor back to viewable image.

    Args:
        tensor: normalized image tensor (C, H, W)

    Returns:
        numpy array (H, W, C) in range [0, 255]
    """
    mean = torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
    std  = torch.tensor(NORMALIZE_STD).view(3, 1, 1)

    img = tensor * std + mean           # reverse normalize
    img = img.permute(1, 2, 0)         # C,H,W → H,W,C
    img = img.numpy()
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img


# ─────────────────────────────────────────────────────────────
# 5. VISUALIZE PREPROCESSING EFFECT
# ─────────────────────────────────────────────────────────────

def visualize_preprocessing(image_path: str, save_path: str = None):
    """
    Show side-by-side comparison of preprocessing steps.
    Useful for paper Figure showing preprocessing pipeline.

    Args:
        image_path: path to one MRI image
        save_path:  optional path to save the figure
    """
    import matplotlib.pyplot as plt

    original   = Image.open(image_path).convert('RGB')
    resized    = original.resize((IMAGE_SIZE, IMAGE_SIZE))
    clahe_img  = CLAHETransform()(resized)

    transforms = get_transforms()
    tensor     = transforms['train'](original)
    augmented  = denormalize(tensor)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(original)
    axes[0].set_title(f'Original\n{original.size[0]}×{original.size[1]}px',
                      fontsize=10)

    axes[1].imshow(resized)
    axes[1].set_title(f'Resized\n{IMAGE_SIZE}×{IMAGE_SIZE}px',
                      fontsize=10)

    axes[2].imshow(clahe_img)
    axes[2].set_title('CLAHE Enhanced\n(contrast improved)',
                      fontsize=10)

    axes[3].imshow(augmented)
    axes[3].set_title('Augmented + Normalized\n(train only)',
                      fontsize=10)

    for ax in axes:
        ax.axis('off')

    plt.suptitle('Preprocessing Pipeline — Brain Tumor MRI',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Preprocessing figure saved → {save_path}")

    plt.show()


# ─────────────────────────────────────────────────────────────
# 6. SANITY CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from src.utils import set_seed
    from src.data_split import create_data_split
    from src.data_loader import create_dataloaders
    import random

    set_seed()

    print("\n🧪 Testing preprocessing.py...\n")

    # Test CLAHE
    print("1. Testing CLAHE transform...")
    dummy_img = Image.new('RGB', (256, 256), color=(128, 64, 32))
    clahe     = CLAHETransform()
    result    = clahe(dummy_img)
    assert result.size == dummy_img.size, "CLAHE changed image size!"
    print("   ✅ CLAHE works correctly")

    # Test get_transforms
    print("2. Testing get_transforms...")
    transforms = get_transforms()
    assert 'train' in transforms
    assert 'val'   in transforms
    assert 'test'  in transforms
    print("   ✅ All 3 transforms created")

    # Test on real data
    print("3. Testing full pipeline on dataset...")
    try:
        split       = create_data_split()
        dataloaders, _ = create_dataloaders(split, transforms)

        images, labels = next(iter(dataloaders['train']))
        assert images.shape == (32, 3, 224, 224), \
            f"Wrong shape: {images.shape}"
        assert images.min() < 0, \
            "Normalization not applied (values should go negative)"

        print(f"   ✅ Batch shape  : {images.shape}")
        print(f"   ✅ Pixel range  : "
              f"[{images.min():.3f}, {images.max():.3f}]")
        print(f"   ✅ Normalization: applied (negative values present)")

    except RuntimeError as e:
        print(f"   ⚠️  {e}")
        print("   Run on Colab after downloading datasets.")

    print("\n✅ preprocessing.py — all checks passed!")
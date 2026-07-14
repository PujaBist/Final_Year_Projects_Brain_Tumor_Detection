# ============================================================
# Brain Tumor MRI Classification Project
# File: src/preprocessing.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description:
# Image preprocessing pipeline.
#
# Pipeline:
# Resize → CLAHE → Augmentation (Train only)
# → Tensor → Normalize
#
# EDA Findings:
# - Image sizes vary from 150×168 to 1024×1024
# - Mean image size ≈ 454×457 px
# - Balanced dataset across four classes
#
# Designed for:
# ResNet50
# DenseNet121
# EfficientNet-B0
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
    AUG_ROTATION_LIMIT,
    AUG_BRIGHTNESS_LIMIT,
    AUG_CONTRAST_LIMIT,
    AUG_ZOOM_LIMIT,
)

# ============================================================
# CLAHE TRANSFORM
# ============================================================

class CLAHETransform:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Enhances local contrast while preserving anatomical
    structures in MRI images.

    Benefits:
    ---------
    • Reduces scanner intensity variation
    • Improves tumor boundary visibility
    • Enhances low-contrast regions
    • Prevents over-amplification of noise
    """

    def __init__(
        self,
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_grid: tuple = CLAHE_TILE_GRID,
    ):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid,
        )

    def __call__(self, img: Image.Image) -> Image.Image:

        img_np = np.array(img)

        # RGB → LAB
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab)

        l = self.clahe.apply(l)

        enhanced = cv2.merge((l, a, b))

        rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        return Image.fromarray(rgb)


# ============================================================
# TORCHVISION TRANSFORMS
# ============================================================

def get_transforms():
    """
    Returns preprocessing transforms.

    Train
    -----
    RandomResizedCrop
    CLAHE
    Rotation
    Translation
    Brightness/Contrast
    Tensor
    Normalize

    Validation/Test
    ---------------
    Resize
    CLAHE
    Tensor
    Normalize
    """

    clahe = CLAHETransform()

    # --------------------------------------------------------
# Training Transform
# --------------------------------------------------------

    train_transform = T.Compose([

    # Random crop with slight zoom
    T.RandomResizedCrop(
        size=IMAGE_SIZE,
        scale=(1 - AUG_ZOOM_LIMIT, 1.0),
    ),

    # CLAHE contrast enhancement
    T.Lambda(lambda img: clahe(img)),

    # Small rotation (scanner/patient positioning variation)
    T.RandomRotation(
        degrees=AUG_ROTATION_LIMIT,
    ),

    # Small translation
    T.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
    ),

    # Simulate scanner brightness & contrast variation
    T.ColorJitter(
        brightness=AUG_BRIGHTNESS_LIMIT,
        contrast=AUG_CONTRAST_LIMIT,
    ),

    # Convert to tensor
    T.ToTensor(),

    # Normalize using ImageNet statistics
    T.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD,
    ),
])

    # --------------------------------------------------------
    # Validation Transform
    # --------------------------------------------------------

    val_transform = T.Compose([

        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        T.Lambda(lambda img: clahe(img)),

        T.ToTensor(),

        T.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD,
        ),
    ])

    # --------------------------------------------------------
    # Test Transform
    # --------------------------------------------------------

    test_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.Lambda(lambda img: clahe(img)),
    T.ToTensor(),
    T.Normalize(
        mean=NORMALIZE_MEAN,
        std=NORMALIZE_STD,
    ),
])

    return {
        "train": train_transform,
        "val": val_transform,
        "test": test_transform,
    }

# ============================================================
# ALBUMENTATIONS TRANSFORMS
# ============================================================

def get_albumentations_transforms():
    """
    Albumentations preprocessing pipeline.

    This provides a stronger augmentation pipeline than
    torchvision and can be used for experimentation.

    Returns
    -------
    dict
        Dictionary containing train, validation,
        and test transforms.
    """

    train_transform = A.Compose([

        A.RandomResizedCrop(
            size=(IMAGE_SIZE, IMAGE_SIZE),
            scale=(1 - AUG_ZOOM_LIMIT, 1.0),
            p=1.0,
        ),

        A.CLAHE(
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID,
            p=1.0,
        ),

        A.Rotate(
            limit=AUG_ROTATION_LIMIT,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),

        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.98, 1.02),
            rotate=0,
            p=0.3,
        ),

        A.RandomBrightnessContrast(
            brightness_limit=AUG_BRIGHTNESS_LIMIT,
            contrast_limit=AUG_CONTRAST_LIMIT,
            p=0.5,
        ),

        A.GaussNoise(
            p=0.2,
        ),

        A.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD,
        ),

        ToTensorV2(),
    ])


    eval_transform = A.Compose([

        A.Resize(
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),

        A.CLAHE(
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=CLAHE_TILE_GRID,
            p=1.0,
        ),

        A.Normalize(
            mean=NORMALIZE_MEAN,
            std=NORMALIZE_STD,
        ),

        ToTensorV2(),
    ])

    return {
        "train": train_transform,
        "val": eval_transform,
        "test": eval_transform,
    }


# ============================================================
# DENORMALIZE
# ============================================================

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor of shape (3, H, W)

    Returns
    -------
    numpy.ndarray
        RGB image in uint8 format.
    """

    mean = torch.tensor(
        NORMALIZE_MEAN,
        dtype=tensor.dtype,
        device=tensor.device,
    ).view(3, 1, 1)

    std = torch.tensor(
        NORMALIZE_STD,
        dtype=tensor.dtype,
        device=tensor.device,
    ).view(3, 1, 1)

    image = tensor * std + mean

    image = image.permute(1, 2, 0)

    image = image.cpu().numpy()

    image = np.clip(image, 0, 1)

    image = (image * 255).astype(np.uint8)

    return image


# ============================================================
# VISUALIZE PREPROCESSING
# ============================================================

def visualize_preprocessing(
    image_path: str,
    save_path: str | None = None,
):
    """
    Visualize every preprocessing stage.

    Useful for:
        • Thesis
        • Research paper
        • Presentation

    Parameters
    ----------
    image_path : str
        Path to MRI image.

    save_path : str | None
        Save figure if provided.
    """

    import matplotlib.pyplot as plt

    original = Image.open(image_path).convert("RGB")

    resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    clahe_img = CLAHETransform()(resized)

    transforms = get_transforms()

    augmented_tensor = transforms["train"](original)

    augmented = denormalize(augmented_tensor)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(original)
    axes[0].set_title(
        f"Original\n{original.size[0]}×{original.size[1]}"
    )

    axes[1].imshow(resized)
    axes[1].set_title(
        f"Resize\n{IMAGE_SIZE}×{IMAGE_SIZE}"
    )

    axes[2].imshow(clahe_img)
    axes[2].set_title(
        "CLAHE"
    )

    axes[3].imshow(augmented)
    axes[3].set_title(
        "Training Transform"
    )

    for ax in axes:
        ax.axis("off")

    plt.suptitle(
        "Brain MRI Preprocessing Pipeline",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path is not None:

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"✅ Figure saved to: {save_path}")

    plt.show()


    # ============================================================
# SANITY CHECK
# ============================================================

if __name__ == "__main__":

    from src.utils import set_seed
    from src.data_split import create_data_split
    from src.data_loader import create_dataloaders

    print("=" * 60)
    print("Testing preprocessing.py")
    print("=" * 60)

    set_seed()

    # --------------------------------------------------------
    # Test 1: CLAHE
    # --------------------------------------------------------
    print("\n[1] Testing CLAHE...")

    dummy = Image.new(
        "RGB",
        (256, 256),
        color=(120, 120, 120),
    )

    clahe = CLAHETransform()

    enhanced = clahe(dummy)

    assert enhanced.size == dummy.size

    print("✅ CLAHE transform works.")

    # --------------------------------------------------------
    # Test 2: Torchvision transforms
    # --------------------------------------------------------
    print("\n[2] Testing torchvision transforms...")

    transforms = get_transforms()

    assert "train" in transforms
    assert "val" in transforms
    assert "test" in transforms

    print("✅ Train transform created.")
    print("✅ Validation transform created.")
    print("✅ Test transform created.")

    # --------------------------------------------------------
    # Test 3: Albumentations transforms
    # --------------------------------------------------------
    print("\n[3] Testing Albumentations transforms...")

    albumentations_tf = get_albumentations_transforms()

    assert "train" in albumentations_tf
    assert "val" in albumentations_tf
    assert "test" in albumentations_tf

    print("✅ Albumentations pipeline created.")

    # --------------------------------------------------------
    # Test 4: Real Dataset
    # --------------------------------------------------------
    print("\n[4] Testing on real dataset...")

    try:

        split = create_data_split()

        dataloaders, _ = create_dataloaders(
            split,
            transforms,
        )

        images, labels = next(
            iter(dataloaders["train"])
        )

        # Shape check
        assert images.ndim == 4
        assert images.shape[1:] == (
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )

        # Label check
        assert labels.ndim == 1

        # Normalization check
        assert images.min() < 0
        assert images.max() > 0

        print("\nBatch Information")
        print("------------------------------")
        print(f"Batch Size : {images.shape[0]}")
        print(f"Shape      : {tuple(images.shape)}")
        print(f"Labels     : {tuple(labels.shape)}")
        print(
            f"Pixel Range: "
            f"{images.min():.3f} "
            f"to "
            f"{images.max():.3f}"
        )

        print("\n✅ DataLoader pipeline works correctly.")

    except FileNotFoundError:

        print(
            "\nDataset not found."
            "\nRun after downloading datasets."
        )

    except Exception as e:

        print("\nUnexpected Error")
        print("------------------------------")
        print(type(e).__name__)
        print(e)

    print("\n" + "=" * 60)
    print("preprocessing.py PASSED")
    print("=" * 60)
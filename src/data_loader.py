"""
============================================================
Brain Tumor MRI Classification
File: src/data_loader.py
Author: Puja Bist

Description:
Loads datasets, applies preprocessing transforms,
and creates PyTorch DataLoaders.
============================================================
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.config import (
    TRAIN_DATA_DIR,
    TEST_DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)
# ============================================================
# Image Transforms
# ============================================================
# Why do we need transforms?

# Your MRI images are not all guaranteed to be the same size. Deep learning models like ResNet, DenseNet, and EfficientNet expect images of a fixed size (224×224 for your project).

# Transforms will:

# Resize the images.
# Convert them to PyTorch tensors.

def get_transforms():
    """
    Create image transforms for training and testing datasets.

    Returns:
        tuple:
            train_transform (torchvision.transforms.Compose)
            test_transform (torchvision.transforms.Compose)
    """

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    return train_transform, test_transform



# Pillow (PIL) is the library used to open and manipulate image files.
# ImageFolder already uses PIL internally, so you don't need to import Image in data_loader.py.
# transforms.ToTensor() converts the PIL image into a PyTorch tensor, which is the format your deep learning models require




# ============================================================
# Load Datasets
# ============================================================

def load_datasets():
    """
    Load training and testing datasets.

    Returns:
        tuple:
            train_dataset (ImageFolder)
            test_dataset (ImageFolder)
    """

    # Get image transforms
    train_transform, test_transform = get_transforms()

    # Load training dataset
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DATA_DIR,
        transform=train_transform
    )

    # Load testing dataset
    test_dataset = datasets.ImageFolder(
        root=TEST_DATA_DIR,
        transform=test_transform
    )

    return train_dataset, test_dataset


def show_dataset_info(train_dataset, test_dataset):
    """
    Display basic information about the datasets.
    """

    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)

    print(f"Training Images : {len(train_dataset)}")
    print(f"Testing Images  : {len(test_dataset)}")

    print(f"\nClasses : {train_dataset.classes}")

    print(f"\nClass Mapping :")
    print(train_dataset.class_to_idx)
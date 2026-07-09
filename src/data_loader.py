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
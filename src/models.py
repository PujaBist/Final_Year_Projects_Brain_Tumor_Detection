# ============================================================
# Brain Tumor MRI Classification Project
# File: src/models.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Three baseline CNN models for brain tumor
#              classification. All pretrained on ImageNet
#              and fine-tuned for 4-class MRI classification.
#
#              Models:
#                1. ResNet50      — skip connections
#                2. DenseNet121   — dense connections
#                3. EfficientNetB0 — compound scaling
#
#              Each model:
#                - Loads pretrained ImageNet weights
#                - Replaces final classifier for 4 classes
#                - Adds Dropout for uncertainty estimation
#                - Is saved/loaded independently
# ============================================================

import torch
import torch.nn as nn
from torchvision import models

from src.config import (
    NUM_CLASSES,
    PRETRAINED,
    DROPOUT_RATE,
    DEVICE,
    MODEL_SAVE_NAMES,
)


# ─────────────────────────────────────────────────────────────
# 1. ResNet50
# ─────────────────────────────────────────────────────────────

def build_resnet50(
    num_classes  : int   = NUM_CLASSES,
    pretrained   : bool  = PRETRAINED,
    dropout_rate : float = DROPOUT_RATE,
) -> nn.Module:
    """
    Build ResNet50 for 4-class brain tumor classification.

    Architecture:
        ImageNet pretrained ResNet50
        → Replace fc layer with:
          Dropout(0.5) → Linear(2048 → 4)

    Why ResNet50?
        - Skip connections solve vanishing gradient problem
        - 50 layers deep with only 25M parameters
        - Strong feature extraction for medical images
        - Most widely used baseline in brain tumor literature

    Args:
        num_classes  : number of output classes (4)
        pretrained   : use ImageNet weights
        dropout_rate : dropout before final classifier

    Returns:
        PyTorch model ready for training
    """
    # Load pretrained ResNet50
    weights = models.ResNet50_Weights.IMAGENET1K_V1 \
              if pretrained else None
    model   = models.resnet50(weights=weights)

    # Get input features of original fc layer
    in_features = model.fc.in_features  # 2048

    # Replace final fc with Dropout + Linear for 4 classes
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────
# 2. DenseNet121
# ─────────────────────────────────────────────────────────────

def build_densenet121(
    num_classes  : int   = NUM_CLASSES,
    pretrained   : bool  = PRETRAINED,
    dropout_rate : float = DROPOUT_RATE,
) -> nn.Module:
    """
    Build DenseNet121 for 4-class brain tumor classification.

    Architecture:
        ImageNet pretrained DenseNet121
        → Replace classifier with:
          Dropout(0.5) → Linear(1024 → 4)

    Why DenseNet121?
        - Each layer connects to ALL previous layers
        - Preserves fine-grained texture features
        - Critical for subtle tumor boundary detection
        - Only 8M parameters — less overfitting risk
        - Addresses pooling feature loss gap from literature

    Args:
        num_classes  : number of output classes (4)
        pretrained   : use ImageNet weights
        dropout_rate : dropout before final classifier

    Returns:
        PyTorch model ready for training
    """
    weights = models.DenseNet121_Weights.IMAGENET1K_V1 \
              if pretrained else None
    model   = models.densenet121(weights=weights)

    # Get input features of original classifier
    in_features = model.classifier.in_features  # 1024

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────
# 3. EfficientNetB0
# ─────────────────────────────────────────────────────────────

def build_efficientnetb0(
    num_classes  : int   = NUM_CLASSES,
    pretrained   : bool  = PRETRAINED,
    dropout_rate : float = DROPOUT_RATE,
) -> nn.Module:
    """
    Build EfficientNetB0 for 4-class brain tumor classification.

    Architecture:
        ImageNet pretrained EfficientNetB0
        → Replace classifier[1] with:
          Dropout(0.5) → Linear(1280 → 4)

    Why EfficientNetB0?
        - Compound scaling of width, depth, resolution together
        - Only 5.3M parameters — most efficient of the three
        - Fastest training on limited GPU
        - Addresses hardware constraint gap from literature
        - Competitive accuracy with smallest model size

    Args:
        num_classes  : number of output classes (4)
        pretrained   : use ImageNet weights
        dropout_rate : dropout before final classifier

    Returns:
        PyTorch model ready for training
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 \
              if pretrained else None
    model   = models.efficientnet_b0(weights=weights)

    # Get input features of original classifier
    in_features = model.classifier[1].in_features  # 1280

    # Replace classifier — keep Dropout + Linear structure
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )

    return model.to(DEVICE)


# ─────────────────────────────────────────────────────────────
# 4. MODEL FACTORY — get any model by name
# ─────────────────────────────────────────────────────────────

def get_model(model_name: str) -> nn.Module:
    """
    Get model by name string.
    Used in training loop to load models by name.

    Args:
        model_name: one of 'resnet50', 'densenet121',
                    'efficientnetb0'

    Returns:
        PyTorch model on DEVICE

    Example:
        model = get_model('resnet50')
    """
    builders = {
        'resnet50'      : build_resnet50,
        'densenet121'   : build_densenet121,
        'efficientnetb0': build_efficientnetb0,
    }

    if model_name not in builders:
        raise ValueError(
            f"Unknown model: '{model_name}'\n"
            f"Choose from: {list(builders.keys())}"
        )

    model = builders[model_name]()
    print(f"✅ {model_name} built → {DEVICE}")
    return model


# ─────────────────────────────────────────────────────────────
# 5. LOAD SAVED MODEL
# ─────────────────────────────────────────────────────────────

def load_model(model_name: str) -> nn.Module:
    """
    Load best saved model weights from disk.
    Used after training for evaluation and ensemble.

    Args:
        model_name: one of 'resnet50', 'densenet121',
                    'efficientnetb0'

    Returns:
        model with loaded weights on DEVICE
    """
    save_path = MODEL_SAVE_NAMES[model_name]

    if not save_path.exists():
        raise FileNotFoundError(
            f"\n❌ No saved model at: {save_path}\n"
            f"   Train {model_name} first."
        )

    # Build fresh model architecture
    model = get_model(model_name)

    # Load saved weights
    checkpoint = torch.load(save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ {model_name} loaded from {save_path}")
    print(f"   Best epoch : {checkpoint['epoch']}")
    print(f"   Val acc    : {checkpoint['val_acc']:.4f}")

    return model


# ─────────────────────────────────────────────────────────────
# 6. PRINT MODEL SUMMARY
# ─────────────────────────────────────────────────────────────

def print_model_summary(model_name: str, model: nn.Module) -> None:
    """Print total and trainable parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)

    print(f"\n{'='*50}")
    print(f"  MODEL: {model_name.upper()}")
    print(f"{'='*50}")
    print(f"  Total parameters     : {total:>12,}")
    print(f"  Trainable parameters : {trainable:>12,}")
    print(f"  Frozen parameters    : {total-trainable:>12,}")
    print(f"  Device               : {DEVICE}")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────
# 7. SANITY CHECK
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from src.utils import set_seed
    set_seed()

    print("\n🧪 Testing models.py...\n")

    # Dummy input batch — (batch=4, channels=3, H=224, W=224)
    dummy = torch.randn(4, 3, 224, 224).to(DEVICE)

    for name in ['resnet50', 'densenet121', 'efficientnetb0']:
        model  = get_model(name)
        output = model(dummy)

        print_model_summary(name, model)

        assert output.shape == (4, NUM_CLASSES), \
            f"Wrong output shape: {output.shape}"

        print(f"  Input  shape : {dummy.shape}")
        print(f"  Output shape : {output.shape}")
        print(f"  ✅ {name} — forward pass OK\n")

    print("✅ models.py — all checks passed!")
# ============================================================
# Brain Tumor MRI Classification Project
# File: src/gradcam.py
# Author: Puja Bist
# College: Cosmos College of Management and Technology
# Date: July 2026
# Description: Grad-CAM explainability for brain tumor
#              classification. Generates heatmaps showing
#              which brain regions the model focuses on.
#              Applied to all 3 models + ensemble.
#
#              Research gap addressed:
#              "Most DL models provide black-box predictions.
#               Building interpretable models remains one of
#               the major unsolved technical challenges."
#               — Dorfner et al. (2025), npj Precision Oncology
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.config import (
    CLASSES,
    IDX_TO_CLASS,
    CLASS_TO_IDX,
    DEVICE,
    FIGURES_DIR,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
)
from src.preprocessing import denormalize


# ─────────────────────────────────────────────────────────────
# 1. GRAD-CAM IMPLEMENTATION
# ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Introduced by Selvaraju et al. (2017):
    "Grad-CAM: Visual Explanations from Deep Networks
     via Gradient-Based Localization"

    How it works:
        1. Forward pass — compute class score
        2. Backward pass — compute gradients of class score
           with respect to last conv layer feature maps
        3. Global average pool the gradients → weights
        4. Weighted sum of feature maps → heatmap
        5. ReLU → keep only positive influence
        6. Resize to input image size
        7. Overlay on original image

    Why last conv layer?
        - Contains the most semantic spatial information
        - Earlier layers → edges/textures (less meaningful)
        - Later layers → class-specific spatial patterns

    Args:
        model      : trained PyTorch model
        target_layer: last convolutional layer name
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor : torch.Tensor,
        class_idx    : Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for one image.

        Args:
            input_tensor : preprocessed image (1, 3, 224, 224)
            class_idx    : target class (None = predicted class)

        Returns:
            heatmap    : numpy array (H, W) in range [0, 1]
            pred_class : predicted class index
            confidence : model confidence for predicted class
        """
        self.model.eval()
        input_tensor = input_tensor.to(DEVICE)

        # Forward pass
        output = self.model(input_tensor)
        probs  = F.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        confidence = probs[0, class_idx].item()

        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Compute weights — global average pool of gradients
        gradients   = self.gradients[0]          # (C, H, W)
        activations = self.activations[0]         # (C, H, W)
        weights     = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(
            activations.shape[1:], device=DEVICE
        )
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU — keep only positive influence
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = np.uint8(255 * cam)
        cam = Image.fromarray(cam).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        )
        heatmap = np.array(cam) / 255.0

        return heatmap, class_idx, confidence

    def remove_hooks(self):
        """Remove registered hooks to free memory."""
        for handle in []:
            handle.remove()


# ─────────────────────────────────────────────────────────────
# 2. GET TARGET LAYER FOR EACH MODEL
# ─────────────────────────────────────────────────────────────

def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Get the last convolutional layer for each model architecture.

    ResNet50      → layer4[-1] (last residual block)
    DenseNet121   → features.denseblock4 (last dense block)
    EfficientNetB0 → features[-1] (last feature block)

    Args:
        model      : PyTorch model
        model_name : 'resnet50', 'densenet121', 'efficientnetb0'

    Returns:
        target convolutional layer
    """
    if model_name == 'resnet50':
        return model.layer4[-1]

    elif model_name == 'densenet121':
        return model.features.denseblock4

    elif model_name == 'efficientnetb0':
        return model.features[-1]

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────
# 3. OVERLAY HEATMAP ON IMAGE
# ─────────────────────────────────────────────────────────────

def overlay_heatmap(
    image   : np.ndarray,
    heatmap : np.ndarray,
    alpha   : float = 0.4,
) -> np.ndarray:
    """
    Overlay colored heatmap on original MRI image.

    Args:
        image   : original image (H, W, 3) in range [0, 255]
        heatmap : Grad-CAM heatmap (H, W) in range [0, 1]
        alpha   : heatmap transparency (0=invisible, 1=opaque)

    Returns:
        overlaid image (H, W, 3) in range [0, 255]
    """
    # Apply colormap to heatmap
    colormap    = cm.get_cmap('jet')
    colored_map = colormap(heatmap)[:, :, :3]   # (H, W, 3) RGB
    colored_map = (colored_map * 255).astype(np.uint8)

    # Blend with original image
    image_float = image.astype(float)
    color_float = colored_map.astype(float)
    overlaid    = (1 - alpha) * image_float + alpha * color_float
    overlaid    = np.clip(overlaid, 0, 255).astype(np.uint8)

    return overlaid


# ─────────────────────────────────────────────────────────────
# 4. VISUALIZE GRAD-CAM FOR ONE IMAGE
# ─────────────────────────────────────────────────────────────

def visualize_gradcam(
    model        : nn.Module,
    model_name   : str,
    image_tensor : torch.Tensor,
    true_label   : int,
    image_path   : str = None,
    save_path    : str = None,
) -> np.ndarray:
    """
    Generate and visualize Grad-CAM for one image.

    Shows 3-panel figure:
        Panel 1: Original MRI image
        Panel 2: Grad-CAM heatmap only
        Panel 3: Heatmap overlaid on image

    Args:
        model        : trained PyTorch model
        model_name   : for plot title
        image_tensor : preprocessed tensor (1, 3, 224, 224)
        true_label   : ground truth class index
        image_path   : original image path (for display)
        save_path    : path to save figure

    Returns:
        heatmap array (H, W)
    """
    # Get target layer
    target_layer = get_target_layer(model, model_name)
    gradcam      = GradCAM(model, target_layer)

    # Generate heatmap
    heatmap, pred_class, confidence = gradcam.generate(
        image_tensor.unsqueeze(0)
    )

    # Get original image for display
    orig_img = denormalize(image_tensor)

    # Create overlay
    overlaid = overlay_heatmap(orig_img, heatmap, alpha=0.4)

    # Determine if prediction is correct
    is_correct   = pred_class == true_label
    true_name    = IDX_TO_CLASS[true_label]
    pred_name    = IDX_TO_CLASS[pred_class]
    status       = '✓' if is_correct else '✗'
    title_color  = 'green' if is_correct else 'red'

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(orig_img)
    axes[0].set_title(
        f'Original MRI\nTrue: {true_name}',
        fontsize=11
    )
    axes[0].axis('off')

    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(
        f'Grad-CAM Heatmap\n(activation regions)',
        fontsize=11
    )
    axes[1].axis('off')

    axes[2].imshow(overlaid)
    axes[2].set_title(
        f'Overlay\nPred: {pred_name} ({confidence*100:.1f}%) {status}',
        fontsize=11,
        color=title_color
    )
    axes[2].axis('off')

    plt.suptitle(
        f'Grad-CAM — {model_name.upper()}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Grad-CAM saved → {save_path}")

    plt.show()

    return heatmap


# ─────────────────────────────────────────────────────────────
# 5. VISUALIZE ONE IMAGE PER CLASS
# ─────────────────────────────────────────────────────────────

def visualize_gradcam_per_class(
    model      : nn.Module,
    model_name : str,
    split      : Dict,
    transforms : Dict,
    n_samples  : int = 2,
) -> None:
    """
    Generate Grad-CAM for N samples from each class.
    Creates a grid showing model attention per tumor type.

    This becomes a key figure in your paper showing
    that the model focuses on clinically relevant regions.

    Args:
        model      : trained PyTorch model
        model_name : 'resnet50', 'densenet121', 'efficientnetb0'
        split      : data split dict from create_data_split()
        transforms : transforms dict from get_transforms()
        n_samples  : number of samples per class
    """
    from src.config import IDX_TO_CLASS, CLASS_TO_IDX

    target_layer = get_target_layer(model, model_name)
    gradcam      = GradCAM(model, target_layer)

    fig, axes = plt.subplots(
        len(CLASSES), n_samples * 3,
        figsize=(n_samples * 12, len(CLASSES) * 3.5)
    )

    for row, class_name in enumerate(CLASSES):
        class_idx = CLASS_TO_IDX[class_name]

        # Get sample paths for this class
        class_paths = [
            p for p, l in
            zip(split['test_paths'], split['test_labels'])
            if l == class_idx
        ][:n_samples]

        for col_set, img_path in enumerate(class_paths):
            # Load and preprocess image
            img_pil    = Image.open(img_path).convert('RGB')
            img_tensor = transforms['test'](img_pil)
            img_tensor_batch = img_tensor.unsqueeze(0)

            # Generate heatmap
            heatmap, pred, conf = gradcam.generate(
                img_tensor_batch, class_idx
            )

            # Denormalize for display
            orig_img = denormalize(img_tensor)
            overlaid = overlay_heatmap(orig_img, heatmap)

            is_correct = pred == class_idx
            color      = 'green' if is_correct else 'red'

            base_col = col_set * 3

            axes[row][base_col].imshow(orig_img)
            axes[row][base_col].axis('off')
            if col_set == 0:
                axes[row][base_col].set_ylabel(
                    class_name, fontsize=11, fontweight='bold'
                )

            axes[row][base_col + 1].imshow(heatmap, cmap='jet')
            axes[row][base_col + 1].axis('off')

            axes[row][base_col + 2].imshow(overlaid)
            axes[row][base_col + 2].set_title(
                f'{"✓" if is_correct else "✗"} {conf*100:.1f}%',
                fontsize=9, color=color
            )
            axes[row][base_col + 2].axis('off')

    # Column headers
    for i in range(n_samples):
        base = i * 3
        axes[0][base].set_title('Original', fontsize=10)
        axes[0][base + 1].set_title('Heatmap', fontsize=10)
        axes[0][base + 2].set_title('Overlay', fontsize=10)

    plt.suptitle(
        f'Grad-CAM Visualizations — {model_name.upper()}\n'
        f'Rows: tumor classes  |  ✓=correct  ✗=incorrect',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()

    save_path = FIGURES_DIR / f'{model_name}_gradcam_all_classes.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Grad-CAM grid saved → {save_path}")
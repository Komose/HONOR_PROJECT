#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grad_cam.py

Grad-CAM implementation for CheXzero CLIP-based foundation models.
Generates attention heatmaps highlighting lesion regions in chest X-rays.

References:
- Selvaraju et al. (2017): Grad-CAM: Visual Explanations from Deep Networks
- Adapted for CLIP Vision Transformer architectures

Author: Generated for HONER_PROJECT
"""
from __future__ import annotations

from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradCAM:
    """
    Grad-CAM for CLIP-style vision models (both ResNet and ViT architectures).

    For Vision Transformer (ViT):
        - Hooks into the last attention block or the output of the visual encoder
        - Generates spatial attention maps from patch-wise features

    For Modified ResNet:
        - Hooks into layer4 (final conv layer before attention pooling)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        use_cuda: bool = True,
    ):
        """
        Args:
            model: The CLIP model (should have .visual attribute)
            target_layer: Specific layer to hook. If None, automatically selects:
                         - ViT: last transformer block
                         - ResNet: layer4
            use_cuda: Whether to use CUDA if available
        """
        self.model = model
        self.model.eval()

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Detect architecture
        self.is_vit = hasattr(self.model.visual, 'transformer')

        # Auto-select target layer if not provided
        if target_layer is None:
            if self.is_vit:
                # For ViT: use last transformer block
                self.target_layer = self.model.visual.transformer.resblocks[-1]
            else:
                # For ResNet: use layer4
                self.target_layer = self.model.visual.layer4
        else:
            self.target_layer = target_layer

        # Storage for forward/backward hooks
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            # Don't detach - we need gradients to flow for backward pass
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            # Don't detach - we need to keep the gradients
            self.gradients = grad_output[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor [B, C, H, W] (preprocessed)

        Returns:
            Image features from encode_image [B, D]
        """
        return self.model.encode_image(x)

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_category: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed input [B, C, H, W]
            target_category: Target class index. If None, uses the predicted class.

        Returns:
            CAM heatmap [B, H, W] in range [0, 1]
        """
        input_tensor = input_tensor.to(self.device)
        batch_size = input_tensor.shape[0]

        # Forward pass
        self.model.zero_grad()
        output = self.forward(input_tensor)  # [B, D]

        # If no target specified, use the norm of the feature vector
        # (For CLIP, we can't use argmax since it's not a classifier)
        if target_category is None:
            # Use the magnitude of the embedding as the target
            score = output.norm(dim=1).sum()
        else:
            # Use specific dimension
            score = output[:, target_category].sum()

        # Backward pass
        score.backward()

        # Get activations and gradients
        activations = self.activations  # Shape varies by architecture
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError("Activations or gradients not captured. Check hooks.")

        # Generate CAM
        if self.is_vit:
            cam = self._generate_vit_cam(activations, gradients, input_tensor.shape[-2:])
        else:
            cam = self._generate_resnet_cam(activations, gradients)

        # Normalize to [0, 1]
        cam = self._normalize_cam(cam)

        return cam

    def _generate_vit_cam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
        input_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Generate CAM for Vision Transformer.

        ViT activations shape: [seq_len, B, D] or [B, seq_len, D]
        We need to extract spatial tokens (excluding CLS token) and reshape to 2D.
        """
        # Handle different tensor formats
        if activations.dim() == 3:
            if activations.shape[0] == 1 or activations.shape[0] > activations.shape[1]:
                # [seq_len, B, D] -> [B, seq_len, D]
                activations = activations.permute(1, 0, 2)
                gradients = gradients.permute(1, 0, 2)

        batch_size = activations.shape[0]

        # Remove CLS token (first token)
        activations = activations[:, 1:, :]  # [B, num_patches, D]
        gradients = gradients[:, 1:, :]

        # For ViT Grad-CAM: compute importance scores per patch
        # Use ReLU on gradients to keep only positive contributions (standard Grad-CAM)
        gradients_positive = F.relu(gradients)

        # Weight activations by positive gradients and sum over feature dimension
        cam = (gradients_positive * activations).sum(dim=2)  # [B, num_patches]

        # Reshape to 2D spatial map
        num_patches = cam.shape[1]
        patch_size = int(np.sqrt(num_patches))
        cam = cam.reshape(batch_size, patch_size, patch_size)  # [B, H', W']

        # Upsample to input resolution
        cam = cam.unsqueeze(1)  # [B, 1, H', W']
        cam = F.interpolate(
            cam,
            size=input_shape,
            mode='bilinear',
            align_corners=False,
        )
        cam = cam.squeeze(1)  # [B, H, W]

        # Apply ReLU to focus on positive contributions
        cam = F.relu(cam)

        return cam.detach().cpu().numpy()

    def _generate_resnet_cam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> np.ndarray:
        """
        Generate CAM for ResNet-style architecture.

        Activations shape: [B, C, H, W]
        """
        # Global average pooling on gradients
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]

        # Weighted combination
        cam = (weights * activations).sum(dim=1)  # [B, H, W]

        # Apply ReLU
        cam = F.relu(cam)

        return cam.detach().cpu().numpy()

    def _normalize_cam(self, cam: np.ndarray) -> np.ndarray:
        """
        Normalize CAM to [0, 1] range per sample.

        Args:
            cam: [B, H, W]

        Returns:
            Normalized CAM [B, H, W]
        """
        batch_size = cam.shape[0]
        cam_normalized = np.zeros_like(cam)

        for i in range(batch_size):
            cam_i = cam[i]
            cam_min = cam_i.min()
            cam_max = cam_i.max()

            if cam_max - cam_min > 1e-8:
                cam_normalized[i] = (cam_i - cam_min) / (cam_max - cam_min)
            else:
                # If constant, set to zero
                cam_normalized[i] = 0.0

        return cam_normalized

    def __call__(self, input_tensor: torch.Tensor, target_category: Optional[int] = None) -> np.ndarray:
        """Shorthand for generate_cam."""
        return self.generate_cam(input_tensor, target_category)


class MultiLabelGradCAM(GradCAM):
    """
    Grad-CAM adapted for multi-label classification (CheXpert 14 labels).

    For each sample, generates a CAM by aggregating across all positive labels.
    """

    def generate_multilabel_cam(
        self,
        input_tensor: torch.Tensor,
        text_features: torch.Tensor,
        target_labels: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Generate CAM for multi-label chest X-ray classification.

        Args:
            input_tensor: Preprocessed input [B, C, H, W]
            text_features: Pre-computed text features [L, D] for L labels
            target_labels: Binary label tensor [B, L]. If None, uses all labels equally.

        Returns:
            CAM heatmap [B, H, W] in range [0, 1]
        """
        input_tensor = input_tensor.to(self.device)
        text_features = text_features.to(self.device)
        batch_size = input_tensor.shape[0]
        num_labels = text_features.shape[0]

        # Forward pass
        self.model.zero_grad()
        image_features = self.forward(input_tensor)  # [B, D]

        # Convert to float32 for consistency (model may be in fp16)
        image_features = image_features.float()
        text_features = text_features.float()

        # For Grad-CAM, we want gradients w.r.t. the image features
        # Use simple magnitude-based score instead of cosine similarity
        # to avoid gradient issues with normalization
        if target_labels is not None:
            target_labels = target_labels.to(self.device).float()
            # Weight features by which labels are active
            # Compute unnormalized similarity
            logits = image_features @ text_features.t()  # [B, L]
            logits = logits * target_labels
            score = logits.sum()
        else:
            # Fall back to feature magnitude
            score = image_features.norm(dim=1).sum()

        # Backward pass
        score.backward()

        # Generate CAM
        activations = self.activations
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError("Activations or gradients not captured.")

        if self.is_vit:
            cam = self._generate_vit_cam(activations, gradients, input_tensor.shape[-2:])
        else:
            cam = self._generate_resnet_cam(activations, gradients)

        cam = self._normalize_cam(cam)

        return cam


def create_lesion_mask(
    cam: np.ndarray,
    threshold: float = 0.5,
    min_size: Optional[int] = None,
) -> np.ndarray:
    """
    Convert Grad-CAM heatmap to binary lesion mask.

    Args:
        cam: Grad-CAM heatmap [B, H, W] in [0, 1]
        threshold: Binarization threshold (keep pixels >= threshold)
        min_size: Minimum connected component size (if None, no filtering)

    Returns:
        Binary mask [B, H, W] with 1 in lesion regions, 0 elsewhere
    """
    mask = (cam >= threshold).astype(np.float32)

    # Optional: remove small connected components
    if min_size is not None:
        from scipy.ndimage import label
        for i in range(mask.shape[0]):
            labeled, num_features = label(mask[i])
            for j in range(1, num_features + 1):
                component = (labeled == j)
                if component.sum() < min_size:
                    mask[i][component] = 0

    return mask


# Convenience function
def generate_chexzero_lesion_mask(
    model: nn.Module,
    input_tensor: torch.Tensor,
    text_features: torch.Tensor,
    target_labels: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
    use_cuda: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    End-to-end: Generate Grad-CAM and lesion mask for CheXzero.

    Args:
        model: CheXzero CLIP model
        input_tensor: Preprocessed input [B, C, H, W]
        text_features: Text features [L, D] for L=14 labels
        target_labels: Ground truth labels [B, L] (optional)
        threshold: Mask binarization threshold
        use_cuda: Use GPU if available

    Returns:
        cam: Grad-CAM heatmap [B, H, W]
        mask: Binary lesion mask [B, H, W]
    """
    grad_cam = MultiLabelGradCAM(model, use_cuda=use_cuda)
    cam = grad_cam.generate_multilabel_cam(input_tensor, text_features, target_labels)
    mask = create_lesion_mask(cam, threshold=threshold)

    return cam, mask

"""Loss functions for binary land-take segmentation.

Shared across all experiments. Self-contained (torch only).
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation with class imbalance.

    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class (0-1)
        gamma: Focusing parameter (gamma >= 0)
        reduction: 'mean', 'sum', or 'none'

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth labels, shape (B, H, W) or (B, 1, H, W)

        Returns:
            Focal loss value
        """
        # Ensure targets have same shape as inputs
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate binary cross entropy
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )

        # Calculate p_t
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Calculate focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate alpha weight
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Apply focal loss formula
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Directly optimizes the Dice coefficient (F1-score), making it inherently
    robust to class imbalance since it normalizes by the predicted and target
    areas.

    DL = 1 - (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)

    Args:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        reduction: 'mean' or 'none'
    """

    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth labels, shape (B, H, W) or (B, 1, H, W)

        Returns:
            Dice loss value (1 - Dice coefficient)
        """
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        probs = torch.sigmoid(inputs)

        # Flatten per sample: (B, N)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        if self.reduction == 'mean':
            return 1.0 - dice.mean()
        else:
            return 1.0 - dice


class FocalDiceLoss(nn.Module):
    """
    Combined Focal + Dice Loss for binary segmentation.

    L = lambda_focal * FocalLoss + lambda_dice * DiceLoss

    Focal provides pixel-level hard-example mining; Dice provides
    region-level overlap optimization. Common in remote sensing
    segmentation with class imbalance.

    Args:
        focal_alpha: Focal loss alpha (positive class weight)
        focal_gamma: Focal loss gamma (focusing parameter)
        lambda_focal: Weight for focal loss term (default: 1.0)
        lambda_dice: Weight for dice loss term (default: 1.0)
        dice_smooth: Dice loss smoothing factor (default: 1.0)
    """

    def __init__(
        self,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')
        self.dice = DiceLoss(smooth=dice_smooth, reduction='mean')
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(inputs, targets)
        dice_loss = self.dice(inputs, targets)
        return self.lambda_focal * focal_loss + self.lambda_dice * dice_loss

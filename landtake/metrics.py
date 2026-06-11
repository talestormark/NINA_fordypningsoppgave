"""Pixel-level segmentation metrics (F1, IoU, precision, recall). Shared across experiments."""

import torch


class Metrics:
    """
    Compute segmentation metrics: F1-score, IoU, Precision, Recall.

    All metrics computed at pixel level for binary segmentation.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset accumulated metrics."""
        self.tp = 0  # True positives
        self.fp = 0  # False positives
        self.tn = 0  # True negatives
        self.fn = 0  # False negatives

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with a batch of predictions and targets.

        Args:
            preds: Model predictions (logits), shape (B, 1, H, W)
            targets: Ground truth labels, shape (B, H, W) or (B, 1, H, W)
        """
        # Convert logits to binary predictions
        preds_binary = (torch.sigmoid(preds) > self.threshold).float()

        # Ensure same shape
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        # Flatten tensors
        preds_flat = preds_binary.view(-1)
        targets_flat = targets.view(-1)

        # Compute confusion matrix components
        self.tp += ((preds_flat == 1) & (targets_flat == 1)).sum().item()
        self.fp += ((preds_flat == 1) & (targets_flat == 0)).sum().item()
        self.tn += ((preds_flat == 0) & (targets_flat == 0)).sum().item()
        self.fn += ((preds_flat == 0) & (targets_flat == 1)).sum().item()

    def compute(self) -> dict:
        """
        Compute final metrics from accumulated values.

        Returns:
            Dictionary with precision, recall, f1, iou, accuracy
        """
        # Avoid division by zero
        epsilon = 1e-7

        precision = self.tp / (self.tp + self.fp + epsilon)
        recall = self.tp / (self.tp + self.fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        iou = self.tp / (self.tp + self.fp + self.fn + epsilon)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + epsilon)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'accuracy': accuracy,
        }

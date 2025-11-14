#!/usr/bin/env python3
"""
General-purpose logging module for training experiments using Weights & Biases.

This module provides a simple interface for:
- Initializing wandb runs with project configuration
- Logging metrics during training
- Logging model architecture and hyperparameters
- Finishing runs properly

Usage:
    from logger import WandbLogger

    # Initialize logger
    logger = WandbLogger(
        project="landtake-detection",
        name="siam_conc_resnet50_seed42",
        config={"lr": 0.01, "batch_size": 4, ...},
        enabled=True  # Set to False to disable wandb
    )

    # Log metrics during training
    logger.log_metrics({
        "train/loss": 0.5,
        "train/f1": 0.7,
        "val/iou": 0.6
    }, step=epoch)

    # Finish run
    logger.finish()
"""

import wandb
from pathlib import Path
from typing import Dict, Any, Optional


class WandbLogger:
    """
    Wrapper class for Weights & Biases logging.

    Provides a clean interface for logging training experiments with
    optional disabling for debugging or offline training.
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        entity: Optional[str] = None,
        enabled: bool = True,
        resume: Optional[str] = None,
        id: Optional[str] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None,
    ):
        """
        Initialize wandb logger.

        Args:
            project: wandb project name (e.g., "landtake-detection")
            name: Run name (e.g., "siam_conc_resnet50_seed42")
            config: Dictionary of hyperparameters and configuration
            entity: wandb team/user name (optional, uses default if not specified)
            enabled: If False, all logging becomes no-ops (useful for debugging)
            resume: Resume mode - "allow", "must", "never", or None
            id: Unique run ID for resuming (optional)
            tags: List of tags for organizing runs
            notes: Text description of the run
        """
        self.enabled = enabled
        self.run = None

        if self.enabled:
            # Initialize wandb run
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                entity=entity,
                resume=resume,
                id=id,
                tags=tags,
                notes=notes,
            )

            print(f"✓ Weights & Biases initialized")
            print(f"  Project: {project}")
            print(f"  Run: {self.run.name}")
            print(f"  URL: {self.run.url}")
        else:
            print("✗ Weights & Biases logging disabled")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names and values
                     e.g., {"train/loss": 0.5, "val/iou": 0.6}
            step: Optional step number (epoch or iteration)
        """
        if self.enabled and self.run is not None:
            self.run.log(metrics, step=step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ):
        """
        Convenience method to log metrics for a complete epoch.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics (loss, f1, iou, precision, recall)
            val_metrics: Validation metrics (loss, f1, iou, precision, recall)
            lr: Current learning rate
        """
        if not self.enabled:
            return

        metrics = {
            "epoch": epoch,
            "lr": lr,
            # Training metrics
            "train/loss": train_metrics.get("loss", 0),
            "train/f1": train_metrics.get("f1", 0),
            "train/iou": train_metrics.get("iou", 0),
            "train/precision": train_metrics.get("precision", 0),
            "train/recall": train_metrics.get("recall", 0),
            "train/accuracy": train_metrics.get("accuracy", 0),
            # Validation metrics
            "val/loss": val_metrics.get("loss", 0),
            "val/f1": val_metrics.get("f1", 0),
            "val/iou": val_metrics.get("iou", 0),
            "val/precision": val_metrics.get("precision", 0),
            "val/recall": val_metrics.get("recall", 0),
            "val/accuracy": val_metrics.get("accuracy", 0),
        }

        self.log_metrics(metrics, step=epoch)

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100):
        """
        Watch model gradients and parameters.

        Args:
            model: PyTorch model to watch
            log: What to log - "gradients", "parameters", "all", or None
            log_freq: Frequency of logging (every N batches)
        """
        if self.enabled and self.run is not None:
            wandb.watch(model, log=log, log_freq=log_freq)

    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str,
        name: Optional[str] = None,
    ):
        """
        Log a file artifact (model checkpoint, config, etc.).

        Args:
            artifact_path: Path to file to upload
            artifact_type: Type of artifact ("model", "dataset", "config", etc.)
            name: Optional name for the artifact
        """
        if self.enabled and self.run is not None:
            artifact = wandb.Artifact(
                name=name or Path(artifact_path).stem,
                type=artifact_type,
            )
            artifact.add_file(artifact_path)
            self.run.log_artifact(artifact)

    def finish(self):
        """Finish the wandb run and upload any remaining data."""
        if self.enabled and self.run is not None:
            self.run.finish()
            print("✓ Weights & Biases run finished")


def create_run_name(model_name: str, encoder_name: str, seed: int) -> str:
    """
    Create a standardized run name.

    Args:
        model_name: Model architecture (e.g., "siam_conc")
        encoder_name: Encoder architecture (e.g., "resnet50")
        seed: Random seed

    Returns:
        Formatted run name (e.g., "siam_conc_resnet50_seed42")
    """
    return f"{model_name}_{encoder_name}_seed{seed}"


def create_tags(model_name: str, encoder_name: str, **kwargs) -> list:
    """
    Create tags for organizing wandb runs.

    Args:
        model_name: Model architecture
        encoder_name: Encoder architecture
        **kwargs: Additional key-value pairs to use as tags

    Returns:
        List of tags
    """
    tags = [model_name, encoder_name]

    # Add any additional tags from kwargs
    for key, value in kwargs.items():
        if value is not None:
            tags.append(f"{key}:{value}")

    return tags

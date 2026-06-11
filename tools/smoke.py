#!/usr/bin/env python3
"""Data-free smoke test for the landtake package — the per-stage refactor gate.

Imports the full shared API, builds a model, runs a forward pass and losses on
random tensors. No data or GPU required. Exits non-zero on any failure.

    python tools/smoke.py
"""

import torch

from landtake.losses import FocalLoss, DiceLoss, FocalDiceLoss
from landtake.metrics import Metrics
from landtake.logger import WandbLogger, create_run_name, create_tags  # noqa: F401
from landtake.models.multitemporal import EarlyFusionUNet, create_multitemporal_model, count_parameters  # noqa: F401
from landtake.models.convlstm import ConvLSTM  # noqa: F401
from landtake.data.multitemporal import get_dataloaders  # noqa: F401  (import only; no data touched)
from landtake import paths, config


def main():
    # losses + metrics on random tensors
    logits = torch.randn(2, 1, 16, 16)
    target = (torch.rand(2, 16, 16) > 0.7).float()
    for loss_fn in (FocalLoss(), DiceLoss(), FocalDiceLoss()):
        assert loss_fn(logits, target).ndim == 0, loss_fn
    m = Metrics()
    m.update(logits, target)
    m.compute()

    # model build + forward + loss
    model = EarlyFusionUNet(encoder_name="resnet18", encoder_weights=None, in_channels=9)
    out = model(torch.randn(1, 2, 9, 64, 64))  # (B, T, C, H, W)
    assert tuple(out.shape) == (1, 1, 64, 64), out.shape
    FocalDiceLoss()(out, (torch.rand(1, 1, 64, 64) > 0.7).float())

    # path/config sanity
    assert paths.REPO_ROOT.is_dir(), paths.REPO_ROOT
    assert config.SENTINEL2_NUM_BANDS == 9

    print("landtake smoke OK")


if __name__ == "__main__":
    main()

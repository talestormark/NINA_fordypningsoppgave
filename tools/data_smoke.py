#!/usr/bin/env python3
"""End-to-end 1-tile data smoke (needs data_v2 + data/processed present, CPU).

Loads one real tile through each input pathway of the refactored pipeline:
  - exp1: multitemporal Sentinel-2  (landtake.data.multitemporal)
  - exp2: spectral Sentinel-2       (landtake.data.spectral, A3_s2_9band)
  - exp3: AlphaEarth                (landtake.data.spectral, D2_alphaearth)
Proves data paths resolve and real files load after the structure reorg.
"""
import numpy as np
import torch

from landtake.paths import REPO_ROOT, DATA_V2_DIR


def _shapes(x):
    if isinstance(x, dict):
        return {k: tuple(v.shape) for k, v in x.items() if hasattr(v, "shape")}
    return tuple(x.shape)


def exp1_multitemporal():
    from landtake.data.multitemporal import MultiTemporalSentinel2Dataset
    refids = [l.strip() for l in open(REPO_ROOT / "preprocessing/outputs/splits/part1/train_refids.txt") if l.strip()][:1]
    ds = MultiTemporalSentinel2Dataset(
        refids=refids,
        sentinel2_dir=DATA_V2_DIR / "Sentinel",
        mask_dir=DATA_V2_DIR / "Land_take_masks_coarse",
        normalization_stats={"mean": np.zeros(9), "std": np.ones(9)},
        temporal_sampling="bi_temporal",
        target_size=64,
    )
    sample = ds[0]
    print("  exp1 multitemporal sample:", _shapes(sample))
    return sample


def exp_spectral(experiment):
    from landtake.data.spectral import get_dataloaders
    loaders = get_dataloaders(
        experiment=experiment, batch_size=1, num_workers=0, image_size=64,
        fold=None, use_precomputed_stats=True, data_dir="epsg3035_10m_v2",
    )
    batch = next(iter(loaders["train"]))
    print(f"  {experiment} batch:", _shapes(batch))
    return batch


def one_step(batch):
    """Real forward + loss + backward on a loaded batch (1 training step)."""
    from landtake.models.multitemporal import EarlyFusionUNet
    from landtake.losses import FocalDiceLoss
    x, y = batch["image"].float(), batch["mask"].float()
    model = EarlyFusionUNet(encoder_name="resnet18", encoder_weights=None, in_channels=x.shape[2])
    loss = FocalDiceLoss()(model(x), y)
    loss.backward()
    print(f"  1-step: loss {loss.item():.4f}")


if __name__ == "__main__":
    print("[exp1] multitemporal Sentinel-2 ...")
    exp1_multitemporal()
    print("[exp2] spectral Sentinel-2 (A3_s2_9band) ...")
    one_step(exp_spectral("A3_s2_9band"))
    print("[exp3] AlphaEarth (D2_alphaearth) ...")
    one_step(exp_spectral("D2_alphaearth"))
    print("DATA SMOKE OK")

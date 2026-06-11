# Data dictionary

The imagery, embeddings and annotation masks are **not redistributed in this repository**
(NINA collaboration; the optical sources have their own licences — see *Data availability* in the
[README](README.md)). This document describes the dataset so the code and results can be understood
and, with access to the data, reproduced.

The dataset documentation below is authored by **Zander Venter (NINA)**, prepared with Google Earth
Engine for the HABLOSS European land-take monitoring project (March 2026).

## Overview

Raster exports for selected manually annotated areas across Europe (REFIDs). Each REFID is a
650 m × 650 m tile where land take / habitat change was manually annotated. As of March 2026 there
are **261 annotated REFIDs** with corresponding satellite imagery. Annotations were resampled to the
Sentinel-2 10 m grid (`Land_take_masks_coarse/`); **101** tiles were also annotated in detail at 1 m
(`Land_take_masks_detailed/`). A metadata CSV records the start and end year of the VHR images used
as the basis for each annotation.

## Folder structure

| Folder | Description | Example file |
|---|---|---|
| `Sentinel/` | Multi-temporal Sentinel-2 mosaics (Q2 + Q3, 2018–24) | `REFID_RGBNIRRSWIRQ_Mosaic.tif` |
| `PlanetScope/` | Multi-temporal PlanetScope mosaics (Q2 + Q3, 2018–24) | `REFID_RGBQ_Mosaic.tif` |
| `VHR_google/` | Google VHR RGB mosaics (start and end years) | `REFID_RGBY_Mosaic.tif` |
| `AlphaEarth/` | AlphaEarth annual embeddings (2018–2024) | `REFID_VEY_Mosaic.tif` |
| `Land_take_masks_coarse/` | Binary change masks from manual annotations, 10 m | `REFID_mask.tif` |
| `Land_take_masks_detailed/` | Binary change masks from manual annotations, 1 m | `REFID_mask.tif` |

All rasters are in **EPSG:3035**.

## Sentinel-2 stack

126 bands = 7 years × 2 quarters × 9 bands. Quarters: Q2 (Apr–Jun), Q3 (Jul–Sep). Bands per quarter:
`blue, green, red, R1, R2, R3, nir, swir1, swir2`. Band naming `<year>_<quarter>_<band>`
(e.g. `2018_2_blue` … `2024_3_swir2`). Resolution 10 m. Units: top-of-atmosphere reflectance × 10000.

## PlanetScope stack

42 bands = 7 years × 2 quarters × 3 bands (`blue, green, red`). Naming `<year>_<quarter>_<band>`.
Resolution 3–5 m.

## VHR Google

6 bands (RGB for start and end years). Naming `<startYear>_R/G/B`, `<endYear>_R/G/B`. Resolution 1 m.

## AlphaEarth embeddings

Annual deep spectral embeddings from `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`. Naming
`<year>_<embedding-vector>`. Resolution 10 m. Learned feature vectors summarising spectral–textural
patterns.

## Land-take masks

Binary rasters (1 = change, 0 = no change), derived from annotation polygons. 10 m (coarse) and 1 m
(detailed).

## Metadata files

- `annotations_metadata_final.csv` — start and end year of the VHR images used as the basis for each
  tile's annotation.
- `land_take_bboxes_650m_v1_filtered.geojson` — bounding boxes of the annotated tiles.

## Band-count summary

| Export | Bands | Resolution | Description |
|---|---|---|---|
| Sentinel-2 | 126 | 10 m | Optical time-series (Q2 + Q3, 2018–2024) |
| PlanetScope | 42 | 3–5 m | Optical time-series (Q2 + Q3, 2018–2024) |
| VHR Google | 6 | 1 m | Start–end RGB mosaics |
| AlphaEarth | varies | 10 m | Annual embeddings |
| Mask (coarse) | 1 | 10 m | Binary change label |
| Mask (detailed) | 1 | 1 m | Binary change label |

## Contact (data)

Zander Venter, Norwegian Institute for Nature Research (NINA) — zander.venter@nina.no



**Author:** Zander Venter (NINA)  
**Project:** HABLOSS – European Land Take Monitoring  
**Data prepared using:** Google Earth Engine (script: https://code.earthengine.google.com/425dee2efda7b2a55da0ea39fbbab84c)  
**Date:** October 2025

---

## OVERVIEW

This dataset contains raster exports from Google Earth Engine for selected annotated areas across Europe (REFIDs). Each REFID corresponds to a 650 m × 650 m tile where land take or habitat change has been manually annotated.

As of October 2025 there are 55 annotated REFIDs with corresponding satellite images. Note that the PlanetScope folder contains all 1967 images, whereas other remote sensing folders only have 55.

Each exported raster stack represents multi-temporal remote sensing data (Sentinel-2, Google VHR, PlanetScope, and AlphaEarth embeddings) clipped to the bounding box of the annotated area.

---

## FOLDER STRUCTURE

| Folder | Description | Example file |
|--------|-------------|--------------|
| Sentinel/ | Multi-temporal Sentinel-2 mosaics (Q2 + Q3, 2018–24) | REFID_RGBNIRRSWIRQ_Mosaic.tif |
| PlanetScope/ | Multi-temporal PlanetScope mosaics (Q2 + Q3, 2018–24) | REFID_RGBQ_Mosaic.tif |
| VHR_google/ | Google VHR RGB mosaics (start and end years) | REFID_RGBY_Mosaic.tif |
| AlphaEarth/ | AlphaEarth annual embeddings (2018–2024) | REFID_VEY_Mosaic.tif |
| Land_take_masks/ | Binary change masks from manual annotations | REFID_mask.tif |

---

## SENTINEL-2 STACK DESCRIPTION

Each Sentinel-2 file contains **126 bands**: 7 years × 2 quarters × 9 bands.

**Quarters:** Q2 (Apr–Jun) and Q3 (Jul–Sep)  
**Bands per quarter:** blue, green, red, R1, R2, R3, nir, swir1, swir2

**Band naming convention:**
```
<year>_<quarter>_<band>
```

**Example:**
```
2018_2_blue, 2018_2_green, 2018_2_red, ..., 2024_3_swir2
```

**Spatial resolution:** 10 m  
**Projection:** EPSG:4326 (WGS 84)  
**Units:** Top-of-atmosphere reflectance × 10000

---

## PlanetScope STACK DESCRIPTION

Each PlanetScope file contains **42 bands**: 7 years × 2 quarters × 3 bands.

**Quarters:** Q2 (Apr–Jun) and Q3 (Jul–Sep)  
**Bands per quarter:** blue, green, red

**Band naming convention:**
```
<year>_<quarter>_<band>
```

**Example:**
```
2018_2_B, 2018_2_G, 2018_2_R
```

**Spatial resolution:** 3 to 5 m  
**Projection:** EPSG:4326 (WGS 84)

---

## VHR GOOGLE IMAGERY

Contains **6 bands** (RGB for start and end years).

**Band naming:**
```
<startYear>_R, <startYear>_G, <startYear>_B,
<endYear>_R, <endYear>_G, <endYear>_B
```

**Spatial resolution:** 1 m  
**Projection:** EPSG:4326 (WGS 84)

---

## ALPHAEARTH EMBEDDINGS

Annual deep spectral embeddings from GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL.

**Band naming:**
```
<year>_<embedding-vector>
```

**Resolution:** 10 m  
**Description:** Learned feature vectors summarizing spectral–textural patterns.

---

## LAND TAKE MASKS

Binary rasters (1 = change, 0 = no change). Derived from annotation polygons.

**Resolution:** 10 m  
**Projection:** EPSG:4326 (WGS 84)

---

## FILE METADATA SUMMARY

| Export | Bands | Resolution | Description |
|--------|-------|------------|-------------|
| Sentinel | 126 | 10 m | Optical time-series (Q2 + Q3, 2018–2024) |
| PlanetScope | 42 | 3-5 m | Optical time-series (Q2 + Q3, 2018–2024) |
| VHR Google | 6 | 1 m | Start–end RGB mosaics |
| AlphaEarth | varies | 10 m | Annual embeddings |
| Mask | 1 | 10 m | Binary change label |

---

## CONTACT

**Zander Venter**  
Norwegian Institute for Nature Research (NINA)  
Email: zander.venter@nina.no</parameter>
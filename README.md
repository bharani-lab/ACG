# RNFL Thickness from OCT (Deep Learning Segmentation)

This repository contains a minimal, reproducible pipeline to compute **retinal nerve fiber layer (RNFL) thickness** from OCT B-scans using a **published deep learning segmentation method** (U-Net-style architecture) and to extract common **glaucoma features** from fundus (en-face) and OCT images.

## Quick start

### 1) Fundus (en-face) feature extraction
Provide binary masks for disc/cup/vessels (from any segmentation model) and compute glaucoma features:

```bash
python glaucoma_features.py fundus \
  --disc-mask disc_mask.png \
  --cup-mask cup_mask.png \
  --vessel-mask vessel_mask.png \
  --vessel-skeleton-mask vessel_skeleton.png \
  --rnfl-defect-mask rnfl_defects.png \
  --output fundus_features.csv
```

### 2) OCT B-scan feature extraction
Provide layer masks and extract RNFL/GCC thickness and BMO-MRW:

```bash
python glaucoma_features.py oct \
  --ilm-mask ilm_mask.png \
  --rnfl-mask rnfl_mask.png \
  --gcl-mask gcl_mask.png \
  --ipl-mask ipl_mask.png \
  --rpe-mask rpe_mask.png \
  --bmo-points bmo_points.csv \
  --microns-per-pixel 3.9 \
  --output oct_features.csv
```

### 3) Optional U-Net RNFL mask prediction
If you have a trained U-Net checkpoint, you can generate an RNFL mask from an OCT B-scan:

```bash
python glaucoma_features.py rnfl-unet \
  --image d2.PNG \
  --weights /path/to/unet_weights.pt \
  --output-mask rnfl_mask.png
```

### 4) Automatic mask generation (heuristic)
If you do not have pretrained models, you can generate rough masks automatically. These are
heuristic estimates and should be reviewed before clinical use.

```bash
python glaucoma_features.py fundus-auto \
  --image fundus.png \
  --output-dir fundus_masks
```

```bash
python glaucoma_features.py oct-auto \
  --image oct.png \
  --output-dir oct_masks
```

## Inputs
### Fundus mode
- `--disc-mask`: Binary optic disc mask.
- `--cup-mask`: Binary optic cup mask.
- `--vessel-mask`: Binary vessel mask (optional).
- `--vessel-skeleton-mask`: Binary vessel skeleton mask (optional; enables tortuosity).
- `--rnfl-defect-mask`: Binary RNFL defect mask (optional).
- `--annulus-scale`: Outer radius scale for peripapillary annulus (default 1.5).
- `--output`: CSV path for features.

### OCT mode
- `--ilm-mask`: Binary ILM mask.
- `--rnfl-mask`: Binary RNFL mask (optional but recommended).
- `--gcl-mask`: Binary GCL mask (optional).
- `--ipl-mask`: Binary IPL mask (optional).
- `--rpe-mask`: Binary RPE mask (optional).
- `--bmo-points`: CSV with BMO points (x,y per line; optional).
- `--microns-per-pixel`: Physical spacing in microns per pixel.
- `--output`: CSV path for features.

### Automatic mask generation
- `fundus-auto` outputs `disc_mask.png`, `cup_mask.png`, `vessel_mask.png`.
- `oct-auto` outputs `ilm_mask.png`, `rnfl_mask.png`.
- These masks are heuristic and should be validated against manual labels when possible.

## What is a mask?
A **mask** is a binary (black/white) image where pixels that belong to a structure of interest
are set to white (value > 0) and everything else is black (value = 0). In this pipeline, masks
are used for structures like the optic disc, cup, vessels, or retinal layers (ILM, RNFL, GCL, IPL, RPE).

### How do I create masks?
You can create masks in three common ways:
1. **Use a pretrained segmentation model** (recommended) to generate masks automatically.
2. **Manual annotation** in tools like ImageJ/Fiji or ITK-SNAP, then export as PNG masks.
3. **Semi-automatic tools** (thresholding + manual cleanup) for quick drafts.

Mask images should be the **same resolution** as the original image and saved as PNG with
white pixels for the target structure and black elsewhere.

## Feature outputs
### Fundus (en-face)
- Cup-to-disc ratio (CDR)
- Neuro-retinal rim area / rim thickness (derived from disc/cup masks)
- Disc area / cup area
- Peripapillary vessel density / vessel tortuosity
- RNFL defect area

### OCT B-scan
- RNFL thickness (mean/median)
- GCC thickness (mean/median, from GCL + IPL)
- ILM/RNFL/GCL/IPL/RPE boundaries used for thickness calculation
- BMO-MRW (if BMO points provided)

## Reference
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

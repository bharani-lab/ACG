# RNFL Thickness from OCT (Deep Learning Segmentation)

This repository contains a minimal, reproducible pipeline to compute **retinal nerve fiber layer (RNFL) thickness** from OCT B-scans using a **published deep learning segmentation method** (U-Net-style architecture). The workflow is:

1. **Segment RNFL** from OCT B-scan with a U-Net segmentation model (Ronneberger et al., 2015) or a heuristic fallback.
2. **Extract RNFL boundaries** per A-scan (image column).
3. **Compute thickness** from the boundary distance and pixel spacing.

> **Note:** You must provide your own trained weights compatible with the included U-Net model. The script expects a binary RNFL mask output.

## Quick start

```bash
python rnfl_thickness.py \
  --image d2.PNG \
  --weights /path/to/unet_weights.pt \
  --microns-per-pixel 3.9 \
  --output rnfl_thickness.csv
```

```bash
python rnfl_thickness.py \
  --image d2.PNG \
  --method heuristic \
  --microns-per-pixel 3.9 \
  --output rnfl_thickness.csv
```

### Inputs
- `--image`: Path to OCT B-scan image (PNG/JPG).
- `--method`: `unet` (default) or `heuristic` segmentation.
- `--weights`: Path to PyTorch `.pt` weights for the included U-Net (required for `unet`).
- `--microns-per-pixel`: Physical spacing in microns per pixel (from OCT device metadata).
- `--output`: CSV file to save per-column thickness values and summary statistics.

### Outputs
- CSV containing per-column RNFL thickness (microns) plus mean/median.

## Method notes
- **Segmentation model**: U-Net (Ronneberger et al., 2015) or a heuristic edge-based detector.
- **Thickness calculation**: For each A-scan (column), the RNFL thickness is computed as the vertical distance between the topmost and bottommost RNFL pixels in that column.

## Glaucoma feature extraction pipeline (fundus + OCT)
This section summarizes a full pipeline for common glaucoma-related features and points to places
where pretrained models are often available. Because this environment cannot browse the web, the
model sources are described at a high level so you can locate the latest checkpoints.

### A) Fundus (en-face) features
**Features to extract**
- Cup-to-disc ratio (CDR)
- Neuro-retinal rim area / rim thickness
- Disc area / cup area
- Peripapillary vessel density / vessel tortuosity
- RNFL defect patterns (wedge defects on en-face)

**Suggested pipeline**
1. **Optic disc/cup segmentation** (disc and cup masks).
2. **Vessel segmentation** for density/tortuosity.
3. **Compute features** from masks (areas, ratios, tortuosity).

**Pretrained model options (typical sources)**
- **Disc/cup segmentation**: U-Net/DeepLab/Mask R-CNN models trained on public datasets
  (e.g., DRISHTI-GS, RIM-ONE, REFUGE).
- **Vessel segmentation**: U-Net variants trained on fundus vessel datasets
  (e.g., DRIVE, STARE, CHASE-DB1).

### B) OCT B-scan features
**Features to extract**
- RNFL thickness (peripapillary)
- GCC thickness (GCL + IPL)
- ILM, RNFL, GCL, IPL, RPE layer boundaries
- Rim width at Bruch’s membrane opening (BMO-MRW)
- ONH morphology metrics

**Suggested pipeline**
1. **Retinal layer segmentation** (ILM, RNFL, GCL, IPL, RPE/BM).
2. **Thickness computation** per A-scan for RNFL and GCC.
3. **ONH/BMO detection** to compute BMO-MRW and rim metrics.

**Pretrained model options (typical sources)**
- **Retinal layer segmentation**: U-Net/FCN models from public OCT layer datasets
  (e.g., Duke/BOE OCT, AROD, RETOUCH where applicable).
- **ONH/BMO detection**: models trained on optic nerve head OCT datasets; some research
  pipelines provide pretrained weights and evaluation scripts.

### Output feature list (examples)
- CDR = cup area / disc area
- Rim area = disc area − cup area
- Vessel density = vessel pixels / peripapillary ROI area
- RNFL thickness profile (microns) and summary stats
- GCC thickness profile (microns) and summary stats
- BMO-MRW (minimum rim width)

## Reference
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

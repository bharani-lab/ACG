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

## Reference
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.

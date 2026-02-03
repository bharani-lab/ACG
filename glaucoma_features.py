import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn


@dataclass
class ThicknessResult:
    per_column_microns: np.ndarray
    mean_microns: float
    median_microns: float


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(256, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        bridge = self.bridge(self.pool3(d3))
        u3 = self.up3(bridge)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.dec3(u3)
        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.dec2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.dec1(u1)
        return self.out_conv(u1)


def load_grayscale_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.array(image, dtype=np.float32) / 255.0


def load_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return (np.array(image) > 0).astype(np.uint8)


def preprocess(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image)[None, None, ...]
    return tensor


def box_filter_1d(data: np.ndarray, axis: int, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return data
    pad = kernel_size // 2
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(data, pad_width, mode="edge")
    cumsum = np.cumsum(padded, axis=axis)
    slice_start = [slice(None)] * data.ndim
    slice_end = [slice(None)] * data.ndim
    slice_start[axis] = slice(0, -kernel_size)
    slice_end[axis] = slice(kernel_size, None)
    window_sum = cumsum[tuple(slice_end)] - cumsum[tuple(slice_start)]
    return window_sum / float(kernel_size)


def smooth_image(image: np.ndarray, kernel_size: int) -> np.ndarray:
    smoothed = box_filter_1d(image, axis=0, kernel_size=kernel_size)
    smoothed = box_filter_1d(smoothed, axis=1, kernel_size=kernel_size)
    return smoothed


def predict_mask(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
    mask = (probs.squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
    return mask


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros_like(mask, dtype=bool)
    best_component: List[Tuple[int, int]] = []
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component: List[Tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if mask[ny, nx] > 0 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                stack.append((ny, nx))
            if len(component) > len(best_component):
                best_component = component

    output = np.zeros_like(mask, dtype=np.uint8)
    for y, x in best_component:
        output[y, x] = 1
    return output


def auto_fundus_masks(
    image: np.ndarray,
    disc_percentile: float,
    cup_percentile: float,
    vessel_percentile: float,
    smooth_kernel: int,
) -> Dict[str, np.ndarray]:
    smoothed = smooth_image(image, smooth_kernel)
    disc_threshold = np.percentile(smoothed, disc_percentile)
    disc_mask = (smoothed >= disc_threshold).astype(np.uint8)
    disc_mask = largest_connected_component(disc_mask)

    cup_threshold = np.percentile(smoothed[disc_mask > 0], cup_percentile) if np.any(disc_mask) else 1.0
    cup_mask = ((smoothed >= cup_threshold) & (disc_mask > 0)).astype(np.uint8)

    background = smooth_image(image, smooth_kernel * 3)
    vessel_response = background - image
    vessel_threshold = np.percentile(vessel_response, vessel_percentile)
    vessel_mask = (vessel_response >= vessel_threshold).astype(np.uint8)

    return {
        "disc": disc_mask,
        "cup": cup_mask,
        "vessel": vessel_mask,
    }


def auto_oct_masks(
    image: np.ndarray,
    search_top_fraction: float,
    min_thickness_px: int,
    max_thickness_px: int,
    smoothing_kernel: int,
) -> Dict[str, np.ndarray]:
    height, width = image.shape
    smoothed = smooth_image(image, smoothing_kernel)
    grad = np.diff(smoothed, axis=0)
    top_limit = max(1, int(height * search_top_fraction))

    ilm_mask = np.zeros((height, width), dtype=np.uint8)
    rnfl_mask = np.zeros((height, width), dtype=np.uint8)

    for col in range(width):
        ilm_idx = int(np.argmax(grad[:top_limit, col]))
        start = ilm_idx + min_thickness_px
        end = min(ilm_idx + max_thickness_px, height - 2)
        if start >= end:
            continue
        rnfl_bottom = start + int(np.argmin(grad[start:end, col]))
        ilm_mask[ilm_idx, col] = 1
        rnfl_mask[ilm_idx : rnfl_bottom + 1, col] = 1

    return {
        "ilm": ilm_mask,
        "rnfl": rnfl_mask,
    }


def compute_thickness(mask: np.ndarray, microns_per_pixel: float) -> ThicknessResult:
    height, width = mask.shape
    per_column = np.zeros(width, dtype=np.float32)
    for col in range(width):
        rows = np.where(mask[:, col] > 0)[0]
        if rows.size == 0:
            per_column[col] = 0.0
            continue
        thickness_px = rows.max() - rows.min() + 1
        per_column[col] = thickness_px * microns_per_pixel
    mean_microns = float(np.mean(per_column))
    median_microns = float(np.median(per_column))
    return ThicknessResult(per_column, mean_microns, median_microns)


def mask_area(mask: np.ndarray) -> float:
    return float(np.sum(mask > 0))


def mask_perimeter(mask: np.ndarray) -> float:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    edges = np.zeros_like(padded)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            edges |= padded != np.roll(np.roll(padded, dy, axis=0), dx, axis=1)
    perimeter = np.sum(edges & padded)
    return float(perimeter)


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def annulus_mask(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    inner_radius: float,
    outer_radius: float,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.indices((height, width))
    dx = xx - center[0]
    dy = yy - center[1]
    dist = np.sqrt(dx**2 + dy**2)
    return ((dist >= inner_radius) & (dist <= outer_radius)).astype(np.uint8)


def skeleton_neighbors(mask: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                if mask[ny, nx] > 0:
                    neighbors.append((ny, nx))
    return neighbors


def bfs_path_length(mask: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> float:
    queue = [start]
    visited = {start: 0}
    while queue:
        current = queue.pop(0)
        if current == end:
            return float(visited[current])
        for neighbor in skeleton_neighbors(mask, *current):
            if neighbor not in visited:
                visited[neighbor] = visited[current] + 1
                queue.append(neighbor)
    return 0.0


def compute_tortuosity(skeleton_mask: np.ndarray) -> float:
    visited = np.zeros_like(skeleton_mask, dtype=bool)
    tortuosities: List[float] = []

    ys, xs = np.where(skeleton_mask > 0)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        stack = [(y, x)]
        component = []
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            component.append((cy, cx))
            for ny, nx in skeleton_neighbors(skeleton_mask, cy, cx):
                if not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(component) < 2:
            continue
        endpoints = [pt for pt in component if len(skeleton_neighbors(skeleton_mask, *pt)) == 1]
        if len(endpoints) < 2:
            continue
        start, end = endpoints[0], endpoints[-1]
        path_length = bfs_path_length(skeleton_mask, start, end)
        straight = np.hypot(start[0] - end[0], start[1] - end[1])
        if straight > 0:
            tortuosities.append(path_length / straight)
    if not tortuosities:
        return float("nan")
    return float(np.mean(tortuosities))


def extract_fundus_features(
    disc_mask: np.ndarray,
    cup_mask: np.ndarray,
    vessel_mask: Optional[np.ndarray],
    rnfl_defect_mask: Optional[np.ndarray],
    vessel_skeleton_mask: Optional[np.ndarray],
    annulus_scale: float,
) -> dict:
    disc_area = mask_area(disc_mask)
    cup_area = mask_area(cup_mask)
    rim_area = max(disc_area - cup_area, 0.0)
    cdr = cup_area / disc_area if disc_area > 0 else float("nan")

    disc_perimeter = mask_perimeter(disc_mask)
    rim_thickness = rim_area / disc_perimeter if disc_perimeter > 0 else float("nan")

    center = mask_centroid(disc_mask)
    disc_radius = np.sqrt(disc_area / np.pi) if disc_area > 0 else 0.0
    annulus = annulus_mask(disc_mask.shape, center, disc_radius, disc_radius * annulus_scale)

    vessel_density = float("nan")
    if vessel_mask is not None:
        vessel_density = mask_area(vessel_mask & annulus) / max(mask_area(annulus), 1.0)

    tortuosity = float("nan")
    if vessel_skeleton_mask is not None:
        tortuosity = compute_tortuosity(vessel_skeleton_mask)

    rnfl_defect_area = float("nan")
    if rnfl_defect_mask is not None:
        rnfl_defect_area = mask_area(rnfl_defect_mask)

    return {
        "disc_area_px": disc_area,
        "cup_area_px": cup_area,
        "rim_area_px": rim_area,
        "cup_to_disc_ratio": cdr,
        "rim_thickness_px": rim_thickness,
        "peripapillary_vessel_density": vessel_density,
        "vessel_tortuosity": tortuosity,
        "rnfl_defect_area_px": rnfl_defect_area,
    }


def boundary_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    top = np.full(width, np.nan, dtype=np.float32)
    bottom = np.full(width, np.nan, dtype=np.float32)
    for col in range(width):
        rows = np.where(mask[:, col] > 0)[0]
        if rows.size == 0:
            continue
        top[col] = rows.min()
        bottom[col] = rows.max()
    return top, bottom


def thickness_between_boundaries(
    top: np.ndarray,
    bottom: np.ndarray,
    microns_per_pixel: float,
) -> ThicknessResult:
    valid = (~np.isnan(top)) & (~np.isnan(bottom))
    per_column = np.zeros_like(top, dtype=np.float32)
    per_column[valid] = (bottom[valid] - top[valid] + 1) * microns_per_pixel
    mean_microns = float(np.mean(per_column))
    median_microns = float(np.median(per_column))
    return ThicknessResult(per_column, mean_microns, median_microns)


def load_points_csv(path: Path) -> List[Tuple[float, float]]:
    points = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            points.append((float(row[0]), float(row[1])))
    return points


def compute_bmo_mrw(bmo_points: List[Tuple[float, float]], ilm_boundary: np.ndarray) -> float:
    boundary_points = [(float(x), float(y)) for x, y in enumerate(ilm_boundary) if not np.isnan(y)]
    if not boundary_points or not bmo_points:
        return float("nan")
    distances = []
    for bx, by in bmo_points:
        min_dist = min(np.hypot(bx - x, by - y) for x, y in boundary_points)
        distances.append(min_dist)
    return float(np.mean(distances))


def extract_oct_features(
    ilm_mask: np.ndarray,
    rnfl_mask: Optional[np.ndarray],
    gcl_mask: Optional[np.ndarray],
    ipl_mask: Optional[np.ndarray],
    rpe_mask: Optional[np.ndarray],
    microns_per_pixel: float,
    bmo_points: Optional[List[Tuple[float, float]]],
) -> dict:
    ilm_top, ilm_bottom = boundary_from_mask(ilm_mask)

    rnfl_thickness = None
    if rnfl_mask is not None:
        rnfl_thickness = compute_thickness(rnfl_mask, microns_per_pixel)

    gcc_thickness = None
    if gcl_mask is not None and ipl_mask is not None:
        gcl_top, _ = boundary_from_mask(gcl_mask)
        _, ipl_bottom = boundary_from_mask(ipl_mask)
        gcc_thickness = thickness_between_boundaries(gcl_top, ipl_bottom, microns_per_pixel)

    rpe_boundary = None
    if rpe_mask is not None:
        rpe_boundary, _ = boundary_from_mask(rpe_mask)

    bmo_mrw = float("nan")
    if bmo_points is not None:
        bmo_mrw = compute_bmo_mrw(bmo_points, ilm_top)

    results = {
        "rnfl_mean_microns": rnfl_thickness.mean_microns if rnfl_thickness else float("nan"),
        "rnfl_median_microns": rnfl_thickness.median_microns if rnfl_thickness else float("nan"),
        "gcc_mean_microns": gcc_thickness.mean_microns if gcc_thickness else float("nan"),
        "gcc_median_microns": gcc_thickness.median_microns if gcc_thickness else float("nan"),
        "bmo_mrw_microns": bmo_mrw * microns_per_pixel if not np.isnan(bmo_mrw) else float("nan"),
    }

    return results


def write_csv(path: Path, rows: Iterable[Tuple[str, float]]) -> None:
    lines = ["feature,value\n"]
    for key, value in rows:
        lines.append(f"{key},{value}\n")
    path.write_text("".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract glaucoma features from fundus and OCT images.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    fundus = subparsers.add_parser("fundus", help="Extract fundus (en-face) features.")
    fundus.add_argument("--disc-mask", type=Path, required=True, help="Binary optic disc mask.")
    fundus.add_argument("--cup-mask", type=Path, required=True, help="Binary optic cup mask.")
    fundus.add_argument("--vessel-mask", type=Path, help="Binary vessel mask.")
    fundus.add_argument("--vessel-skeleton-mask", type=Path, help="Binary vessel skeleton mask.")
    fundus.add_argument("--rnfl-defect-mask", type=Path, help="Binary RNFL defect mask.")
    fundus.add_argument(
        "--annulus-scale",
        type=float,
        default=1.5,
        help="Outer radius scale for peripapillary annulus (relative to disc radius).",
    )
    fundus.add_argument("--output", type=Path, required=True, help="Output CSV path.")

    oct_parser = subparsers.add_parser("oct", help="Extract OCT B-scan features.")
    oct_parser.add_argument("--ilm-mask", type=Path, required=True, help="Binary ILM mask.")
    oct_parser.add_argument("--rnfl-mask", type=Path, help="Binary RNFL mask.")
    oct_parser.add_argument("--gcl-mask", type=Path, help="Binary GCL mask.")
    oct_parser.add_argument("--ipl-mask", type=Path, help="Binary IPL mask.")
    oct_parser.add_argument("--rpe-mask", type=Path, help="Binary RPE mask.")
    oct_parser.add_argument("--bmo-points", type=Path, help="CSV with BMO points (x,y per line).")
    oct_parser.add_argument(
        "--microns-per-pixel",
        type=float,
        required=True,
        help="Microns per pixel from OCT device metadata.",
    )
    oct_parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")

    unet_parser = subparsers.add_parser("rnfl-unet", help="Predict RNFL mask using U-Net.")
    unet_parser.add_argument("--image", type=Path, required=True, help="OCT B-scan image.")
    unet_parser.add_argument("--weights", type=Path, required=True, help="Path to U-Net weights.")
    unet_parser.add_argument("--output-mask", type=Path, required=True, help="Output RNFL mask path.")

    auto_fundus = subparsers.add_parser("fundus-auto", help="Auto-generate fundus masks.")
    auto_fundus.add_argument("--image", type=Path, required=True, help="Fundus image.")
    auto_fundus.add_argument("--output-dir", type=Path, required=True, help="Directory to save masks.")
    auto_fundus.add_argument("--disc-percentile", type=float, default=85.0, help="Percentile for disc mask.")
    auto_fundus.add_argument("--cup-percentile", type=float, default=95.0, help="Percentile for cup mask.")
    auto_fundus.add_argument(
        "--vessel-percentile",
        type=float,
        default=95.0,
        help="Percentile for vessel mask response.",
    )
    auto_fundus.add_argument("--smoothing-kernel", type=int, default=11, help="Smoothing kernel size.")

    auto_oct = subparsers.add_parser("oct-auto", help="Auto-generate OCT masks (ILM/RNFL).")
    auto_oct.add_argument("--image", type=Path, required=True, help="OCT B-scan image.")
    auto_oct.add_argument("--output-dir", type=Path, required=True, help="Directory to save masks.")
    auto_oct.add_argument(
        "--search-top-fraction",
        type=float,
        default=0.4,
        help="Fraction of image height to search for the ILM boundary.",
    )
    auto_oct.add_argument("--min-thickness-px", type=int, default=2, help="Minimum RNFL thickness (px).")
    auto_oct.add_argument("--max-thickness-px", type=int, default=80, help="Maximum RNFL thickness (px).")
    auto_oct.add_argument("--smoothing-kernel", type=int, default=5, help="Smoothing kernel size.")

    return parser


def run_fundus(args: argparse.Namespace) -> None:
    disc_mask = load_mask(args.disc_mask)
    cup_mask = load_mask(args.cup_mask)
    vessel_mask = load_mask(args.vessel_mask) if args.vessel_mask else None
    vessel_skeleton_mask = load_mask(args.vessel_skeleton_mask) if args.vessel_skeleton_mask else None
    rnfl_defect_mask = load_mask(args.rnfl_defect_mask) if args.rnfl_defect_mask else None

    features = extract_fundus_features(
        disc_mask,
        cup_mask,
        vessel_mask,
        rnfl_defect_mask,
        vessel_skeleton_mask,
        annulus_scale=args.annulus_scale,
    )
    write_csv(args.output, features.items())


def run_oct(args: argparse.Namespace) -> None:
    ilm_mask = load_mask(args.ilm_mask)
    rnfl_mask = load_mask(args.rnfl_mask) if args.rnfl_mask else None
    gcl_mask = load_mask(args.gcl_mask) if args.gcl_mask else None
    ipl_mask = load_mask(args.ipl_mask) if args.ipl_mask else None
    rpe_mask = load_mask(args.rpe_mask) if args.rpe_mask else None
    bmo_points = load_points_csv(args.bmo_points) if args.bmo_points else None

    features = extract_oct_features(
        ilm_mask,
        rnfl_mask,
        gcl_mask,
        ipl_mask,
        rpe_mask,
        microns_per_pixel=args.microns_per_pixel,
        bmo_points=bmo_points,
    )
    write_csv(args.output, features.items())


def run_unet(args: argparse.Namespace) -> None:
    image = load_grayscale_image(args.image)
    tensor = preprocess(image)
    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    mask = predict_mask(model, tensor)
    Image.fromarray((mask * 255).astype(np.uint8)).save(args.output_mask)


def run_auto_fundus(args: argparse.Namespace) -> None:
    image = load_grayscale_image(args.image)
    masks = auto_fundus_masks(
        image,
        disc_percentile=args.disc_percentile,
        cup_percentile=args.cup_percentile,
        vessel_percentile=args.vessel_percentile,
        smooth_kernel=args.smoothing_kernel,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        Image.fromarray((mask * 255).astype(np.uint8)).save(args.output_dir / f\"{name}_mask.png\")


def run_auto_oct(args: argparse.Namespace) -> None:
    image = load_grayscale_image(args.image)
    masks = auto_oct_masks(
        image,
        search_top_fraction=args.search_top_fraction,
        min_thickness_px=args.min_thickness_px,
        max_thickness_px=args.max_thickness_px,
        smoothing_kernel=args.smoothing_kernel,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, mask in masks.items():
        Image.fromarray((mask * 255).astype(np.uint8)).save(args.output_dir / f\"{name}_mask.png\")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "fundus":
        run_fundus(args)
    elif args.mode == "oct":
        run_oct(args)
    elif args.mode == "fundus-auto":
        run_auto_fundus(args)
    elif args.mode == "oct-auto":
        run_auto_oct(args)
    else:
        run_unet(args)


if __name__ == "__main__":
    main()

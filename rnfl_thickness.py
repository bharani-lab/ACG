import argparse
from dataclasses import dataclass
from pathlib import Path

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


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L")
    return np.array(image, dtype=np.float32) / 255.0


def preprocess(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image)[None, None, ...]
    return tensor


def predict_mask(model: nn.Module, image_tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
    mask = (probs.squeeze(0).squeeze(0).cpu().numpy() > 0.5).astype(np.uint8)
    return mask


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


def write_csv(path: Path, result: ThicknessResult) -> None:
    header = "column_index,thickness_microns\n"
    lines = [header]
    for idx, value in enumerate(result.per_column_microns):
        lines.append(f"{idx},{value:.4f}\n")
    lines.append(f"mean,{result.mean_microns:.4f}\n")
    lines.append(f"median,{result.median_microns:.4f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute RNFL thickness from OCT images.")
    parser.add_argument("--image", type=Path, required=True, help="Path to OCT B-scan image.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to PyTorch weights.")
    parser.add_argument(
        "--microns-per-pixel",
        type=float,
        required=True,
        help="Microns per pixel from OCT device metadata.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    image = load_image(args.image)
    tensor = preprocess(image)

    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    mask = predict_mask(model, tensor)
    result = compute_thickness(mask, args.microns_per_pixel)
    write_csv(args.output, result)

    print(f"Mean RNFL thickness: {result.mean_microns:.2f} microns")
    print(f"Median RNFL thickness: {result.median_microns:.2f} microns")


if __name__ == "__main__":
    main()

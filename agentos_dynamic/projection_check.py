import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=ROOT)
ROOT = parser.parse_args().root.resolve()
for index in (0, 40, 79):
    rgb = cv2.imread(str(ROOT / "clip/front" / f"{index:05d}.png"))
    depth = np.load(ROOT / "clip/lidar_depth" / f"{index:05d}.npy")
    colors = cv2.applyColorMap((np.clip(depth / 5, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = rgb.copy()
    mask = depth > 0
    overlay[mask] = (rgb[mask] * .35 + colors[mask] * .65).astype(np.uint8)
    cv2.imwrite(str(ROOT / f"lidar_overlay_{index:05d}.jpg"), overlay)
    print(index, "anchor fraction", mask.mean(), "depth quantiles", np.quantile(depth[mask], [.1, .5, .9]))

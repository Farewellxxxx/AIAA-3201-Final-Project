#!/usr/bin/env python3
"""Run a reproducible StereoSGBM baseline on Middlebury Stereo v3 trainQ."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


SCENES = [
    "Adirondack", "ArtL", "Jadeplant", "Motorcycle", "MotorcycleE",
    "Piano", "PianoL", "Pipes", "Playroom", "Playtable", "PlaytableP",
    "Recycle", "Shelves", "Teddy", "Vintage",
]


def read_pfm(path: Path) -> np.ndarray:
    """Read a Middlebury PFM disparity map as float32, top row first."""
    with path.open("rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"Unsupported PFM header {header!r} in {path}")
        dims = f.readline().decode("ascii").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("ascii").strip()
        width, height = map(int, dims.split())
        scale = float(f.readline().decode("ascii").strip())
        dtype = "<f4" if scale < 0 else ">f4"
        channels = 3 if header == "PF" else 1
        data = np.fromfile(f, dtype=dtype)
    expected = width * height * channels
    if data.size != expected:
        raise ValueError(f"PFM size mismatch in {path}: expected {expected}, got {data.size}")
    shape = (height, width, channels) if channels == 3 else (height, width)
    image = np.flipud(data.reshape(shape)).astype(np.float32)
    # Middlebury's disp0GT.pfm files store disparity directly in pixel units;
    # the header magnitude is a PFM scale metadata value (not a multiplier for
    # these benchmark disparities). Its sign selects endianness above.
    return image


def read_calibration(path: Path) -> dict[str, float]:
    text = path.read_text()
    values: dict[str, float] = {}
    for key in ("doffs", "baseline", "ndisp", "vmin", "vmax", "width", "height"):
        match = re.search(rf"(?m)^\s*{key}\s*=\s*([-+\d.eE]+)", text)
        if match:
            values[key] = float(match.group(1))
    cam0 = re.search(r"cam0\s*=\s*\[([^\]]+)\]", text)
    if not cam0:
        raise ValueError(f"Missing cam0 matrix in {path}")
    first_row = cam0.group(1).split(";")[0].split()
    values["focal_length_px"] = float(first_row[0])
    required = ("doffs", "baseline", "ndisp", "vmin", "vmax", "width", "height")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing calibration values {missing} in {path}")
    return values


def color_disparity(disparity: np.ndarray, valid: np.ndarray,
                    low: float, high: float) -> np.ndarray:
    denom = max(high - low, 1e-6)
    gray = np.clip((disparity - low) / denom * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    return color


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/middlebury"))
    parser.add_argument("--output-root", type=Path, default=Path("results/stereo"))
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--uniqueness-ratio", type=int, default=10)
    parser.add_argument("--speckle-window-size", type=int, default=100)
    args = parser.parse_args()

    input_root = args.data_root / "MiddEval3-data-Q" / "MiddEval3" / "trainingQ"
    gt_root = args.data_root / "MiddEval3-GT0-Q" / "MiddEval3" / "trainingQ"
    out = args.output_root
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "visualizations").mkdir(parents=True, exist_ok=True)

    rows = []
    for scene in SCENES:
        inp = input_root / scene
        gt_dir = gt_root / scene
        left = cv2.imread(str(inp / "im0.png"), cv2.IMREAD_COLOR)
        right = cv2.imread(str(inp / "im1.png"), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise FileNotFoundError(f"Could not read stereo pair for {scene} under {inp}")
        calib = read_calibration(inp / "calib.txt")
        gt_disp = read_pfm(gt_dir / "disp0GT.pfm")
        if left.shape[:2] != right.shape[:2] or left.shape[:2] != gt_disp.shape[:2]:
            raise ValueError(
                f"Resolution mismatch for {scene}: left={left.shape[:2]}, "
                f"right={right.shape[:2]}, gt={gt_disp.shape[:2]}"
            )

        # Middlebury defines ndisp as the allowed search range 0..ndisp-1.
        # OpenCV requires numDisparities to be divisible by 16.
        num_disparities = int(math.ceil(calib["ndisp"] / 16.0) * 16)
        block_size = args.block_size if args.block_size % 2 else args.block_size + 1
        channels = 1
        p1 = 8 * channels * block_size**2
        p2 = 32 * channels * block_size**2
        matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=1,
            uniquenessRatio=args.uniqueness_ratio,
            speckleWindowSize=args.speckle_window_size,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        # OpenCV StereoSGBM returns signed fixed-point disparity with 4 fractional bits.
        pred_raw = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        pred_valid = np.isfinite(pred_raw) & (pred_raw > 0)
        pred_disp = pred_raw.copy()
        pred_disp[~pred_valid] = np.nan

        gt_valid = np.isfinite(gt_disp) & (gt_disp > 0)
        overlap = gt_valid & pred_valid
        coverage = float(overlap.sum() / max(gt_valid.sum(), 1))
        mae = float(np.abs(pred_disp[overlap] - gt_disp[overlap]).mean()) if overlap.any() else float("nan")

        depth_valid = pred_valid & ((pred_disp + calib["doffs"]) > 0)
        depth_m = np.full(pred_disp.shape, np.nan, dtype=np.float32)
        depth_m[depth_valid] = (
            calib["focal_length_px"] * calib["baseline"]
            / (pred_disp[depth_valid] + calib["doffs"]) / 1000.0
        )

        np.save(out / "predictions" / f"{scene}_disparity_px.npy", pred_disp)
        np.save(out / "predictions" / f"{scene}_depth_m.npy", depth_m)
        cv2.imwrite(str(out / "predictions" / f"{scene}_invalid.png"), (~pred_valid).astype(np.uint8) * 255)

        vis_low, vis_high = calib["vmin"], calib["vmax"]
        pred_color = color_disparity(pred_raw, pred_valid, vis_low, vis_high)
        gt_color = color_disparity(gt_disp, gt_valid, vis_low, vis_high)
        invalid_vis = np.zeros_like(left)
        invalid_vis[~pred_valid] = (0, 0, 255)  # invalid pixels shown red
        top = np.hstack((left, pred_color))
        bottom = np.hstack((gt_color, invalid_vis))
        panel = np.vstack((top, bottom))
        cv2.putText(panel, "Left RGB", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, "SGBM disparity (invalid=black)", (left.shape[1] + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, "Ground-truth disparity", (12, left.shape[0] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(panel, "Invalid prediction mask (red)", (left.shape[1] + 12, left.shape[0] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(out / "visualizations" / f"{scene}.png"), panel)

        rows.append({
            "scene": scene,
            "focal_length_px": calib["focal_length_px"],
            "baseline_mm": calib["baseline"],
            "doffs_px": calib["doffs"],
            "ndisp": int(calib["ndisp"]),
            "num_disparities": num_disparities,
            "valid_gt_pixels": int(gt_valid.sum()),
            "valid_prediction_coverage": coverage,
            "disparity_mae_px_on_valid_overlap": mae,
            "invalid_prediction_fraction_on_valid_gt": 1.0 - coverage,
        })
        print(f"{scene:12s} MAE={mae:.3f}px coverage={coverage:.3%} ndisp={num_disparities}")

    with (out / "per_scene.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "OpenCV StereoSGBM",
        "opencv_version": cv2.__version__,
        "scenes": len(rows),
        "parameters": {
            "min_disparity": 0,
            "num_disparities": "ceil(ndisp / 16) * 16 per scene",
            "block_size": block_size,
            "uniqueness_ratio": args.uniqueness_ratio,
            "speckle_window_size": args.speckle_window_size,
            "speckle_range": 2,
            "mode": "SGBM_3WAY",
            "invalid_prediction_rule": "non-finite or disparity <= 0 px",
            "metric_depth_rule": "f_px * baseline_mm / (disparity_px + doffs_px) / 1000",
            "disparity_mae_rule": "mean absolute error on finite positive GT and valid predicted disparity overlap",
        },
        "mean_disparity_mae_px_on_valid_overlap": float(np.nanmean([r["disparity_mae_px_on_valid_overlap"] for r in rows])),
        "mean_valid_prediction_coverage": float(np.mean([r["valid_prediction_coverage"] for r in rows])),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nMean MAE on valid overlap: {summary['mean_disparity_mae_px_on_valid_overlap']:.3f}px")
    print(f"Mean valid-prediction coverage: {summary['mean_valid_prediction_coverage']:.3%}")
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()

"""Shared video panels and metadata for relative-depth frame alignment."""

import hashlib
import json

import cv2
import numpy as np


def source_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_depth_metadata(path, digest, fps, count):
    metadata = {
        "source_sha256": digest,
        "fps": float(fps),
        "frame_indices": list(range(count)),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_depths(path, digest, fps, *, allow_legacy=False):
    with np.load(path, allow_pickle=False) as archive:
        depths = archive["depths"]
    if depths.ndim != 3 or not depths.size or depths.dtype != np.float32:
        raise ValueError(f"{path}: expected nonempty float32 depths (frames, height, width)")
    if not np.isfinite(depths).all():
        raise ValueError(f"{path}: depths contain nonfinite values")

    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        if allow_legacy:
            return depths, False
        raise ValueError(f"Missing metadata: {metadata_path}; see docs/team-handoff.md")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["source_sha256"] != digest:
        raise ValueError(f"{path}: source SHA256 differs from the supplied video")
    recorded_fps = float(metadata["fps"])
    if not np.isfinite(recorded_fps) or abs(recorded_fps - fps) > .01:
        raise ValueError(f"{path}: FPS differs from the supplied video")
    if metadata["frame_indices"] != list(range(len(depths))):
        raise ValueError(f"{path}: frame order must be 0 through N-1")
    return depths, True


def colorize(depth, limits):
    lo, hi = limits
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-8), 0, 1)
    colors = cv2.applyColorMap(np.round(normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)


def panel(image, label):
    bar = np.full((38, image.shape[1], 3), 22, dtype=np.uint8)
    cv2.putText(bar, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .56, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([bar, image], axis=0)


def comparison_row(rgb, dav2_colors, vda_colors):
    return np.concatenate([
        panel(rgb, "RGB input"),
        panel(dav2_colors, "Depth Anything V2-S | relative"),
        panel(vda_colors, "Video Depth Anything-S | relative"),
    ], axis=1)

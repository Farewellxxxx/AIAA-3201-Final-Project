"""Compose existing DA V2 and VDA depth arrays on the CPU."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import _bootstrap  # Add the repository root to the import path.
import cv2
import numpy as np

from depth_project.protocol import fixed_range
from depth_project.video_render import colorize, comparison_row, load_depths, source_sha256


def compose(source, dav2_path, vda_path, output, *, allow_legacy=False):
    if output.suffix.lower() != ".mp4" or output.resolve() == source.resolve():
        raise ValueError("Choose a separate .mp4 output path")
    source_digest = source_sha256(source)
    capture = cv2.VideoCapture(str(source))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Cannot read video FPS: {source}")
        dav2, dav2_verified = load_depths(dav2_path, source_digest, fps, allow_legacy=allow_legacy)
        vda, vda_verified = load_depths(vda_path, source_digest, fps, allow_legacy=allow_legacy)
        if dav2.shape != vda.shape:
            raise ValueError(f"Depth frame counts or dimensions differ: {dav2.shape} vs {vda.shape}")

        count, height, width = dav2.shape
        output_size = (3 * width, height + 38)
        if height % 2 or width % 2:
            raise ValueError("MP4 export requires even depth-map height and width")
        dav2_limits, vda_limits = fixed_range(dav2), fixed_range(vda)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as temp_dir:
            temp_video = Path(temp_dir) / "comparison.mp4"
            writer = cv2.VideoWriter(str(temp_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, output_size)
            try:
                if not writer.isOpened():
                    raise ValueError("Could not open the MP4 writer")
                for i, (image_depth, video_depth) in enumerate(zip(dav2, vda, strict=True)):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError(f"Source video ended before depth frame {i}")
                    source_h, source_w = frame.shape[:2]
                    if abs(source_w * height / source_h - width) > 2:
                        raise ValueError("Depth-map aspect ratio differs from the source video")
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (width, height))
                    row = comparison_row(rgb, colorize(image_depth, dav2_limits), colorize(video_depth, vda_limits))
                    writer.write(cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()

            rendered = cv2.VideoCapture(str(temp_video))
            decoded = 0
            try:
                if not rendered.isOpened() or abs(rendered.get(cv2.CAP_PROP_FPS) - fps) > .01:
                    raise ValueError("Could not verify the rendered video FPS")
                while True:
                    ok, frame = rendered.read()
                    if not ok:
                        break
                    if frame.shape[:2] != (output_size[1], output_size[0]):
                        raise ValueError("Rendered video dimensions differ")
                    decoded += 1
            finally:
                rendered.release()
            if decoded != count:
                raise ValueError(f"Expected {count} rendered frames, decoded {decoded}")
            temp_video.replace(output)
    finally:
        capture.release()

    return {
        "frames": count,
        "fps": fps,
        "width": output_size[0],
        "height": output_size[1],
        "source_sha256": source_digest,
        "metadata_verified": {"dav2": dav2_verified, "vda": vda_verified},
        "normalization": {"dav2": dav2_limits, "vda": vda_limits},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dav2", type=Path, required=True)
    parser.add_argument("--vda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-legacy", action="store_true",
                        help="Accept old caches without source, FPS and frame-order metadata")
    args = parser.parse_args()
    result = compose(args.source, args.dav2, args.vda, args.output, allow_legacy=args.allow_legacy)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

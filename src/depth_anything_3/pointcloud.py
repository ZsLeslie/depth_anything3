"""CLI: DA3 images/prediction NPZ -> PLY and pixel-linked point cloud NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from depth_anything_3.utils.pointcloud import depth_to_pointcloud, write_ply


def predict_images(args: argparse.Namespace) -> dict[str, np.ndarray]:
    # Keep conversion of existing predictions usable with NumPy alone.
    import torch
    from depth_anything_3.api import DepthAnything3

    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    root = args.images
    paths = (
        sorted(p for p in root.iterdir() if p.suffix.lower() in extensions)
        if root.is_dir() else [root]
    )
    if not paths or any(not p.is_file() or p.suffix.lower() not in extensions for p in paths):
        raise ValueError(f"No supported images found at {root}")
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable; check the GPU environment or use --prediction with an existing NPZ"
        )
    model = DepthAnything3.from_pretrained(args.model_dir).to(device)
    prediction = model.inference(
        [str(p) for p in paths], process_res=args.process_res,
        process_res_method="upper_bound_resize",
    )
    data = {"depth": prediction.depth, "is_metric": np.asarray(prediction.is_metric),
            "source_images": np.asarray([str(p.resolve()) for p in paths])}
    for key, value in {
        "image": prediction.processed_images, "conf": prediction.conf,
        "intrinsics": prediction.intrinsics, "extrinsics": prediction.extrinsics,
    }.items():
        if value is not None:
            data[key] = value
    return data


def export_prediction(data: dict[str, np.ndarray], args: argparse.Namespace) -> dict:
    if "depth" not in data or "intrinsics" not in data:
        raise ValueError(
            "Prediction requires depth and intrinsics; camera intrinsics are not guessed"
        )
    depth = np.asarray(data["depth"])
    if depth.ndim == 2:
        depth = depth[None]
    if depth.ndim != 3 or not depth.size:
        raise ValueError("depth must have shape (N, H, W) or (H, W)")
    n, h, w = depth.shape
    arrays = {}
    for key, tails in {
        "intrinsics": [(3, 3)], "extrinsics": [(3, 4), (4, 4)],
        "image": [(h, w, 3)], "conf": [(h, w)],
    }.items():
        if key not in data:
            continue
        value = np.asarray(data[key])
        if n == 1 and value.shape in tails:
            value = value[None]
        if value.shape not in [(n, *tail) for tail in tails]:
            raise ValueError(f"{key} shape {value.shape} does not match depth {depth.shape}")
        arrays[key] = value
    if args.coordinates == "world" and "extrinsics" not in arrays:
        raise ValueError("World coordinates require DA3 world-to-camera extrinsics")

    summary = {
        "coordinates": args.coordinates,
        "camera_axes": "x right, y down, z forward",
        "is_metric": bool(np.asarray(data["is_metric"]).item()) if "is_metric" in data else None,
        "depth_shape": list(depth.shape),
        "confidence_min": args.confidence_min, "max_depth": args.max_depth,
        "stride": args.stride, "frames": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cloud = depth_to_pointcloud(
            depth[i], arrays["intrinsics"][i],
            image=arrays["image"][i] if "image" in arrays else None,
            confidence=arrays["conf"][i] if "conf" in arrays else None,
            confidence_min=args.confidence_min, max_depth=args.max_depth, stride=args.stride,
            world_to_camera=arrays["extrinsics"][i] if args.coordinates == "world" else None,
        )
        stem = f"frame_{i:04d}"
        write_ply(args.output_dir / f"{stem}.ply", cloud["points"], cloud.get("colors"))
        np.savez_compressed(args.output_dir / f"{stem}.npz", **cloud,
                            coordinates=np.asarray(args.coordinates))
        count = len(cloud["points"])
        info = {"frame": i, "point_count": count, "ply": f"{stem}.ply", "npz": f"{stem}.npz"}
        if "source_images" in data:
            info["source_image"] = str(data["source_images"][i])
        summary["frames"].append(info)
        print(f"{stem}: {count}/{h * w} points -> {args.output_dir / (stem + '.ply')}")
        if count == 0:
            print("Warning: empty point cloud; check depth validity and filter thresholds.")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prediction", type=Path, help="DA3 npz/mini_npz results.npz")
    source.add_argument("--images", type=Path, help="Image file or directory; run DA3 first")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pointcloud"))
    parser.add_argument("--model-dir", default="depth-anything/DA3-BASE")
    parser.add_argument("--device", default="cuda", help="Inference device (default: cuda)")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--coordinates", choices=("camera", "world"), default="camera")
    parser.add_argument(
        "--confidence-min", type=float, help="Raw DA3 confidence threshold, not a probability"
    )
    parser.add_argument("--max-depth", type=float, help="Maximum camera z in prediction units")
    parser.add_argument(
        "--stride", type=int, default=1, help="Sample every Nth pixel in each axis"
    )
    args = parser.parse_args()
    if args.process_res < 14 or args.stride < 1:
        parser.error("process-res must be >= 14 and stride must be >= 1")
    try:
        if args.prediction is not None:
            with np.load(args.prediction, allow_pickle=False) as archive:
                data = dict(archive)
        else:
            data = predict_images(args)
        export_prediction(data, args)
        if args.images is not None:
            np.savez_compressed(args.output_dir / "prediction.npz", **data)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

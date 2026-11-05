"""CLI: DA3 images/prediction NPZ -> PLY and pixel-linked point cloud NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from depth_anything_3.utils.pointcloud import depth_to_pointcloud, write_ply


def load_intrinsics(path: Path) -> np.ndarray:
    """Read a shared or per-frame pinhole K from JSON or a NumPy array."""
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, dict):
            value = value.get("intrinsics", value.get("K"))
    elif path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
    else:
        raise ValueError("intrinsics must be a .json or .npy file")
    try:
        matrices = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("intrinsics must contain numeric 3x3 matrices") from exc
    if matrices.shape == (3, 3):
        batch = matrices[None]
    elif matrices.ndim == 3 and matrices.shape[1:] == (3, 3) and len(matrices):
        batch = matrices
    else:
        raise ValueError("intrinsics must have shape (3, 3) or (N, 3, 3)")
    if not np.isfinite(batch).all():
        raise ValueError("intrinsics must be finite")
    if (not np.allclose(batch[:, 2], [0, 0, 1])
            or np.any(batch[:, 0, 0] <= 0) or np.any(batch[:, 1, 1] <= 0)):
        raise ValueError("intrinsics must be pinhole matrices with positive focal lengths")
    if np.any(np.linalg.slogdet(batch)[0] == 0):
        raise ValueError("intrinsics must be invertible")
    return matrices


def intrinsics_for_frames(intrinsics: np.ndarray, count: int) -> np.ndarray:
    matrices = intrinsics[None] if intrinsics.ndim == 2 else intrinsics
    if len(matrices) == 1:
        return np.repeat(matrices, count, axis=0)
    if len(matrices) != count:
        raise ValueError(f"intrinsics contains {len(matrices)} matrices for {count} frames")
    return matrices.copy()


def resize_intrinsics(intrinsics: np.ndarray, source_size, target_size) -> np.ndarray:
    """Map K between image resolutions for resize-only transforms (sizes are W,H)."""
    result = intrinsics.copy()
    sx, sy = np.asarray(target_size, dtype=float) / source_size
    result[..., 0, :] *= sx
    result[..., 1, :] *= sy
    return result


def predict_images(
    args: argparse.Namespace, intrinsics: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    initialization_start = perf_counter()
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
    inference_kwargs = {}
    if intrinsics is not None:
        matrices = intrinsics_for_frames(intrinsics, len(paths))
        if args.intrinsics_size is not None:
            from PIL import Image
            for i, path in enumerate(paths):
                with Image.open(path) as image:
                    matrices[i] = resize_intrinsics(
                        matrices[i], args.intrinsics_size, image.size,
                    )
        # The API applies the same resize and batch center-crop to images and K.
        inference_kwargs["intrinsics"] = matrices
    device = args.device
    use_cuda = device.startswith("cuda")
    if use_cuda and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable; check the GPU environment or use --prediction with an existing NPZ"
        )
    print(
        "Prediction initialization (imports, input discovery, device check): "
        f"{perf_counter() - initialization_start:.3f} s"
    )
    load_start = perf_counter()
    model = DepthAnything3.from_pretrained(args.model_dir).to(device)
    if use_cuda:
        torch.cuda.synchronize(device)
    print(f"Model loading ({device}): {perf_counter() - load_start:.3f} s")

    inference_start = perf_counter()
    prediction = model.inference(
        [str(p) for p in paths], process_res=args.process_res,
        process_res_method="upper_bound_resize",
        **inference_kwargs,
    )
    if use_cuda:
        # CUDA work is asynchronous; wait for completion before recording elapsed time.
        torch.cuda.synchronize(device)
    print(f"Inference (including pre/postprocessing): {perf_counter() - inference_start:.3f} s")
    finalization_start = perf_counter()
    data = {"depth": prediction.depth, "is_metric": np.asarray(prediction.is_metric),
            "source_images": np.asarray([str(p.resolve()) for p in paths])}
    for key, value in {
        "image": prediction.processed_images, "conf": prediction.conf,
        "intrinsics": prediction.intrinsics, "extrinsics": prediction.extrinsics,
    }.items():
        if value is not None:
            data[key] = value
    # Releasing model parameters also takes time and used to be absent from stage timings.
    del prediction, model
    print(f"Prediction packing and model cleanup: {perf_counter() - finalization_start:.3f} s")
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
        conversion_start = perf_counter()
        cloud = depth_to_pointcloud(
            depth[i], arrays["intrinsics"][i],
            image=arrays["image"][i] if "image" in arrays else None,
            confidence=arrays["conf"][i] if "conf" in arrays else None,
            confidence_min=args.confidence_min, max_depth=args.max_depth, stride=args.stride,
            world_to_camera=arrays["extrinsics"][i] if args.coordinates == "world" else None,
        )
        conversion_seconds = perf_counter() - conversion_start
        save_start = perf_counter()
        stem = f"frame_{i:04d}"
        write_ply(args.output_dir / f"{stem}.ply", cloud["points"], cloud.get("colors"))
        np.savez_compressed(args.output_dir / f"{stem}.npz", **cloud,
                            coordinates=np.asarray(args.coordinates))
        save_seconds = perf_counter() - save_start
        count = len(cloud["points"])
        info = {"frame": i, "point_count": count, "ply": f"{stem}.ply", "npz": f"{stem}.npz"}
        if "source_images" in data:
            info["source_image"] = str(data["source_images"][i])
        summary["frames"].append(info)
        print(
            f"{stem}: {count}/{h * w} points -> {args.output_dir / (stem + '.ply')} "
            f"(conversion: {conversion_seconds:.3f} s, saving: {save_seconds:.3f} s)"
        )
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
    parser.add_argument(
        "--intrinsics", type=Path,
        help="Custom K: JSON/NPY (3,3) or (N,3,3); defaults to original-image pixels "
             "with --images, depth-image pixels with --prediction",
    )
    parser.add_argument(
        "--intrinsics-size", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"),
        help="Calibration image size for custom K; rescale K from this resolution",
    )
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
    if args.intrinsics_size is not None:
        if args.intrinsics is None:
            parser.error("intrinsics-size requires --intrinsics")
        if min(args.intrinsics_size) < 1:
            parser.error("intrinsics-size width and height must be positive")
    total_start = perf_counter()
    try:
        intrinsics = load_intrinsics(args.intrinsics) if args.intrinsics is not None else None
        if args.prediction is not None:
            with np.load(args.prediction, allow_pickle=False) as archive:
                data = dict(archive)
            if intrinsics is not None:
                if "depth" not in data:
                    raise ValueError("Prediction requires depth")
                depth = np.asarray(data["depth"])
                if depth.ndim not in (2, 3) or not depth.size:
                    raise ValueError("depth must have shape (N, H, W) or (H, W)")
                matrices = intrinsics_for_frames(intrinsics, 1 if depth.ndim == 2 else len(depth))
                if args.intrinsics_size is not None:
                    matrices = resize_intrinsics(
                        matrices, args.intrinsics_size, depth.shape[-2:][::-1],
                    )
                data["intrinsics"] = matrices
            source_seconds = perf_counter() - total_start
            print(f"Prediction loading: {source_seconds:.3f} s")
        else:
            data = predict_images(args, intrinsics)
            source_seconds = perf_counter() - total_start
            print(f"Image prediction total (includes stages above): {source_seconds:.3f} s")
        export_start = perf_counter()
        export_prediction(data, args)
        export_seconds = perf_counter() - export_start
        print(f"Point cloud export (includes per-frame conversion/saving): {export_seconds:.3f} s")
        prediction_save_seconds = 0.0
        if args.images is not None:
            save_start = perf_counter()
            np.savez_compressed(args.output_dir / "prediction.npz", **data)
            prediction_save_seconds = perf_counter() - save_start
            print(f"Prediction saving: {prediction_save_seconds:.3f} s")
        total_seconds = perf_counter() - total_start
        # Top-level stages partition the total; nested timings must not be added again.
        other_seconds = total_seconds - source_seconds - export_seconds - prediction_save_seconds
        print(f"Other overhead (logging/control flow): {other_seconds:.3f} s")
        print(f"Total processing time: {total_seconds:.3f} s")
        print(
            "  = input/prediction + point cloud export + prediction saving + other overhead "
            "(subtotals above are included; values rounded to 0.001 s)"
        )
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

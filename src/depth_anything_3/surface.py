"""Fit a visible surface from a pixel-linked DA3 point cloud; no segmentation needed."""

from __future__ import annotations

from time import perf_counter

# Include dependency imports in the CLI's total execution time.
_SCRIPT_START = perf_counter()

import argparse
import json
from pathlib import Path
import numpy as np

from depth_anything_3.utils.pointcloud import write_ply
from depth_anything_3.utils.surface import fit_surface, triangulate_surface


def load_cloud(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    required = {"points", "pixels_uv", "intrinsics", "valid_mask", "coordinates"}
    if missing := required - data.keys():
        raise ValueError(f"Point cloud is missing fields: {sorted(missing)}")
    if data["valid_mask"].ndim != 2:
        raise ValueError("valid_mask must have shape (H, W)")
    coordinates = str(data["coordinates"].item())
    if coordinates == "world":
        if "world_to_camera" not in data:
            raise ValueError("World-coordinate input requires world_to_camera")
        ext = np.asarray(data["world_to_camera"], float)
        if ext.shape not in ((3, 4), (4, 4)) or not np.isfinite(ext).all():
            raise ValueError("world_to_camera must be a finite (3, 4) or (4, 4) matrix")
        data["points"] = data["points"] @ ext[:3, :3].T + ext[:3, 3]
    elif coordinates != "camera":
        raise ValueError(f"Unknown coordinate system: {coordinates}")
    if "colors" in data and (
        data["colors"].shape != data["points"].shape or data["colors"].dtype != np.uint8
    ):
        raise ValueError("colors must be uint8 RGB with shape (N, 3)")
    return data


def save_diagnostics(output: Path, arrays: dict, report: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    h, w = arrays["image_shape"]
    u, v = arrays["pixels_uv"].T

    def raster(values, fill=np.nan):
        result = np.full((h, w, *values.shape[1:]), fill, dtype=float)
        result[v, u] = values
        return result

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    if "colors" in arrays:
        axes[0, 0].imshow(raster(arrays["colors"] / 255, 1))
    else:
        axes[0, 0].imshow(raster(arrays["eligible"].astype(float)), vmin=0, vmax=1)
    axes[0, 0].set_title("Input RGB / observed support")
    depth = arrays["points"][:, 2]
    lo, hi = np.percentile(depth[arrays["eligible"]], [1, 99])
    for ax, values, title in (
        (axes[0, 1], depth, "DA3 depth (prediction units)"),
        (axes[0, 2], arrays["fitted_points"][:, 2], "Fitted surface depth (same scale)"),
    ):
        im = ax.imshow(raster(values), vmin=lo, vmax=hi, cmap="viridis")
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)
    im = axes[1, 0].imshow(raster(100 * arrays["relative_depth_residual"]),
                           cmap="magma", vmin=0, vmax=100 * report["relative_threshold"])
    fig.colorbar(im, ax=axes[1, 0], shrink=0.8, label="% of DA3 z depth")
    axes[1, 0].set_title("Absolute relative depth residual")
    axes[1, 1].imshow(raster((arrays["normals"] + 1) / 2, 1))
    axes[1, 1].set_title("Analytic camera-facing normals (XYZ to RGB)")
    colors = np.full((len(u), 3), [0.65, 0.65, 0.65])
    colors[arrays["eligible"]] = [0.9, 0.2, 0.15]
    colors[arrays["inlier"]] = [0.15, 0.45, 0.85]
    colors[arrays["inlier"] & arrays["validation"]] = [0.15, 0.8, 0.55]
    axes[1, 2].imshow(raster(colors, 1))
    axes[1, 2].set_title("Fit support (not physical surface labels)")
    axes[1, 2].legend(handles=[
        Patch(color=[0.15, 0.45, 0.85], label="inlier"),
        Patch(color=[0.15, 0.8, 0.55], label="held-out inlier"),
        Patch(color=[0.9, 0.2, 0.15], label="residual rejected"),
        Patch(color=[0.65, 0.65, 0.65], label="excluded"),
    ], fontsize=8, loc="lower right")
    for ax in axes.ravel():
        ax.set_xlabel("processed image u")
        ax.set_ylabel("v")
    fig.savefig(output / "diagnostics.png", dpi=150)
    plt.close(fig)

    faces = arrays["faces"]
    if not len(faces):
        return
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for index, (title, key) in enumerate([
        ("Observed DA3 geometry", "points"), ("Fitted visible surface", "fitted_points"),
    ], 1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        triangles = arrays[key][faces]
        collection = Poly3DCollection(triangles, linewidths=0, rasterized=True)
        if "colors" in arrays:
            collection.set_facecolor(arrays["colors"][faces].mean(axis=1) / 255)
        else:
            collection.set_facecolor("steelblue")
        ax.add_collection3d(collection)
        bounds = arrays["points"][arrays["inlier"]]
        lower, upper = bounds.min(axis=0), bounds.max(axis=0)
        ax.set(xlim=(lower[0], upper[0]), ylim=(upper[1], lower[1]),
               zlim=(lower[2], upper[2]), xlabel="X", ylabel="Y", zlabel="Z", title=title)
        ax.set_box_aspect(np.maximum(upper - lower, 1e-6))
        ax.view_init(elev=20, azim=-65)
    fig.savefig(output / "surface_3d.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcloud", type=Path, required=True, help="Step-one frame_XXXX.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/surface"))
    parser.add_argument("--controls-u", type=int, default=16)
    parser.add_argument("--controls-v", type=int, default=10)
    parser.add_argument("--smoothness", type=float, default=0.05)
    parser.add_argument("--relative-threshold", type=float, default=0.05)
    parser.add_argument("--confidence-min", type=float)
    parser.add_argument("--max-samples", type=int, default=16000)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--validation-block", type=int, default=8)
    parser.add_argument("--mesh-step", type=int, default=3)
    parser.add_argument("--max-relative-jump", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    try:
        data = load_cloud(args.pointcloud)
        model, arrays, report = fit_surface(
            data["points"], data["pixels_uv"], data["intrinsics"], data["valid_mask"].shape,
            confidence=data.get("confidence"), confidence_min=args.confidence_min,
            controls=(args.controls_u, args.controls_v), smoothness=args.smoothness,
            max_samples=args.max_samples, iterations=args.iterations,
            relative_threshold=args.relative_threshold, validation_block=args.validation_block,
            seed=args.seed,
        )
        if "colors" in data:
            arrays["colors"] = data["colors"]
        if "confidence" in data:
            arrays["confidence"] = data["confidence"]
        arrays["faces"] = triangulate_surface(
            arrays, step=args.mesh_step, max_relative_jump=args.max_relative_jump,
        )
        report.update({"input": str(args.pointcloud.resolve()), "coordinates": "camera",
                       "mesh_step": args.mesh_step, "max_relative_jump": args.max_relative_jump,
                       "face_count": len(arrays["faces"]),
                       "units": "input prediction units; no metric scale inferred"})
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        for name, payload in (("model.json", model), ("report.json", report)):
            (output / name).write_text(
                json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
            )
        np.savez_compressed(output / "surface.npz", **arrays, coordinates=np.asarray("camera"))
        keep = arrays["inlier"]
        colors = arrays["colors"][keep] if "colors" in arrays else None
        write_ply(output / "surface_points.ply", arrays["fitted_points"][keep], colors)
        used = np.unique(arrays["faces"])
        remap = np.full(len(keep), -1, dtype=np.int32)
        remap[used] = np.arange(len(used))
        write_ply(output / "surface_mesh.ply", arrays["fitted_points"][used],
                  arrays["colors"][used] if "colors" in arrays else None,
                  faces=remap[arrays["faces"]])
        if not args.no_plots:
            save_diagnostics(output, arrays, report)
        print(json.dumps({"inliers": int(keep.sum()), "faces": len(arrays["faces"]),
                          "validation": report["validation"], "output": str(output)}, indent=2))
        if not keep.any() or not len(arrays["faces"]):
            print("Warning: no supported surface mesh; inspect report and point coverage.")
        if not report["converged"]:
            print(
                "Warning: IRLS iteration limit reached; inspect the fit or increase --iterations."
            )
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    try:
        main()
    finally:
        print(f"Total execution time: {perf_counter() - _SCRIPT_START:.3f} s")

"""Estimate and visually validate a straight structure-axis candidate from step two."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from depth_anything_3.utils.axis import (
    axis_basis, camera_z_axis, direction_from_parallel_lines, estimate_axis, mesh_sections,
    section_diagnostics, unit_direction,
)
from depth_anything_3.utils.pointcloud import write_ply


def load_surface(path):
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    required = {"points", "fitted_points", "normals", "pixels_uv", "intrinsics",
                "image_shape", "inlier", "validation", "faces", "coordinates"}
    if missing := required - data.keys():
        raise ValueError(f"Surface archive is missing {sorted(missing)}")
    if str(data["coordinates"].item()) != "camera":
        raise ValueError("Expected camera-frame output from step-two surface fitting")
    n = len(data["points"])
    for key in ("points", "fitted_points", "normals"):
        if data[key].shape != (n, 3):
            raise ValueError(f"Invalid shape for {key}")
    for key in ("inlier", "validation"):
        if data[key].shape != (n,) or data[key].dtype.kind != "b":
            raise ValueError(f"{key} must be a boolean array of length N")
    if data["image_shape"].shape != (2,) or np.any(data["image_shape"] < 2):
        raise ValueError("image_shape must contain H,W >= 2")
    h, w = data["image_shape"]
    uv = data["pixels_uv"]
    if uv.shape != (n, 2) or uv.dtype.kind not in "iu":
        raise ValueError("pixels_uv must have integer shape (N,2)")
    if np.any(uv < 0) or np.any(uv >= [w, h]):
        raise ValueError("pixels_uv falls outside image_shape")
    K = data["intrinsics"]
    if K.shape != (3, 3) or not np.isfinite(K).all() or abs(np.linalg.det(K)) < 1e-12:
        raise ValueError("intrinsics must be a finite invertible 3x3 matrix")
    if not np.allclose(K[2], [0, 0, 1]) or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Expected pinhole intrinsics with positive focal lengths")
    selected = data["inlier"]
    if not selected.any():
        raise ValueError("Surface has no inlier points")
    for key in ("points", "fitted_points"):
        points = data[key][selected]
        if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
            raise ValueError(f"Inlier {key} must have finite positive camera depth")
        hom = points @ K.T
        if np.max(np.abs(hom[:, :2] / hom[:, 2:] - uv[selected])) > 0.05:
            raise ValueError(f"{key} does not match pixels and intrinsics")
    faces = data["faces"]
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("faces must be integer triangle indices")
    if not len(faces) or np.any(faces < 0) or np.any(faces >= n):
        raise ValueError("A nonempty supported mesh with valid face indices is required")
    if not data["inlier"][faces].all():
        raise ValueError("Mesh faces must only reference inlier surface points")
    if "colors" in data and (data["colors"].shape != (n, 3)
                              or data["colors"].dtype != np.uint8):
        raise ValueError("colors must be uint8 RGB with shape (N,3)")
    return data


def image_from_surface(data):
    h, w = data["image_shape"]
    rgb = np.full((h, w, 3), 240, dtype=np.uint8)
    u, v = data["pixels_uv"].T
    if "colors" in data:
        rgb[v, u] = data["colors"]
    else:
        rgb[v, u] = [140, 150, 160]
    return rgb


def annotate_lines(data, path):
    """Optional desktop Matplotlib point picker; not invoked during headless runs."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(image_from_surface(data))
    lines, current, artists = [], [], []
    saved = []

    def redraw(message=""):
        for artist in artists:
            artist.remove()
        artists.clear()
        for i, line in enumerate(lines + ([current] if current else [])):
            uv = np.asarray(line)
            artists.extend(ax.plot(uv[:, 0], uv[:, 1], "o-", label=f"line {i + 1}"))
        ax.set_title("Mark physically parallel straight seams only\n"
                     "Click: add | Enter: finish line | Backspace: undo | S: save | Esc: cancel\n"
                     + message)
        fig.canvas.draw_idle()

    def click(event):
        if event.inaxes is ax and event.button == 1 and event.xdata is not None:
            current.append([float(event.xdata), float(event.ydata)])
            redraw()

    def key(event):
        if event.key == "enter":
            if len(current) >= 2:
                lines.append(current.copy())
                current.clear()
                redraw()
        elif event.key == "backspace":
            if current:
                current.pop()
            elif lines:
                current.extend(lines.pop())
            redraw()
        elif event.key == "escape":
            plt.close(fig)
        elif event.key in ("s", "S"):
            complete = lines + ([current.copy()] if len(current) >= 2 else [])
            annotation = {"image_size": data["image_shape"][::-1].tolist(),
                          "parallel_lines": complete}
            try:
                direction_from_parallel_lines(annotation, data["intrinsics"], data["image_shape"])
            except ValueError as exc:
                redraw(str(exc))
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
            saved.append(annotation)
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", click)
    fig.canvas.mpl_connect("key_press_event", key)
    redraw()
    plt.show()
    if not saved:
        raise ValueError("Annotation cancelled; no axis estimation was run")
    return saved[0]


def project(points, K):
    points = np.asarray(points)
    output = np.full((*points.shape[:-1], 2), np.nan)
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 1e-8)
    hom = points[valid] @ K.T
    output[valid] = hom[:, :2] / hom[:, 2:]
    return output


def save_visualizations(output, data, axis, report, geometry):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    section_colors = plt.get_cmap("turbo")(np.linspace(0.08, 0.92, len(geometry["levels"])))
    seg = geometry["segments"]
    ids = geometry["section_ids"]
    axis_uv = project(geometry["axis_line"], data["intrinsics"])
    segments_uv = project(seg, data["intrinsics"])
    visible = np.isfinite(segments_uv).all(axis=(1, 2))
    h, w = data["image_shape"]

    def overlay(ax):
        ax.imshow(image_from_surface(data))
        ax.add_collection(LineCollection(segments_uv[visible], colors=section_colors[ids[visible]],
                                         linewidths=1.6))
        fixed_z = axis.get("axis_source") == "camera_z"
        if not fixed_z:
            ax.plot(axis_uv[:, 0], axis_uv[:, 1], "--", color="#ff42e0", linewidth=2,
                    label="candidate axis (through empty space)")
        a = np.asarray(axis["direction"])
        vp = data["intrinsics"] @ a
        vp = vp[:2] / vp[2]
        if 0 <= vp[0] < w and 0 <= vp[1] < h:
            ax.plot(*vp, marker="x", color="#ff42e0" if fixed_z else "cyan", markersize=9,
                    label="camera Z axis / principal point" if fixed_z else "direction vanishing point (model)")
        for line in report.get("annotations", {}).get("processed_lines", []):
            uv = np.asarray(line)
            ax.plot(uv[:, 0], uv[:, 1], "o-", color="yellow", linewidth=1)
        ax.set(xlim=(0, w - 1), ylim=(h - 1, 0), xlabel="processed image u", ylabel="v",
               title="Fixed camera Z axis: constant-depth sections" if fixed_z else "Candidate axis and supported cross-sections")
        ax.legend(fontsize=8, loc="lower left")

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    overlay(ax)
    fig.savefig(output / "axis_overlay.png", dpi=150)
    plt.close(fig)
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
    overlay(fig.add_subplot(2, 2, 1))
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    triangles = data["fitted_points"][data["faces"]]
    mesh = Poly3DCollection(triangles, linewidths=0, alpha=0.55, rasterized=True)
    if "colors" in data:
        mesh.set_facecolor(data["colors"][data["faces"]].mean(axis=1) / 255)
    else:
        mesh.set_facecolor("silver")
    ax.add_collection3d(mesh)
    ax.add_collection3d(Line3DCollection(seg, colors=section_colors[ids], linewidths=2))
    line = geometry["axis_line"]
    ax.plot(*line.T, "--", color="#ff42e0", linewidth=2, label="candidate axis")
    centers = np.asarray(axis["point"]) + geometry["levels"][:, None] * axis["direction"]
    ax.scatter(*centers.T, color=section_colors, s=20)
    bounds = np.vstack((data["fitted_points"][data["inlier"]], line))
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    ax.set(xlim=(lower[0], upper[0]), ylim=(upper[1], lower[1]), zlim=(lower[2], upper[2]),
           xlabel="X", ylabel="Y", zlabel="Z", title="Axis / surface intersection in camera coordinates")
    ax.set_box_aspect(np.maximum(upper - lower, 1e-6))
    ax.view_init(elev=22, azim=-65)
    ax = fig.add_subplot(2, 2, 3)
    e1, e2 = axis_basis(axis["direction"])
    center = np.asarray(axis["point"])
    transverse = np.stack(((seg - center) @ e1, (seg - center) @ e2), axis=-1)
    ax.add_collection(LineCollection(transverse, colors=section_colors[ids], linewidths=1.2))
    ax.plot(0, 0, "+", color="black", markersize=12, label="axis center")
    for item in report["sections"]:
        if item["circle_reliable"]:
            delta = np.asarray(item["circle_center"]) - center
            ax.plot(delta @ e1, delta @ e2, "x", color=section_colors[item["section"]])
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set(xlabel="transverse coordinate 1", ylabel="transverse coordinate 2",
           title="Observed cross-sections; crosses: supported circle centers")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    stats, stability = report["selected_validation"], report["stability"]
    residual_text = (
        f"Median / P90 model residual: {100 * stats['median_error_over_radius']:.1f}% / "
        f"{100 * stats['p90_error_over_radius']:.1f}% of radius"
        if stats is not None else "Model residual: not assessed (no axis fitting)"
    )
    text = [f"Status: {report['status']} (physical axis not verified)",
            f"Selected hypothesis: {axis['candidate_model']}",
            f"Direction: {np.round(axis['direction'], 4)}",
            f"Reference point: {np.round(axis['point'], 4)}",
            residual_text,
            f"Resamples: {stability['successful']}/{stability['requested']}",
            "P95 direction change: " + (
                f"{stability['direction_p95_degrees']:.3f} deg"
                if stability['direction_p95_degrees'] is not None else "not assessed"
            ),
            "", "Review notes:"]
    import textwrap
    for warning in report["warnings"]:
        text.extend(textwrap.wrap("- " + warning, width=65))
    ax.text(0, 1, "\n".join(text), va="top", fontsize=10, family="monospace")
    fig.savefig(output / "axis_validation.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, default=Path("outputs/surface/surface.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/axis"))
    parser.add_argument("--direction", nargs=3, type=float, metavar=("AX", "AY", "AZ"))
    parser.add_argument("--axis-point", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--axis-json", type=Path, help="Load point/direction as fixed manual constraints")
    parser.add_argument("--camera-z-axis", action="store_true",
                        help="Use the fixed camera axis x=y=0, direction [0,0,1]; skip axis fitting")
    annotations = parser.add_mutually_exclusive_group()
    annotations.add_argument("--annotations", type=Path, help="JSON of physically parallel lines")
    annotations.add_argument("--annotate", action="store_true", help="Open desktop line point picker")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--bootstrap", type=int, default=12)
    parser.add_argument("--sections", type=int, default=5)
    parser.add_argument("--normal-weight", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.sections < 2 or args.sections > 30:
        parser.error("sections must be between 2 and 30")
    if (args.annotations or args.annotate) and (args.direction or args.axis_json):
        parser.error("Use annotations OR explicit direction/axis-json, not both")
    if args.camera_z_axis and (args.direction or args.axis_point or args.axis_json or args.annotations or args.annotate):
        parser.error("camera-z-axis cannot be combined with manual axis or reference-line options")
    try:
        data = load_surface(args.surface)
        direction, point = args.direction, args.axis_point
        if args.axis_json:
            manual = json.loads(args.axis_json.read_text())
            if manual.get("coordinates") != "camera":
                raise ValueError("Manual axis JSON must use camera coordinates")
            direction = direction if direction is not None else manual["direction"]
            point = point if point is not None else manual["point"]
        annotation = None
        if args.annotate:
            annotation = annotate_lines(data, args.output_dir / "annotations.json")
        elif args.annotations:
            annotation = json.loads(args.annotations.read_text())
        annotation_report = None
        if annotation is not None:
            direction, annotation_report = direction_from_parallel_lines(
                annotation, data["intrinsics"], data["image_shape"],
            )
        keep = data["inlier"]
        if args.camera_z_axis:
            axis, report, bootstrap_axes = camera_z_axis(data["fitted_points"][np.unique(data["faces"])])
        else:
            axis, report, bootstrap_axes = estimate_axis(
                data["points"][keep], data["normals"][keep], validation=data["validation"][keep],
                max_samples=args.max_samples, bootstrap=args.bootstrap, seed=args.seed,
                direction=direction, axis_point=point, normal_weight=args.normal_weight,
            )
        a, c = unit_direction(axis["direction"]), np.asarray(axis["point"])
        # Only the supported mesh contributes to section placement.
        used = np.unique(data["faces"])
        axial = (data["fitted_points"][used] - c) @ a
        lo, hi = np.percentile(axial, [12, 88])
        if hi - lo < 1e-8:
            raise ValueError("Mesh has no usable axial span")
        levels = np.linspace(lo, hi, args.sections)
        segments, section_ids = mesh_sections(data["fitted_points"], data["faces"], axis, levels)
        sections = section_diagnostics(segments, section_ids, axis, levels)
        report["sections"] = sections
        reliable = [s for s in sections if s["circle_reliable"]]
        if len(reliable) < max(2, (args.sections + 1) // 2):
            report["warnings"].append("Too few well-supported circular cross-sections to verify center position")
        if reliable and np.median([s["center_offset_over_radius"] for s in reliable]) > 0.15:
            report["warnings"].append("Supported section centers do not agree with the proposed axis")
        if annotation_report:
            report["annotations"] = annotation_report
            if max(annotation_report["line_plane_errors_degrees"]) > 3:
                report["warnings"].append("Annotated parallel lines disagree about vanishing direction")
        if report["warnings"]:
            report["status"] = "needs_review"
        report.update({"surface": str(args.surface.resolve()),
                       "residual_units": None if args.camera_z_axis else "geometric distance divided by median model radius",
                       "scope": "Diagnostic axis and sections; not a final inspection grid"})
        axis.update({"s_range": [float(lo), float(hi)], "section_levels": levels.tolist(),
                     "status": report["status"], "accepted_as_physical_axis": False,
                     "intrinsics": data["intrinsics"].tolist(),
                     "image_shape": data["image_shape"].tolist()})
        if args.camera_z_axis:
            axis["section_depths"] = (c[2] + levels).tolist()
        axis_line = c + np.linspace(lo, hi, 200)[:, None] * a
        geometry = {"segments": segments, "section_ids": section_ids, "levels": levels,
                    "axis_line": axis_line, "bootstrap_axes": bootstrap_axes,
                    "segments_uv": project(segments, data["intrinsics"])}
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        for name, payload in (("axis.json", axis), ("report.json", report)):
            (out / name).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        template = {"image_size": data["image_shape"][::-1].tolist(), "parallel_lines": [],
                    "instructions": "Add >=2 physically parallel straight lines, each with >=2 [u,v] points"}
        (out / "annotations.example.json").write_text(json.dumps(template, indent=2), encoding="utf-8")
        if annotation is not None:
            (out / "annotations.json").write_text(json.dumps(annotation, indent=2), encoding="utf-8")
        np.savez_compressed(out / "sections.npz", **geometry, coordinates=np.asarray("camera"))
        write_ply(out / "axis_points.ply", axis_line,
                  np.tile(np.array([255, 60, 220], np.uint8), (len(axis_line), 1)))
        write_ply(out / "section_points.ply", segments.reshape(-1, 3))
        if not args.no_plots:
            save_visualizations(out, data, axis, report, geometry)
        print(json.dumps({"status": report["status"], "axis": axis,
                          "validation": report["selected_validation"],
                          "stability": report["stability"], "warnings": report["warnings"],
                          "output": str(out)}, indent=2))
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

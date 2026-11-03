"""Preview section/grid curves on a fitted surface or an unmodified DA3 depth mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from depth_anything_3.axis import image_from_surface, load_surface, project
from depth_anything_3.utils.grid import build_grid, pack_paths, visible_fragments
from depth_anything_3.utils.pointcloud import write_ply


def load_raw_mesh(path, *, max_relative_jump=0.1, confidence_min=None):
    """Build adjacent-pixel triangles without fitting or smoothing raw vertices.

    fitted_points/inlier are compatibility keys for the shared renderer: here
    they mean raw camera points/observed validity, never spline fit or residuals.
    """
    from depth_anything_3.surface import load_cloud
    from depth_anything_3.utils.surface import triangulate_surface

    data = load_cloud(path)
    points, uv = data["points"], data["pixels_uv"]
    h, w = data["valid_mask"].shape
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if uv.shape != (len(points), 2) or uv.dtype.kind not in "iu":
        raise ValueError("pixels_uv must be integer (N,2)")
    if np.any(uv < 0) or np.any(uv >= [w, h]):
        raise ValueError("pixels_uv falls outside valid_mask")
    if len(np.unique(uv[:, 1]*w + uv[:, 0])) != len(uv):
        raise ValueError("Duplicate pixels in point cloud")
    K = data["intrinsics"]
    if (K.shape != (3, 3) or not np.isfinite(K).all() or
            not np.allclose(K[2], [0, 0, 1]) or min(K[0, 0], K[1, 1]) <= 0 or
            abs(np.linalg.det(K)) < 1e-12):
        raise ValueError("Expected finite invertible pinhole intrinsics")
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    valid &= data["valid_mask"][uv[:, 1], uv[:, 0]].astype(bool)
    if confidence_min is not None:
        if not np.isfinite(confidence_min) or "confidence" not in data:
            raise ValueError("confidence-min requires a finite threshold and saved confidence")
        confidence = data["confidence"]
        if confidence.shape != (len(points),):
            raise ValueError("confidence must have shape (N,)")
        valid &= np.isfinite(confidence) & (confidence >= confidence_min)
    if not valid.any():
        raise ValueError("Point cloud has no valid camera-depth observations")
    if np.max(np.abs(project(points[valid], K) - uv[valid])) > 0.05:
        raise ValueError("Point cloud does not match pixels and intrinsics")
    data.update(image_shape=np.array([h, w]), fitted_points=points, inlier=valid,
                coordinates=np.asarray("camera"))
    data["faces"] = triangulate_surface(data, step=1, max_relative_jump=max_relative_jump)
    if not len(data["faces"]):
        raise ValueError("No adjacent-pixel mesh remains; use a dense point cloud or inspect filtering")
    return data


def curve_color(family, line):
    if family == 0:
        return (40, 210, 240)
    return (255, 70, 90) if line == 0 else (255, 190, 50)


def image_transform(shape, original_size, *, resize_only=False, matrix=None):
    h, w = shape
    ow, oh = original_size
    if matrix is not None:
        transform = np.asarray(matrix, float)
    elif resize_only:
        transform = np.diag([ow/w, oh/h, 1.])
    elif (ow, oh) == (w, h):
        transform = np.eye(3)
    else:
        raise ValueError("Original size differs: explicitly use --resize-only or --processed-to-original")
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("processed-to-original must be a finite 3x3 matrix")
    if abs(np.linalg.det(transform)) < 1e-12:
        raise ValueError("processed-to-original must be invertible")
    corners = np.array([[0, 0, 1], [w-1, 0, 1], [0, h-1, 1], [w-1, h-1, 1]]) @ transform.T
    if np.any(np.abs(corners[:, 2]) < 1e-10) or np.ptp(np.sign(corners[:, 2])) > 0:
        raise ValueError("Image transform crosses a projective horizon")
    return transform


def map_pixels(uv, matrix):
    hom = np.column_stack((uv, np.ones(len(uv)))) @ matrix.T
    return hom[:, :2] / hom[:, 2:]


def render_overlay(rgb, paths_uv, families, lines, support, matrix, width):
    import cv2
    if width < 1:
        raise ValueError("line width must be >= 1")
    output = np.asarray(rgb).copy()
    for uv, family, line in zip(paths_uv, families, lines):
        pixels = map_pixels(uv, matrix)
        if not np.isfinite(pixels).all() or np.max(np.abs(pixels), initial=0) > 1e8:
            raise ValueError("Image transform produces unreasonable coordinates")
        pixels = np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [pixels], False, curve_color(family, line), width, cv2.LINE_AA)
    # Restrict stroke width as well as its centerline to the supported image region.
    mask = cv2.warpPerspective(support.astype(np.uint8), matrix,
                               (output.shape[1], output.shape[0]), flags=cv2.INTER_NEAREST)
    output[mask == 0] = rgb[mask == 0]
    return output


def write_obj(path, paths, families, lines):
    with path.open("w", encoding="ascii") as handle:
        handle.write("# Camera coordinates; separate surface grid curves, no gap bridging\n")
        index = 1
        for component, (points, family, line) in enumerate(zip(paths, families, lines)):
            handle.write(f"g family_{family}_line_{line}_component_{component}\n")
            for x, y, z in points:
                handle.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")
            handle.write("l " + " ".join(map(str, range(index, index + len(points)))) + "\n")
            index += len(points)


def save_validation(out, data, fields, frame, paths, families, lines, overlay, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(overlay)
    ax.set_title("Sparse preview on the processed image")
    legend = [Line2D([], [], color=np.array(curve_color(0, 0))/255, label="sections")]
    if not report.get("sections_only", False):
        legend += [
                       Line2D([], [], color=np.array(curve_color(1, 1))/255, label="longitudinal"),
                       Line2D([], [], color=np.array(curve_color(1, 0))/255, label="theta=0")]
    ax.legend(handles=legend, fontsize=8, loc="lower left")
    h, w = data["image_shape"]
    u, v = data["pixels_uv"].T
    theta = np.full((h, w), np.nan)
    valid = data["inlier"] & fields["parameter_valid"]
    theta[v[valid], u[valid]] = np.degrees(fields["theta"][valid])
    ax = fig.add_subplot(2, 2, 2)
    im = ax.imshow(theta, vmin=0, vmax=360, cmap="hsv")
    fig.colorbar(im, ax=ax, shrink=0.8, label="theta (degrees)")
    ax.set_title("Circumferential coordinate on supported pixels")
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    triangles = data["fitted_points"][data["faces"]]
    mesh = Poly3DCollection(triangles, linewidths=0, alpha=0.4, rasterized=True)
    mesh.set_facecolor(data["colors"][data["faces"]].mean(axis=1)/255
                      if "colors" in data else "silver")
    ax.add_collection3d(mesh)
    for p, family, line in zip(paths, families, lines):
        segments = np.stack((p[:-1], p[1:]), axis=1)
        ax.add_collection3d(Line3DCollection(segments,
                           colors=np.array(curve_color(family, line))/255, linewidths=1.5))
    axis_line = np.asarray(frame["center"]) + np.array(frame["s_range"])[:, None]*frame["direction"]
    ax.plot(*axis_line.T, "--", color="magenta", linewidth=1)
    points = data["fitted_points"][np.unique(data["faces"])]
    lower, upper = points.min(axis=0), points.max(axis=0)
    ax.set(xlim=(lower[0], upper[0]), ylim=(upper[1], lower[1]), zlim=(lower[2], upper[2]),
           xlabel="X", ylabel="Y", zlabel="Z", title="Visible curves on the supported surface")
    ax.set_box_aspect(np.maximum(upper-lower, 1e-8))
    ax.view_init(elev=22, azim=-65)
    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    import textwrap
    text = ["PREVIEW - candidate axis is not physically verified",
            f"Geometry: {report.get('geometry_source', 'fitted_surface')}",
            f"Axis state: {report['axis_status']}",
            f"Sections / angular sectors: {report['sections']} / {report['sectors']}",
            f"Mesh-connected curves: {report['geometric_components']}",
            f"Visible fragments: {report['visible_fragments']}",
            f"Multiple-radius longitudinal IDs: {report['ambiguous_longitudinal_ids']}",
            "", "Review notes:"]
    for warning in report["warnings"]:
        text.extend(textwrap.wrap("- " + warning, width=62))
    ax.text(0, 1, "\n".join(text), va="top", family="monospace", fontsize=10)
    fig.savefig(out / "grid_validation.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--surface", type=Path, help="Fitted surface archive")
    source.add_argument("--pointcloud", type=Path, help="Raw pixel-linked point cloud archive")
    parser.add_argument("--sections-only", action="store_true", help="Only intersect transverse planes")
    parser.add_argument("--max-relative-jump", type=float, default=0.1,
                        help="Raw mesh relative depth discontinuity threshold (default 0.1)")
    parser.add_argument("--confidence-min", type=float, help="Optional raw point confidence threshold")
    parser.add_argument("--axis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grid"))
    parser.add_argument("--sections", type=int, default=5)
    parser.add_argument("--sectors", type=int, default=12)
    parser.add_argument("--s-range", nargs=2, type=float)
    parser.add_argument("--zero-angle-deg", type=float, default=0.)
    parser.add_argument("--zero-pixel", nargs=2, type=float, metavar=("U", "V"))
    parser.add_argument("--visibility-tolerance", type=float, default=0.03)
    parser.add_argument("--min-length-px", type=float, default=2.)
    parser.add_argument("--line-width", type=int, default=1, help="Processed image stroke width")
    parser.add_argument("--original-image", type=Path)
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument("--resize-only", action="store_true", help="Explicitly assert no crop")
    mapping.add_argument("--processed-to-original", type=Path, help="JSON 3x3 image transform")
    parser.add_argument("--no-plots", action="store_true", help="Skip diagnostic figure, keep overlays")
    args = parser.parse_args()
    if not 2 <= args.sections <= 100 or not 3 <= args.sectors <= 120:
        parser.error("sections must be 2..100 and sectors must be 3..120")
    if not np.isfinite(args.min_length_px) or args.min_length_px < 0 or args.line_width < 1:
        parser.error("min-length-px must be finite and nonnegative, line-width >= 1")
    if not np.isfinite(args.visibility_tolerance) or args.visibility_tolerance < 0:
        parser.error("visibility-tolerance must be finite and nonnegative")
    if (args.resize_only or args.processed_to_original) and args.original_image is None:
        parser.error("Image mapping requires --original-image")
    if args.surface and (args.confidence_min is not None or args.max_relative_jump != 0.1):
        parser.error("confidence-min and max-relative-jump apply only to --pointcloud")
    try:
        from PIL import Image
        data = (load_raw_mesh(args.pointcloud, max_relative_jump=args.max_relative_jump,
                              confidence_min=args.confidence_min)
                if args.pointcloud else load_surface(args.surface))
        axis = json.loads(args.axis.read_text())
        if axis.get("coordinates") != "camera":
            raise ValueError("Axis must be in camera coordinates")
        if axis.get("image_shape") != data["image_shape"].tolist() or not np.allclose(
            np.asarray(axis["intrinsics"]), data["intrinsics"], rtol=1e-6, atol=1e-6
        ):
            raise ValueError("Axis and surface image size/intrinsics do not match")
        h, w = data["image_shape"]
        u, v = data["pixels_uv"].T
        support = np.zeros((h, w), bool)
        support[v, u] = data["inlier"]
        depth = np.full((h, w), np.nan)
        depth[v, u] = data["fitted_points"][:, 2]
        zero_vector = None
        if args.zero_pixel is not None:
            pixel = np.asarray(args.zero_pixel)
            if not np.isfinite(pixel).all() or np.any(pixel < 0) or np.any(pixel > [w-1, h-1]):
                raise ValueError("zero-pixel must be a finite pixel inside the processed image")
            x, y = np.rint(pixel).astype(int)
            if not support[y, x]:
                raise ValueError("zero-pixel must belong to a supported surface point")
            index = np.flatnonzero((u == x) & (v == y))[0]
            zero_vector = data["fitted_points"][index] - axis["point"]
        frame, fields, paths, families, lines, details = build_grid(
            data["fitted_points"], data["faces"], axis, sections=args.sections, sectors=args.sectors,
            zero_vector=zero_vector, zero_angle_degrees=args.zero_angle_deg, s_range=args.s_range,
            sections_only=args.sections_only,
        )
        visible, vf, vl = [], [], []
        for path, family, line in zip(paths, families, lines):
            fragments = visible_fragments(path, data["intrinsics"], support, depth,
                                           tolerance=args.visibility_tolerance)
            for fragment in fragments:
                uv = project(fragment, data["intrinsics"])
                if np.linalg.norm(np.diff(uv, axis=0), axis=1).sum() >= args.min_length_px:
                    visible.append(fragment)
                    vf.append(family)
                    vl.append(line)
        vf, vl = np.asarray(vf, np.int32), np.asarray(vl, np.int32)
        projected_paths = [project(p, data["intrinsics"]) for p in visible]
        overlay = render_overlay(image_from_surface(data), projected_paths, vf, vl, support,
                                  np.eye(3), args.line_width)
        original, transform = None, None
        if args.original_image:
            with Image.open(args.original_image) as image:
                rgb = np.asarray(image.convert("RGB"))
            matrix = (
                json.loads(args.processed_to_original.read_text())
                if args.processed_to_original else None
            )
            transform = image_transform((h, w), (rgb.shape[1], rgb.shape[0]),
                                         resize_only=args.resize_only, matrix=matrix)
            width = max(args.line_width, int(round(args.line_width * rgb.shape[1] / w)))
            original = render_overlay(rgb, projected_paths, vf, vl, support, transform, width)
        warnings = []
        if axis.get("status") == "needs_review" or not axis.get("accepted_as_physical_axis", False):
            warnings.append(
                "Axis is an unverified candidate; this preview does not validate its physical position"
            )
        ambiguous = [item["line_id"] for item in details if item.get("multiple_radius_samples")]
        if ambiguous:
            warnings.append(
                "Multiple radii found at one s/theta: branches remain separate; "
                "no wall/web identity inferred"
            )
        if any(item["coplanar_triangles"] for item in details):
            warnings.append(
                "Some cut planes coincide with triangles; these ambiguous surface patches were omitted"
            )
        if any(item["junction_nodes"] for item in details):
            warnings.append(
                "Curve junctions were split; continuation through different surfaces was not guessed"
            )
        if not visible:
            warnings.append("No visible grid fragments remain; inspect surface support and axial range")
        report = {"surface": str(args.surface.resolve()) if args.surface else None,
                  "pointcloud": str(args.pointcloud.resolve()) if args.pointcloud else None,
                  "geometry_source": "raw_depth_mesh" if args.pointcloud else "fitted_surface",
                  "mesh_faces": len(data["faces"]),
                  "mesh_vertices_used": len(np.unique(data["faces"])),
                  "max_relative_jump": args.max_relative_jump if args.pointcloud else None,
                  "confidence_min": args.confidence_min,
                  "sections_only": args.sections_only, "axis": str(args.axis.resolve()),
                  "axis_status": axis.get("status", "unknown"), "preview_only": True,
                  "sections": args.sections, "sectors": 0 if args.sections_only else args.sectors,
                  "section_spacing": float(np.diff(frame["section_levels"])[0]),
                  "geometric_components": len(paths), "visible_fragments": len(visible),
                  "visible_longitudinal_ids": sorted(set(map(int, vl[vf == 1]))),
                  "ambiguous_longitudinal_ids": ambiguous, "warnings": warnings,
                  "visibility_tolerance": args.visibility_tolerance, "min_length_px": args.min_length_px,
                  "zero_pixel": args.zero_pixel, "lines": details,
                  "original_image": str(args.original_image.resolve()) if args.original_image else None,
                  "processed_to_original": transform.tolist() if transform is not None else None}
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        if args.pointcloud:
            # Preserve original point order and vertices for reproducible mesh inspection.
            np.savez_compressed(out / "raw_mesh.npz", points=data["points"], faces=data["faces"],
                                pixels_uv=data["pixels_uv"], valid=data["inlier"],
                                intrinsics=data["intrinsics"], image_shape=data["image_shape"],
                                coordinates=np.asarray("camera"))
            used = np.unique(data["faces"])
            lookup = np.full(len(data["points"]), -1, dtype=np.int32)
            lookup[used] = np.arange(len(used))
            write_ply(out / "raw_mesh.ply", data["points"][used],
                      data["colors"][used] if "colors" in data else None,
                      faces=lookup[data["faces"]])
        Image.fromarray(overlay).save(out / "grid_overlay.png")
        if original is not None:
            Image.fromarray(original).save(out / "grid_overlay_original.png")
        for name, value in (("frame.json", frame), ("report.json", report)):
            (out / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        vertices, offsets, closed = pack_paths(paths)
        vv, vo, vc = pack_paths(visible)
        np.savez_compressed(out / "grid.npz", vertices=vertices, offsets=offsets, closed=closed,
                            family_ids=families, line_ids=lines, visible_vertices=vv,
                            visible_offsets=vo, visible_closed=vc, visible_family_ids=vf,
                            visible_line_ids=vl, visible_pixels_uv=project(vv, data["intrinsics"]),
                            coordinates=np.asarray("camera"))
        np.savez_compressed(out / "coordinates.npz", **fields, inlier=data["inlier"],
                            pixels_uv=data["pixels_uv"], coordinates=np.asarray("camera"))
        write_obj(out / "grid_curves.obj", visible, vf, vl)
        colors = np.concatenate([np.tile(curve_color(f, l), (len(p), 1))
                                 for p, f, l in zip(visible, vf, vl)]).astype(np.uint8) if visible else None
        write_ply(out / "grid_points.ply", vv, colors)
        if not args.no_plots:
            save_validation(out, data, fields, frame, visible, vf, vl, overlay, report)
        print(json.dumps({k: report[k] for k in ("axis_status", "sections", "sectors",
                         "geometric_components", "visible_fragments", "ambiguous_longitudinal_ids",
                         "warnings")}, indent=2))
        print(f"Saved sparse grid preview to {out}")
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

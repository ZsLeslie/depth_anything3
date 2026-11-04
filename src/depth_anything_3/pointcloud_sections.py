"""Raw pixel-linked point cloud -> camera-Z sections -> image overlays, without fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from depth_anything_3.axis import project
from depth_anything_3.grid import image_transform, map_pixels, write_obj
from depth_anything_3.utils.grid import pack_paths, plane_curves, visible_fragments
from depth_anything_3.utils.pointcloud import triangulate_pointcloud, write_ply


def load_raw_cloud(path, *, confidence_min=None):
    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    required = {"points", "pixels_uv", "valid_mask", "intrinsics", "coordinates"}
    if missing := required-data.keys():
        raise ValueError(f"Point cloud is missing fields: {sorted(missing)}")
    p, uv, mask, K = data["points"], data["pixels_uv"], data["valid_mask"], data["intrinsics"]
    if p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ValueError("points must be a nonempty (N,3) array")
    if mask.ndim != 2 or mask.dtype.kind != "b" or min(mask.shape) < 2:
        raise ValueError("valid_mask must be a boolean image of shape (H,W)")
    h, w = mask.shape
    if uv.shape != (len(p), 2) or uv.dtype.kind not in "iu" or np.any(uv < 0) or np.any(uv >= [w, h]):
        raise ValueError("pixels_uv must contain integer (N,2) coordinates inside valid_mask")
    if (K.shape != (3, 3) or not np.isfinite(K).all() or not np.allclose(K[2], [0, 0, 1])
            or min(K[0, 0], K[1, 1]) <= 0 or abs(np.linalg.det(K)) < 1e-12):
        raise ValueError("Expected finite invertible pinhole intrinsics")
    coordinates = str(data["coordinates"].item())
    if coordinates == "world":
        ext = np.asarray(data.get("world_to_camera"), float)
        if ext.shape not in ((3, 4), (4, 4)) or not np.isfinite(ext).all():
            raise ValueError("World-coordinate input requires finite world_to_camera")
        if ext.shape == (4, 4) and not np.allclose(ext[3], [0, 0, 0, 1]):
            raise ValueError("world_to_camera has an invalid homogeneous last row")
        p = p @ ext[:3, :3].T + ext[:3, 3]
    elif coordinates != "camera":
        raise ValueError("Expected camera or world point cloud coordinates")
    valid = np.isfinite(p).all(axis=1) & (p[:, 2] > 0) & mask[uv[:, 1], uv[:, 0]]
    if "confidence" in data:
        conf = data["confidence"]
        if conf.shape != (len(p),):
            raise ValueError("confidence must have shape (N,)")
        valid &= np.isfinite(conf)
        if confidence_min is not None:
            valid &= conf >= confidence_min
    elif confidence_min is not None:
        raise ValueError("confidence-min requires saved point confidence")
    if confidence_min is not None and not np.isfinite(confidence_min):
        raise ValueError("confidence-min must be finite")
    if not valid.any():
        raise ValueError("No valid raw point observations remain")
    if np.max(np.abs(project(p[valid], K)-uv[valid])) > .05:
        raise ValueError("Raw points do not match their pixels and camera intrinsics")
    if "colors" in data and (data["colors"].shape != p.shape or data["colors"].dtype != np.uint8):
        raise ValueError("colors must contain uint8 RGB for every point")
    return {**data, "points": p, "valid": valid, "image_shape": np.array([h, w]),
            "input_coordinates": coordinates, "coordinates": np.asarray("camera")}


def section_depths(data, faces, *, count=5, z_range=None, depths=None, reference_axis=None):
    if reference_axis is not None:
        if reference_axis.get("coordinates") != "camera":
            raise ValueError("Reference axis must be in camera coordinates")
        direction = np.asarray(reference_axis["direction"], float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-10:
            raise ValueError("Invalid reference axis direction")
        if not np.allclose(direction/np.linalg.norm(direction), [0, 0, 1], atol=1e-8):
            raise ValueError("Reference axis must have direction [0,0,1]")
        if reference_axis.get("image_shape") != data["image_shape"].tolist() or not np.allclose(
            np.asarray(reference_axis["intrinsics"]), data["intrinsics"], rtol=1e-6, atol=1e-6
        ):
            raise ValueError("Reference axis and raw cloud image size/intrinsics do not match")
        depths = reference_axis.get("section_depths")
        if depths is None:
            depths = np.asarray(reference_axis["section_levels"])+reference_axis["point"][2]
    if depths is None:
        bounds = (np.percentile(data["points"][np.unique(faces), 2], [12, 88])
                  if z_range is None else np.asarray(z_range, float))
        if np.shape(bounds) != (2,) or not np.isfinite(bounds).all() or bounds[0] <= 0 or bounds[1] <= bounds[0]:
            raise ValueError("z-range must contain increasing positive finite bounds")
        depths = np.linspace(*bounds, count)
    depths = np.asarray(depths, float)
    if (depths.ndim != 1 or not 2 <= len(depths) <= 100 or not np.isfinite(depths).all()
            or np.any(depths <= 0) or np.any(np.diff(depths) <= 0)):
        raise ValueError("Section depths must be 2..100 increasing positive finite values")
    return depths


def section_colors(count):
    palette = np.array([[65, 70, 180], [35, 195, 210], [150, 220, 70], [245, 165, 45], [185, 40, 20]])
    return np.rint(np.column_stack([np.interp(np.linspace(0, 1, count), np.linspace(0, 1, len(palette)),
                                              palette[:, channel]) for channel in range(3)])).astype(np.uint8)


def render_sections(rgb, paths_uv, ids, colors, support, matrix, width):
    import cv2
    result = rgb.copy()
    for uv, index in zip(paths_uv, ids):
        mapped = map_pixels(uv, matrix)
        if not np.isfinite(mapped).all() or np.max(np.abs(mapped), initial=0) > 1e8:
            raise ValueError("Image transform produces unreasonable coordinates")
        cv2.polylines(result, [np.rint(mapped).astype(np.int32).reshape(-1, 1, 2)], False,
                      tuple(map(int, colors[index])), width, cv2.LINE_AA)
    mask = cv2.warpPerspective(support.astype(np.uint8), matrix, (rgb.shape[1], rgb.shape[0]),
                               flags=cv2.INTER_NEAREST).astype(bool)
    result[~mask] = rgb[~mask]
    return result


def save_validation(out, data, faces, paths, ids, colors, overlay, depth, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(overlay)
    ax.set_title("Raw point-cloud sections: camera Z axis")
    ax = fig.add_subplot(2, 2, 2)
    lo, hi = np.percentile(data["points"][data["valid"], 2], [1, 99])
    im = ax.imshow(depth, vmin=lo, vmax=hi, cmap="viridis")
    fig.colorbar(im, ax=ax, shrink=.8, label="Raw camera Z (prediction units)")
    for path, index in zip(paths, ids):
        uv = project(path, data["intrinsics"])
        ax.add_collection(LineCollection(np.stack((uv[:-1], uv[1:]), axis=1),
                                          colors=colors[index]/255, linewidths=.8))
    ax.set_title("Original depth and visible contours")
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    # Display at most 20k faces; the saved mesh and all intersections use every face.
    shown = faces[::max(1, int(np.ceil(len(faces)/20000)))]
    triangles = data["points"][shown]
    mesh = Poly3DCollection(triangles, alpha=.35, linewidths=0, rasterized=True)
    mesh.set_facecolor(data["colors"][shown].mean(axis=1)/255 if "colors" in data else "silver")
    ax.add_collection3d(mesh)
    for path, index in zip(paths, ids):
        ax.add_collection3d(Line3DCollection(np.stack((path[:-1], path[1:]), axis=1),
                                             colors=colors[index]/255, linewidths=1.4))
    used = data["points"][np.unique(faces)]
    lower, upper = used.min(axis=0), used.max(axis=0)
    ax.plot([0, 0], [0, 0], [lower[2], upper[2]], "--", color="magenta", label="camera Z")
    lower[:2] = np.minimum(lower[:2], 0)
    upper[:2] = np.maximum(upper[:2], 0)
    ax.set(xlim=(lower[0], upper[0]), ylim=(upper[1], lower[1]), zlim=(lower[2], upper[2]),
           xlabel="X", ylabel="Y", zlabel="Z", title="Original vertices; mesh display subsampled")
    ax.set_box_aspect(np.maximum(upper-lower, 1e-8))
    ax.view_init(elev=22, azim=-65)
    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    notes = ["RAW DEPTH - no surface fitting / smoothing", "Axis: x=y=0, direction [0,0,1]",
             f"Mesh faces: {len(faces)}", f"Section depths: {np.round(report['section_depths'], 5)}",
             f"Geometric components: {report['geometric_components']}",
             f"Visible fragments: {report['visible_fragments']}",
             f"Depth jump limit: {report['max_relative_jump']}",
             "Holes and rejected discontinuities remain disconnected.",
             "Only adjacent observed pixels are triangulated.",
             "Image position does not validate true wall geometry."]
    ax.text(0, 1, "\n".join(notes), va="top", fontsize=10)
    fig.savefig(out / "sections_validation.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcloud", type=Path, required=True, help="Pixel-linked frame_XXXX.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pointcloud_sections_z"))
    parser.add_argument("--sections", type=int, default=5)
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument("--z-range", nargs=2, type=float, help="Absolute Z bounds")
    placement.add_argument("--section-depths", nargs="+", type=float, help="Explicit depths, overrides --sections")
    placement.add_argument("--depths-from-axis", type=Path, help="Use matching camera-Z axis JSON section depths")
    parser.add_argument("--max-relative-jump", type=float, default=.1)
    parser.add_argument("--confidence-min", type=float)
    parser.add_argument("--visibility-tolerance", type=float, default=.03)
    parser.add_argument("--min-length-px", type=float, default=2.)
    parser.add_argument("--line-width", type=int, default=1)
    parser.add_argument("--original-image", type=Path)
    mapping = parser.add_mutually_exclusive_group()
    mapping.add_argument("--resize-only", action="store_true")
    mapping.add_argument("--processed-to-original", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if not 2 <= args.sections <= 100:
        parser.error("sections must be between 2 and 100")
    if args.line_width < 1 or not np.isfinite(args.min_length_px) or args.min_length_px < 0:
        parser.error("line-width must be positive and min-length-px finite/nonnegative")
    if not np.isfinite(args.visibility_tolerance) or args.visibility_tolerance < 0:
        parser.error("visibility-tolerance must be finite and nonnegative")
    if (args.resize_only or args.processed_to_original) and args.original_image is None:
        parser.error("Image mapping requires --original-image")
    try:
        from PIL import Image
        data = load_raw_cloud(args.pointcloud, confidence_min=args.confidence_min)
        points, uv, valid, K = data["points"], data["pixels_uv"], data["valid"], data["intrinsics"]
        faces = triangulate_pointcloud(points, uv, data["image_shape"], valid=valid,
                                       max_relative_jump=args.max_relative_jump)
        if not len(faces):
            raise ValueError("No adjacent-pixel triangles remain; use a dense --stride 1 point cloud or inspect depth filtering")
        reference = json.loads(args.depths_from_axis.read_text()) if args.depths_from_axis else None
        depths = section_depths(data, faces, count=args.sections, z_range=args.z_range,
                                depths=args.section_depths, reference_axis=reference)
        colors = section_colors(len(depths))
        h, w = data["image_shape"]
        support, depth = np.zeros((h, w), bool), np.full((h, w), np.nan)
        support[uv[:, 1], uv[:, 0]] = valid
        depth[uv[valid, 1], uv[valid, 0]] = points[valid, 2]
        rgb = np.full((h, w, 3), 235, np.uint8)
        if "colors" in data:
            rgb[uv[:, 1], uv[:, 0]] = data["colors"]
        paths, ids, visible, visible_ids, details = [], [], [], [], []
        for index, level in enumerate(depths):
            current, info = plane_curves(points, faces, np.zeros(3), np.array([0., 0., 1.]), level)
            info.update(section=index, depth=float(level), components=len(current))
            details.append(info)
            paths.extend(current)
            ids.extend([index]*len(current))
            for path in current:
                for fragment in visible_fragments(path, K, support, depth, tolerance=args.visibility_tolerance):
                    pixel_length = np.linalg.norm(np.diff(project(fragment, K), axis=0), axis=1).sum()
                    if pixel_length >= args.min_length_px:
                        visible.append(fragment)
                        visible_ids.append(index)
        ids, visible_ids = np.asarray(ids, np.int32), np.asarray(visible_ids, np.int32)
        projected = [project(path, K) for path in visible]
        overlay = render_sections(rgb, projected, visible_ids, colors, support, np.eye(3), args.line_width)
        original, transform = None, None
        if args.original_image:
            with Image.open(args.original_image) as image:
                original_rgb = np.asarray(image.convert("RGB"))
            matrix = json.loads(args.processed_to_original.read_text()) if args.processed_to_original else None
            transform = image_transform((h, w), (original_rgb.shape[1], original_rgb.shape[0]),
                                         resize_only=args.resize_only, matrix=matrix)
            width = max(args.line_width, int(round(args.line_width*original_rgb.shape[1]/w)))
            original = render_sections(original_rgb, projected, visible_ids, colors, support, transform, width)
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        warnings = []
        if not visible:
            warnings.append("No visible sections remain; inspect depth range and point support")
        if any(d["coplanar_triangles"] for d in details):
            warnings.append("Coplanar triangles have no unique section line and were omitted")
        if any(d["junction_nodes"] for d in details):
            warnings.append("Branch junctions remain split; no physical surface identity inferred")
        report = dict(pointcloud=str(args.pointcloud.resolve()), geometry_source="raw_pointcloud",
                      surface_fitting=False, smoothing=False, axis_direction=[0., 0., 1.],
                      input_coordinates=data["input_coordinates"], image_shape=[int(h), int(w)],
                      point_count=len(points), valid_points=int(valid.sum()), mesh_faces=len(faces),
                      mesh_vertices_used=len(np.unique(faces)), max_relative_jump=args.max_relative_jump,
                      confidence_min=args.confidence_min, section_depths=depths.tolist(),
                      depths_from_axis=str(args.depths_from_axis.resolve()) if args.depths_from_axis else None,
                      geometric_components=len(paths), visible_fragments=len(visible), lines=details,
                      visibility_tolerance=args.visibility_tolerance, min_length_px=args.min_length_px,
                      section_colors_rgb=colors.tolist(), preview_only=True, warnings=warnings,
                      original_image=str(args.original_image.resolve()) if args.original_image else None,
                      processed_to_original=transform.tolist() if transform is not None else None)
        reference_z = float(np.median(points[np.unique(faces), 2]))
        axis = dict(point=[0., 0., reference_z], direction=[0., 0., 1.], coordinates="camera",
                    axis_source="camera_z", section_depths=depths.tolist(),
                    section_levels=(depths-reference_z).tolist(),
                    s_range=[float(depths[0]-reference_z), float(depths[-1]-reference_z)],
                    intrinsics=K.tolist(), image_shape=[int(h), int(w)],
                    status="needs_review", accepted_as_physical_axis=False)
        (out / "axis.json").write_text(json.dumps(axis, indent=2, allow_nan=False))
        (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        vertices, offsets, closed = pack_paths(paths)
        vv, vo, vc = pack_paths(visible)
        segments = np.concatenate([np.stack((p[:-1], p[1:]), axis=1) for p in paths]) if paths else np.empty((0, 2, 3))
        segment_ids = np.concatenate([np.full(len(p)-1, index, np.int32) for p, index in zip(paths, ids)]) if paths else np.empty(0, np.int32)
        np.savez_compressed(out / "sections.npz", vertices=vertices, offsets=offsets, closed=closed,
                            path_section_ids=ids, segments=segments, section_ids=segment_ids,
                            segments_uv=project(segments, K), section_depths=depths,
                            visible_vertices=vv, visible_offsets=vo, visible_closed=vc,
                            visible_section_ids=visible_ids, visible_pixels_uv=project(vv, K),
                            intrinsics=K, image_shape=data["image_shape"], coordinates=np.asarray("camera"))
        np.savez_compressed(out / "raw_mesh.npz", points=points, faces=faces, valid=valid,
                            pixels_uv=uv, intrinsics=K, image_shape=data["image_shape"], coordinates=np.asarray("camera"))
        used = np.unique(faces)
        remap = np.full(len(points), -1, np.int32)
        remap[used] = np.arange(len(used))
        write_ply(out / "raw_mesh.ply", points[used], data["colors"][used] if "colors" in data else None,
                  faces=remap[faces])
        visible_colors = np.concatenate([np.tile(colors[index], (len(p), 1)) for p, index in zip(visible, visible_ids)]) if visible else None
        write_ply(out / "section_points.ply", vv, visible_colors)
        write_obj(out / "sections.obj", visible, np.zeros(len(visible), np.int32), visible_ids)
        Image.fromarray(overlay).save(out / "sections_overlay.png")
        if original is not None:
            Image.fromarray(original).save(out / "sections_overlay_original.png")
        if not args.no_plots:
            save_validation(out, data, faces, visible, visible_ids, colors, overlay, depth, report)
        print(json.dumps({k: report[k] for k in ("point_count", "mesh_faces", "section_depths",
                         "geometric_components", "visible_fragments", "warnings")}, indent=2))
        print(f"Saved raw point-cloud Z sections to {out}")
    except (ValueError, OSError, ImportError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

"""Recover visible surface points from DA3 z-depth, without segmentation masks."""

from __future__ import annotations

from pathlib import Path
import numpy as np


def confidence_to_colors(confidence: np.ndarray, value_range) -> np.ndarray:
    """Map raw confidence to RGB: low red, midpoint yellow, high green."""
    values = np.asarray(confidence, dtype=np.float64)
    bounds = np.asarray(value_range, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("confidence colors require a finite 1D confidence array")
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or bounds[1] < bounds[0]):
        raise ValueError("confidence color range must contain finite MIN <= MAX")
    if bounds[0] == bounds[1]:
        normalized = np.full(values.shape, 0.5)
    else:
        normalized = np.clip((values-bounds[0])/(bounds[1]-bounds[0]), 0, 1)
    colors = np.column_stack((
        np.interp(normalized, [0, 0.5, 1], [255, 255, 0]),
        np.interp(normalized, [0, 0.5, 1], [0, 255, 255]),
        np.zeros(len(values)),
    ))
    return np.rint(colors).astype(np.uint8)


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    *,
    image: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    confidence_min: float | None = None,
    max_depth: float | None = None,
    stride: int = 1,
    world_to_camera: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Backproject one frame, retaining pixel correspondence for later grid projection.

    Depth is camera z (not ray length). K, RGB and confidence must describe the
    *processed* depth image, not the original image before resize/crop. Camera
    coordinates are OpenCV: x right, y down, z forward. Supplying w2c transforms
    the result to world coordinates; depth thresholds still use camera z units.
    No scale normalization, smoothing or guessed camera intrinsics is applied.
    """
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("depth must be a nonempty (H, W) array")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("intrinsics must be a finite (3, 3) matrix")
    if not np.allclose(K[2], [0, 0, 1]) or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("intrinsics must be a pinhole camera matrix with positive focal lengths")
    if not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if max_depth is not None and (not np.isfinite(max_depth) or max_depth <= 0):
        raise ValueError("max_depth must be finite and positive")
    if confidence_min is not None and not np.isfinite(confidence_min):
        raise ValueError("confidence_min must be finite")
    if image is not None:
        image = np.asarray(image)
        if image.shape != (*depth.shape, 3) or image.dtype != np.uint8:
            raise ValueError("image must be uint8 RGB with shape (H, W, 3) matching depth")
    if confidence is not None:
        confidence = np.asarray(confidence)
        if confidence.shape != depth.shape:
            raise ValueError("confidence must have the same shape as depth")
    elif confidence_min is not None:
        raise ValueError("confidence_min requires a confidence map")

    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth <= max_depth
    if confidence is not None:
        valid &= np.isfinite(confidence)
        if confidence_min is not None:
            valid &= confidence >= confidence_min
    sampled = np.zeros(depth.shape, dtype=bool)
    sampled[::stride, ::stride] = True
    valid &= sampled
    v, u = np.nonzero(valid)
    pixels = np.column_stack((u, v, np.ones(len(u))))
    try:
        rays = np.linalg.solve(K, pixels.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("intrinsics must be invertible") from exc
    points = rays * depth[v, u, None]

    if world_to_camera is not None:
        ext = np.asarray(world_to_camera, dtype=np.float64)
        if ext.shape == (3, 4):
            ext = np.vstack((ext, [0, 0, 0, 1]))
        if ext.shape != (4, 4) or not np.isfinite(ext).all():
            raise ValueError("world_to_camera must be a finite (3, 4) or (4, 4) matrix")
        if not np.allclose(ext[3], [0, 0, 0, 1]):
            raise ValueError("world_to_camera must have homogeneous last row [0, 0, 0, 1]")
        try:
            c2w = np.linalg.inv(ext)
        except np.linalg.LinAlgError as exc:
            raise ValueError("world_to_camera must be invertible") from exc
        points = points @ c2w[:3, :3].T + c2w[:3, 3]

    if not np.isfinite(points).all():
        raise ValueError(
            "backprojection produced nonfinite points; check depth and camera matrices"
        )
    result = {
        "points": points,
        "pixels_uv": np.column_stack((u, v)).astype(np.int32),
        "valid_mask": valid,
        "intrinsics": K,
    }
    if image is not None:
        result["colors"] = image[v, u]
    if confidence is not None:
        result["confidence"] = confidence[v, u]
    if world_to_camera is not None:
        result["world_to_camera"] = ext
    return result


def triangulate_pointcloud(
    points: np.ndarray, pixels_uv: np.ndarray, image_shape: tuple[int, int], *,
    valid: np.ndarray | None = None, max_relative_jump: float = 0.1,
) -> np.ndarray:
    """Connect original adjacent-pixel vertices, without moving or fitting them.

    A pixel quad needs all four observations. Triangles crossing a relative
    camera-depth discontinuity are discarded. Sparse stride>1 observations
    are not connected across unsampled pixels.
    """
    points, uv = np.asarray(points), np.asarray(pixels_uv)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if len(image_shape) != 2 or any(int(n) != n or n < 2 for n in image_shape):
        raise ValueError("image_shape must contain integer H,W >= 2")
    h, w = map(int, image_shape)
    if uv.shape != (len(points), 2) or uv.dtype.kind not in "iu":
        raise ValueError("pixels_uv must have integer shape (N,2)")
    if np.any(uv < 0) or np.any(uv >= [w, h]):
        raise ValueError("pixels_uv falls outside image_shape")
    if len(np.unique(uv[:, 1]*w+uv[:, 0])) != len(uv):
        raise ValueError("Duplicate pixel observations are not supported")
    if not np.isfinite(max_relative_jump) or max_relative_jump <= 0:
        raise ValueError("max_relative_jump must be finite and positive")
    eligible = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    if valid is not None:
        valid = np.asarray(valid)
        if valid.shape != (len(points),) or valid.dtype.kind != "b":
            raise ValueError("valid must be a boolean array of length N")
        eligible &= valid
    lookup = np.full((h, w), -1, np.int32)
    lookup[uv[eligible, 1], uv[eligible, 0]] = np.flatnonzero(eligible)
    corners = np.stack((lookup[:-1, :-1], lookup[:-1, 1:],
                        lookup[1:, :-1], lookup[1:, 1:]), axis=-1).reshape(-1, 4)
    corners = corners[(corners >= 0).all(axis=1)]
    faces = np.concatenate((corners[:, [0, 2, 1]], corners[:, [1, 2, 3]]))
    if len(faces):
        z = points[faces, 2]
        faces = faces[np.ptp(z, axis=1)/z.min(axis=1) <= max_relative_jump]
    return faces.astype(np.int32)


def write_ply(
    path: str | Path, points: np.ndarray, colors: np.ndarray | None = None,
    *, faces: np.ndarray | None = None,
) -> None:
    """Write binary PLY points or a triangle mesh without requiring Open3D/plyfile."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite with shape (N, 3)")
    fields = [(axis, "<f8") for axis in ("x", "y", "z")]
    properties = [f"property double {axis}" for axis in ("x", "y", "z")]
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != points.shape or colors.dtype != np.uint8:
            raise ValueError("colors must be uint8 RGB with shape (N, 3)")
        fields.extend((channel, "u1") for channel in ("red", "green", "blue"))
        properties.extend(f"property uchar {c}" for c in ("red", "green", "blue"))
    vertices = np.empty(len(points), dtype=fields)
    for i, axis in enumerate(("x", "y", "z")):
        vertices[axis] = points[:, i]
    if colors is not None:
        for i, channel in enumerate(("red", "green", "blue")):
            vertices[channel] = colors[:, i]
    face_properties = []
    if faces is not None:
        faces = np.asarray(faces)
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
            raise ValueError("faces must be integer triangle indices with shape (F, 3)")
        if np.any(faces < 0) or np.any(faces >= len(points)):
            raise ValueError("faces contain out-of-range vertex indices")
        face_properties = [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", f"element vertex {len(points)}",
        *properties, *face_properties, "end_header", "",
    ])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)
        if faces is not None:
            records = np.empty(len(faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))])
            records["count"] = 3
            records["indices"] = faces
            records.tofile(handle)

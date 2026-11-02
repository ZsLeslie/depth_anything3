"""Robust perspective B-spline fitting of a single visible depth surface.

The parameter domain is the processed image, not physical axial/circumferential
coordinates. No cylinder, hidden geometry, or semantic surface split is assumed.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve


def _knots(count: int) -> np.ndarray:
    return np.r_[np.zeros(4), np.linspace(0, 1, count - 2)[1:-1], np.ones(4)]


def _design(uv: np.ndarray, knots_u: np.ndarray, knots_v: np.ndarray) -> sparse.csr_matrix:
    """Tensor-product cubic basis; each row has at most 16 nonzero entries."""
    bu = BSpline.design_matrix(uv[:, 0], knots_u, 3)
    bv = BSpline.design_matrix(uv[:, 1], knots_v, 3)
    nu = len(knots_u) - 4
    ui, vi = bu.indices.reshape(-1, 4), bv.indices.reshape(-1, 4)
    values = (bv.data.reshape(-1, 4, 1) * bu.data.reshape(-1, 1, 4)).ravel()
    columns = (vi[:, :, None] * nu + ui[:, None, :]).ravel()
    rows = np.arange(0, 16 * len(uv) + 1, 16)
    return sparse.csr_matrix((values, columns, rows), shape=(len(uv), nu * (len(knots_v) - 4)))


def _bending_penalty(nu: int, nv: int) -> sparse.csr_matrix:
    du = sparse.diags([np.ones(nu - 2), -2 * np.ones(nu - 2), np.ones(nu - 2)],
                      [0, 1, 2], shape=(nu - 2, nu))
    dv = sparse.diags([np.ones(nv - 2), -2 * np.ones(nv - 2), np.ones(nv - 2)],
                      [0, 1, 2], shape=(nv - 2, nv))
    return sparse.vstack([sparse.kron(sparse.eye(nv), du),
                          sparse.kron(dv, sparse.eye(nu))], format="csr")


def evaluate_surface(
    model: dict, pixels_uv: np.ndarray, intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate camera-frame points, camera-facing normals and positive-depth validity.

    Returns NaNs outside the image domain or where the fitted inverse depth is
    nonpositive. Geometric validity here does not imply observed support: callers
    must additionally apply the saved observation/inlier masks before rendering.
    """
    pixels = np.asarray(pixels_uv, dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_uv must have shape (N, 2)")
    h, w = model["image_shape"]
    normalized = pixels / [w - 1, h - 1]
    inside = np.isfinite(normalized).all(axis=1) & (normalized >= 0).all(axis=1)
    inside &= (normalized <= 1).all(axis=1)
    points = np.full((len(pixels), 3), np.nan)
    normals = np.full_like(points, np.nan)
    valid = np.zeros(len(pixels), dtype=bool)
    idx = np.flatnonzero(inside)
    if not len(idx):
        return points, normals, valid
    ku, kv = np.asarray(model["knots_u"]), np.asarray(model["knots_v"])
    coeff = np.asarray(model["coefficients"])
    nu, nv = len(ku) - 4, len(kv) - 4
    su, sv = BSpline(ku, np.eye(nu), 3), BSpline(kv, np.eye(nv), 3)
    u, v = normalized[idx].T
    bu, bv = su(u), sv(v)
    g = np.einsum("ni,ij,nj->n", bv, coeff, bu)
    gu = np.einsum("ni,ij,nj->n", bv, coeff, su(u, nu=1)) / (w - 1)
    gv = np.einsum("ni,ij,nj->n", sv(v, nu=1), coeff, bu) / (h - 1)
    good = np.isfinite(g) & (g > 1e-8)
    idx, g, gu, gv = idx[good], g[good], gu[good], gv[good]
    Kinv = np.linalg.inv(np.asarray(intrinsics, dtype=float))
    rays = np.column_stack((pixels[idx], np.ones(len(idx)))) @ Kinv.T
    z = model["depth_scale"] / g
    du = -model["depth_scale"] * gu / g**2
    dv = -model["depth_scale"] * gv / g**2
    fitted = rays * z[:, None]
    pu = rays * du[:, None] + z[:, None] * Kinv[:, 0]
    pv = rays * dv[:, None] + z[:, None] * Kinv[:, 1]
    n = np.cross(pu, pv)
    lengths = np.linalg.norm(n, axis=1)
    good = np.isfinite(fitted).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-12)
    idx, fitted, n, lengths = idx[good], fitted[good], n[good], lengths[good]
    n /= lengths[:, None]
    n[np.einsum("ij,ij->i", n, fitted) > 0] *= -1
    points[idx], normals[idx], valid[idx] = fitted, n, True
    return points, normals, valid


def fit_surface(
    points: np.ndarray,
    pixels_uv: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: tuple[int, int],
    *,
    confidence: np.ndarray | None = None,
    confidence_min: float | None = None,
    controls: tuple[int, int] = (16, 10),
    smoothness: float = 0.05,
    max_samples: int = 16000,
    iterations: int = 30,
    relative_threshold: float = 0.05,
    validation_block: int = 8,
    seed: int = 42,
) -> tuple[dict, dict[str, np.ndarray], dict]:
    """Fit normalized inverse depth with Huber IRLS and control-net regularization.

    Deterministic spatial blocks (20% of eligible pixels) are held out throughout
    fitting. Reported validation residuals measure agreement with DA3, not truth.
    Input must be a single camera's pixel-linked cloud from step one.
    """
    points, pixels = np.asarray(points, float), np.asarray(pixels_uv)
    K = np.asarray(intrinsics, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 30:
        raise ValueError("At least 30 points with shape (N, 3) are required")
    if pixels.shape != (len(points), 2) or not np.isfinite(pixels).all():
        raise ValueError("pixels_uv must be finite with shape (N, 2)")
    if len(image_shape) != 2 or min(image_shape) < 2:
        raise ValueError("image_shape must be (H, W), both >= 2")
    h, w = image_shape
    if not np.all(pixels == np.floor(pixels)):
        raise ValueError("Input pixels_uv must contain integer pixel indices")
    if np.any(pixels < 0) or np.any(pixels >= [w, h]):
        raise ValueError("pixels_uv falls outside image_shape")
    pixels = pixels.astype(np.int32)
    if len(np.unique(pixels[:, 1] * w + pixels[:, 0])) != len(pixels):
        raise ValueError("Duplicate pixel indices are not supported")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("intrinsics must be a finite (3, 3) matrix")
    if not np.allclose(K[2], [0, 0, 1]) or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Expected pinhole intrinsics with positive focal lengths")
    if abs(np.linalg.det(K)) < 1e-12:
        raise ValueError("intrinsics must be invertible")
    nu, nv = controls
    if any(not isinstance(n, (int, np.integer)) or n < 4 for n in controls):
        raise ValueError("Each control count must be an integer >= 4")
    if not np.isfinite(smoothness) or smoothness < 0:
        raise ValueError("smoothness must be finite and nonnegative")
    if not np.isfinite(relative_threshold) or relative_threshold <= 0:
        raise ValueError("relative_threshold must be finite and positive")
    if max_samples < 2 * nu * nv or iterations < 1 or validation_block < 1:
        raise ValueError(
            "Need max_samples >= 2 * controls_u * controls_v and positive iterations/block"
        )
    eligible = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    if confidence is not None:
        confidence = np.asarray(confidence, float)
        if confidence.shape != (len(points),):
            raise ValueError("confidence must have shape (N,)")
        eligible &= np.isfinite(confidence)
    if confidence_min is not None:
        if confidence is None or not np.isfinite(confidence_min):
            raise ValueError("A finite confidence_min requires confidence values")
        eligible &= confidence >= confidence_min
    if eligible.sum() < 2 * nu * nv:
        raise ValueError("Too few eligible points; reduce controls or relax confidence threshold")
    projected = points[eligible] @ K.T
    error = np.max(np.abs(projected[:, :2] / projected[:, 2:] - pixels[eligible]))
    if error > 0.05:
        raise ValueError(
            "Points and pixels disagree; supply camera coordinates and matching intrinsics"
        )
    blocks = pixels[:, 0] // validation_block + 2 * (pixels[:, 1] // validation_block)
    holdout = blocks % 5 == 0
    validation = holdout & eligible
    train_indices = np.flatnonzero(~holdout & eligible)
    if len(train_indices) < 2 * nu * nv or validation.sum() < 10:
        raise ValueError(
            "Insufficient train/validation support; reduce controls or validation-block"
        )
    rng = np.random.default_rng(seed)
    if len(train_indices) > max_samples:
        train_indices = np.sort(rng.choice(train_indices, max_samples, replace=False))
    train = np.zeros(len(points), dtype=bool)
    train[train_indices] = True
    scale = float(np.median(points[train, 2]))
    ku, kv = _knots(nu), _knots(nv)
    A = _design(pixels[train] / [w - 1, h - 1], ku, kv)
    target = scale / points[train, 2]
    penalty = _bending_penalty(nu, nv)
    regularization = smoothness * len(target) / (nu * nv)
    prior = regularization * (penalty.T @ penalty) + 1e-10 * sparse.eye(nu * nv)
    weights = np.ones(len(target))
    previous = None
    converged = False
    for iteration in range(iterations):
        weighted = A.multiply(weights[:, None])
        coefficients = spsolve(
            (A.T @ weighted + prior).tocsc(), A.T @ (weights * target)
        )
        if not np.isfinite(coefficients).all():
            raise ValueError(
                "Spline solve failed; increase regularization or improve point coverage"
            )
        residual = A @ coefficients - target
        sigma = max(1.4826 * np.median(np.abs(residual - np.median(residual))), 1e-5)
        weights = np.minimum(1.0, 1.345 * sigma / np.maximum(np.abs(residual), 1e-12))
        if previous is not None and np.linalg.norm(coefficients - previous) < 1e-6 * max(
            1.0, np.linalg.norm(previous)
        ):
            converged = True
            break
        previous = coefficients.copy()
    model = {
        "type": "perspective_inverse_depth_bspline", "degree": 3,
        "coordinates": "camera", "image_shape": [int(h), int(w)],
        "parameterization": "u/(W-1), v/(H-1); not physical axial/circumferential coordinates",
        "depth_scale": scale, "knots_u": ku.tolist(), "knots_v": kv.tolist(),
        "coefficients": coefficients.reshape(nv, nu).tolist(), "intrinsics": K.tolist(),
        "support": "Use eligible/inlier masks in surface.npz; do not extrapolate hidden surfaces",
    }
    fitted, normals, evaluated = evaluate_surface(model, pixels, K)
    dz = fitted[:, 2] - points[:, 2]
    relative = np.full(len(points), np.nan)
    measurable = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) & evaluated
    relative[measurable] = np.abs(dz[measurable]) / points[measurable, 2]
    inlier = eligible & evaluated & (relative <= relative_threshold)
    arrays = {
        "points": points, "fitted_points": fitted, "normals": normals,
        "pixels_uv": pixels, "intrinsics": K, "image_shape": np.array([h, w]),
        "eligible": eligible, "evaluated": evaluated, "inlier": inlier,
        "train": train, "validation": validation,
        "depth_residual": dz, "relative_depth_residual": relative,
        "point_displacement": np.linalg.norm(fitted - points, axis=1),
    }

    def metrics(mask):
        observed = mask & measurable
        values = relative[observed]
        return {
            "count": int(mask.sum()), "evaluated_count": int(observed.sum()),
            "median_relative_depth_error": float(np.median(values)) if len(values) else None,
            "p95_relative_depth_error": float(np.percentile(values, 95)) if len(values) else None,
            "depth_rmse": float(np.sqrt(np.mean(dz[observed]**2))) if len(values) else None,
            "inlier_fraction": float(inlier[mask].mean()) if mask.any() else None,
        }

    report = {
        "method": model["type"], "controls_uv": [nu, nv], "smoothness": smoothness,
        "relative_threshold": relative_threshold, "confidence_min": confidence_min,
        "seed": seed, "iterations": iteration + 1, "converged": converged,
        "validation_block_pixels": validation_block,
        "all_eligible": metrics(eligible), "train": metrics(train),
        "validation": metrics(validation),
        "limitations": [
            "Residuals compare the fitted surface to DA3 predictions, not measured geometry.",
            "A continuous visible-depth surface does not identify physical surface boundaries.",
            "Axial direction, circumference coordinates and metric scale are not estimated here.",
        ],
    }
    return model, arrays, report


def triangulate_surface(
    arrays: dict[str, np.ndarray], *, step: int = 3, max_relative_jump: float = 0.1,
) -> np.ndarray:
    """Image-grid triangles; omit holes, outliers and large observed/fitted z jumps.

    Every pixel inside a coarse cell must be supported. Thus increasing step
    cannot silently bridge a small hole between its corners.
    """
    if step < 1 or not isinstance(step, (int, np.integer)):
        raise ValueError("step must be a positive integer")
    if not np.isfinite(max_relative_jump) or max_relative_jump <= 0:
        raise ValueError("max_relative_jump must be finite and positive")
    h, w = arrays["image_shape"]
    lookup = np.full((h, w), -1, dtype=np.int32)
    uv = arrays["pixels_uv"]
    valid = arrays["inlier"]
    lookup[uv[valid, 1], uv[valid, 0]] = np.flatnonzero(valid)
    z = np.full((h, w), np.nan)
    z[uv[valid, 1], uv[valid, 0]] = arrays["points"][valid, 2]
    bad = lookup < 0
    for axis in (0, 1):
        first, second = [slice(None), slice(None)], [slice(None), slice(None)]
        first[axis], second[axis] = slice(None, -1), slice(1, None)
        first, second = tuple(first), tuple(second)
        jump = np.abs(z[first] - z[second]) / np.minimum(z[first], z[second])
        bad[first] |= jump > max_relative_jump
        bad[second] |= jump > max_relative_jump
    integral = np.pad(bad.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ys, xs = np.meshgrid(np.arange(0, h - step, step),
                         np.arange(0, w - step, step), indexing="ij")
    y, x = ys.ravel(), xs.ravel()
    blocked = (integral[y + step + 1, x + step + 1] - integral[y, x + step + 1]
               - integral[y + step + 1, x] + integral[y, x]) > 0
    corners = np.column_stack((lookup[y, x], lookup[y, x + step],
                               lookup[y + step, x], lookup[y + step, x + step]))
    corners = corners[~blocked]
    faces = np.concatenate((corners[:, [0, 2, 1]], corners[:, [1, 2, 3]]), axis=0)
    if not len(faces):
        return faces.astype(np.int32)
    for key in ("points", "fitted_points"):
        depths = arrays[key][faces, 2]
        keep = (depths.max(axis=1) - depths.min(axis=1)) / depths.min(axis=1)
        faces = faces[keep <= max_relative_jump]
    return faces.astype(np.int32)

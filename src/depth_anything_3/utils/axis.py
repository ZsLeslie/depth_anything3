"""Candidate straight axes, stability diagnostics and supported mesh sections.

Circular cylinder/cone models are hypotheses for axis initialization, not a
replacement for the fitted surface or proof of the object's physical axis.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def unit_direction(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("Direction must be a finite 3-vector")
    length = np.linalg.norm(value)
    if length < 1e-10:
        raise ValueError("Direction must be nonzero")
    value = value / length
    if value[2] < 0:
        value = -value
    if value[2] < 0.1:
        raise ValueError("This forward-looking axis model requires direction z >= 0.1")
    return value


def axis_basis(direction):
    a = unit_direction(direction)
    reference = np.eye(3)[np.argmin(np.abs(a))]
    first = reference - reference.dot(a) * a
    first /= np.linalg.norm(first)
    return first, np.cross(a, first)


def camera_z_axis(points):
    """Fixed camera optical axis; use a positive reference depth for visualization."""
    points = np.asarray(points, float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or
            not np.isfinite(points).all() or np.any(points[:, 2] <= 0)):
        raise ValueError("Camera Z preview requires finite points with positive depth")
    z = float(np.median(points[:, 2]))
    axis = dict(point=[0., 0., z], direction=[0., 0., 1.], coordinates="camera",
                reference_z=z, candidate_model="fixed_camera_z", axis_source="camera_z",
                manual_direction=True, manual_position=True)
    report = dict(status="needs_review", accepted_as_physical_axis=False,
                  axis_source="camera_z", selected_validation=None,
                  stability=dict(requested=0, successful=0, direction_p95_degrees=None,
                                 position_p95_over_radius=None, competing_candidate_max_angle_degrees=None,
                                 interpretation="Not applicable: the camera Z axis is fixed, not estimated"),
                  warnings=["Camera Z is a fixed viewing-axis reference; it is not an estimated structure axis"])
    return axis, report, np.empty((0, 6))


def direction_from_parallel_lines(annotation, intrinsics, image_shape):
    """Estimate direction from explicitly annotated physically parallel straight lines.

    Coordinates are in the processed image. Original-image annotation requires
    an explicit original_to_processed homography, including any resize/crop.
    Pixel depth is never used for a vanishing point or an empty-space axis point.
    """
    h, w = image_shape
    transform = annotation.get("original_to_processed")
    if transform is None:
        if annotation.get("image_size") != [int(w), int(h)]:
            raise ValueError(
                "Annotation image_size must match [W,H] or supply original_to_processed"
            )
        transform = np.eye(3)
    transform = np.asarray(transform, float)
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("original_to_processed must be a finite 3x3 matrix")
    lines = annotation.get("parallel_lines", [])
    if len(lines) < 2:
        raise ValueError("Annotate at least two physically parallel longitudinal lines")
    Kinv = np.linalg.inv(intrinsics)
    plane_normals, processed_lines = [], []
    for line in lines:
        points = np.asarray(line, float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("Each parallel line needs at least two [u,v] points")
        if not np.isfinite(points).all():
            raise ValueError("Annotation points must be finite")
        hom = np.column_stack((points, np.ones(len(points)))) @ transform.T
        if np.any(np.abs(hom[:, 2]) < 1e-12):
            raise ValueError("Annotation transformation sends a reference point to infinity")
        uv = hom[:, :2] / hom[:, 2:]
        if np.any(uv < 0) or np.any(uv > [w - 1, h - 1]):
            raise ValueError("Annotated points fall outside the processed image")
        if np.linalg.norm(np.ptp(uv, axis=0)) < 5:
            raise ValueError(
                "Reference line is too short; use points spanning at least five pixels"
            )
        rays = np.column_stack((uv, np.ones(len(uv)))) @ Kinv.T
        _, _, vt = np.linalg.svd(rays, full_matrices=True)
        plane_normals.append(vt[-1])
        processed_lines.append(uv.tolist())
    normals = np.asarray(plane_normals)
    _, singular, vt = np.linalg.svd(normals, full_matrices=True)
    if singular[1] < 0.03 * singular[0]:
        raise ValueError("Reference lines are nearly coincident; choose spatially separated lines")
    direction = unit_direction(vt[-1])
    angular = np.degrees(np.arcsin(np.clip(np.abs(normals @ direction), 0, 1)))
    return direction, {
        "processed_lines": processed_lines,
        "line_plane_errors_degrees": angular.tolist(),
        "assumption": "Marked lines are physically straight and parallel to the structure axis",
    }


def _unpack(params):
    direction = np.r_[params[:2], 1.0]
    direction /= np.linalg.norm(direction)
    return direction, np.r_[params[2:4], 0.0], params[4], params[5]


def _geometry(params, points):
    direction, center, radius, slope = _unpack(params)
    delta = points - center
    axial = delta @ direction
    radial = delta - axial[:, None] * direction
    rho = np.linalg.norm(radial, axis=1)
    residual = (rho - radius - slope * axial) / np.sqrt(1 + slope*slope)
    return residual, axial, radial, rho


def _optimize(points, normals, initial, kind, fixed, normal_weight, max_nfev=180):
    lower = np.array([-2., -2., -3., -3., 0.02, -2.])
    upper = np.array([2., 2., 3., 3., 3., 2.])
    initial = np.clip(np.asarray(initial, float), lower + 1e-6, upper - 1e-6)
    fixed = dict(fixed)
    if kind == "cylinder":
        fixed[5] = 0.0
    free = np.array([i for i in range(6) if i not in fixed])
    for index, value in fixed.items():
        initial[index] = value

    def complete(x):
        params = initial.copy()
        params[free] = x
        return params

    def residual(x):
        params = complete(x)
        distances, axial, radial, rho = _geometry(params, points)
        a, _, radius, slope = _unpack(params)
        output = [distances]
        if normals is not None and normal_weight > 0:
            expected = radial / np.maximum(rho[:, None], 1e-12) - slope * a
            expected /= np.sqrt(1 + slope*slope)
            cosine = np.abs(np.einsum("ij,ij->i", expected, normals))
            output.append(normal_weight * (1 - np.clip(cosine, 0, 1)))
        # Discourage the negative-radius branch of the infinite cone.
        radii = radius + slope * np.array([axial.min(), axial.max()])
        output.append(np.sqrt(len(points)) * np.minimum(radii - 0.005, 0))
        return np.concatenate(output)

    fit = least_squares(residual, initial[free], bounds=(lower[free], upper[free]),
                        loss="soft_l1", f_scale=0.01, max_nfev=max_nfev)
    params = complete(fit.x)
    bound_hit = np.any(np.minimum(fit.x - lower[free], upper[free] - fit.x) < 1e-3)
    return params, bool(fit.success), bool(bound_hit)


def _statistics(params, points):
    residual, axial, _, _ = _geometry(params, points)
    radius = params[4] + params[5] * axial
    denominator = max(float(np.median(radius)), 1e-6)
    relative = np.abs(residual) / denominator
    return {
        "median_error_over_radius": float(np.median(relative)),
        "p90_error_over_radius": float(np.percentile(relative, 90)),
        "median_radius_normalized": denominator,
        "min_radius_normalized": float(radius.min()),
        "axial_span_over_radius": float(np.ptp(axial) / denominator),
        # Truncated RMS avoids allowing a few remaining artifacts to dominate ranking.
        "score": float(np.sqrt(np.mean(np.minimum(relative, 0.5)**2))),
    }


def estimate_axis(
    points, normals=None, *, validation=None, max_samples=5000, bootstrap=12,
    seed=42, direction=None, axis_point=None, normal_weight=0.03,
):
    """Fit multi-start cylinder/cone candidates and audit resampling stability.

    Direction is parameterized by [tx,ty,1] and automatic slopes are bounded to
    [-2,2]. Axis point is gauged at the median camera z plane. Manual inputs are
    fixed constraints, not automatically verified physical measurements.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 200:
        raise ValueError("At least 200 finite camera-frame points are required")
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
        raise ValueError("Axis input points must be finite with positive camera z")
    if not isinstance(max_samples, int) or max_samples < 200:
        raise ValueError("max_samples must be an integer >= 200")
    if not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    if not np.isfinite(normal_weight) or normal_weight < 0:
        raise ValueError("normal_weight must be finite and nonnegative")
    if normals is not None:
        normals = np.asarray(normals, float)
        if normals.shape != points.shape or not np.isfinite(normals).all():
            raise ValueError("normals must be finite and match points")
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths < 1e-8):
            raise ValueError("Normals must be nonzero")
        normals = normals / lengths[:, None]
    rng = np.random.default_rng(seed)
    if validation is None:
        validation = np.zeros(len(points), bool)
        validation[rng.choice(len(points), len(points) // 5, replace=False)] = True
    validation = np.asarray(validation, bool)
    if validation.shape != (len(points),) or min(validation.sum(), (~validation).sum()) < 40:
        raise ValueError("Need at least 40 points in each train/validation partition")
    train_idx = np.flatnonzero(~validation)
    train_idx = rng.choice(train_idx, min(max_samples, len(train_idx)), replace=False)
    val_idx = np.flatnonzero(validation)
    val_idx = rng.choice(val_idx, min(max_samples, len(val_idx)), replace=False)
    origin = np.median(points[train_idx], axis=0)
    scale = float(np.linalg.norm(np.ptp(np.percentile(points[train_idx], [5, 95], axis=0), axis=0)))
    if scale < 1e-8:
        raise ValueError("Point cloud has insufficient spatial extent")
    fixed = {}
    if axis_point is not None:
        axis_point = np.asarray(axis_point, float)
        if axis_point.shape != (3,) or not np.isfinite(axis_point).all():
            raise ValueError("axis_point must be a finite camera-frame 3-vector")
        origin[2] = axis_point[2]
        fixed.update({2: (axis_point[0] - origin[0]) / scale,
                      3: (axis_point[1] - origin[1]) / scale})
    if direction is not None:
        a = unit_direction(direction)
        fixed.update({0: a[0] / a[2], 1: a[1] / a[2]})
    q = (points - origin) / scale
    train, check = q[train_idx], q[val_idx]
    ns = normals[train_idx] if normals is not None else None
    seeds = [("camera_forward", np.array([0., 0., 1.]))]
    far = points[train_idx][points[train_idx, 2] >= np.percentile(points[train_idx, 2], 90)]
    far_direction = np.median(far, axis=0)
    if far_direction[2] / np.linalg.norm(far_direction) >= 0.1:
        seeds.append(("far_depth_direction", unit_direction(far_direction)))
    if ns is not None:
        _, _, vt = np.linalg.svd(ns - ns.mean(axis=0), full_matrices=False)
        if abs(vt[-1, 2]) >= 0.35:
            seeds.append(("normal_variation", unit_direction(vt[-1])))
    for x, y in [(-0.6, 0), (0.6, 0), (0, -0.6), (0, 0.6)]:
        seeds.append((f"tilted_{x}_{y}", unit_direction([x, y, 1])))
    if direction is not None:
        seeds = [("manual_direction", unit_direction(direction))]
    candidates = []
    for kind in ("cylinder", "cone"):
        for name, start in seeds:
            initial = np.r_[start[:2] / start[2], 0., 0., 0.3, 0.]
            params, success, bound = _optimize(train, ns, initial, kind, fixed, normal_weight)
            metrics = _statistics(params, check)
            candidates.append({"model": kind, "seed": name, "params": params,
                               "converged": success, "bound_hit": bound, **metrics})
    usable = [c for c in candidates if c["converged"] and c["min_radius_normalized"] > 0]
    if not usable:
        raise ValueError("No converged positive-radius axis candidate; provide manual constraints")
    best = min(usable, key=lambda c: c["score"])
    # Prefer a cylinder if a cone improves the validation score by less than 5%.
    cylinders = [c for c in usable if c["model"] == "cylinder"]
    if best["model"] == "cone" and cylinders:
        cylinder = min(cylinders, key=lambda c: c["score"])
        if cylinder["score"] <= 1.05 * best["score"]:
            best = cylinder
    a, center, radius, slope = _unpack(best["params"])
    c = origin + scale * center
    axis = {"point": c.tolist(), "direction": a.tolist(), "coordinates": "camera",
            "reference_z": float(c[2]), "candidate_model": best["model"],
            "radius_at_reference": float(radius * scale), "radius_slope": float(slope),
            "manual_direction": direction is not None, "manual_position": axis_point is not None}
    samples, angular, offsets = [], [], []
    for _ in range(bootstrap):
        indices = rng.choice(len(train), min(2000, len(train)), replace=True)
        params, success, bound = _optimize(
            train[indices], ns[indices] if ns is not None else None, best["params"],
            best["model"], fixed, normal_weight, max_nfev=100,
        )
        if not success or bound:
            continue
        aa, cc, _, _ = _unpack(params)
        samples.append(np.r_[origin + scale * cc, aa])
        angular.append(np.degrees(np.arccos(np.clip(aa @ a, -1, 1))))
        offsets.append(np.linalg.norm(cc - center) / best["median_radius_normalized"])
    comparable = [candidate for candidate in usable
                  if candidate["score"] <= max(best["score"] * 1.10, best["score"] + 1e-5)]
    disagreement = max(float(np.degrees(np.arccos(np.clip(
        _unpack(candidate["params"])[0] @ a, -1, 1
    )))) for candidate in comparable)
    stability = {
        "requested": bootstrap, "successful": len(samples),
        "direction_p95_degrees": float(np.percentile(angular, 95)) if angular else None,
        "position_p95_over_radius": float(np.percentile(offsets, 95)) if offsets else None,
        "competing_candidate_max_angle_degrees": disagreement,
        "interpretation": (
            "Conditional resampling sensitivity; not physical accuracy or a confidence interval"
        ),
    }
    warnings = []
    if best["bound_hit"]:
        warnings.append("Selected fit reached parameter bounds; possible geometric degeneracy")
    if best["median_error_over_radius"] > 0.10 or best["p90_error_over_radius"] > 0.25:
        warnings.append("Circular model poorly explains the observed surface")
    if best["axial_span_over_radius"] < 1:
        warnings.append("Observed axial span is too short relative to the estimated radius")
    if len(samples) < max(3, int(0.7 * bootstrap)):
        warnings.append("Insufficient successful resamples to assess stability")
    if angular and np.percentile(angular, 95) > 5:
        warnings.append("Axis direction changes substantially under resampling")
    if offsets and np.percentile(offsets, 95) > 0.15:
        warnings.append("Axis position changes substantially under resampling")
    if disagreement > 10:
        warnings.append("Similarly scoring candidates disagree about axis direction")
    if direction is not None or axis_point is not None:
        warnings.append("Manual constraints are fixed; reported stability does not validate them")
    report = {
        "status": "needs_review" if warnings else "candidate_consistent",
        "accepted_as_physical_axis": False, "seed": seed,
        "train_count": len(train_idx), "validation_count": len(val_idx),
        "normal_weight": normal_weight, "normalization_scale": scale,
        "selected_validation": {key: best[key] for key in (
            "median_error_over_radius", "p90_error_over_radius", "score", "axial_span_over_radius")},
        "stability": stability, "warnings": warnings,
        "candidates": [{key: value.tolist() if isinstance(value, np.ndarray) else value
                        for key, value in candidate.items() if key != "params"}
                       | {"direction": _unpack(candidate["params"])[0].tolist(),
                          "point": (origin + scale * _unpack(candidate["params"])[1]).tolist()}
                       for candidate in candidates],
    }
    return axis, report, np.asarray(samples, float).reshape(-1, 6)


def mesh_sections(points, faces, axis, levels):
    """Intersect each supported triangle with planes normal to the candidate axis.

    Return separate line segments, never connecting across mesh holes. Fully
    coplanar triangles and zero-length intersections are omitted.
    """
    points, faces = np.asarray(points, float), np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("faces must be integer triangle indices")
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("faces contain out-of-range indices")
    triangles = points[faces]
    if not np.isfinite(triangles).all():
        raise ValueError("Mesh triangles must contain finite points")
    a = unit_direction(axis["direction"])
    c = np.asarray(axis["point"], float)
    levels = np.asarray(levels, float)
    if c.shape != (3,) or not np.isfinite(c).all() or not np.isfinite(levels).all():
        raise ValueError("Axis point and section levels must be finite")
    axial = (triangles - c) @ a
    segments, ids = [], []
    for section_id, level in enumerate(levels):
        distances = axial - level
        crosses = []
        intersections = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            cross = (distances[:, i] >= 0) != (distances[:, j] >= 0)
            t = np.divide(distances[:, i], distances[:, i] - distances[:, j],
                          out=np.zeros(len(faces)), where=cross)
            intersections.append(triangles[:, i] + t[:, None] * (triangles[:, j] - triangles[:, i]))
            crosses.append(cross)
        crosses = np.stack(crosses, axis=1)
        intersections = np.stack(intersections, axis=1)
        valid = crosses.sum(axis=1) == 2
        current = intersections[valid][crosses[valid]].reshape(-1, 2, 3)
        current = current[np.linalg.norm(current[:, 1] - current[:, 0], axis=1) > 1e-10]
        segments.append(current)
        ids.append(np.full(len(current), section_id, dtype=np.int32))
    if not segments:
        return np.empty((0, 2, 3)), np.empty(0, dtype=np.int32)
    return np.concatenate(segments), np.concatenate(ids)


def section_diagnostics(segments, section_ids, axis, levels):
    """Fit circles to observed intersections to expose off-center/partial sections."""
    a, c = unit_direction(axis["direction"]), np.asarray(axis["point"])
    e1, e2 = axis_basis(a)
    output = []
    for i, level in enumerate(levels):
        section = segments[section_ids == i].reshape(-1, 3)
        if len(section):
            section = np.unique(np.round(section, 9), axis=0)
        item = {"section": i, "s": float(level), "point_count": len(section),
                "circle_reliable": False}
        if len(section) < 20:
            item["reason"] = "Too few observed intersection points"
            output.append(item)
            continue
        delta = section - (c + level * a)
        xy = np.column_stack((delta @ e1, delta @ e2))
        scale = max(float(np.linalg.norm(np.ptp(xy, axis=0))), 1e-8)
        q = xy / scale
        A = np.column_stack((2 * q, np.ones(len(q))))
        coeff, _, _, _ = np.linalg.lstsq(A, (q*q).sum(axis=1), rcond=None)
        radius = np.sqrt(max(coeff[2] + coeff[:2] @ coeff[:2], 0.01))
        start = np.clip(np.r_[coeff[:2], radius], [-3, -3, 0.01], [3, 3, 3])
        fit = least_squares(lambda p: np.linalg.norm(q - p[:2], axis=1) - p[2],
                            start, bounds=([-3, -3, 0.005], [3, 3, 3]),
                            loss="soft_l1", f_scale=0.01, max_nfev=100)
        center2, radius = fit.x[:2] * scale, fit.x[2] * scale
        angles = np.sort(np.mod(np.arctan2(xy[:, 1] - center2[1], xy[:, 0] - center2[0]), 2*np.pi))
        coverage = 360 - np.degrees(np.diff(np.r_[angles, angles[0] + 2*np.pi]).max())
        residual = np.abs(np.linalg.norm(xy - center2, axis=1) - radius) / radius
        bound = np.any(np.abs(fit.x[:2]) > 2.99) or fit.x[2] > 2.99
        reliable = bool(fit.success and not bound and coverage >= 120
                        and np.median(residual) < 0.1 and np.linalg.cond(A) < 1000)
        item.update({"circle_center": (c + level*a + center2[0]*e1 + center2[1]*e2).tolist(),
                     "circle_radius": float(radius), "angular_coverage_degrees": float(coverage),
                     "median_circle_error_over_radius": float(np.median(residual)),
                     "center_offset_over_radius": float(np.linalg.norm(center2) / radius),
                     "circle_reliable": reliable})
        output.append(item)
    return output

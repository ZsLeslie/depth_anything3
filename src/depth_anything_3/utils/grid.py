"""Axis-based surface coordinates and topology-preserving sparse grid curves."""

from __future__ import annotations

import numpy as np

from depth_anything_3.utils.axis import unit_direction


def circumferential_frame(direction, *, zero_vector=None, zero_angle_degrees=0.0):
    """Right-handed frame; default zero points along camera X projected off the axis."""
    a = unit_direction(direction)
    if not np.isfinite(zero_angle_degrees):
        raise ValueError("zero_angle_degrees must be finite")
    reference = np.array([1., 0., 0.]) if zero_vector is None else np.asarray(zero_vector, float)
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("zero_vector must be a finite 3-vector")
    e1 = reference - (reference @ a) * a
    if np.linalg.norm(e1) < 1e-8:
        raise ValueError("Zero reference is parallel to the axis; select a surface point off the axis")
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    angle = np.radians(zero_angle_degrees)
    first = np.cos(angle) * e1 + np.sin(angle) * e2
    return a, first, np.cross(a, first)


def surface_coordinates(points, center, direction, e1, e2, min_radius=1e-8):
    delta = np.asarray(points, float) - np.asarray(center, float)
    s = delta @ direction
    x, y = delta @ e1, delta @ e2
    radius = np.hypot(x, y)
    valid = np.isfinite(delta).all(axis=1) & (radius > min_radius)
    theta = np.full(len(delta), np.nan)
    theta[valid] = np.mod(np.arctan2(y[valid], x[valid]), 2 * np.pi)
    return s, theta, radius, valid


def _join_segments(segments, keys):
    """Join only shared mesh edge/vertex intersections, never nearby unrelated points."""
    nodes, positions, edges, adjacency = {}, [], [], []
    unique_edges = set()
    for segment, pair in zip(segments, keys):
        ids = []
        for point, key in zip(segment, pair):
            if key not in nodes:
                nodes[key] = len(positions)
                positions.append(point)
                adjacency.append([])
            ids.append(nodes[key])
        edge_key = tuple(sorted(ids))
        if ids[0] == ids[1] or edge_key in unique_edges:
            continue
        unique_edges.add(edge_key)
        edge_id = len(edges)
        edges.append(tuple(ids))
        for node in ids:
            adjacency[node].append(edge_id)
    positions = np.asarray(positions)
    seen, paths = set(), []

    def follow(node, edge):
        sequence = [node]
        while edge not in seen:
            seen.add(edge)
            left, right = edges[edge]
            node = right if node == left else left
            sequence.append(node)
            if node == sequence[0] or len(adjacency[node]) != 2:
                break
            choices = [i for i in adjacency[node] if i not in seen]
            if not choices:
                break
            edge = choices[0]
        paths.append(positions[sequence])

    for node, neighbors in enumerate(adjacency):
        if len(neighbors) != 2:
            for edge in neighbors:
                if edge not in seen:
                    follow(node, edge)
    for edge, (node, _) in enumerate(edges):
        if edge not in seen:
            follow(node, edge)
    return paths, sum(len(neighbors) > 2 for neighbors in adjacency)


def plane_curves(points, faces, center, normal, offset=0.0, *, constraints=()):
    """Intersect a triangle mesh with a plane and clip by linear half-spaces.

    constraints contains (vector, minimum) for (P-center).vector >= minimum.
    Whole coplanar triangles have no unique intersection curve: omit and report.
    At graph junctions paths are split instead of guessing surface continuation.
    """
    points, faces = np.asarray(points, float), np.asarray(faces)
    center, normal = np.asarray(center, float), np.asarray(normal, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("faces must contain integer triangle indices")
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("faces contain out-of-range vertex indices")
    if center.shape != (3,) or normal.shape != (3,) or not np.isfinite([center, normal]).all():
        raise ValueError("Plane center and normal must be finite 3-vectors")
    if not np.isclose(np.linalg.norm(normal), 1) or not np.isfinite(offset):
        raise ValueError("Plane normal must be unit length and offset must be finite")
    triangles = points[faces]
    if not np.isfinite(triangles).all():
        raise ValueError("Supported mesh must be finite")
    if not len(faces):
        return [], {"segments": 0, "junction_nodes": 0, "coplanar_triangles": 0}
    scale = max(float(np.linalg.norm(np.ptp(triangles.reshape(-1, 3), axis=0))), 1e-8)
    eps = scale * 1e-10
    distance = (triangles - center) @ normal - offset
    distance[np.abs(distance) <= eps] = 0
    coplanar = (distance == 0).all(axis=1)
    selected = np.flatnonzero((distance.min(axis=1) <= 0) &
                             (distance.max(axis=1) >= 0) & ~coplanar)
    segments, keys = [], []
    for face_id in selected:
        tri, d, vertex_ids = triangles[face_id], distance[face_id], faces[face_id]
        hits = {}
        for i in range(3):
            if d[i] == 0:
                hits[(int(vertex_ids[i]), int(vertex_ids[i]))] = tri[i]
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if d[i] * d[j] < 0:
                if vertex_ids[i] > vertex_ids[j]:
                    i, j = j, i
                key = (int(vertex_ids[i]), int(vertex_ids[j]))
                hits[key] = tri[i] + d[i] / (d[i] - d[j]) * (tri[j] - tri[i])
        if len(hits) != 2:
            continue
        # A cut along a shared mesh edge can be emitted by both incident faces.
        # Canonical endpoint order also makes clipped endpoint IDs identical.
        pair = sorted(hits)
        source_pair = tuple(pair)
        segment = np.array([hits[key] for key in pair])
        low, high = 0., 1.
        for vector, minimum in constraints:
            values = (segment - center) @ vector - minimum
            values[np.abs(values) <= eps] = 0
            if (values < 0).all():
                high = -1
                break
            if values[0] < 0:
                low = max(low, float(values[0] / (values[0] - values[1])))
            elif values[1] < 0:
                high = min(high, float(values[0] / (values[0] - values[1])))
        if high <= low:
            continue
        delta = segment[1] - segment[0]
        clipped = segment[0] + np.array([low, high])[:, None] * delta
        if np.linalg.norm(clipped[1] - clipped[0]) <= eps:
            continue
        if low > 1e-10:
            pair[0] = ("clip", source_pair, 0)
        if high < 1 - 1e-10:
            pair[1] = ("clip", source_pair, 1)
        segments.append(clipped)
        keys.append(pair)
    paths, junctions = _join_segments(segments, keys)
    return paths, {"segments": len(segments), "junction_nodes": junctions,
                   "coplanar_triangles": int(coplanar.sum())}


def radial_ambiguity(paths, center, direction, radial_direction, s_range, tolerance):
    """Find sampled axial locations with multiple distinct radii for one theta."""
    if not paths:
        return []
    segments = np.concatenate([np.stack((p[:-1], p[1:]), axis=1) for p in paths])
    s = (segments - center) @ direction
    r = (segments - center) @ radial_direction
    locations = np.linspace(*s_range, 42)[1:-1]
    result = []
    for level in locations:
        crosses = (s[:, 0] > level) != (s[:, 1] > level)
        if not crosses.any():
            continue
        t = (level - s[crosses, 0]) / (s[crosses, 1] - s[crosses, 0])
        radii = np.sort(r[crosses, 0] + t * (r[crosses, 1] - r[crosses, 0]))
        distinct = np.r_[True, np.diff(radii) > tolerance]
        if distinct.sum() > 1:
            result.append({"s": float(level), "radii": radii[distinct].tolist()})
    return result


def build_grid(points, faces, axis, *, sections=5, sectors=12, zero_vector=None,
               zero_angle_degrees=0., s_range=None):
    points, faces = np.asarray(points, float), np.asarray(faces)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu":
        raise ValueError("faces must contain integer triangle indices")
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("faces contain out-of-range vertex indices")
    if not isinstance(sections, int) or sections < 2:
        raise ValueError("sections must be an integer >= 2")
    if not isinstance(sectors, int) or sectors < 3:
        raise ValueError("sectors must be an integer >= 3")
    center = np.asarray(axis["point"], float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Axis point must be a finite 3-vector")
    a, e1, e2 = circumferential_frame(axis["direction"], zero_vector=zero_vector,
                                     zero_angle_degrees=zero_angle_degrees)
    bounds = np.asarray(axis["s_range"] if s_range is None else s_range, float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("s_range must contain increasing finite bounds")
    used = points[np.unique(faces)]
    if not len(used):
        raise ValueError("A nonempty supported mesh is required")
    scale = float(np.linalg.norm(np.ptp(used, axis=0)))
    min_radius = max(scale * 1e-6, 1e-10)
    s, theta, radius, valid = surface_coordinates(points, center, a, e1, e2, min_radius)
    levels = np.linspace(*bounds, sections)
    angles = np.arange(sectors) * (2 * np.pi / sectors)
    paths, family_ids, line_ids, diagnostics = [], [], [], []
    for family, values in enumerate((levels, angles)):
        for line_id, value in enumerate(values):
            constraints = ()
            if family == 0:
                normal, offset = a, value
            else:
                radial = np.cos(value)*e1 + np.sin(value)*e2
                normal, offset = np.cross(a, radial), 0.
                constraints = ((radial, min_radius), (a, bounds[0]), (-a, -bounds[1]))
            current, info = plane_curves(points, faces, center, normal, offset,
                                          constraints=constraints)
            info.update({"family": "section" if family == 0 else "longitudinal",
                         "line_id": line_id, "value": float(value), "components": len(current)})
            if family == 1:
                info["multiple_radius_samples"] = radial_ambiguity(
                    current, center, a, radial, bounds, max(scale * 1e-4, 1e-8),
                )
            diagnostics.append(info)
            paths.extend(current)
            family_ids.extend([family] * len(current))
            line_ids.extend([line_id] * len(current))
    frame = {"center": center.tolist(), "direction": a.tolist(),
             "e1": e1.tolist(), "e2": e2.tolist(), "s_range": bounds.tolist(),
             "section_levels": levels.tolist(), "angles_radians": angles.tolist(),
             "zero_angle_degrees": float(zero_angle_degrees), "min_radius": min_radius,
             "coordinates": "camera", "angular_direction": "right-handed about +axis"}
    fields = {"s": s, "theta": theta, "radius": radius, "parameter_valid": valid}
    return (frame, fields, paths, np.asarray(family_ids, np.int32),
            np.asarray(line_ids, np.int32), diagnostics)


def visible_fragments(path, intrinsics, support, depth, *, tolerance=0.03, sample_step=0.5):
    """Sample perspective-correctly, splitting at unsupported or inconsistent pixels.

    This uses the observed fitted depth as a visibility proxy. It does not infer
    occluders missing from DA3 and does not bridge holes even along one polyline.
    """
    if not np.isfinite(tolerance) or tolerance < 0 or sample_step <= 0:
        raise ValueError("Visibility tolerance must be nonnegative and step positive")
    path = np.asarray(path, float)
    if len(path) < 2 or not np.isfinite(path).all() or np.any(path[:, 2] <= 0):
        return []
    hom = path @ intrinsics.T
    uv = hom[:, :2] / hom[:, 2:]
    samples = []
    for i in range(len(path) - 1):
        count = max(1, int(np.ceil(np.linalg.norm(uv[i+1] - uv[i]) / sample_step)))
        fraction = np.arange(count) / count
        t = fraction * path[i, 2] / ((1-fraction)*path[i+1, 2] + fraction*path[i, 2])
        samples.append(path[i] + t[:, None] * (path[i+1] - path[i]))
    samples.append(path[-1:])
    samples = np.concatenate(samples)
    projected = samples @ intrinsics.T
    pixels = np.rint(projected[:, :2] / projected[:, 2:]).astype(np.int64)
    h, w = support.shape
    inside = (pixels >= 0).all(axis=1) & (pixels < [w, h]).all(axis=1)
    good = np.zeros(len(samples), bool)
    u, v = pixels[inside].T
    reference = depth[v, u]
    good[inside] = support[v, u] & np.isfinite(reference) & (reference > 0)
    good[inside] &= np.abs(samples[inside, 2] - reference) <= tolerance * reference
    starts = np.flatnonzero(good & ~np.r_[False, good[:-1]])
    ends = np.flatnonzero(good & ~np.r_[good[1:], False]) + 1
    return [samples[start:end] for start, end in zip(starts, ends) if end-start >= 2]


def pack_paths(paths):
    offsets = np.r_[0, np.cumsum([len(p) for p in paths])].astype(np.int64)
    vertices = np.concatenate(paths) if paths else np.empty((0, 3))
    closed = np.array([np.linalg.norm(p[0] - p[-1]) < 1e-9 for p in paths], bool)
    return vertices, offsets, closed

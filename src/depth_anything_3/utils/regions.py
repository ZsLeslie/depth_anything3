"""Validated polygon annotations and pixel-center region membership."""

from __future__ import annotations

import numpy as np


def _polygon(vertices, size):
    p = np.asarray(vertices, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("Polygon vertices must be finite [u,v] pairs")
    if len(p) > 1 and np.array_equal(p[0], p[-1]):
        p = p[:-1]
    if len(p) < 3 or len(np.unique(p, axis=0)) != len(p):
        raise ValueError("A polygon requires at least three distinct vertices")
    if np.any(p < 0) or np.any(p > np.asarray(size)-1):
        raise ValueError("Polygon vertex falls outside its annotation image")

    def cross(a, b):
        return a[0]*b[1] - a[1]*b[0]

    def on(a, b, x):
        return abs(cross(b-a, x-a)) < 1e-8 and np.all(x >= np.minimum(a, b)-1e-8) and np.all(x <= np.maximum(a, b)+1e-8)

    for i in range(len(p)):
        a, b = p[i], p[(i+1) % len(p)]
        for j in range(i+1, len(p)):
            if j == i+1 or (i == 0 and j == len(p)-1):
                continue
            c, d = p[j], p[(j+1) % len(p)]
            ab = cross(b-a, c-a)*cross(b-a, d-a)
            cd = cross(d-c, a-c)*cross(d-c, b-c)
            if (ab < 0 and cd < 0) or on(a, b, c) or on(a, b, d) or on(c, d, a) or on(c, d, b):
                raise ValueError("Self-intersecting polygons are not supported")
    area = abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1)))/2
    if area < 1e-8:
        raise ValueError("Polygon has zero area")
    return p


def prepare_regions(annotation, image_shape):
    """Return polygons in processed coordinates; require an explicit resize/crop map."""
    if not isinstance(annotation, dict):
        raise ValueError("Region annotation must be a JSON object")
    size = np.asarray(annotation.get("image_size", []), float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 2) or np.any(size != np.floor(size)):
        raise ValueError("Annotation image_size must contain integer [W,H] >= 2")
    h, w = image_shape
    matrix = annotation.get("original_to_processed")
    if matrix is None:
        if not np.array_equal(size, [w, h]):
            raise ValueError("Annotation image size differs; supply original_to_processed including any crop")
        matrix = np.eye(3)
    matrix = np.asarray(matrix, float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("original_to_processed must be a finite invertible 3x3 matrix")

    def transform(vertices):
        p = _polygon(vertices, size)
        hom = np.column_stack((p, np.ones(len(p)))) @ matrix.T
        if np.any(np.abs(hom[:, 2]) < 1e-10) or np.ptp(np.sign(hom[:, 2])):
            raise ValueError("Polygon transformation crosses a projective horizon")
        return (hom[:, :2]/hom[:, 2:]).tolist()

    source_regions = annotation.get("regions", [])
    if not isinstance(source_regions, list):
        raise ValueError("regions must be a list of region objects")
    regions, ids = [], set()
    for region in source_regions:
        if not isinstance(region, dict):
            raise ValueError("Each region must be a JSON object")
        index = region.get("id")
        if not isinstance(index, int) or isinstance(index, bool) or not 1 <= index <= 65535 or index in ids:
            raise ValueError("Region IDs must be unique integers in 1..65535; 0 means ignored")
        ids.add(index)
        name = region.get("name", f"region_{index}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Region names must be nonempty strings")
        if not isinstance(region.get("polygons", []), list) or not isinstance(region.get("holes", []), list):
            raise ValueError("polygons and holes must be lists of polygons")
        polygons = [transform(p) for p in region.get("polygons", [])]
        if not polygons:
            raise ValueError(f"Region {name} requires at least one polygon")
        regions.append(dict(id=index, name=name, polygons=polygons,
                            holes=[transform(p) for p in region.get("holes", [])]))
    if not regions:
        raise ValueError("Annotate at least one polygon region")
    if not isinstance(annotation.get("exclude_polygons", []), list):
        raise ValueError("exclude_polygons must be a list of polygons")
    return dict(image_size=[int(w), int(h)], regions=regions,
                exclude_polygons=[transform(p) for p in annotation.get("exclude_polygons", [])])


def _contains(pixels, vertices):
    """Even/odd membership including the polygon boundary; supports fractional pixels."""
    p = np.asarray(vertices, float)
    candidate = np.isfinite(pixels).all(axis=1) & (pixels >= p.min(axis=0)-1e-8).all(axis=1)
    candidate &= (pixels <= p.max(axis=0)+1e-8).all(axis=1)
    idx = np.flatnonzero(candidate)
    result = np.zeros(len(pixels), bool)
    x, y = pixels[idx].T
    inside, boundary = np.zeros(len(idx), bool), np.zeros(len(idx), bool)
    for a, b in zip(p, np.roll(p, -1, axis=0)):
        dx, dy = b-a
        cross = dx*(y-a[1])-dy*(x-a[0])
        boundary |= ((np.abs(cross) <= 1e-8*max(1., np.hypot(dx, dy))) &
                     (x >= min(a[0], b[0])-1e-8) & (x <= max(a[0], b[0])+1e-8) &
                     (y >= min(a[1], b[1])-1e-8) & (y <= max(a[1], b[1])+1e-8))
        if dy != 0:
            inside ^= ((a[1] > y) != (b[1] > y)) & (x < a[0]+(y-a[1])*dx/dy)
    result[idx] = inside | boundary
    return result


def region_ids_at_pixels(processed_annotation, pixels_uv):
    pixels = np.asarray(pixels_uv, float)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels_uv must have shape (N,2)")
    size = processed_annotation["image_size"]
    inside = np.isfinite(pixels).all(axis=1) & (pixels >= 0).all(axis=1) & (pixels <= np.asarray(size)-1).all(axis=1)
    labels = np.zeros(len(pixels), np.uint16)
    excluded = np.zeros(len(pixels), bool)
    for p in processed_annotation.get("exclude_polygons", []):
        excluded |= _contains(pixels, p)
    for region in processed_annotation["regions"]:
        selected = np.zeros(len(pixels), bool)
        for p in region["polygons"]:
            selected |= _contains(pixels, p)
        for p in region.get("holes", []):
            selected &= ~_contains(pixels, p)
        selected &= inside & ~excluded
        if np.any(selected & (labels != 0)):
            raise ValueError(f"Region {region['name']} overlaps another region; leave a boundary gap")
        labels[selected] = region["id"]
    return labels


def rasterize_regions(annotation, image_shape):
    processed = prepare_regions(annotation, image_shape)
    h, w = image_shape
    v, u = np.mgrid[:h, :w]
    labels = region_ids_at_pixels(processed, np.column_stack((u.ravel(), v.ravel()))).reshape(h, w)
    for region in processed["regions"]:
        if not np.any(labels == region["id"]):
            raise ValueError(f"Region {region['name']} covers no processed pixels")
    return labels, processed

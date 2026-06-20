"""Dependency-light topology and geometry QA for Open3D triangle meshes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class MeshComponent:
    rank: int
    vertices: int
    triangles: int
    vertex_fraction: float
    triangle_fraction: float
    surface_area_m2: float
    centroid_m: tuple[float, float, float]
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    extents_m: tuple[float, float, float]


@dataclass(frozen=True)
class DenseMeshReport:
    source_mesh: str
    vertices: int
    triangles: int
    surface_area_m2: float
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    extents_m: tuple[float, float, float]
    diagonal_m: float
    connected_components: int
    meaningful_components: int
    largest_component_vertex_fraction: float
    second_component_vertex_fraction: float
    largest_two_centroid_distance_m: float | None
    largest_two_sampled_gap_m: float | None
    boundary_edges: int
    non_manifold_edges: int
    degenerate_triangles: int
    watertight: bool
    components: tuple[MeshComponent, ...]
    gates: dict[str, bool]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_open3d_triangle_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the fixed-width binary PLY layout emitted by Open3D."""
    source = Path(path)
    with source.open("rb") as handle:
        header: list[str] = []
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PLY header is incomplete: {source}")
            line = raw.decode("ascii", errors="strict").strip()
            header.append(line)
            if line == "end_header":
                break

        if "format binary_little_endian 1.0" not in header:
            raise ValueError("only binary little-endian PLY meshes are supported")

        vertex_count = _element_count(header, "vertex")
        face_count = _element_count(header, "face")
        expected_vertex_properties = (
            "property double x",
            "property double y",
            "property double z",
            "property double nx",
            "property double ny",
            "property double nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        )
        if not all(item in header for item in expected_vertex_properties):
            raise ValueError("unsupported PLY vertex layout; expected Open3D XYZ/normal/RGB fields")
        if "property list uchar uint vertex_indices" not in header:
            raise ValueError("unsupported PLY face layout")

        vertex_dtype = np.dtype(
            [
                ("x", "<f8"),
                ("y", "<f8"),
                ("z", "<f8"),
                ("nx", "<f8"),
                ("ny", "<f8"),
                ("nz", "<f8"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        records = np.fromfile(handle, dtype=vertex_dtype, count=vertex_count)
        if len(records) != vertex_count:
            raise ValueError("PLY ended before all vertices were read")
        vertices = np.column_stack((records["x"], records["y"], records["z"]))

        face_dtype = np.dtype([("size", "u1"), ("indices", "<u4", (3,))])
        face_records = np.fromfile(handle, dtype=face_dtype, count=face_count)
        if len(face_records) != face_count:
            raise ValueError("PLY ended before all faces were read")
        if np.any(face_records["size"] != 3):
            raise ValueError("only triangle faces are supported")
        triangles = np.asarray(face_records["indices"], dtype=np.int64)

    if np.any(~np.isfinite(vertices)):
        raise ValueError("mesh contains non-finite vertices")
    if triangles.size and (np.min(triangles) < 0 or np.max(triangles) >= len(vertices)):
        raise ValueError("mesh contains out-of-range vertex indices")
    return vertices, triangles


def _element_count(header: list[str], name: str) -> int:
    prefix = f"element {name} "
    for line in header:
        if line.startswith(prefix):
            return int(line.removeprefix(prefix))
    raise ValueError(f"PLY is missing the {name} element")


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[item] != item:
            parent = int(self.parent[item])
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: int, second: int) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1


def analyze_dense_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    source: Path,
) -> DenseMeshReport:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError("vertices must be a non-empty Nx3 array")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles):
        raise ValueError("triangles must be a non-empty Mx3 array")

    first = vertices[triangles[:, 0]]
    second = vertices[triangles[:, 1]]
    third = vertices[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(second - first, third - first), axis=1)
    degenerate = int(np.count_nonzero(areas <= 1e-12))

    union_find = _UnionFind(len(vertices))
    for a, b, c in triangles:
        union_find.union(int(a), int(b))
        union_find.union(int(a), int(c))
    roots = np.fromiter((union_find.find(i) for i in range(len(vertices))), dtype=np.int64)

    component_vertices: dict[int, list[int]] = {}
    for index, root in enumerate(roots):
        component_vertices.setdefault(int(root), []).append(index)
    face_roots = roots[triangles[:, 0]]
    face_counts = dict(zip(*np.unique(face_roots, return_counts=True), strict=False))
    area_by_root = {
        int(root): float(np.sum(areas[face_roots == root])) for root in np.unique(face_roots)
    }

    ranked = sorted(component_vertices.items(), key=lambda item: len(item[1]), reverse=True)
    components: list[MeshComponent] = []
    component_arrays: list[np.ndarray] = []
    for rank, (root, indices) in enumerate(ranked, start=1):
        index_array = np.asarray(indices, dtype=np.int64)
        component_arrays.append(index_array)
        points = vertices[index_array]
        lower = np.min(points, axis=0)
        upper = np.max(points, axis=0)
        triangle_count = int(face_counts.get(root, 0))
        components.append(
            MeshComponent(
                rank=rank,
                vertices=len(index_array),
                triangles=triangle_count,
                vertex_fraction=len(index_array) / len(vertices),
                triangle_fraction=triangle_count / len(triangles),
                surface_area_m2=area_by_root.get(root, 0.0),
                centroid_m=tuple(float(value) for value in np.mean(points, axis=0)),
                bounds_min_m=tuple(float(value) for value in lower),
                bounds_max_m=tuple(float(value) for value in upper),
                extents_m=tuple(float(value) for value in upper - lower),
            )
        )

    edges = np.concatenate(
        (triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]), axis=0
    )
    edges.sort(axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(edge_counts == 1))
    non_manifold_edges = int(np.count_nonzero(edge_counts > 2))

    largest_fraction = components[0].vertex_fraction
    second_fraction = components[1].vertex_fraction if len(components) > 1 else 0.0
    centroid_distance = None
    sampled_gap = None
    if len(components) > 1:
        centroid_distance = float(
            np.linalg.norm(
                np.asarray(components[0].centroid_m) - np.asarray(components[1].centroid_m)
            )
        )
        sampled_gap = _sampled_component_gap(
            vertices[component_arrays[0]], vertices[component_arrays[1]]
        )

    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    meaningful = sum(component.vertex_fraction >= 0.01 for component in components)
    gates = {
        "single_dominant_component_at_least_95_percent": largest_fraction >= 0.95,
        "no_secondary_component_over_5_percent": second_fraction <= 0.05,
        "no_non_manifold_edges": non_manifold_edges == 0,
        "no_degenerate_triangles": degenerate == 0,
    }
    return DenseMeshReport(
        source_mesh=str(source),
        vertices=len(vertices),
        triangles=len(triangles),
        surface_area_m2=float(np.sum(areas)),
        bounds_min_m=tuple(float(value) for value in lower),
        bounds_max_m=tuple(float(value) for value in upper),
        extents_m=tuple(float(value) for value in upper - lower),
        diagonal_m=float(np.linalg.norm(upper - lower)),
        connected_components=len(components),
        meaningful_components=meaningful,
        largest_component_vertex_fraction=largest_fraction,
        second_component_vertex_fraction=second_fraction,
        largest_two_centroid_distance_m=centroid_distance,
        largest_two_sampled_gap_m=sampled_gap,
        boundary_edges=boundary_edges,
        non_manifold_edges=non_manifold_edges,
        degenerate_triangles=degenerate,
        watertight=boundary_edges == 0 and non_manifold_edges == 0,
        components=tuple(components[:10]),
        gates=gates,
    )


def _sampled_component_gap(first: np.ndarray, second: np.ndarray, limit: int = 4000) -> float:
    first_sample = _even_sample(first, limit)
    second_sample = _even_sample(second, limit)
    minimum_squared = math.inf
    for start in range(0, len(first_sample), 250):
        chunk = first_sample[start : start + 250]
        distances = np.sum((chunk[:, None, :] - second_sample[None, :, :]) ** 2, axis=2)
        minimum_squared = min(minimum_squared, float(np.min(distances)))
    return math.sqrt(minimum_squared)


def _even_sample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points
    return points[np.linspace(0, len(points) - 1, limit, dtype=np.int64)]


def write_dense_mesh_qa(mesh_path: Path, output_dir: Path) -> DenseMeshReport:
    vertices, triangles = read_open3d_triangle_mesh(mesh_path)
    report = analyze_dense_mesh(vertices, triangles, Path(mesh_path))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "dense_mesh_qa_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_component_preview(vertices, triangles, report, destination / "dense_components.png")
    return report


def _write_component_preview(
    vertices: np.ndarray,
    triangles: np.ndarray,
    report: DenseMeshReport,
    destination: Path,
) -> None:
    union_find = _UnionFind(len(vertices))
    for a, b, c in triangles:
        union_find.union(int(a), int(b))
        union_find.union(int(a), int(c))
    roots = np.fromiter((union_find.find(i) for i in range(len(vertices))), dtype=np.int64)
    unique, counts = np.unique(roots, return_counts=True)
    ordered_roots = unique[np.argsort(counts)[::-1]]
    rank_by_root = {int(root): rank for rank, root in enumerate(ordered_roots)}
    ranks = np.fromiter((rank_by_root[int(root)] for root in roots), dtype=np.int64)

    centered = vertices - np.mean(vertices, axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes[:2].T
    width, height, margin = 1400, 900, 80
    lower = np.min(projected, axis=0)
    upper = np.max(projected, axis=0)
    span = np.maximum(upper - lower, 1e-12)
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    pixels = np.empty_like(projected)
    pixels[:, 0] = margin + (projected[:, 0] - lower[0]) * scale
    pixels[:, 1] = height - margin - (projected[:, 1] - lower[1]) * scale
    pixels = np.clip(np.rint(pixels).astype(np.int32), 0, [[width - 1, height - 1]])

    canvas = np.full((height, width, 3), 242, dtype=np.uint8)
    palette = np.asarray(
        [
            (55, 105, 230),
            (230, 120, 45),
            (70, 180, 95),
            (180, 75, 190),
            (80, 185, 205),
        ],
        dtype=np.uint8,
    )
    colors = palette[np.minimum(ranks, len(palette) - 1)]
    canvas[pixels[:, 1], pixels[:, 0]] = colors
    title = (
        f"Dense mesh topology | {report.vertices:,} vertices | "
        f"{report.connected_components} components | "
        f"largest {report.largest_component_vertex_fraction:.1%}"
    )
    cv2.putText(
        canvas,
        title,
        (margin, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    if report.largest_two_sampled_gap_m is not None:
        cv2.putText(
            canvas,
            f"sampled gap between two largest components: {report.largest_two_sampled_gap_m:.2f} m",
            (margin, height - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(destination), canvas):
        raise RuntimeError(f"failed to write component preview: {destination}")

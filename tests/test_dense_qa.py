from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from reconbot.dense_qa import analyze_dense_mesh, read_open3d_triangle_mesh, write_dense_mesh_qa


def _two_tetrahedra() -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)], dtype=np.float64
    )
    second = first + np.asarray((10, 0, 0), dtype=np.float64)
    faces = np.asarray([(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)], dtype=np.int64)
    return np.vstack((first, second)), np.vstack((faces, faces + 4))


def _write_open3d_ply(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Created by Open3D\n"
        f"element vertex {len(vertices)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property double nx\nproperty double ny\nproperty double nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"element face {len(triangles)}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for x, y, z in vertices:
            handle.write(struct.pack("<6d3B", x, y, z, 0, 0, 1, 200, 210, 220))
        for a, b, c in triangles:
            handle.write(struct.pack("<BIII", 3, a, b, c))


def test_analyze_dense_mesh_finds_two_watertight_components(tmp_path):
    vertices, triangles = _two_tetrahedra()

    report = analyze_dense_mesh(vertices, triangles, tmp_path / "mesh.ply")

    assert report.connected_components == 2
    assert report.meaningful_components == 2
    assert report.largest_component_vertex_fraction == pytest.approx(0.5)
    assert report.second_component_vertex_fraction == pytest.approx(0.5)
    assert report.largest_two_sampled_gap_m == pytest.approx(9.0)
    assert report.boundary_edges == 0
    assert report.non_manifold_edges == 0
    assert report.watertight is True
    assert report.gates["single_dominant_component_at_least_95_percent"] is False


def test_open3d_ply_round_trip_and_artifacts(tmp_path):
    vertices, triangles = _two_tetrahedra()
    mesh = tmp_path / "mesh.ply"
    output = tmp_path / "qa"
    _write_open3d_ply(mesh, vertices, triangles)

    loaded_vertices, loaded_triangles = read_open3d_triangle_mesh(mesh)
    report = write_dense_mesh_qa(mesh, output)

    assert loaded_vertices == pytest.approx(vertices)
    assert np.array_equal(loaded_triangles, triangles)
    assert report.vertices == 8
    assert (output / "dense_components.png").is_file()
    payload = json.loads((output / "dense_mesh_qa_report.json").read_text(encoding="utf-8"))
    assert payload["connected_components"] == 2

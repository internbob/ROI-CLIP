from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np


ICO_LEVEL_TO_VERTICES = {
    0: 12,
    1: 42,
    2: 162,
    3: 642,
    4: 2562,
    5: 10242,
    6: 40962,
    7: 163842,
}


def normalize_rows(values: np.ndarray) -> np.ndarray:
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)


def infer_ico_level(num_vertices: int) -> int:
    for level, count in ICO_LEVEL_TO_VERTICES.items():
        if count == int(num_vertices):
            return level
    supported = ", ".join(str(v) for v in ICO_LEVEL_TO_VERTICES.values())
    raise ValueError(f"Unsupported num_vertices={num_vertices}. Supported: {supported}")


def create_icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )
    return normalize_rows(vertices), faces


def midpoint_index(
    i: int,
    j: int,
    vertices: List[np.ndarray],
    midpoint_cache: Dict[Tuple[int, int], int],
) -> int:
    key = (min(i, j), max(i, j))
    if key not in midpoint_cache:
        midpoint = (vertices[i] + vertices[j]) / 2.0
        midpoint = midpoint / (np.linalg.norm(midpoint) + 1e-8)
        vertices.append(midpoint.astype(np.float32))
        midpoint_cache[key] = len(vertices) - 1
    return midpoint_cache[key]


@lru_cache(maxsize=8)
def generate_icosphere(level: int) -> Tuple[np.ndarray, np.ndarray]:
    vertices, faces = create_icosahedron()
    vertex_list = [vertex.astype(np.float32) for vertex in vertices]

    for _ in range(int(level)):
        midpoint_cache: Dict[Tuple[int, int], int] = {}
        refined_faces = []
        for tri in faces:
            a, b, c = [int(v) for v in tri]
            ab = midpoint_index(a, b, vertex_list, midpoint_cache)
            bc = midpoint_index(b, c, vertex_list, midpoint_cache)
            ca = midpoint_index(c, a, vertex_list, midpoint_cache)
            refined_faces.extend(
                [
                    (a, ab, ca),
                    (b, bc, ab),
                    (c, ca, bc),
                    (ab, bc, ca),
                ]
            )
        faces = np.asarray(refined_faces, dtype=np.int64)

    vertices = normalize_rows(np.asarray(vertex_list, dtype=np.float32))
    return vertices, faces


def build_adjacency(num_vertices: int, faces: np.ndarray) -> List[List[int]]:
    adjacency = [set() for _ in range(int(num_vertices))]
    for tri in faces:
        a, b, c = [int(v) for v in tri]
        adjacency[a].update([b, c])
        adjacency[b].update([a, c])
        adjacency[c].update([a, b])
    return [sorted(neigh) for neigh in adjacency]


def sorted_one_ring(vertices: np.ndarray, adjacency: List[List[int]]) -> np.ndarray:
    one_ring = np.zeros((vertices.shape[0], 7), dtype=np.int64)
    for idx, neighs in enumerate(adjacency):
        center = vertices[idx]
        normal = center / (np.linalg.norm(center) + 1e-8)
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(normal, reference))) > 0.9:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis_x = np.cross(normal, reference)
        axis_x = axis_x / (np.linalg.norm(axis_x) + 1e-8)
        axis_y = np.cross(normal, axis_x)

        ordered = []
        for neigh_idx in neighs:
            rel = vertices[neigh_idx] - center
            angle = np.arctan2(float(np.dot(rel, axis_y)), float(np.dot(rel, axis_x)))
            ordered.append((angle, int(neigh_idx)))
        ordered = [neigh_idx for angle, neigh_idx in sorted(ordered, key=lambda item: item[0])]
        while len(ordered) < 6:
            ordered.append(int(idx))
        one_ring[idx, 0] = int(idx)
        one_ring[idx, 1:] = np.asarray(ordered[:6], dtype=np.int64)
    return one_ring


@lru_cache(maxsize=8)
def get_ico_topology(level: int) -> Tuple[np.ndarray, np.ndarray]:
    vertices, faces = generate_icosphere(level)
    adjacency = build_adjacency(vertices.shape[0], faces)
    return vertices, sorted_one_ring(vertices, adjacency)


def build_resample_index(source_coords: np.ndarray, target_coords: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
    source = normalize_rows(np.asarray(source_coords, dtype=np.float32))
    target = normalize_rows(np.asarray(target_coords, dtype=np.float32))
    source_t = source.transpose(1, 0)
    indices = []
    for start in range(0, target.shape[0], int(chunk_size)):
        chunk = target[start : start + int(chunk_size)]
        scores = chunk @ source_t
        indices.append(scores.argmax(axis=1).astype(np.int64))
    return np.concatenate(indices, axis=0)

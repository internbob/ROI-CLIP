import numpy as np

from data.dkt_topology import mesh_boundary_graph, reorder_topology, row_normalize_boundary


def test_mesh_boundary_graph_counts_unique_triangle_edges():
    # Two triangles share edge (1,2). Labels 0/1 meet along (0,2), (1,2),
    # and (1,3); the shared triangle edge is counted only once.
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    adjacency, boundary = mesh_boundary_graph(faces, labels, num_rois=2)
    np.testing.assert_array_equal(adjacency, np.asarray([[0, 1], [1, 0]], dtype=np.uint8))
    assert boundary[0, 1] == 3
    assert boundary[1, 0] == 3


def test_boundary_strength_is_row_normalized_and_zero_for_isolate():
    boundary = np.asarray([[0, 2, 1], [2, 0, 0], [1, 0, 0]], dtype=np.int64)
    strength = row_normalize_boundary(boundary)
    np.testing.assert_allclose(strength.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(strength[0], [0.0, 2.0 / 3.0, 1.0 / 3.0])


def test_reorder_topology_preserves_exact_homology():
    topology = {
        "region_names": ("lh_a", "lh_b", "rh_a", "rh_b"),
        "adjacency": np.asarray([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
        "boundary_edge_count": np.asarray([[0, 2, 0, 0], [2, 0, 0, 0], [0, 0, 0, 3], [0, 0, 3, 0]]),
        "boundary_strength": np.asarray([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.float32),
        "homology": np.asarray([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]),
    }
    target = ("lh_b", "lh_a", "rh_b", "rh_a")
    reordered = reorder_topology(topology, target)
    assert tuple(reordered["region_names"]) == target
    np.testing.assert_array_equal(reordered["homolog_index"], [2, 3, 0, 1])
    assert np.all(np.asarray(reordered["homology"]).sum(axis=1) == 1)

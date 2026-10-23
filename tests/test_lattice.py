import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from forest_fire.lattice import Boundary, Neighbourhood, has_burning_neighbour, label_clusters

E, T, B = 0, 1, 2


def grid(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def test_von_neumann_periodic_neighbours_wrap_around_edges() -> None:
    g = grid(
        [
            [B, E, E, E],
            [E, E, E, E],
            [E, E, E, E],
            [E, E, E, E],
        ]
    )
    expected = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    )
    result = has_burning_neighbour(g, Neighbourhood.VON_NEUMANN, Boundary.PERIODIC)
    np.testing.assert_array_equal(result, expected)


def test_fixed_boundary_neighbours_do_not_wrap() -> None:
    g = grid(
        [
            [B, E, E, E],
            [E, E, E, E],
            [E, E, E, E],
            [E, E, E, E],
        ]
    )
    expected = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )
    result = has_burning_neighbour(g, Neighbourhood.VON_NEUMANN, Boundary.FIXED)
    np.testing.assert_array_equal(result, expected)


def test_moore_neighbourhood_includes_diagonals() -> None:
    g = grid(
        [
            [E, E, E, E, E],
            [E, E, E, E, E],
            [E, E, B, E, E],
            [E, E, E, E, E],
            [E, E, E, E, E],
        ]
    )
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=bool,
    )
    result = has_burning_neighbour(g, Neighbourhood.MOORE, Boundary.FIXED)
    np.testing.assert_array_equal(result, expected)


def cluster_sizes(
    trees: list[list[int]], neighbourhood: Neighbourhood, boundary: Boundary
) -> list[int]:
    labels, n = label_clusters(np.array(trees, dtype=bool), neighbourhood, boundary)
    assert set(np.unique(labels)) <= set(range(n + 1))
    return sorted(int(np.sum(labels == k)) for k in range(1, n + 1))


EDGE_SPANNING = [
    [0, 0, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 0, 0],
    [0, 1, 0, 0],
]


def test_periodic_clusters_merge_across_edges() -> None:
    # (1,0)-(1,1) joins (1,3) across the left/right edge; (3,1) joins nothing.
    assert cluster_sizes(EDGE_SPANNING, Neighbourhood.VON_NEUMANN, Boundary.PERIODIC) == [1, 3]


def test_fixed_clusters_do_not_merge_across_edges() -> None:
    assert cluster_sizes(EDGE_SPANNING, Neighbourhood.VON_NEUMANN, Boundary.FIXED) == [1, 1, 2]


@pytest.mark.parametrize("neighbourhood", list(Neighbourhood))
@given(
    trees=arrays(np.bool_, st.tuples(st.integers(1, 8), st.integers(1, 8))),
    shift=st.tuples(st.integers(0, 7), st.integers(0, 7)),
)
def test_periodic_cluster_sizes_are_translation_invariant(
    neighbourhood: Neighbourhood, trees: np.ndarray, shift: tuple[int, int]
) -> None:
    rolled = np.roll(trees, shift, axis=(0, 1))
    assert cluster_sizes(trees.tolist(), neighbourhood, Boundary.PERIODIC) == cluster_sizes(
        rolled.tolist(), neighbourhood, Boundary.PERIODIC
    )

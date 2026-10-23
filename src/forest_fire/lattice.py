"""The lattice: cell states, neighbourhoods and boundaries shared by both simulation modes."""

from enum import IntEnum, StrEnum

import numpy as np
import numpy.typing as npt
from scipy import ndimage, sparse
from scipy.sparse import csgraph

Grid = npt.NDArray[np.int8]
Mask = npt.NDArray[np.bool_]
Labels = npt.NDArray[np.intp]


class State(IntEnum):
    EMPTY = 0
    TREE = 1
    BURNING = 2


class Neighbourhood(StrEnum):
    VON_NEUMANN = "von-neumann"
    MOORE = "moore"


class Boundary(StrEnum):
    PERIODIC = "periodic"
    FIXED = "fixed"


_OFFSETS = {
    Neighbourhood.VON_NEUMANN: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    Neighbourhood.MOORE: [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)],
}

_STRUCTURES = {
    Neighbourhood.VON_NEUMANN: ndimage.generate_binary_structure(2, 1),
    Neighbourhood.MOORE: ndimage.generate_binary_structure(2, 2),
}


def has_burning_neighbour(grid: Grid, neighbourhood: Neighbourhood, boundary: Boundary) -> Mask:
    """Mask of cells with at least one burning cell in their neighbourhood."""
    burning: Mask = grid == State.BURNING
    result = np.zeros_like(burning)
    for dr, dc in _OFFSETS[neighbourhood]:
        result |= _shift(burning, dr, dc, boundary)
    return result


def label_clusters(
    trees: Mask, neighbourhood: Neighbourhood, boundary: Boundary
) -> tuple[Labels, int]:
    """Label connected clusters of trees.

    Returns ``(labels, n)``: ``labels`` is 0 off-tree and 1..n on trees, one label per cluster.
    """
    labels, n = ndimage.label(trees, structure=_STRUCTURES[neighbourhood])
    if boundary is Boundary.FIXED or n == 0:
        return labels, n

    # Clusters that touch across a wrapped edge are one cluster: union their labels.
    pairs = [
        np.stack([labels, np.roll(labels, (dr, dc), axis=(0, 1))]).reshape(2, -1)
        for dr, dc in _OFFSETS[neighbourhood]
    ]
    a, b = np.concatenate(pairs, axis=1)
    touching = (a > 0) & (b > 0) & (a != b)
    graph = sparse.coo_matrix(
        (np.ones(touching.sum()), (a[touching], b[touching])), shape=(n + 1, n + 1)
    )
    _, component = csgraph.connected_components(graph, directed=False)
    # Label 0 (off-tree) touches nothing, so it is its own component; renumber the rest 1..m.
    _, merged = np.unique(component[1:], return_inverse=True)
    relabel = np.concatenate([[0], merged + 1])
    return relabel[labels], int(merged.max()) + 1


def _shift(mask: Mask, dr: int, dc: int, boundary: Boundary) -> Mask:
    """Shift a mask by (dr, dc); cells shifted in from beyond a fixed edge are False."""
    if boundary is Boundary.PERIODIC:
        return np.roll(mask, (dr, dc), axis=(0, 1))
    rows, cols = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    return padded[1 - dr : 1 - dr + rows, 1 - dc : 1 - dc + cols]

"""Instantaneous mode: the time-scale-separated limit, advanced in events (see ADR 0001).

Each event makes ``theta`` planting attempts on random cells, then strikes one random cell;
if the struck cell holds a tree, its whole cluster burns at once and the fire size is recorded.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from forest_fire.lattice import Boundary, Grid, Neighbourhood, State, neighbour_offsets

# Random draws are made in chunks of events, bounding memory to CHUNK * theta indices.
_CHUNK = 1024


@dataclass(frozen=True)
class Params:
    theta: int
    """Planting attempts per lightning strike (p/f)."""
    neighbourhood: Neighbourhood = Neighbourhood.VON_NEUMANN
    boundary: Boundary = Boundary.PERIODIC


@dataclass(frozen=True)
class Result:
    fire_sizes: npt.NDArray[np.int64]
    """Size of each fire, in order; strikes on empty cells cause no fire and are not recorded."""
    tree_density: npt.NDArray[np.float64]
    """Tree density after each event."""
    grid: Grid
    """The lattice after the final event."""


def simulate(grid: Grid, params: Params, n_events: int, rng: np.random.Generator) -> Result:
    """Run ``n_events`` events from ``grid`` (which must hold only empty and tree cells)."""
    if np.any(grid == State.BURNING):
        raise ValueError("instantaneous mode takes a grid with no burning cells")
    grid = grid.copy()
    flat = grid.reshape(-1)
    burn = _Burner(grid.shape, params)
    fire_sizes: list[int] = []
    tree_density = np.empty(n_events)
    n_trees = int(np.count_nonzero(flat == State.TREE))

    for start in range(0, n_events, _CHUNK):
        stop = min(start + _CHUNK, n_events)
        attempts = rng.integers(0, flat.size, size=(stop - start, params.theta))
        strikes = rng.integers(0, flat.size, size=stop - start)
        for i, event in enumerate(range(start, stop)):
            planted = attempts[i][flat[attempts[i]] == State.EMPTY]
            flat[planted] = State.TREE
            n_trees += np.unique(planted).size

            if flat[strikes[i]] == State.TREE:
                size = burn(flat, int(strikes[i]))
                fire_sizes.append(size)
                n_trees -= size
            tree_density[event] = n_trees / flat.size

    return Result(np.array(fire_sizes, dtype=np.int64), tree_density, grid)


class _Burner:
    """Burns the cluster containing a struck tree by breadth-first flood fill.

    The whole fire front advances in one vectorised operation, so the cost scales with the
    fire's size and diameter rather than with the lattice area.
    """

    def __init__(self, shape: tuple[int, ...], params: Params) -> None:
        self.rows, self.cols = shape
        offsets = np.array(neighbour_offsets(params.neighbourhood), dtype=np.intp)
        self.dr, self.dc = offsets[:, 0], offsets[:, 1]
        self.periodic = params.boundary is Boundary.PERIODIC

    def __call__(self, flat: Grid, strike: int) -> int:
        """Empty every tree connected to ``strike`` in place; return how many burned."""
        flat[strike] = State.EMPTY
        front = np.array([strike], dtype=np.intp)
        size = 1
        while front.size:
            r, c = np.divmod(front, self.cols)
            nr = r[:, None] + self.dr
            nc = c[:, None] + self.dc
            if self.periodic:
                neighbours = (nr % self.rows) * self.cols + nc % self.cols
            else:
                inside = (nr >= 0) & (nr < self.rows) & (nc >= 0) & (nc < self.cols)
                neighbours = (nr * self.cols + nc)[inside]
            neighbours = neighbours.ravel()
            front = np.unique(neighbours[flat[neighbours] == State.TREE])
            flat[front] = State.EMPTY
            size += front.size
        return size

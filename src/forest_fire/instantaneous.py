"""Instantaneous mode: the time-scale-separated limit, advanced in events (see ADR 0001).

Each event makes ``theta`` planting attempts on random cells, then strikes one random cell;
if the struck cell holds a tree, its whole cluster burns at once and the fire size is recorded.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from forest_fire.lattice import Boundary, Grid, Neighbourhood, State, label_clusters


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
    grid = grid.copy()
    flat = grid.reshape(-1)
    n_cells = flat.size
    fire_sizes: list[int] = []
    tree_density = np.empty(n_events)
    n_trees = int(np.count_nonzero(flat == State.TREE))

    for event in range(n_events):
        planted = rng.integers(0, n_cells, size=params.theta)
        flat[planted] = State.TREE
        n_trees = int(np.count_nonzero(flat == State.TREE))

        strike = int(rng.integers(0, n_cells))
        if flat[strike] == State.TREE:
            labels, _ = label_clusters(grid == State.TREE, params.neighbourhood, params.boundary)
            burnt = labels.reshape(-1) == labels.reshape(-1)[strike]
            flat[burnt] = State.EMPTY
            size = int(np.count_nonzero(burnt))
            fire_sizes.append(size)
            n_trees -= size

        tree_density[event] = n_trees / n_cells

    return Result(np.array(fire_sizes, dtype=np.int64), tree_density, grid)

"""Stepwise mode: the four Drossel-Schwabl rules applied simultaneously to every cell."""

from dataclasses import dataclass

import numpy as np

from forest_fire.lattice import Boundary, Grid, Neighbourhood, State, has_burning_neighbour


@dataclass(frozen=True)
class Params:
    p: float
    """Growth probability per empty cell per step."""
    f: float
    """Lightning probability per tree per step."""
    neighbourhood: Neighbourhood = Neighbourhood.VON_NEUMANN
    boundary: Boundary = Boundary.PERIODIC


def step(grid: Grid, params: Params, rng: np.random.Generator) -> Grid:
    """Advance the lattice one time step. Every rule reads the old grid; returns a new grid."""
    draw = rng.random(grid.shape)
    tree = grid == State.TREE
    empty = grid == State.EMPTY
    struck = draw < params.f
    catches = tree & (struck | has_burning_neighbour(grid, params.neighbourhood, params.boundary))
    grows = empty & (draw < params.p)

    new = grid.copy()
    new[grid == State.BURNING] = State.EMPTY
    new[catches] = State.BURNING
    new[grows] = State.TREE
    return new

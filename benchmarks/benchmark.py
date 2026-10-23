"""Time one stepwise-mode step: the original per-cell loop against the vectorised version.

Run with ``python benchmarks/benchmark.py``; prints a Markdown table for the README.
"""

import timeit

import numpy as np

from forest_fire import stepwise
from forest_fire.lattice import Boundary, Neighbourhood, random_lattice


def moore_neighbourhood(x, y, grid):  # type: ignore[no-untyped-def]
    """The original implementation's neighbourhood lookup, verbatim apart from formatting."""
    x_lower, x_upper = 1, 1
    y_lower, y_upper = 1, 1
    grid_length = grid.shape[0]
    if (x == 0) & (y == 0):
        x_lower, y_lower = 0, 0
    elif x == 0:
        x_lower = 0
    elif y == 0:
        y_lower = 0
    if (x == grid_length) & (y == grid_length):
        x_upper, y_upper = 0, 0
    elif x == grid_length:
        x_upper = 0
    elif y == grid_length:
        y_upper = 0
    return grid[x - x_lower : x + x_upper + 1, y - y_lower : y + y_upper + 1]


def original_step(grid: np.ndarray, p: float, f: float, rng: np.random.Generator) -> np.ndarray:
    """One step of the original script's update loop (Moore neighbourhood, fixed boundary)."""
    grid = grid.copy()
    grid_old = grid.copy()
    grid[grid_old == 2] = 0
    x, y = np.where(grid_old == 1)
    trees = list(zip(x, y, strict=True))
    grid[grid_old == 1] = rng.choice([1, 2], len(trees), p=[1 - f, f])
    grid[grid_old == 0] = rng.choice([0, 1], np.sum(grid_old == 0), p=[1 - p, p])
    for tx, ty in trees:
        if np.any(moore_neighbourhood(tx, ty, grid_old) == 2):
            grid[tx, ty] = 2
    return grid


def main() -> None:
    p, f = 0.01, 0.0001
    params = stepwise.Params(p, f, Neighbourhood.MOORE, Boundary.FIXED)
    print("| L | original (ms/step) | vectorised (ms/step) | speed-up |")
    print("|---|---:|---:|---:|")
    for size in (100, 200, 400):
        rng = np.random.default_rng(0)
        grid = random_lattice(size, 0.5, rng)
        grid[rng.random(grid.shape) < 0.01] = 2
        repeats = 3 if size <= 200 else 1
        old = (
            timeit.timeit(lambda g=grid, r=rng: original_step(g, p, f, r), number=repeats) / repeats
        )
        new = timeit.timeit(lambda g=grid, r=rng: stepwise.step(g, params, r), number=50) / 50
        print(f"| {size} | {old * 1e3:,.1f} | {new * 1e3:.2f} | {old / new:,.0f}x |")


if __name__ == "__main__":
    main()

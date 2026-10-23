"""Animating stepwise mode with matplotlib."""

import numpy as np
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.image import AxesImage

from forest_fire import stepwise
from forest_fire.lattice import Grid

# Empty ground, tree, burning: indexed by State.
CMAP = colors.ListedColormap(["#1b1b1b", "#2d6a4f", "#f77f00"])


def animate(
    grid: Grid, params: stepwise.Params, steps: int, rng: np.random.Generator
) -> FuncAnimation:
    """An animation of ``steps`` stepwise-mode time steps starting from ``grid``."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    image = ax.imshow(grid, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    state = {"grid": grid}

    def update(_frame: int) -> tuple[AxesImage]:
        state["grid"] = stepwise.step(state["grid"], params, rng)
        image.set_data(state["grid"])
        return (image,)

    return FuncAnimation(fig, update, frames=steps, interval=50, blit=True)

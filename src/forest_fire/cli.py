"""Simulate the Drossel-Schwabl forest-fire model."""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from forest_fire import instantaneous, powerlaw, stepwise
from forest_fire.lattice import Boundary, Neighbourhood, random_lattice


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.command(args)


def _animate(args: argparse.Namespace) -> None:
    import matplotlib

    if not args.live:
        matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from forest_fire.animate import animate

    rng = np.random.default_rng(args.seed)
    params = stepwise.Params(args.p, args.f, args.neighbourhood, args.boundary)
    grid = random_lattice(args.size, args.density, rng)
    animation = animate(grid, params, args.steps, rng)
    if args.live:
        plt.show()
    else:
        animation.save(args.save, writer="pillow", fps=args.fps)
        print(f"saved {args.steps} frames to {args.save}")


def _run(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    params = instantaneous.Params(args.theta, args.neighbourhood, args.boundary)
    grid = np.zeros((args.size, args.size), dtype=np.int8)

    burn_in = instantaneous.simulate(grid, params, args.burn_in, rng)
    result = instantaneous.simulate(burn_in.grid, params, args.events, rng)
    np.savez_compressed(args.out, fire_sizes=result.fire_sizes, tree_density=result.tree_density)

    f = powerlaw.fit(result.fire_sizes, s_max=args.s_max)
    print(f"{result.fire_sizes.size} fires, mean tree density {result.tree_density.mean():.4f}")
    print(f"tau = {f.tau:.3f} ± {f.sigma:.3f}  (s_min={f.s_min}, s_max={f.s_max}, n={f.n})")
    print(f"saved to {args.out}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forest-fire", description=__doc__)
    commands = parser.add_subparsers(required=True)

    lattice = argparse.ArgumentParser(add_help=False)
    lattice.add_argument(
        "--neighbourhood",
        type=Neighbourhood,
        default=Neighbourhood.VON_NEUMANN,
        choices=list(Neighbourhood),
    )
    lattice.add_argument(
        "--boundary", type=Boundary, default=Boundary.PERIODIC, choices=list(Boundary)
    )
    lattice.add_argument("--seed", type=int, default=None, help="random seed (default: random)")

    animate = commands.add_parser(
        "animate", parents=[lattice], help="animate stepwise mode as a GIF or live window"
    )
    animate.set_defaults(command=_animate)
    animate.add_argument("--size", type=int, default=200, help="lattice side length L")
    animate.add_argument("--p", type=float, default=0.01, help="growth probability")
    animate.add_argument("--f", type=float, default=0.0001, help="lightning probability")
    animate.add_argument("--density", type=float, default=0.5, help="initial tree density")
    animate.add_argument("--steps", type=int, default=200)
    animate.add_argument("--fps", type=int, default=20)
    output = animate.add_mutually_exclusive_group(required=True)
    output.add_argument("--save", type=Path, help="write the animation to this GIF")
    output.add_argument("--live", action="store_true", help="show it in a window instead")

    run = commands.add_parser(
        "run", parents=[lattice], help="collect fire sizes in instantaneous mode and fit tau"
    )
    run.set_defaults(command=_run)
    run.add_argument("--size", type=int, default=256, help="lattice side length L")
    run.add_argument("--theta", type=int, default=500, help="planting attempts per strike (p/f)")
    run.add_argument("--events", type=int, default=100_000)
    run.add_argument(
        "--burn-in", type=int, default=20_000, help="events discarded before measuring"
    )
    run.add_argument("--s-max", type=int, default=None, help="upper end of the fitting window")
    run.add_argument("--out", type=Path, default=Path("fires.npz"))
    return parser

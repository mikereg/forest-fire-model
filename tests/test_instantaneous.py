import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from forest_fire.instantaneous import Params, simulate
from forest_fire.lattice import Boundary, Neighbourhood, label_clusters

E, T = 0, 1

# Two clusters (sizes 3 and 5) that are not connected under the von Neumann neighbourhood.
TWO_CLUSTERS = np.array(
    [
        [T, T, T, E, E, E],
        [E, E, E, E, E, E],
        [E, E, T, T, T, T],
        [E, E, E, E, E, T],
    ],
    dtype=np.int8,
)
SMALL = TWO_CLUSTERS * (np.arange(6) < 3)
LARGE = TWO_CLUSTERS - SMALL


def test_a_strike_burns_exactly_the_struck_cluster() -> None:
    rng = np.random.default_rng(1)
    params = Params(theta=0, boundary=Boundary.FIXED)

    # Strike until something burns; with theta=0 nothing is ever planted.
    result = simulate(TWO_CLUSTERS, params, n_events=50, rng=rng)
    first_fire = int(result.fire_sizes[0])

    assert first_fire in (3, 5)
    burnt, survivor = (SMALL, LARGE) if first_fire == 3 else (LARGE, SMALL)
    assert np.all(result.grid[burnt == T] == E)
    # The other cluster either survives intact or burns whole later; never partially.
    assert np.sum(result.grid[survivor == T]) in (0, np.sum(survivor))
    assert np.sum(result.fire_sizes) + np.sum(result.grid) == 8


def test_stationary_mean_fire_size_balances_planting() -> None:
    # Trees planted per event, theta * (1 - rho), balance trees burnt, rho * <s>.
    theta, size = 50, 64
    rng = np.random.default_rng(7)
    start = np.zeros((size, size), dtype=np.int8)

    burn_in = simulate(start, Params(theta=theta), n_events=5_000, rng=rng)
    result = simulate(burn_in.grid, Params(theta=theta), n_events=20_000, rng=rng)

    rho = result.tree_density.mean()
    expected = theta * (1 - rho) / rho
    assert abs(result.fire_sizes.mean() - expected) / expected < 0.1


def test_simulation_is_reproducible_for_a_given_seed() -> None:
    start = np.zeros((16, 16), dtype=np.int8)
    a = simulate(start, Params(theta=10), n_events=500, rng=np.random.default_rng(3))
    b = simulate(start, Params(theta=10), n_events=500, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(a.fire_sizes, b.fire_sizes)
    np.testing.assert_array_equal(a.grid, b.grid)


@given(
    trees=arrays(np.bool_, st.tuples(st.integers(1, 12), st.integers(1, 12))),
    neighbourhood=st.sampled_from(Neighbourhood),
    boundary=st.sampled_from(Boundary),
    seed=st.integers(0, 2**32 - 1),
)
def test_every_fire_burns_one_whole_cluster(
    trees: np.ndarray, neighbourhood: Neighbourhood, boundary: Boundary, seed: int
) -> None:
    grid = trees.astype(np.int8)
    params = Params(theta=0, neighbourhood=neighbourhood, boundary=boundary)
    result = simulate(grid, params, n_events=1, rng=np.random.default_rng(seed))

    burnt = (grid == T) & (result.grid == E)
    if not burnt.any():
        assert result.fire_sizes.size == 0
        return
    labels, _ = label_clusters(trees, neighbourhood, boundary)
    struck = np.unique(labels[burnt])
    assert struck.size == 1
    np.testing.assert_array_equal(burnt, labels == struck[0])
    assert result.fire_sizes.tolist() == [int(burnt.sum())]

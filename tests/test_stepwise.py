import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from forest_fire.lattice import Boundary, Neighbourhood
from forest_fire.stepwise import Params, step

E, T, B = 0, 1, 2

NO_GROWTH_NO_LIGHTNING = Params(p=0.0, f=0.0, boundary=Boundary.FIXED)


def grid(rows: list[list[int]]) -> np.ndarray:
    return np.array(rows, dtype=np.int8)


def test_fire_advances_one_cell_per_step_and_burnt_cells_empty() -> None:
    rng = np.random.default_rng(0)
    g = grid([[B, T, T, T]])

    g = step(g, NO_GROWTH_NO_LIGHTNING, rng)
    np.testing.assert_array_equal(g, grid([[E, B, T, T]]))

    g = step(g, NO_GROWTH_NO_LIGHTNING, rng)
    np.testing.assert_array_equal(g, grid([[E, E, B, T]]))


def test_certain_lightning_ignites_every_tree() -> None:
    g = grid([[T, E], [T, T]])
    result = step(g, Params(p=0.0, f=1.0), np.random.default_rng(0))
    np.testing.assert_array_equal(result, grid([[B, E], [B, B]]))


def test_certain_growth_fills_every_empty_cell_including_beside_fire() -> None:
    g = grid([[E, B], [E, E]])
    result = step(g, Params(p=1.0, f=0.0), np.random.default_rng(0))
    np.testing.assert_array_equal(result, grid([[T, E], [T, T]]))


grids = arrays(
    np.int8, st.tuples(st.integers(1, 10), st.integers(1, 10)), elements=st.integers(0, 2)
)
params = st.builds(
    Params,
    p=st.floats(0, 1),
    f=st.floats(0, 1),
    neighbourhood=st.sampled_from(Neighbourhood),
    boundary=st.sampled_from(Boundary),
)


@given(g=grids, params=params, seed=st.integers(0, 2**32 - 1))
def test_step_invariants(g: np.ndarray, params: Params, seed: int) -> None:
    result = step(g, params, np.random.default_rng(seed))

    assert result.shape == g.shape
    assert set(np.unique(result)) <= {E, T, B}
    # A burning cell always burns out; only trees can catch fire; empty cells never burn.
    assert np.all(result[g == B] == E)
    assert np.all(g[result == B] == T)
    assert np.all(np.isin(result[g == E], [E, T]))


@given(g=grids, params=params, seed=st.integers(0, 2**32 - 1))
def test_step_is_reproducible_for_a_given_seed(g: np.ndarray, params: Params, seed: int) -> None:
    first = step(g, params, np.random.default_rng(seed))
    second = step(g, params, np.random.default_rng(seed))
    np.testing.assert_array_equal(first, second)

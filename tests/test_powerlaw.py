import numpy as np
import pytest

from forest_fire.powerlaw import Fit, cdf, fit


def sample_power_law(
    tau: float, s_min: int, s_max: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Exact samples from P(s) proportional to s^-tau on [s_min, s_max]."""
    support = np.arange(s_min, s_max + 1)
    weights = support.astype(float) ** -tau
    return rng.choice(support, size=n, p=weights / weights.sum())


@pytest.mark.parametrize("tau", [1.2, 1.5, 2.0, 2.5])
def test_recovers_known_exponent_within_its_error_bar(tau: float) -> None:
    rng = np.random.default_rng(int(tau * 100))
    sizes = sample_power_law(tau, s_min=1, s_max=10_000, n=20_000, rng=rng)

    result = fit(sizes, s_min=1, s_max=10_000)

    assert abs(result.tau - tau) < 3 * result.sigma
    assert 0 < result.sigma < 0.05
    assert result.n == 20_000


def test_chooses_s_min_above_a_non_power_law_head() -> None:
    rng = np.random.default_rng(11)
    tail = sample_power_law(1.8, s_min=30, s_max=100_000, n=10_000, rng=rng)
    head = rng.integers(1, 30, size=10_000)  # uniform, not a power law
    sizes = np.concatenate([head, tail])

    result = fit(sizes, s_max=100_000)

    assert 20 <= result.s_min <= 60
    assert abs(result.tau - 1.8) < 3 * result.sigma


def test_unbounded_tail_fit_recovers_exponent() -> None:
    rng = np.random.default_rng(5)
    sizes = sample_power_law(2.5, s_min=5, s_max=1_000_000, n=10_000, rng=rng)

    result = fit(sizes, s_min=5)

    assert result.s_max is None
    assert abs(result.tau - 2.5) < 3 * result.sigma


@pytest.mark.parametrize("tau", [0.8, 1.0 + 1e-9, 1.7])
def test_cdf_rises_monotonically_to_one_at_s_max(tau: float) -> None:
    f = Fit(tau=tau, sigma=0.0, s_min=3, s_max=500, n=0)
    values = cdf(np.arange(3, 501), f)
    assert np.all(np.diff(values) > 0)
    assert values[-1] == pytest.approx(1.0)

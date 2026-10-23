"""Maximum-likelihood fitting of discrete power laws (Clauset, Shalizi & Newman 2009)."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import optimize, special


@dataclass(frozen=True)
class Fit:
    tau: float
    """Estimated exponent in P(s) proportional to s^-tau."""
    sigma: float
    """Standard error of tau, from the Fisher information."""
    s_min: int
    s_max: int | None
    """Upper end of the fitting window; None for an unbounded tail."""
    n: int
    """Number of sizes inside the fitting window."""


def fit(sizes: npt.ArrayLike, s_min: int | None = None, s_max: int | None = None) -> Fit:
    """Fit a discrete power law to the sizes in [s_min, s_max] by maximum likelihood.

    With ``s_min=None`` it is chosen, as in Clauset et al., to minimise the Kolmogorov-Smirnov
    distance between the data and the fitted model in the window.
    """
    s = np.asarray(sizes)
    if s_max is not None:
        s = s[s <= s_max]
    if s_min is not None:
        return _fit_window(s[s >= s_min], s_min, s_max)

    # Candidate s_min values: distinct sizes leaving enough data above them, at most
    # _MAX_CANDIDATES of them, log-spaced so large sizes are not over-represented.
    distinct = np.unique(s)
    distinct = distinct[np.searchsorted(np.sort(s), distinct) <= s.size - _MIN_TAIL]
    if distinct.size == 0:
        raise ValueError(f"need at least {_MIN_TAIL} sizes to choose s_min")
    if distinct.size > _MAX_CANDIDATES:
        picks = np.unique(np.geomspace(1, distinct.size, _MAX_CANDIDATES).astype(int) - 1)
        distinct = distinct[picks]

    fits = [_fit_window(s[s >= m], int(m), s_max) for m in distinct]
    distances = [_ks_distance(s[s >= m.s_min], m) for m in fits]
    return fits[int(np.argmin(distances))]


_MIN_TAIL = 50
_MAX_CANDIDATES = 100


def _fit_window(window: npt.NDArray[np.integer], s_min: int, s_max: int | None) -> Fit:
    n = window.size
    if n < 2:
        raise ValueError(f"need at least 2 sizes in the fitting window, got {n}")
    sum_log = float(np.sum(np.log(window)))

    def log_norm(tau: float) -> float:
        return _log_normaliser(tau, s_min, s_max)

    def negative_log_likelihood(tau: float) -> float:
        return tau * sum_log + n * log_norm(tau)

    lower = 1.0 + 1e-6 if s_max is None else 0.0
    tau = float(
        optimize.minimize_scalar(
            negative_log_likelihood, bounds=(lower, 6.0), method="bounded", options={"xatol": 1e-8}
        ).x
    )
    # Fisher information per sample is the second derivative of log Z: the variance of log s.
    h = 1e-4
    curvature = (log_norm(tau + h) - 2 * log_norm(tau) + log_norm(tau - h)) / h**2
    sigma = 1.0 / np.sqrt(n * curvature)
    return Fit(tau=tau, sigma=float(sigma), s_min=s_min, s_max=s_max, n=n)


def _ks_distance(window: npt.NDArray[np.integer], model: Fit) -> float:
    """Largest gap between the empirical and the fitted CDF over the window."""
    values, counts = np.unique(window, return_counts=True)
    empirical = np.cumsum(counts) / window.size
    return float(np.max(np.abs(empirical - cdf(values, model))))


def cdf(s: npt.ArrayLike, model: Fit) -> npt.NDArray[np.float64]:
    """P(S <= s) under the fitted power law, for s inside the fitting window."""
    s = np.asarray(s, dtype=np.float64)
    log_z = _log_normaliser(model.tau, model.s_min, model.s_max)
    if model.tau > 1:
        below = special.zeta(model.tau, model.s_min) - special.zeta(model.tau, s + 1)
        return np.asarray(below / np.exp(log_z), dtype=np.float64)
    assert model.s_max is not None  # unbounded fits always have tau > 1
    k = np.arange(model.s_min, model.s_max + 1, dtype=np.float64)
    cumulative = np.cumsum(np.exp(-model.tau * np.log(k) - log_z))
    return np.asarray(cumulative[s.astype(np.intp) - model.s_min], dtype=np.float64)


def _log_normaliser(tau: float, s_min: int, s_max: int | None) -> float:
    """log Z, where Z is the sum of k^-tau over the fitting window."""
    if s_max is None:
        return float(np.log(special.zeta(tau, s_min)))
    k = np.arange(s_min, s_max + 1, dtype=np.float64)
    return float(special.logsumexp(-tau * np.log(k)))

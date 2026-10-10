"""The significance block for a series whose observations are not independent.

An ordinary t-statistic assumes each observation carries fresh information.
Daily returns, per-trade outcomes and rolling scans rarely do, and when
neighbouring values move together the ordinary statistic overstates how much
evidence a sample holds. Reporting it alone is how a result that is noise
reads as significant.

``newey_west_summary`` reports both, side by side, so the gap between them is
visible rather than a matter of which estimator someone picked. The
heteroskedasticity-and-autocorrelation-consistent variance of the mean is

``Var(mean) = (1/n) * [gamma_0 + 2 * sum_{k=1}^{L} w_k * gamma_k]``

where ``gamma_k`` is the lag-k autocovariance, averaged over the ``n - k``
pairs that lag has, and ``w_k = 1 - k/(L+1)`` are the Bartlett weights. The
lag count ``L`` comes from ``newey_west_lag``. The framework is Andrews
(1991) and the operational formula is Newey and West (1994).

The lag index is whatever the caller's series index is. Calendar days for a
daily series, trade order for a ledger, where lag 1 is one trade cycle rather
than one day. The weights do not care and the interpretation does, so a caller
documents its own units.

Three guards keep the function total rather than raising into code that has
already checked its own inputs. A sample of fewer than two observations
returns an all-zero summary, a standard error of zero reports a t-statistic of
zero, and the robust variance is floored at zero.

That third guard is reachable. Bartlett weights keep the estimate
non-negative only when every autocovariance divides by ``n``. This
implementation divides each ``gamma_k`` by its ``n - k`` pairs instead, which
inflates every lag term by ``n / (n - k)`` and is enough to lose the
guarantee. Taking ``gamma_0`` from ``np.var(ddof=1)`` does not lose it, since
that only enlarges the leading term. Random draws rarely find the case, and a
search over 152,000 normal samples at every length from 2 to 39 reached no
negative value. A series built for it does, from 23 observations upward, and
the function then reports a robust t-statistic of zero whatever the naive one
reads. The standard estimator divides by ``n`` and cannot go negative. This
one keeps ``n - k`` because consumers have already committed numbers computed
with it, which the next paragraph explains.

The arithmetic is pinned. Consumers commit numbers that trace to these exact
floating-point operations, so reordering a sum or swapping ``np.mean`` for a
hand-written loop moves results that are already written down. Change the body
only when a re-pin is the intention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np


def newey_west_lag(n: int) -> int:
    """The auto-lag rule ``L = int(4 * (n / 100) ** (2 / 9))``.

    Separate from the estimator so a caption or a report can quote the same
    ``L`` the estimator used rather than recomputing it and drifting.
    """
    return int(4 * (n / 100) ** (2 / 9))


class NeweyWestSummary(NamedTuple):
    """One series' mean tested against zero, by both estimators.

    ``t_naive`` assumes independent observations and is the one that
    overstates the evidence under autocorrelation. ``t_newey_west`` is the
    robust counterpart. ``lag`` is the ``L`` actually used, so a reader can
    re-derive the result rather than trust it.
    """

    n: int
    mean: float
    var: float
    t_naive: float
    t_newey_west: float
    lag: int


def newey_west_summary(x: np.ndarray | Sequence[float]) -> NeweyWestSummary:
    """Both t-statistics of a series' mean against the null that it is zero.

    ``var`` is the ``ddof=1`` sample variance, which is the ``gamma_0`` the
    module docstring's formula names. See that docstring for the estimator,
    the lag rule, and the two guards on short or degenerate samples.
    """
    arr = np.asarray(x, dtype=float)
    n = arr.size
    if n < 2:
        return NeweyWestSummary(n, float(arr.mean()) if n else 0.0, 0.0, 0.0, 0.0, 0)
    mean = float(np.mean(arr))
    var0 = float(np.var(arr, ddof=1))
    se_naive = math.sqrt(var0 / n) if var0 > 0 else 0.0
    t_naive = mean / se_naive if se_naive > 0 else 0.0
    lag = newey_west_lag(n)
    nw_sum = 0.0
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1)
        nw_sum += weight * float(np.mean((arr[:-k] - mean) * (arr[k:] - mean)))
    var_mean = (var0 + 2.0 * nw_sum) / n
    se = math.sqrt(max(var_mean, 0.0))
    t_nw = mean / se if se > 0 else 0.0
    return NeweyWestSummary(n, mean, var0, t_naive, t_nw, lag)


def newey_west_t(x: np.ndarray | Sequence[float]) -> float:
    """The robust t-statistic alone, for a caller that reports nothing else."""
    return newey_west_summary(x).t_newey_west

"""
Horizon Index
-------------
Measures whether current policy trajectory is sustainable.

Detects sugar-rush policies: improvements to headline metrics that come
at the cost of degrading the foundations those metrics depend on. A
planner that reduces Gini by destroying employment gets a low Horizon
Index. A planner that reduces Gini by growing the productive base gets
a high one.

Architecture:
    - Smooth sigmoid scoring (no hard boolean thresholds)
    - Blended short/long horizon trends (12 and 60 months)
    - Weighted metric-foundation pairs
    - Sustainability logic: metric improvement without foundation
      support is penalized as a sugar rush, not rewarded

Range: 0.0 (self-defeating trajectory) to 1.0 (fully sustainable)
Sweet spot: 0.7+ (building on solid foundations)
Warning: 0.3-0.5 (some sugar-rushing)
Crisis: < 0.3 (policy trajectory is self-defeating)

Usage in objective functions:
    from horizon_index import compute_horizon_index
    hi = compute_horizon_index(model)
    reward = base_reward * hi  # unsustainable policies get crushed

Usage in metrics:
    m["horizon_index"] = compute_horizon_index(model)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from environment import EconomicModel


# ── Lookback windows ────────────────────────────────────────────
SHORT_LOOKBACK = 12
LONG_LOOKBACK = 60

# ── Pair weights (sum to 1.0) ───────────────────────────────────
PAIR_WEIGHTS = {
    "gini": 0.20,
    "unemployment": 0.25,
    "wealth": 0.20,
    "revenue": 0.15,
    "population": 0.10,
    "agency": 0.10,
}

# ── Sigmoid calibration ────────────────────────────────────────
# Trends are normalized by (scale * lookback), producing values
# in the range ~1e-4 to ~1e-2 for typical economic dynamics.
# The sigmoid multiplier maps that range onto [0, 1] with the
# inflection point at zero (no change = 0.5).
#
# At SIGMOID_K = 200:
#   trend =  0.005  ->  score ~ 0.73  (moderate improvement)
#   trend =  0.010  ->  score ~ 0.88  (strong improvement)
#   trend = -0.005  ->  score ~ 0.27  (moderate decline)
#   trend =  0.000  ->  score = 0.50  (flat)
#
# Adjust if your simulation produces trends outside this range.
SIGMOID_K = 200


# ── Trend computation ──────────────────────────────────────────

def _compute_trend(history, key, lookback):
    """Normalized trend over a single lookback window."""
    if len(history) < lookback + 1:
        return 0.0

    now = history[-1].get(key, 0)
    then = history[-(lookback + 1)].get(key, 0)

    if not isinstance(now, (int, float)):
        return 0.0
    if not np.isfinite(now) or not np.isfinite(then):
        return 0.0

    scale = max(abs(now), abs(then), 1.0)
    return (now - then) / (scale * lookback)


def _blended_trend(history, key):
    """
    Blend short and long trends.

    Short trend (60% weight) captures recent policy effects.
    Long trend (40% weight) captures structural trajectory.
    Falls back to short-only when history is insufficient for long.
    """
    short = _compute_trend(history, key, SHORT_LOOKBACK)
    long_ = _compute_trend(history, key, LONG_LOOKBACK)
    return 0.6 * short + 0.4 * long_


# ── Sigmoid scoring ────────────────────────────────────────────

def _sigmoid_positive(x):
    """Score for metrics where higher is better. trend > 0 scores > 0.5."""
    return float(1.0 / (1.0 + np.exp(-x * SIGMOID_K)))


def _sigmoid_negative(x):
    """Score for metrics where lower is better. trend < 0 scores > 0.5."""
    return 1.0 - _sigmoid_positive(x)


# ── Foundation and metric scoring ──────────────────────────────

def _foundation_score(history, keys, lower_is_better=None):
    """
    Aggregate foundation health as the mean of individual sigmoid scores.

    Using mean rather than AND logic: one weak foundation drags the
    score down proportionally rather than collapsing everything to zero.
    """
    if lower_is_better is None:
        lower_is_better = set()

    scores = []
    for k in keys:
        t = _blended_trend(history, k)
        if k in lower_is_better:
            scores.append(_sigmoid_negative(t))
        else:
            scores.append(_sigmoid_positive(t))

    return float(np.mean(scores)) if scores else 0.5


def _metric_score(history, key, lower_is_better=False):
    """Sigmoid score for a single headline metric."""
    t = _blended_trend(history, key)
    if lower_is_better:
        return _sigmoid_negative(t)
    return _sigmoid_positive(t)


# ── Sustainability logic ───────────────────────────────────────

def _pair_sustainability(metric, foundation):
    """
    Sustainability rule for a metric-foundation pair.

    The core diagnostic question: is this trajectory self-defeating?

    - metric * (0.4 + 0.6 * foundation) means:
      - Metric improvement gets ZERO credit at foundation = 0
        (pure sugar rush is worthless)
      - Metric improvement gets PARTIAL credit with weak foundations
        (improvement is real but fragile)
      - Metric improvement gets FULL credit with strong foundations
        (sustainable trajectory)
      - No metric improvement = no score regardless of foundations
        (this index measures trajectory, not static health)

    This avoids the flaw of awarding a floor score for foundation
    strength alone: the index answers "is the current direction
    sustainable?" not "are foundations healthy?"
    """
    return metric * (0.4 + 0.6 * foundation)


# ── Main computation ───────────────────────────────────────────

def compute_horizon_index(model: "EconomicModel") -> float:
    """
    Compute the Horizon Index: are current improvements sustainable?

    Returns 0.5 (neutral) until enough history accumulates for the
    long lookback window. After that, returns 0.0 to 1.0.
    """
    history = model.metrics_history

    if len(history) < LONG_LOOKBACK + 1:
        return 0.5  # not enough data, assume neutral

    scores = []

    # 1. Gini (lower is better) depends on employment, wealth, population
    m = _metric_score(history, "all_gini", lower_is_better=True)
    f = _foundation_score(
        history,
        ["unemployment_rate", "worker_mean", "n_workers"],
        lower_is_better={"unemployment_rate"},
    )
    scores.append(PAIR_WEIGHTS["gini"] * _pair_sustainability(m, f))

    # 2. Unemployment (lower is better) depends on firms, profitability
    m = _metric_score(history, "unemployment_rate", lower_is_better=True)
    f = _foundation_score(
        history,
        ["n_firms", "mean_firm_profit"],
    )
    scores.append(PAIR_WEIGHTS["unemployment"] * _pair_sustainability(m, f))

    # 3. Mean wealth (higher is better) depends on production, skills, employment
    m = _metric_score(history, "worker_mean")
    f = _foundation_score(
        history,
        ["total_production", "mean_skill", "unemployment_rate"],
        lower_is_better={"unemployment_rate"},
    )
    scores.append(PAIR_WEIGHTS["wealth"] * _pair_sustainability(m, f))

    # 4. Tax revenue (higher is better) depends on broad tax base
    m = _metric_score(history, "planner_tax_revenue")
    f = _foundation_score(
        history,
        ["n_workers", "n_firms", "mean_wage"],
    )
    scores.append(PAIR_WEIGHTS["revenue"] * _pair_sustainability(m, f))

    # 5. Population (higher is better) depends on wealth floor, employment
    m = _metric_score(history, "n_workers")
    f = _foundation_score(
        history,
        ["worker_min", "unemployment_rate"],
        lower_is_better={"unemployment_rate"},
    )
    scores.append(PAIR_WEIGHTS["population"] * _pair_sustainability(m, f))

    # 6. Agency floor (higher is better) depends on wealth floor, skills, employment
    m = _metric_score(history, "agency_floor")
    f = _foundation_score(
        history,
        ["worker_min", "mean_skill", "unemployment_rate"],
        lower_is_better={"unemployment_rate"},
    )
    scores.append(PAIR_WEIGHTS["agency"] * _pair_sustainability(m, f))

    hi = float(np.sum(scores))
    return float(np.clip(hi, 0.0, 1.0))


# ── Integration Guide ─────────────────────────────────────────
#
# THREE WAYS TO USE THE HORIZON INDEX:
#
# 1. AS A METRIC (monitoring only, no effect on planner):
#    In metrics.py collect_step_metrics():
#        from horizon_index import compute_horizon_index
#        m["horizon_index"] = compute_horizon_index(model)
#
# 2. AS AN OBJECTIVE MULTIPLIER (shapes planner learning):
#    In any objective function:
#        from horizon_index import compute_horizon_index
#        hi = compute_horizon_index(model)
#        reward = base_reward * hi
#
#    This crushes rewards from unsustainable policies.
#    A sugar-rush path with hi=0.15 keeps only 15% of its reward.
#    A sustainable path with hi=0.85 keeps 85%.
#    The planner learns that sustainable improvements are worth
#    ~5x more than sugar rushes.
#
# 3. AS PART OF THE IDEAL BASELINE:
#    In _compute_ideal_score():
#        ideal includes hi=1.0 (fully sustainable)
#    In _learning_step():
#        current score includes actual hi
#
#    The planner is penalized for unsustainability as part of
#    its distance-from-ideal calculation.
#
# Option 2 is the cleanest integration. One line in each objective.

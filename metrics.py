"""
metrics.py
----------
Metric collection, computation, and emergence detection.

Tracked every timestep:
  - wealth statistics (min, median, mean, Gini, power-law exponent)
  - top 1/5/10 % wealth share
  - firm market concentration (HHI, top firm share)
  - cartel count
  - rent extraction rate
  - debt concentration
  - worker mobility / unemployment
  - trade network centralisation
  - agency floor (for JAM objective)
  - planner policy snapshot

Emergence detection:
  - monopoly (single firm >40% market share)
  - cartel formation
  - power-law wealth distribution (Pareto fit)
  - persistent poverty trap (bottom quintile static)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Any, Optional

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from environment import EconomicModel


# ---------------------------------------------------------------------------
# Step-level metric collection
# ---------------------------------------------------------------------------

def collect_step_metrics(model: "EconomicModel") -> Dict[str, Any]:
    """Return a flat dict of all metrics for the current step."""
    m: Dict[str, Any] = {"step": model.current_step}

    # Wealth stats
    all_w = model.get_all_agent_wealths()
    worker_w = model.get_worker_wealths()

    m.update(_wealth_stats(all_w, prefix="all"))
    m.update(_wealth_stats(worker_w, prefix="worker"))

    # Firm concentration
    m.update(_firm_stats(model))

    # Cartel detection
    m["n_active_cartels"] = sum(
        1 for members in model.active_cartels.values() if len(members) >= 2
    )

    # Rent extraction
    m["total_rent_collected"] = sum(lo.total_rent_collected for lo in model.landowners)
    m["mean_rent_rate"] = float(np.mean([lo.rent_rate for lo in model.landowners])) if model.landowners else 0.0

    # Debt
    m["total_debt"] = model.economy.total_debt_outstanding
    m["total_defaults"] = model.economy.total_defaults
    worker_debts = np.array([w.debt for w in model.workers], dtype=float)
    m["debt_gini"] = _gini(worker_debts)
    m["fraction_in_debt"] = float(np.mean(worker_debts > 0)) if len(worker_debts) else 0.0

    # Labour market
    employed = [w for w in model.workers if w.employed]
    m["n_workers"] = len(model.workers)
    m["n_employed"] = len(employed)
    m["unemployment_rate"] = 1.0 - len(employed) / max(1, len(model.workers))
    m["mean_wage"] = float(np.mean([w.wage for w in employed])) if employed else 0.0

    # Worker mobility
    m["mean_consecutive_unemployed"] = float(
        np.mean([w.consecutive_unemployed_steps for w in model.workers])
    ) if model.workers else 0.0

    # Trade network
    net_stats = model.economy.get_network_stats()
    m["trade_nodes"] = net_stats["nodes"]
    m["trade_edges"] = net_stats["edges"]
    m["trade_density"] = net_stats["density"]
    m["trade_max_centrality"] = net_stats["max_centrality"]

    # Agency floor (JAM)
    if model.workers:
        agencies = [w.compute_agency() for w in model.workers]
        m["agency_floor"] = float(min(agencies))
        m["agency_mean"] = float(np.mean(agencies))
    else:
        m["agency_floor"] = 0.0
        m["agency_mean"] = 0.0

    # Planner
    m["planner_objective_value"] = model.planner.last_objective_value
    m["planner_tax_revenue"] = model.planner.tax_revenue
    m["planner_ubi"] = model.planner.policy["ubi_payment"]
    m["planner_min_wage"] = model.planner.policy["min_wage"]
    m["planner_tax_worker"] = model.planner.policy["tax_rate_worker"]
    m["planner_tax_firm"] = model.planner.policy["tax_rate_firm"]
    m["planner_agriculture_inv"]    = model.planner.policy["agriculture_investment"]
    m["planner_infrastructure_inv"] = model.planner.policy["infrastructure_investment"]
    m["planner_healthcare_inv"]     = model.planner.policy["healthcare_investment"]
    m["planner_education_inv"]      = model.planner.policy["education_investment"]

    # Public investment effects (current bonus levels)
    m["infrastructure_level"]  = model._infrastructure_level
    m["healthcare_bonus"]      = model._healthcare_bonus
    m["education_quality"]     = model._education_quality
    m["agriculture_bonus"]     = model._agriculture_bonus

    # Water resource
    m["mean_water"] = float(np.mean(model.water_grid))
    m["min_water"]  = float(np.min(model.water_grid))

    # Pollution
    m["mean_pollution"]    = float(np.mean(model.pollution_grid))
    m["max_pollution"]     = float(np.max(model.pollution_grid))
    m["total_pollution"]   = float(np.sum(model.pollution_grid))
    m["n_polluted_cells"]  = int(np.sum(model.pollution_grid > 1.0))
    m["total_firm_emissions"] = float(sum(f.total_pollution_emitted for f in model.firms if not f.defunct))
    # Health burden: extra metabolism cost workers bear from pollution this step
    health_burden = 0.0
    for w in model.workers:
        if w.pos is not None:
            p = float(model.pollution_grid[int(w.pos[0]), int(w.pos[1])])
            health_burden += p * 0.05
    m["pollution_health_burden"] = health_burden
    m["planner_pollution_tax"]      = model.planner.policy["pollution_tax"]
    m["planner_cleanup_investment"] = model.planner.policy["cleanup_investment"]

    # Power-law exponent (Pareto tail)
    m["wealth_power_law_alpha"] = _pareto_alpha(all_w)

    # Emergence flags
    m["monopoly_detected"] = _detect_monopoly(model)
    m["cartel_detected"] = m["n_active_cartels"] > 0
    m["poverty_trap_detected"] = _detect_poverty_trap(worker_w)

    return m


# ---------------------------------------------------------------------------
# Wealth statistics
# ---------------------------------------------------------------------------

def _wealth_stats(w: np.ndarray, prefix: str) -> Dict[str, float]:
    if len(w) == 0:
        return {f"{prefix}_{k}": 0.0 for k in
                ["min", "median", "mean", "max", "std", "gini",
                 "top1_share", "top5_share", "top10_share"]}
    w = w[w > 0] if len(w[w > 0]) > 0 else np.array([1e-9])
    total = w.sum()
    top1 = np.percentile(w, 99)
    top5 = np.percentile(w, 95)
    top10 = np.percentile(w, 90)
    return {
        f"{prefix}_min": float(w.min()),
        f"{prefix}_median": float(np.median(w)),
        f"{prefix}_mean": float(w.mean()),
        f"{prefix}_max": float(w.max()),
        f"{prefix}_std": float(w.std()),
        f"{prefix}_gini": _gini(w),
        f"{prefix}_top1_share": float(w[w >= top1].sum() / max(total, 1e-9)),
        f"{prefix}_top5_share": float(w[w >= top5].sum() / max(total, 1e-9)),
        f"{prefix}_top10_share": float(w[w >= top10].sum() / max(total, 1e-9)),
    }


def _gini(w: np.ndarray) -> float:
    """Compute Gini coefficient."""
    w = np.sort(w[w > 0])
    if len(w) == 0:
        return 0.0
    n = len(w)
    cumsum = np.cumsum(w)
    return float((2 * np.sum((np.arange(1, n + 1)) * w) - (n + 1) * w.sum())
                 / (n * w.sum()))


def _pareto_alpha(w: np.ndarray) -> float:
    """
    Estimate the Pareto tail exponent using the Hill estimator.
    Uses the top 10% of the wealth distribution.
    Returns NaN if insufficient data.
    """
    w = w[w > 0]
    if len(w) < 20:
        return float("nan")
    threshold = np.percentile(w, 90)
    tail = w[w >= threshold]
    if len(tail) < 5:
        return float("nan")
    # Hill estimator
    alpha = len(tail) / np.sum(np.log(tail / threshold))
    return float(alpha)


# ---------------------------------------------------------------------------
# Firm statistics
# ---------------------------------------------------------------------------

def _firm_stats(model: "EconomicModel") -> Dict[str, float]:
    active_firms = [f for f in model.firms if not f.defunct]
    if not active_firms:
        return {
            "n_firms": 0, "hhi": 0.0, "top_firm_share": 0.0,
            "mean_firm_profit": 0.0, "total_production": 0.0,
        }

    shares = np.array([f.market_share for f in active_firms])
    hhi = float(np.sum(shares ** 2))
    top_share = float(shares.max()) if len(shares) else 0.0
    profits = np.array([f.profit for f in active_firms])
    total_prod = sum(f.production_this_step for f in active_firms)

    return {
        "n_firms": len(active_firms),
        "hhi": hhi,
        "top_firm_share": top_share,
        "mean_firm_profit": float(profits.mean()),
        "total_production": float(total_prod),
    }


# ---------------------------------------------------------------------------
# Emergence detection
# ---------------------------------------------------------------------------

def _detect_monopoly(model: "EconomicModel") -> bool:
    """True if any single firm has >40% market share."""
    return any(f.market_share > 0.40 for f in model.firms if not f.defunct)


def _detect_poverty_trap(worker_w: np.ndarray, window: int = 50) -> bool:
    """
    Heuristic: bottom quintile trapped if their average is < 2x survival threshold.
    """
    from agents import SURVIVAL_THRESHOLD
    if len(worker_w) == 0:
        return False
    bottom20 = np.percentile(worker_w, 20)
    return float(bottom20) < SURVIVAL_THRESHOLD * 2.0


def detect_power_law(w: np.ndarray) -> Dict[str, Any]:
    """
    Statistical test for power-law distribution in upper tail.
    Returns dict with alpha, p_value, and is_power_law flag.
    """
    w = np.sort(w[w > 0])
    if len(w) < 50:
        return {"alpha": float("nan"), "p_value": float("nan"), "is_power_law": False}

    threshold = np.percentile(w, 80)
    tail = w[w >= threshold]
    if len(tail) < 10:
        return {"alpha": float("nan"), "p_value": float("nan"), "is_power_law": False}

    # Fit log-normal (null) vs Pareto (alternative)
    log_tail = np.log(tail)
    # Kolmogorov-Smirnov test against fitted lognormal
    mu, sigma = log_tail.mean(), log_tail.std()
    if sigma < 1e-10:
        return {"alpha": float("nan"), "p_value": 1.0, "is_power_law": False}

    ks_stat, p_val = stats.kstest(tail, "lognorm", args=(sigma, 0, math.exp(mu)))
    alpha = _pareto_alpha(w)

    return {
        "alpha": alpha,
        "p_value": float(p_val),
        "is_power_law": alpha < 3.0 and p_val > 0.05,
    }


# ---------------------------------------------------------------------------
# Episode-level summary
# ---------------------------------------------------------------------------

def episode_summary(metrics_history: List[Dict]) -> Dict[str, Any]:
    """
    Compute aggregate statistics over a full episode.
    Returns a flat dict suitable for DataFrame storage.
    """
    if not metrics_history:
        return {}

    # Use the final 20% of steps (steady-state estimate)
    n = len(metrics_history)
    tail = metrics_history[max(0, int(n * 0.8)):]

    def avg(key):
        vals = [m[key] for m in tail if key in m and not (isinstance(m[key], float) and math.isnan(m[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    def last(key):
        for m in reversed(tail):
            if key in m:
                return m[key]
        return float("nan")

    summary = {}
    scalar_keys = [
        "all_gini", "all_min", "all_mean", "all_top10_share",
        "worker_gini", "worker_min", "worker_mean", "worker_top1_share",
        "worker_top5_share", "worker_top10_share",
        "wealth_power_law_alpha",
        "hhi", "top_firm_share", "n_active_cartels",
        "unemployment_rate", "mean_wage",
        "total_rent_collected", "total_debt",
        "fraction_in_debt", "debt_gini",
        "agency_floor", "agency_mean",
        "trade_density", "trade_max_centrality",
        "planner_ubi", "planner_min_wage",
        "planner_tax_worker", "planner_tax_firm",
        "planner_agriculture_inv", "planner_infrastructure_inv",
        "planner_healthcare_inv", "planner_education_inv",
        "infrastructure_level", "healthcare_bonus",
        "education_quality", "agriculture_bonus",
        "mean_water", "min_water",
        "mean_pollution", "max_pollution", "total_pollution",
        "n_polluted_cells", "pollution_health_burden",
        "planner_pollution_tax", "planner_cleanup_investment",
        "monopoly_detected", "cartel_detected", "poverty_trap_detected",
    ]
    for key in scalar_keys:
        summary[key] = avg(key)

    # Fraction of steps with specific emergent phenomena
    summary["frac_monopoly"] = float(np.mean([m.get("monopoly_detected", False) for m in metrics_history]))
    summary["frac_cartel"] = float(np.mean([m.get("cartel_detected", False) for m in metrics_history]))
    summary["frac_poverty_trap"] = float(np.mean([m.get("poverty_trap_detected", False) for m in metrics_history]))

    # Terminal values
    summary["terminal_n_workers"] = last("n_workers")
    summary["terminal_all_gini"] = last("all_gini")
    summary["terminal_worker_min"] = last("worker_min")
    summary["terminal_agency_floor"] = last("agency_floor")

    return summary

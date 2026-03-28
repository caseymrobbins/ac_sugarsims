"""
metrics.py
----------
Metric collection, computation, and emergence detection.

Includes economic, information, banking, horizon index,
and sustainable capitalism metrics.

Changes:
  - Integrated horizon_index (compute_horizon_index)
  - Added missing metrics: mean_firm_floor, min_firm_floor,
    total_production, median_agency, agency_gini,
    mean_worker_age, population_growth_rate, price snapshots,
    trade_volume, loan metrics, resource levels
  - Added collect_animation_frame for step-by-step visualization
  - Fixed poverty trap detection (structural trap, not near-death)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Tuple

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from environment import EconomicModel


# ---------------------------------------------------------------------------
# Diversity / identity metrics
# ---------------------------------------------------------------------------

IDENTITY_TYPES = (
    ("A", "A"),
    ("A", "B"),
    ("A", "C"),
    ("B", "B"),
    ("B", "C"),
    ("C", "C"),
)

def _normalize_identity(identity):
    if identity is None:
        return None
    try:
        a, b = identity
        return tuple(sorted((str(a), str(b))))
    except Exception:
        return None

def _identity_entropy_from_workers(workers) -> float:
    counts = {t: 0 for t in IDENTITY_TYPES}
    for w in workers:
        key = _normalize_identity(getattr(w, "identity", None))
        if key in counts:
            counts[key] += 1
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    p = np.array([c / total for c in counts.values() if c > 0], dtype=np.float64)
    return float(-np.sum(p * np.log(p)))

def _hybrid_fraction(workers) -> float:
    total = 0
    hybrids = 0
    for w in workers:
        key = _normalize_identity(getattr(w, "identity", None))
        if key is None:
            continue
        total += 1
        if key[0] != key[1]:
            hybrids += 1
    return float(hybrids / max(total, 1))

def _identity_segregation(model: "EconomicModel") -> float:
    same = 0
    total = 0
    for w in model.workers:
        pos = getattr(w, "pos", None)
        if pos is None:
            continue
        key_w = _normalize_identity(getattr(w, "identity", None))
        if key_w is None:
            continue
        neighbours = model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=1)
        for cell in neighbours:
            for other in model.grid.get_cell_list_contents([cell]):
                if other is w or not hasattr(other, "identity"):
                    continue
                key_o = _normalize_identity(getattr(other, "identity", None))
                if key_o is None:
                    continue
                total += 1
                if key_o == key_w:
                    same += 1
    if total == 0:
        return 0.0
    return float(same / total)

# ---------------------------------------------------------------------------
# Step-level metric collection
# ---------------------------------------------------------------------------

def collect_step_metrics(model: "EconomicModel") -> Dict[str, Any]:
    """Return a flat dict of all metrics for the current step."""
    m: Dict[str, Any] = {"step": model.current_step}

    all_w = model.get_all_agent_wealths()
    worker_w = model.get_worker_wealths()

    m.update(_wealth_stats(all_w, prefix="all"))
    m.update(_wealth_stats(worker_w, prefix="worker"))
    m.update(_firm_stats(model))

    m["n_active_cartels"] = sum(
        1 for members in model.active_cartels.values() if len(members) >= 2
    )

    m["total_rent_collected"] = sum(lo.total_rent_collected for lo in model.landowners)
    m["mean_rent_rate"] = float(np.mean([lo.rent_rate for lo in model.landowners])) if model.landowners else 0.0

    m["total_debt"] = model.economy.total_debt_outstanding
    m["total_defaults"] = model.economy.total_defaults
    worker_debts = np.array([w.debt for w in model.workers], dtype=float)
    m["debt_gini"] = _gini(worker_debts)
    m["fraction_in_debt"] = float(np.mean(worker_debts > 0)) if len(worker_debts) else 0.0

    employed = [w for w in model.workers if w.employed]
    m["n_workers"] = len(model.workers)
    m["n_employed"] = len(employed)
    m["unemployment_rate"] = 1.0 - len(employed) / max(1, len(model.workers))
    m["mean_wage"] = float(np.mean([w.wage for w in employed])) if employed else 0.0

    m["mean_consecutive_unemployed"] = float(
        np.mean([w.consecutive_unemployed_steps for w in model.workers])
    ) if model.workers else 0.0

    net_stats = model.economy.get_network_stats()
    m["trade_nodes"] = net_stats["nodes"]
    m["trade_edges"] = net_stats["edges"]
    m["trade_density"] = net_stats["density"]
    m["trade_max_centrality"] = net_stats["max_centrality"]
    m["trade_volume"] = model.economy.trade_volume_this_step

    # Agency metrics
    if model.workers:
        agencies = [w.compute_agency() for w in model.workers]
        agencies_arr = np.array(agencies, dtype=np.float64)
        m["agency_floor"] = float(np.min(agencies_arr))
        m["agency_mean"] = float(np.mean(agencies_arr))
        m["agency_median"] = float(np.median(agencies_arr))
        m["agency_gini"] = _gini(agencies_arr)
    else:
        m["agency_floor"] = 0.0
        m["agency_mean"] = 0.0
        m["agency_median"] = 0.0
        m["agency_gini"] = 0.0

    # Planner policy snapshot
    m["planner_objective_value"] = model.planner.last_objective_value
    m["planner_tax_revenue"] = model.planner.tax_revenue
    m["planner_ubi"] = model.planner.policy["ubi_payment"]
    m["planner_min_wage"] = model.planner.policy["min_wage"]
    m["planner_tax_worker"] = model.planner.policy["tax_rate_worker"]
    m["planner_tax_firm"] = model.planner.policy["tax_rate_firm"]
    m["planner_agriculture_inv"] = model.planner.policy["agriculture_investment"]
    m["planner_infrastructure_inv"] = model.planner.policy["infrastructure_investment"]
    m["planner_healthcare_inv"] = model.planner.policy["healthcare_investment"]
    m["planner_education_inv"] = model.planner.policy["education_investment"]
    m["planner_inheritance_tax"] = model.planner.policy.get("inheritance_tax", 0.0)

    m["infrastructure_level"] = model._infrastructure_level
    m["healthcare_bonus"] = model._healthcare_bonus
    m["education_quality"] = model._education_quality
    m["agriculture_bonus"] = model._agriculture_bonus

    # Resource levels
    m["mean_food"] = float(np.mean(model.food_grid))
    m["mean_raw"] = float(np.mean(model.raw_grid))
    m["mean_water"] = float(np.mean(model.water_grid))
    m["min_water"] = float(np.min(model.water_grid))
    m["mean_capital"] = float(np.mean(model.capital_grid))

    # Pollution
    m["mean_pollution"] = float(np.mean(model.pollution_grid))
    m["max_pollution"] = float(np.max(model.pollution_grid))
    m["total_pollution"] = float(np.sum(model.pollution_grid))
    m["n_polluted_cells"] = int(np.sum(model.pollution_grid > 1.0))
    m["total_firm_emissions"] = float(sum(
        f.total_pollution_emitted for f in model.firms if not f.defunct
    ))
    health_burden = 0.0
    for w in model.workers:
        if w.pos is not None:
            p = float(model.pollution_grid[int(w.pos[0]), int(w.pos[1])])
            health_burden += p * 0.05
    m["pollution_health_burden"] = health_burden
    m["planner_pollution_tax"] = model.planner.policy["pollution_tax"]
    m["planner_cleanup_investment"] = model.planner.policy["cleanup_investment"]
    m["planner_media_funding"] = model.planner.policy.get("media_funding", 0.0)
    m["planner_antitrust_enforcement"] = model.planner.policy.get("antitrust_enforcement", 0.0)

    m["wealth_power_law_alpha"] = _pareto_alpha(all_w)

    # Emergence detection
    m["monopoly_detected"] = _detect_monopoly(model)
    m["cartel_detected"] = m["n_active_cartels"] > 0
    m["poverty_trap_detected"] = _detect_poverty_trap(worker_w)

    # Skill distribution
    if model.workers:
        skills = np.array([w.skill for w in model.workers])
        m["mean_skill"] = float(np.mean(skills))
        m["skill_gini"] = _gini(skills)
    else:
        m["mean_skill"] = 0.0
        m["skill_gini"] = 0.0

    # Investment metrics
    if model.workers:
        total_invested = sum(sum(w.investments.values()) for w in model.workers
                           if hasattr(w, 'investments'))
        n_investors = sum(1 for w in model.workers
                        if hasattr(w, 'investments') and w.investments)
        m["total_investment"] = total_invested
        m["n_investors"] = n_investors
        m["investment_concentration"] = n_investors / max(len(model.workers), 1)
    else:
        m["total_investment"] = 0.0
        m["n_investors"] = 0
        m["investment_concentration"] = 0.0

    # Sustainable capitalism: firm stakeholder floors
    try:
        from sustainable_capitalism import compute_stakeholder_scores
        firm_floors = []
        firm_floors_raw = []
        firm_S, firm_E, firm_V, firm_C = [], [], [], []
        for f in model.firms:
            if not f.defunct:
                scores = compute_stakeholder_scores(f)
                firm_floors.append(scores['floor'])
                firm_S.append(scores['S'])
                firm_E.append(scores['E'])
                firm_V.append(scores['V'])
                firm_C.append(scores['C'])
                # Raw floor from unnormalized scores
                raw_floor = min(scores.get('S_raw', scores['S']),
                                scores.get('E_raw', scores['E']),
                                scores.get('V_raw', scores['V']),
                                scores.get('C_raw', scores['C']))
                firm_floors_raw.append(raw_floor)
        if firm_floors:
            m["mean_firm_floor"] = float(np.mean(firm_floors))
            m["min_firm_floor"] = float(np.min(firm_floors))
            m["mean_firm_floor_norm"] = float(np.mean(firm_floors))
            m["mean_firm_floor_raw"] = float(np.mean(firm_floors_raw))
            m["mean_firm_S"] = float(np.mean(firm_S))
            m["mean_firm_E"] = float(np.mean(firm_E))
            m["mean_firm_V"] = float(np.mean(firm_V))
            m["mean_firm_C"] = float(np.mean(firm_C))
        else:
            m["mean_firm_floor"] = 0.0
            m["min_firm_floor"] = 0.0
            m["mean_firm_floor_norm"] = 0.0
            m["mean_firm_floor_raw"] = 0.0
            m["mean_firm_S"] = 0.0
            m["mean_firm_E"] = 0.0
            m["mean_firm_V"] = 0.0
            m["mean_firm_C"] = 0.0
    except ImportError:
        pass

    # Green R&D and pollution metrics (Task 2)
    active_firms_list = [f for f in model.firms if not f.defunct]
    if active_firms_list:
        m["mean_green_rd_priority"] = float(np.mean([
            getattr(f, 'green_rd_priority', 0.5) for f in active_firms_list
        ]))
        m["mean_pollution_factor"] = float(np.mean([
            f.pollution_factor for f in active_firms_list
        ]))
    else:
        m["mean_green_rd_priority"] = 0.0
        m["mean_pollution_factor"] = 0.0

    # Per-firm SEVC adoption metrics (Task 4)
    if active_firms_list:
        sevc_firms = [f for f in active_firms_list if getattr(f, 'is_sevc', True)]
        vanilla_firms = [f for f in active_firms_list if not getattr(f, 'is_sevc', True)]
        m["sevc_adoption_rate"] = len(sevc_firms) / len(active_firms_list)
        m["sevc_market_share"] = float(sum(f.market_share for f in sevc_firms))
        m["sevc_mean_profit"] = float(np.mean([f.profit for f in sevc_firms])) if sevc_firms else 0.0
        m["vanilla_mean_profit"] = float(np.mean([f.profit for f in vanilla_firms])) if vanilla_firms else 0.0
        m["sevc_mean_workers"] = float(np.mean([len(f.workers) for f in sevc_firms])) if sevc_firms else 0.0
        m["vanilla_mean_workers"] = float(np.mean([len(f.workers) for f in vanilla_firms])) if vanilla_firms else 0.0
    else:
        m["sevc_adoption_rate"] = 0.0
        m["sevc_market_share"] = 0.0
        m["sevc_mean_profit"] = 0.0
        m["vanilla_mean_profit"] = 0.0
        m["sevc_mean_workers"] = 0.0
        m["vanilla_mean_workers"] = 0.0

    # Horizon Index
    try:
        from horizon_index import compute_horizon_index
        m["horizon_index"] = compute_horizon_index(model)
    except ImportError:
        pass

    # Price levels
    for good, price in model.economy.prices.items():
        m[f"price_{good}"] = float(price)

    # Loan book stats
    m["n_active_loans"] = len(model.economy.loans)
    m["mean_loan_rate"] = float(np.mean(
        [l.interest_rate for l in model.economy.loans]
    )) if model.economy.loans else 0.0

    # Demographics
    if model.workers:
        ages = np.array([w.age for w in model.workers], dtype=np.float64)
        m["mean_worker_age"] = float(np.mean(ages))
        m["max_worker_age"] = float(np.max(ages))
    else:
        m["mean_worker_age"] = 0.0
        m["max_worker_age"] = 0.0

    # Population growth rate (requires history)
    if len(model.metrics_history) >= 1:
        prev_pop = model.metrics_history[-1].get("n_workers", len(model.workers))
        if prev_pop > 0:
            m["population_growth_rate"] = (len(model.workers) - prev_pop) / prev_pop
        else:
            m["population_growth_rate"] = 0.0
    else:
        m["population_growth_rate"] = 0.0

    # Diversity metrics from the document.
    m["identity_entropy"] = _identity_entropy_from_workers(model.workers)
    m["hybrid_fraction"] = _hybrid_fraction(model.workers)
    m["identity_segregation"] = _identity_segregation(model)

    return m


# ---------------------------------------------------------------------------
# Animation frame collection
# ---------------------------------------------------------------------------

def collect_animation_frame(model: "EconomicModel") -> Dict[str, Any]:
    """
    Collect a lightweight snapshot of the simulation state for animation.

    Returns positions, wealth, employment status, firm locations,
    resource grids (downsampled), and pollution for each step.
    Designed to be stored in a list and replayed as an animation.
    """
    frame: Dict[str, Any] = {"step": model.current_step}

    # Worker data: position, wealth, employed, skill, agency
    workers_data = []
    for w in model.workers:
        if w.pos is not None:
            workers_data.append({
                "x": int(w.pos[0]),
                "y": int(w.pos[1]),
                "wealth": float(w.wealth),
                "employed": w.employed,
                "skill": float(w.skill),
                "in_debt": w.debt > 0,
            })
    frame["workers"] = workers_data

    # Firm data: position, wealth, n_workers, defunct, cartel
    firms_data = []
    for f in model.firms:
        if not f.defunct and f.pos is not None:
            firms_data.append({
                "x": int(f.pos[0]),
                "y": int(f.pos[1]),
                "wealth": float(f.wealth),
                "n_workers": len(f.workers),
                "production": float(f.production_this_step),
                "in_cartel": f.cartel_id is not None,
                "pollution_factor": float(f.pollution_factor),
            })
    frame["firms"] = firms_data

    # Landowner territories (only collect occasionally, they change slowly)
    if model.current_step % 50 == 1 or model.current_step == 1:
        lo_data = []
        for lo in model.landowners:
            lo_data.append({
                "cells": [(int(c[0]), int(c[1])) for c in lo.controlled_cells],
                "rent_rate": float(lo.rent_rate),
                "wealth": float(lo.wealth),
            })
        frame["landowners"] = lo_data

    # Resource grids: downsample to 20x20 for animation performance
    ds = 4  # downsample factor (80/4 = 20)
    frame["food_grid"] = model.food_grid[::ds, ::ds].tolist()
    frame["pollution_grid"] = model.pollution_grid[::ds, ::ds].tolist()
    frame["water_grid"] = model.water_grid[::ds, ::ds].tolist()

    # Key aggregate metrics for overlay display
    if model.metrics_history:
        latest = model.metrics_history[-1]
        frame["overlay"] = {
            "gini": latest.get("all_gini", 0),
            "unemployment": latest.get("unemployment_rate", 0),
            "agency_floor": latest.get("agency_floor", 0),
            "horizon_index": latest.get("horizon_index", 0.5),
            "mean_firm_floor": latest.get("mean_firm_floor", 0),
            "n_workers": latest.get("n_workers", 0),
            "n_firms": latest.get("n_firms", 0),
            "n_cartels": latest.get("n_active_cartels", 0),
            "trust_planner": latest.get("trust_planner", 0.5),
            "trust_institutional": latest.get("trust_institutional", 0.5),
        }

    return frame


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
    w = np.sort(w[w > 0])
    if len(w) == 0:
        return 0.0
    n = len(w)
    cumsum = np.cumsum(w)
    return float((2 * np.sum((np.arange(1, n + 1)) * w) - (n + 1) * w.sum())
                 / (n * w.sum()))


def _pareto_alpha(w: np.ndarray) -> float:
    w = w[w > 0]
    if len(w) < 20:
        return float("nan")
    threshold = np.percentile(w, 90)
    tail = w[w >= threshold]
    if len(tail) < 5:
        return float("nan")
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
            "mean_firm_wage": 0.0, "total_wages_paid": 0.0,
            "mean_pollution_factor": 0.0,
        }

    shares = np.array([f.market_share for f in active_firms])
    hhi = float(np.sum(shares ** 2))
    top_share = float(shares.max()) if len(shares) else 0.0
    profits = np.array([f.profit for f in active_firms])
    total_prod = sum(f.production_this_step for f in active_firms)
    wages = np.array([f.offered_wage for f in active_firms])
    total_wages = sum(f.total_wages_paid for f in active_firms)
    poll_factors = np.array([f.pollution_factor for f in active_firms])

    return {
        "n_firms": len(active_firms),
        "hhi": hhi,
        "top_firm_share": top_share,
        "mean_firm_profit": float(profits.mean()),
        "total_production": float(total_prod),
        "mean_firm_wage": float(wages.mean()),
        "total_wages_paid": float(total_wages),
        "mean_pollution_factor": float(poll_factors.mean()),
    }


# ---------------------------------------------------------------------------
# Emergence detection
# ---------------------------------------------------------------------------

def _detect_monopoly(model: "EconomicModel") -> bool:
    return any(f.market_share > 0.40 for f in model.firms if not f.defunct)


def _detect_poverty_trap(worker_w: np.ndarray, window: int = 50) -> bool:
    """
    Poverty trap = bottom quintile stuck near subsistence.
    Not 'almost dead' but 'alive and unable to escape.'
    Trapped when bottom 20% wealth is < 10% of median.
    """
    if len(worker_w) < 10:
        return False
    bottom20 = np.percentile(worker_w, 20)
    median = np.median(worker_w)
    if median <= 0:
        return False
    return float(bottom20) < max(median * 0.10, 20.0)


def detect_power_law(w: np.ndarray) -> Dict[str, Any]:
    w = np.sort(w[w > 0])
    if len(w) < 50:
        return {"alpha": float("nan"), "p_value": float("nan"), "is_power_law": False}

    threshold = np.percentile(w, 80)
    tail = w[w >= threshold]
    if len(tail) < 10:
        return {"alpha": float("nan"), "p_value": float("nan"), "is_power_law": False}

    log_tail = np.log(tail)
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
    """Compute aggregate statistics over a full episode."""
    if not metrics_history:
        return {}

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

    def max_val(key):
        vals = [m[key] for m in metrics_history if key in m
                and not (isinstance(m[key], float) and math.isnan(m[key]))]
        return float(np.max(vals)) if vals else float("nan")

    summary = {}

    # Economic metrics
    scalar_keys = [
        "all_gini", "all_min", "all_mean", "all_top10_share",
        "worker_gini", "worker_min", "worker_mean", "worker_top1_share",
        "worker_top5_share", "worker_top10_share",
        "wealth_power_law_alpha",
        "hhi", "top_firm_share", "n_active_cartels",
        "unemployment_rate", "mean_wage",
        "total_rent_collected", "total_debt",
        "fraction_in_debt", "debt_gini",
        "agency_floor", "agency_mean", "agency_median", "agency_gini",
        "trade_density", "trade_max_centrality", "trade_volume",
        "planner_ubi", "planner_min_wage",
        "planner_tax_worker", "planner_tax_firm",
        "planner_agriculture_inv", "planner_infrastructure_inv",
        "planner_healthcare_inv", "planner_education_inv",
        "planner_inheritance_tax",
        "infrastructure_level", "healthcare_bonus",
        "education_quality", "agriculture_bonus",
        "mean_food", "mean_raw", "mean_water", "min_water", "mean_capital",
        "mean_pollution", "max_pollution", "total_pollution",
        "n_polluted_cells", "pollution_health_burden",
        "planner_pollution_tax", "planner_cleanup_investment",
        "monopoly_detected", "cartel_detected", "poverty_trap_detected",
        "mean_skill", "skill_gini",
        "total_investment", "n_investors", "investment_concentration",
        "total_production", "mean_firm_wage", "total_wages_paid",
        "mean_pollution_factor",
        "mean_firm_floor", "min_firm_floor",
        "mean_firm_floor_norm", "mean_firm_floor_raw",
        "mean_firm_S", "mean_firm_E", "mean_firm_V", "mean_firm_C",
        "mean_green_rd_priority", "mean_pollution_factor",
        "sevc_adoption_rate", "sevc_market_share",
        "sevc_mean_profit", "vanilla_mean_profit",
        "sevc_mean_workers", "vanilla_mean_workers",
        "horizon_index",
        "mean_worker_age", "population_growth_rate",
        "n_active_loans", "mean_loan_rate",
        "trust_worker_mean", "trust_worker_min", "trust_worker_std",
        "trust_firm_mean", "trust_firm_min",
        "trust_bank_mean", "trust_news_mean", "trust_news_capture_gap",
        "trust_landowner_mean", "trust_planner", "trust_institutional",
    ]
    for key in scalar_keys:
        summary[key] = avg(key)

    # Emergence fractions
    summary["frac_monopoly"] = float(np.mean([m.get("monopoly_detected", False) for m in metrics_history]))
    summary["frac_cartel"] = float(np.mean([m.get("cartel_detected", False) for m in metrics_history]))
    summary["frac_poverty_trap"] = float(np.mean([m.get("poverty_trap_detected", False) for m in metrics_history]))

    # Terminal values
    summary["terminal_n_workers"] = last("n_workers")
    summary["terminal_all_gini"] = last("all_gini")
    summary["terminal_worker_min"] = last("worker_min")
    summary["terminal_agency_floor"] = last("agency_floor")
    summary["terminal_horizon_index"] = last("horizon_index")
    summary["terminal_mean_firm_floor"] = last("mean_firm_floor")
    summary["terminal_trust_planner"] = last("trust_planner")
    summary["terminal_trust_institutional"] = last("trust_institutional")

    # Information layer metrics
    info_keys = [
        "mean_authority_trust", "min_authority_trust",
        "weight_polarization", "info_r0",
        "n_news_firms", "n_captured_news",
        "n_accurate_news", "n_captured_accurate",
        "epistemic_health", "trust_gini", "pct_low_trust",
    ]
    for key in info_keys:
        summary[key] = avg(key)

    summary["terminal_epistemic_health"] = last("epistemic_health")
    summary["terminal_authority_trust"] = last("mean_authority_trust")
    summary["terminal_polarization"] = last("weight_polarization")
    summary["terminal_info_r0"] = last("info_r0")
    summary["max_info_r0"] = max_val("info_r0")
    summary["max_captured_news"] = max_val("n_captured_news")

    # Banking metrics
    banking_keys = [
        "n_banks", "total_bank_deposits", "total_bank_loans",
        "bank_leverage_ratio", "bank_default_rate", "bank_profit",
        "deposit_concentration",
    ]
    for key in banking_keys:
        summary[key] = avg(key)

    # Horizon index trajectory stats
    hi_vals = [m.get("horizon_index", float("nan")) for m in metrics_history
               if "horizon_index" in m and np.isfinite(m.get("horizon_index", float("nan")))]
    if hi_vals:
        summary["horizon_index_min"] = float(np.min(hi_vals))
        summary["horizon_index_max"] = float(np.max(hi_vals))
        summary["horizon_index_final_20pct"] = float(np.mean(hi_vals[max(0, int(len(hi_vals)*0.8)):]))

    return summary

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

from agents import _identity_similarity


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
        firm_floors_norm = []
        firm_floors_raw = []
        firm_S, firm_E, firm_V, firm_C = [], [], [], []
        firm_bindings = {"S": 0, "E": 0, "V": 0, "C": 0}
        for f in model.firms:
            if not f.defunct:
                scores = compute_stakeholder_scores(f)
                firm_floors_norm.append(scores['floor'])
                firm_S.append(scores['S'])
                firm_E.append(scores['E'])
                firm_V.append(scores['V'])
                firm_C.append(scores['C'])
                raw_floor = min(scores.get('S_raw', scores['S']),
                                scores.get('E_raw', scores['E']),
                                scores.get('V_raw', scores['V']),
                                scores.get('C_raw', scores['C']))
                firm_floors_raw.append(raw_floor)
                binding = scores.get('binding', 'S')
                if binding in firm_bindings:
                    firm_bindings[binding] += 1
        n_scored = max(len(firm_floors_norm), 1)
        if firm_floors_norm:
            m["mean_firm_floor"] = float(np.mean(firm_floors_norm))
            m["min_firm_floor"] = float(np.min(firm_floors_norm))
            m["mean_firm_floor_norm"] = float(np.mean(firm_floors_norm))
            m["mean_firm_floor_raw"] = float(np.mean(firm_floors_raw))
            m["mean_firm_S"] = float(np.mean(firm_S))
            m["mean_firm_E"] = float(np.mean(firm_E))
            m["mean_firm_V"] = float(np.mean(firm_V))
            m["mean_firm_C"] = float(np.mean(firm_C))
            m["mean_firm_binding_S"] = firm_bindings["S"] / n_scored
            m["mean_firm_binding_E"] = firm_bindings["E"] / n_scored
            m["mean_firm_binding_V"] = firm_bindings["V"] / n_scored
            m["mean_firm_binding_C"] = firm_bindings["C"] / n_scored
        else:
            m["mean_firm_floor"] = 0.0; m["min_firm_floor"] = 0.0
            m["mean_firm_floor_norm"] = 0.0; m["mean_firm_floor_raw"] = 0.0
            m["mean_firm_S"] = 0.0; m["mean_firm_E"] = 0.0
            m["mean_firm_V"] = 0.0; m["mean_firm_C"] = 0.0
            m["mean_firm_binding_S"] = 0.0; m["mean_firm_binding_E"] = 0.0
            m["mean_firm_binding_V"] = 0.0; m["mean_firm_binding_C"] = 0.0
    except ImportError:
        pass

    # Firm-level Horizon Index metrics (Task 7)
    active_firms_list = [f for f in model.firms if not f.defunct]
    if active_firms_list:
        firm_his = [getattr(f, 'horizon_index', 1.0) for f in active_firms_list]
        m["mean_firm_hi"] = float(np.mean(firm_his))
        m["min_firm_hi"] = float(np.min(firm_his))
        m["n_firms_declining"] = int(sum(1 for h in firm_his if h < 0.7))
        m["n_firms_critical"] = int(sum(1 for h in firm_his if h < 0.4))
    else:
        m["mean_firm_hi"] = 1.0
        m["min_firm_hi"] = 1.0
        m["n_firms_declining"] = 0
        m["n_firms_critical"] = 0

    # Capacity-driven mitosis and headroom metrics (Task 13)
    if active_firms_list:
        n_workers_total = max(len(model.workers), 1)
        m["firms_per_worker"] = len(active_firms_list) / n_workers_total
        stabs = [getattr(f, 'headroom_stability', 0.0) for f in active_firms_list]
        m["mean_headroom_stability"] = float(np.mean(stabs))
        m["n_mitosis_events"] = int(sum(getattr(f, '_mitosis_events', 0) for f in active_firms_list))
        # Profit acceleration and headroom gap (from firms with enough history)
        accels = []; gaps = []
        for f in active_firms_list:
            ph = getattr(f, '_profit_history', None)
            if ph and len(ph) >= 20:
                h = list(ph); early = h[:10]; late = h[-10:]
                eg = (early[-1] - early[0]) / max(abs(early[0]) + 1, 1)
                lg = (late[-1] - late[0]) / max(abs(late[0]) + 1, 1)
                accels.append(lg - eg)
            sc = getattr(f, '_prev_scores', None)
            if sc:
                s_raw = sc.get('S_raw', sc.get('S', 0.5))
                gaps.append(s_raw - sc.get('floor', 0.5))
        m["mean_profit_acceleration"] = float(np.mean(accels)) if accels else 0.0
        m["mean_headroom_gap"] = float(np.mean(gaps)) if gaps else 0.0
    else:
        m["firms_per_worker"] = 0.0; m["mean_headroom_stability"] = 0.0
        m["n_mitosis_events"] = 0; m["mean_profit_acceleration"] = 0.0
        m["mean_headroom_gap"] = 0.0

    # Green R&D and pollution metrics (Task 2)
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

    # Production-capture ratio metrics (Task 11)
    if active_firms_list:
        capture_ratios = [getattr(f, 'capture_ratio', 0.5) for f in active_firms_list]
        cr_arr = np.array(capture_ratios, dtype=np.float64)
        m["mean_capture_ratio"]   = float(np.mean(cr_arr))
        m["median_capture_ratio"] = float(np.median(cr_arr))
        m["min_firm_capture_ratio"] = float(np.min(cr_arr))
        cr_pos = cr_arr[cr_arr > 0]
        if len(cr_pos) > 1 and cr_pos.sum() > 0:
            n = len(cr_pos); cr_s = np.sort(cr_pos)
            m["capture_gini"] = float((2*np.sum(np.arange(1,n+1)*cr_s)-(n+1)*cr_s.sum())/(n*cr_s.sum()))
        else:
            m["capture_gini"] = 0.0
        total_wages_step = sum(getattr(f, 'wages_this_step', 0.0) for f in active_firms_list)
        total_rev_step   = sum(f.revenue for f in active_firms_list)
        m["total_wages_to_revenue"] = float(total_wages_step / max(total_rev_step, 1e-9))
    else:
        m["mean_capture_ratio"] = 0.0; m["median_capture_ratio"] = 0.0
        m["min_firm_capture_ratio"] = 0.0; m["capture_gini"] = 0.0
        m["total_wages_to_revenue"] = 0.0
    m["planner_min_capture_ratio"] = float(model.planner.policy.get("min_capture_ratio", 0.0))

    # CEO compensation metrics (Task 12)
    if active_firms_list:
        ceo_comps   = [getattr(f, 'ceo_compensation_this_step', 0.0) for f in active_firms_list]
        ceo_bases   = [getattr(f, 'ceo_base_salary', 0.0) for f in active_firms_list]
        ceo_bonuses = [getattr(f, 'ceo_bonus', 0.0) for f in active_firms_list]
        ceo_pots    = [getattr(f, 'ceo_potential_bonus', 0.0) for f in active_firms_list]
        ceo_equities= [getattr(f, 'ceo_equity_value', 0.0) for f in active_firms_list]
        m["mean_ceo_compensation"] = float(np.mean(ceo_comps))
        # Bonus realisation: actual/potential (how much of the possible bonus did CEO capture?)
        total_pot = sum(ceo_pots)
        m["mean_ceo_bonus_realisation"] = float(sum(ceo_bonuses) / total_pot) if total_pot > 0 else 0.0
        # CEO/floor-worker ratio: ceo_total / ceo_base (1.0 = equal; 2.0 = 2× floor)
        ratios = [(c / b) for c, b in zip(ceo_comps, ceo_bases) if b > 0]
        m["mean_ceo_floor_ratio"] = float(np.mean(ratios)) if ratios else 1.0
        m["mean_ceo_equity_value"] = float(np.mean(ceo_equities))
    else:
        m["mean_ceo_compensation"] = 0.0
        m["mean_ceo_bonus_realisation"] = 0.0
        m["mean_ceo_floor_ratio"] = 1.0
        m["mean_ceo_equity_value"] = 0.0

    # Government / election metrics (Task 8)
    m["gov_type"] = getattr(model, 'gov_type', 'authoritarian')
    planner = model.planner
    m["election_winner"] = getattr(planner, '_last_election_winner', 'none')
    for platform in ('redistribution', 'growth', 'education', 'environment', 'security'):
        m["voter_turnout_" + platform] = getattr(planner, '_vote_shares', {}).get(platform, 0.0)

    # Planner SEVC dimensions (Task 10)
    pdims = getattr(planner, '_planner_sevc_dims', None)
    if pdims:
        m["planner_S_pop"] = pdims.get("S", 0.0)
        m["planner_E_pop"] = pdims.get("E", 0.0)
        m["planner_V_pop"] = pdims.get("V", 0.0)
        m["planner_C_pop"] = pdims.get("C", 0.0)
        vals = [pdims["S"], pdims["E"], pdims["V"], pdims["C"]]
        m["planner_sevc_score"] = min(vals)
        m["planner_binding_dimension"] = ["S", "E", "V", "C"][int(np.argmin(vals))]
    else:
        m["planner_S_pop"] = 0.0; m["planner_E_pop"] = 0.0
        m["planner_V_pop"] = 0.0; m["planner_C_pop"] = 0.0
        m["planner_sevc_score"] = 0.0; m["planner_binding_dimension"] = "none"

    # Election-planner alignment: does binding dim match election winner?
    _dim_map = {"education": "E", "redistribution": "S", "security": "C",
                "growth": "S", "environment": "V"}
    winner = getattr(planner, '_last_election_winner', 'none')
    binding = m.get("planner_binding_dimension", "none")
    voter_dim = _dim_map.get(winner, "none")
    m["election_planner_aligned"] = 1.0 if (voter_dim == binding and voter_dim != "none") else 0.0
    m["election_weight"] = getattr(model, 'election_weight', 0.0)

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

    # Conflict, aggression, and enforcement metrics
    if model.workers:
        agg_vals = np.array([getattr(w, 'aggression', 0.0) for w in model.workers], dtype=np.float64)
        m["mean_aggression"] = float(np.mean(agg_vals))
        m["max_aggression"] = float(np.max(agg_vals))
    else:
        m["mean_aggression"] = 0.0; m["max_aggression"] = 0.0
    m["crime_events"] = getattr(model, '_total_crime_events', 0)
    m["riot_events"] = getattr(model, '_total_riot_events', 0)
    m["rebellion_events"] = getattr(model, '_rebellion_events', 0)
    if hasattr(model, 'conflict_grid'):
        cg = model.conflict_grid
        m["mean_conflict"] = float(np.mean(cg))
        m["max_conflict"] = float(np.max(cg))
        m["conflict_variance"] = float(np.var(cg))
        m["n_conflict_hotspots"] = int(np.sum(cg > 0.5))
    else:
        m["mean_conflict"] = 0.0; m["max_conflict"] = 0.0
        m["conflict_variance"] = 0.0; m["n_conflict_hotspots"] = 0
    if hasattr(model, 'legitimacy_grid'):
        lg = model.legitimacy_grid
        m["legitimacy_mean"] = float(np.mean(lg))
        m["legitimacy_min"] = float(np.min(lg))
        m["legitimacy_variance"] = float(np.var(lg))
        m["n_low_legitimacy"] = int(np.sum(lg < 0.3))
    else:
        m["legitimacy_mean"] = 0.7; m["legitimacy_min"] = 0.7
        m["legitimacy_variance"] = 0.0; m["n_low_legitimacy"] = 0
    # Enforcement metrics
    m["n_enforcers"] = len(getattr(model, 'enforcers', []))
    m["total_arrests"] = sum(getattr(e, 'arrests', 0) for e in getattr(model, 'enforcers', []))
    m["surveillance_level"] = getattr(model, '_surveillance_level', 0.0)
    # Identity conflict index: mean(aggression * (1 - identity_similarity_to_neighbors))
    if model.workers:
        id_conflict_vals = []
        for w in model.workers[:100]:  # sample for performance
            agg = getattr(w, 'aggression', 0.0)
            if w.pos is not None:
                neighbors = model.grid.get_neighborhood(
                    (int(w.pos[0]), int(w.pos[1])), moore=True, include_center=False, radius=2)
                nearby = [a for cell in neighbors[:8] for a in model.grid.get_cell_list_contents([cell])
                         if hasattr(a, 'identity') and a.unique_id != w.unique_id]
                if nearby:
                    avg_sim = sum(_identity_similarity(w.identity, getattr(n, 'identity', w.identity))
                                for n in nearby[:5]) / min(len(nearby), 5)
                    id_conflict_vals.append(agg * (1.0 - avg_sim))
        m["identity_conflict_index"] = float(np.mean(id_conflict_vals)) if id_conflict_vals else 0.0
    else:
        m["identity_conflict_index"] = 0.0

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
                "aggression": float(getattr(w, 'aggression', 0.0)),
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
                "is_sevc": bool(getattr(f, 'is_sevc', True)),
                "firm_hi": float(getattr(f, 'horizon_index', 1.0)),
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
    if hasattr(model, 'conflict_grid'):
        frame["conflict_grid"] = model.conflict_grid[::ds, ::ds].tolist()
    if hasattr(model, 'legitimacy_grid'):
        frame["legitimacy_grid"] = model.legitimacy_grid[::ds, ::ds].tolist()

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
            "mean_firm_hi": latest.get("mean_firm_hi", 1.0),
            "gov_type": latest.get("gov_type", ""),
            "election_winner": latest.get("election_winner", "none"),
            "sevc_adoption_rate": latest.get("sevc_adoption_rate", 0),
            "total_pollution": latest.get("total_pollution", 0),
            "mean_aggression": latest.get("mean_aggression", 0),
            "crime_events": latest.get("crime_events", 0),
            "riot_events": latest.get("riot_events", 0),
            "mean_conflict": latest.get("mean_conflict", 0),
            "legitimacy_mean": latest.get("legitimacy_mean", 0.7),
            "identity_conflict_index": latest.get("identity_conflict_index", 0),
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
        "mean_capture_ratio", "median_capture_ratio", "min_firm_capture_ratio",
        "capture_gini", "total_wages_to_revenue", "planner_min_capture_ratio",
        "mean_ceo_compensation", "mean_ceo_bonus_realisation",
        "mean_ceo_floor_ratio", "mean_ceo_equity_value",
        "mean_firm_hi", "min_firm_hi", "n_firms_declining", "n_firms_critical",
        "horizon_index",
        "election_winner", "voter_turnout_redistribution", "voter_turnout_growth",
        "voter_turnout_education", "voter_turnout_environment", "voter_turnout_security",
        "mean_worker_age", "population_growth_rate",
        "n_active_loans", "mean_loan_rate",
        "trust_worker_mean", "trust_worker_min", "trust_worker_std",
        "trust_firm_mean", "trust_firm_min",
        "trust_bank_mean", "trust_news_mean", "trust_news_capture_gap",
        "trust_landowner_mean", "trust_planner", "trust_institutional",
        "mean_aggression", "max_aggression",
        "crime_events", "riot_events", "rebellion_events",
        "mean_conflict", "max_conflict", "conflict_variance", "n_conflict_hotspots",
        "legitimacy_mean", "legitimacy_min", "legitimacy_variance", "n_low_legitimacy",
        "n_enforcers", "total_arrests", "surveillance_level",
        "identity_conflict_index",
        "planner_S_pop", "planner_E_pop", "planner_V_pop", "planner_C_pop",
        "planner_sevc_score",
        "mean_firm_binding_S", "mean_firm_binding_E",
        "mean_firm_binding_V", "mean_firm_binding_C",
        "election_planner_aligned", "election_weight",
        "firms_per_worker", "mean_headroom_stability",
        "n_mitosis_events", "mean_profit_acceleration", "mean_headroom_gap",
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
    summary["mean_capture_ratio"] = avg("mean_capture_ratio")
    summary["terminal_capture_ratio"] = last("mean_capture_ratio")
    summary["terminal_total_wages_to_revenue"] = last("total_wages_to_revenue")
    summary["mean_ceo_compensation"] = avg("mean_ceo_compensation")
    summary["mean_ceo_bonus_realisation"] = avg("mean_ceo_bonus_realisation")
    summary["mean_ceo_floor_ratio"] = avg("mean_ceo_floor_ratio")
    summary["mean_ceo_equity_value"] = avg("mean_ceo_equity_value")
    summary["terminal_ceo_compensation"] = last("mean_ceo_compensation")
    summary["terminal_ceo_floor_ratio"] = last("mean_ceo_floor_ratio")
    summary["terminal_trust_planner"] = last("trust_planner")
    summary["terminal_trust_institutional"] = last("trust_institutional")

    # Information layer metrics
    info_keys = [
        "mean_authority_trust", "min_authority_trust",
        "weight_polarization", "info_r0",
        "n_news_firms", "n_captured_news",
        "n_accurate_news", "n_captured_accurate",
        "trust_gini", "pct_low_trust",
        # Four-variable EH decomposition
        "system_M", "system_VE", "system_CI", "system_tau_c",
        "epistemic_health_mean", "epistemic_health_floor",
        "epistemic_health_median", "eh_gini", "pct_low_eh",
    ]
    for key in info_keys:
        summary[key] = avg(key)

    summary["terminal_epistemic_health_mean"] = last("epistemic_health_mean")
    summary["terminal_epistemic_health_floor"] = last("epistemic_health_floor")
    summary["terminal_authority_trust"] = last("mean_authority_trust")
    summary["terminal_polarization"] = last("weight_polarization")
    summary["terminal_info_r0"] = last("info_r0")
    summary["terminal_system_M"] = last("system_M")
    summary["terminal_system_VE"] = last("system_VE")
    summary["terminal_system_CI"] = last("system_CI")
    summary["terminal_system_tau_c"] = last("system_tau_c")
    summary["max_info_r0"] = max_val("info_r0")
    summary["max_captured_news"] = max_val("n_captured_news")
    summary["max_system_M"] = max_val("system_M")

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

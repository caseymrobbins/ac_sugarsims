"""
trust.py - canonical trust engine for the simulation.

This module provides two things:
  1) latent trust updates from outcomes
  2) observed trust reads with optional measurement noise

The intent is to keep the underlying trust state stable while allowing
experiments to inject fuzzy observation without mutating the true score.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np

TRUST_DEFAULT = 0.50
TRUST_MIN = 0.00
TRUST_MAX = 1.00
TRUST_FLOOR = 0.05
TRUST_CEILING = 0.95


def _clip01(value: float) -> float:
    return float(np.clip(value, TRUST_MIN, TRUST_MAX))


def _safe_get(obj: Any, name: str, default: float = TRUST_DEFAULT) -> float:
    value = getattr(obj, name, default)
    try:
        return float(value)
    except Exception:
        return float(default)


def observed_trust(agent: Any, model: Any = None, default: float = TRUST_DEFAULT) -> float:
    """
    Return the trust level that an observer perceives.

    If trust is disabled or frozen, the observable value is neutral.
    If the model defines trust_noise, the returned value is noisy but the
    underlying agent.trust_score is not changed.
    """
    if model is not None and (getattr(model, "_trust_frozen", False) or not getattr(model, "use_trust", True)):
        return TRUST_DEFAULT

    base = _clip01(_safe_get(agent, "trust_score", default))
    if model is None:
        return base

    noise = float(getattr(model, "trust_noise", 0.0) or 0.0)
    if noise <= 0:
        return base

    rng = getattr(model, "rng", None)
    if rng is None:
        rng = np.random.default_rng()
    return _clip01(base + float(rng.normal(0.0, noise)))


def _blend(current: float, target: float, rate: float) -> float:
    rate = float(np.clip(rate, 0.0, 1.0))
    return float(np.clip((1.0 - rate) * current + rate * target, TRUST_FLOOR, TRUST_CEILING))


def _worker_trust_target(worker, model) -> float:
    from information import ACTIONS

    current_income = float(getattr(worker, "income_last_step", 0.0))
    prev_income = float(getattr(worker, "income_prev_step", 0.0))
    income_delta = current_income - prev_income

    employed = 1.0 if getattr(worker, "employed", False) else 0.0
    unemployment_pressure = min(1.0, float(getattr(worker, "consecutive_unemployed_steps", 0)) / 12.0)
    debt = float(getattr(worker, "debt", 0.0))
    risk_tolerance = float(getattr(worker, "risk_tolerance", 0.5))
    authority = float(getattr(worker, "authority_trust", 0.5))

    # Local conditions
    pollution = 0.0
    if getattr(worker, "pos", None) is not None and hasattr(model, "pollution_grid"):
        x, y = int(worker.pos[0]), int(worker.pos[1])
        pollution = float(model.pollution_grid[x, y])

    wage_signal = np.tanh(current_income / 25.0)
    momentum = np.tanh(income_delta / 20.0)
    debt_drag = np.tanh(debt / 120.0)

    target = (
        0.50
        + 0.12 * employed
        - 0.10 * unemployment_pressure
        + 0.10 * wage_signal
        + 0.06 * momentum
        - 0.10 * debt_drag
        - 0.05 * np.tanh(pollution / 4.0)
        + 0.08 * (authority - 0.5)
        + 0.04 * (risk_tolerance - 0.5)
    )
    return float(np.clip(target, TRUST_FLOOR, TRUST_CEILING))


def _firm_trust_target(firm, model) -> float:
    stakeholder_floor = 0.5
    try:
        from sustainable_capitalism import compute_stakeholder_scores
        stakeholder_floor = float(compute_stakeholder_scores(firm).get("floor", 0.5))
    except Exception:
        pass

    offered_wage = float(getattr(firm, "offered_wage", 0.0))
    profit = float(getattr(firm, "profit", 0.0))
    pollution_factor = float(getattr(firm, "pollution_factor", 0.0))
    cartel_penalty = 0.15 if getattr(firm, "cartel_id", None) is not None else 0.0
    wage_fairness = np.tanh(offered_wage / max(1.0, abs(profit) / 10.0 + 1.0))
    pollution_penalty = np.tanh(pollution_factor / 0.2)

    target = 0.50 + 0.30 * (stakeholder_floor - 0.5) + 0.08 * wage_fairness - 0.10 * pollution_penalty - cartel_penalty
    return float(np.clip(target, TRUST_FLOOR, TRUST_CEILING))


def _landowner_trust_target(landowner, model) -> float:
    rent_rate = float(getattr(landowner, "rent_rate", 0.1))
    occupied = 0.0
    controlled = getattr(landowner, "controlled_cells", [])
    if controlled and hasattr(model, "grid"):
        hits = 0
        for cell in controlled[:50]:
            try:
                contents = model.grid.get_cell_list_contents([cell])
                if contents:
                    hits += 1
            except Exception:
                pass
        occupied = hits / max(1, min(len(controlled), 50))
    target = 0.50 + 0.12 * (0.35 - rent_rate) + 0.08 * occupied
    return float(np.clip(target, TRUST_FLOOR, TRUST_CEILING))


def _planner_trust_target(planner, model) -> float:
    latest = getattr(model, "metrics_history", [])
    if latest:
        m = latest[-1]
    else:
        m = {}

    agency_floor = float(m.get("agency_floor", 0.5))
    all_gini = float(m.get("all_gini", 0.35))
    horizon = float(m.get("horizon_index", 1.0))
    unemployment = float(m.get("unemployment_rate", 0.1))
    pollution = float(m.get("mean_pollution", 0.0))

    target = (
        0.50
        + 0.18 * (agency_floor - 0.5)
        - 0.12 * (all_gini - 0.35)
        + 0.12 * (horizon - 0.5)
        - 0.08 * unemployment
        - 0.04 * np.tanh(pollution / 5.0)
    )
    return float(np.clip(target, TRUST_FLOOR, TRUST_CEILING))


def _news_trust_target(news_firm, model) -> float:
    accuracy = float(getattr(news_firm, "accuracy", 0.5))
    audience_capture = float(getattr(news_firm, "audience_size", 0)) / max(1.0, float(len(getattr(model, "workers", [])) or 1))
    captured = 1.0 if getattr(news_firm, "captured_by_cartel", None) is not None else 0.0
    target = 0.40 + 0.45 * accuracy - 0.12 * audience_capture - 0.20 * captured
    return float(np.clip(target, TRUST_FLOOR, TRUST_CEILING))


def update_trust_scores(model) -> None:
    """
    Update latent trust values in-place.

    This function is intentionally conservative: trust moves slowly and is
    clipped to a stable interval so it behaves like a reputation state rather
    than a volatile signal.
    """
    if getattr(model, "_trust_frozen", False) or not getattr(model, "use_trust", True):
        for collection_name in ("workers", "firms", "landowners", "news_firms"):
            for agent in getattr(model, collection_name, []) or []:
                if hasattr(agent, "trust_score"):
                    agent.trust_score = TRUST_DEFAULT
        planner = getattr(model, "planner", None)
        if planner is not None and hasattr(planner, "trust_score"):
            planner.trust_score = TRUST_DEFAULT
        model.institutional_trust = TRUST_DEFAULT
        return

    workers = list(getattr(model, "workers", []) or [])
    firms = [f for f in (getattr(model, "firms", []) or []) if not getattr(f, "defunct", False)]
    landowners = list(getattr(model, "landowners", []) or [])
    news_firms = list(getattr(model, "news_firms", []) or [])
    planner = getattr(model, "planner", None)

    # Workers
    for worker in workers:
        if not hasattr(worker, "trust_score"):
            worker.trust_score = TRUST_DEFAULT
        target = _worker_trust_target(worker, model)
        worker.trust_score = _blend(float(worker.trust_score), target, 0.08)

    # Firms
    for firm in firms:
        if not hasattr(firm, "trust_score"):
            firm.trust_score = TRUST_DEFAULT
        target = _firm_trust_target(firm, model)
        firm.trust_score = _blend(float(firm.trust_score), target, 0.06)

    # Landowners
    for landowner in landowners:
        if not hasattr(landowner, "trust_score"):
            landowner.trust_score = TRUST_DEFAULT
        target = _landowner_trust_target(landowner, model)
        landowner.trust_score = _blend(float(landowner.trust_score), target, 0.05)

    # News firms
    for news_firm in news_firms:
        if not hasattr(news_firm, "trust_score"):
            news_firm.trust_score = TRUST_DEFAULT
        target = _news_trust_target(news_firm, model)
        news_firm.trust_score = _blend(float(news_firm.trust_score), target, 0.05)

    # Planner / institution
    if planner is not None:
        if not hasattr(planner, "trust_score"):
            planner.trust_score = TRUST_DEFAULT
        target = _planner_trust_target(planner, model)
        planner.trust_score = _blend(float(planner.trust_score), target, 0.04)

    # Institutional aggregate trust
    trust_values: List[float] = []
    for collection_name in ("workers", "firms", "landowners", "news_firms"):
        for agent in getattr(model, collection_name, []) or []:
            if hasattr(agent, "trust_score"):
                trust_values.append(float(agent.trust_score))
    if planner is not None and hasattr(planner, "trust_score"):
        trust_values.append(float(planner.trust_score))

    model.institutional_trust = float(np.mean(trust_values)) if trust_values else TRUST_DEFAULT


def trust_population_values(model) -> np.ndarray:
    """Return all trust-bearing agent scores as a numeric array."""
    values: List[float] = []
    for collection_name in ("workers", "firms", "landowners", "news_firms"):
        for agent in getattr(model, collection_name, []) or []:
            if hasattr(agent, "trust_score"):
                values.append(float(agent.trust_score))
    planner = getattr(model, "planner", None)
    if planner is not None and hasattr(planner, "trust_score"):
        values.append(float(planner.trust_score))
    return np.array(values, dtype=np.float64)

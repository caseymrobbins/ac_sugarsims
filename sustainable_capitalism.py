"""
sustainable_capitalism.py
-------------------------
Floor-raising incentive structure for firms.

Changes:
  - "innovate" strategy added to context scoring
  - Innovation boosts Company health (C) and Shareholder (S) dimensions
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from agents import FirmAgent
    from environment import EconomicModel


def compute_stakeholder_scores(firm: "FirmAgent") -> Dict[str, float]:
    model = firm.model

    # S: Shareholder
    profit_delta = firm.profit - firm.prev_profit
    profit_signal = math.tanh(profit_delta / max(abs(firm.profit) + 1, 1))
    capital_health = min(1.0, firm.capital_stock / max(firm.wealth * 0.3 + 1, 1))
    # Tech bonus: innovative firms have higher shareholder potential
    tech_bonus = min(0.1, (getattr(firm, 'tech_level', 1.0) - 1.0) * 0.1)
    S = 0.5 + 0.3 * profit_signal + 0.2 * capital_health + tech_bonus
    S = max(0.01, min(1.0, S))

    # E: Employee
    n_workers = len(firm.workers)
    if n_workers == 0:
        E = 0.1
    else:
        rev_per_worker = firm.revenue / max(n_workers, 1)
        if rev_per_worker > 0:
            wage_share = firm.offered_wage / rev_per_worker
            wage_fairness = min(1.0, wage_share / 0.5)
        else:
            wage_fairness = 0.3
        worker_skills = [w.skill for w in firm.workers.values()]
        mean_skill = np.mean(worker_skills) if worker_skills else 0.3
        skill_health = min(1.0, mean_skill / 0.6)
        worker_wealths = [w.wealth for w in firm.workers.values()]
        min_worker_wealth = min(worker_wealths) if worker_wealths else 0
        wealth_floor = min(1.0, min_worker_wealth / 50.0)
        employment = min(1.0, n_workers / 5.0)
        E = (wage_fairness * 0.35 + skill_health * 0.25
             + wealth_floor * 0.25 + employment * 0.15)
        E = max(0.01, min(1.0, E))

    # V: Environmental
    pollution_score = max(0.01, 1.0 - firm.pollution_factor * 2.0)
    if firm.cartel_id is not None:
        pollution_score *= 0.7
    V = max(0.01, min(1.0, pollution_score))

    # C: Company health
    if n_workers > 0:
        capital_per_worker = firm.capital_stock / max(n_workers, 1)
        capital_adequacy = min(1.0, capital_per_worker / 20.0)
    else:
        capital_adequacy = min(1.0, firm.capital_stock / 50.0)
    stability = 1.0 if firm._consecutive_losses == 0 else max(0.1, 1.0 / (1 + firm._consecutive_losses))
    longevity = min(1.0, firm.age / 100.0)
    # Tech level contributes to company health (future readiness)
    tech_readiness = min(0.15, (getattr(firm, 'tech_level', 1.0) - 1.0) * 0.1)
    C = (capital_adequacy * 0.35 + stability * 0.35 + longevity * 0.15 + tech_readiness)
    C = max(0.01, min(1.0, C))

    # Store raw scores for metrics/debugging
    raw = {'S_raw': S, 'E_raw': E, 'V_raw': V, 'C_raw': C}

    # EMA normalization: z-score each dimension using firm's own history,
    # then map to [0,1] via sigmoid so min() compares comparable scales.
    alpha = 0.05  # slow adaptation
    dims = {'S': S, 'E': E, 'V': V, 'C': C}
    normed = {}
    for d, val in dims.items():
        ema_mean = firm.sevc_ema_mean[d]
        ema_var = firm.sevc_ema_var[d]
        # Update EMA stats
        firm.sevc_ema_mean[d] = (1 - alpha) * ema_mean + alpha * val
        firm.sevc_ema_var[d] = (1 - alpha) * ema_var + alpha * (val - ema_mean) ** 2
        # Z-score then sigmoid to [0,1]
        std = max(math.sqrt(firm.sevc_ema_var[d]), 0.01)
        z = (val - firm.sevc_ema_mean[d]) / std
        normed[d] = 1.0 / (1.0 + math.exp(-z))

    S_n, E_n, V_n, C_n = normed['S'], normed['E'], normed['V'], normed['C']
    floor = min(S_n, E_n, V_n, C_n)
    result = {'S': S_n, 'E': E_n, 'V': V_n, 'C': C_n, 'floor': floor}
    result.update(raw)
    return result


def compute_firm_horizon_index(firm: "FirmAgent") -> float:
    """
    Firm-level Horizon Index: detects sugar-rush firm behavior.

    Compares recent floor trend against longer-term baseline.
    A firm with rising profit but declining floor is consuming
    its foundations (the Boeing pattern).

    Returns a value in [0, 1] where:
      1.0 = floor is stable or improving (sustainable)
      0.0 = floor is in rapid decline (sugar rush, collapse imminent)
    """
    history = firm.floor_history
    n = len(history)
    if n < 10:
        return 1.0  # not enough data

    # Recent window: last 20 steps (or available)
    recent_len = min(20, n)
    recent_mean = sum(list(history)[-recent_len:]) / recent_len

    # Baseline window: last 60-100 steps (or full history)
    baseline_mean = sum(history) / n

    # Ratio: recent vs baseline
    ratio = recent_mean / max(baseline_mean, 0.01)
    firm_hi = min(1.0, max(0.0, ratio))

    # Acceleration penalty: is the decline accelerating?
    if n >= 20:
        last_10 = sum(list(history)[-10:]) / 10
        prev_10 = sum(list(history)[-20:-10]) / 10
        if prev_10 > 0.01:
            accel_ratio = last_10 / max(prev_10, 0.01)
            if accel_ratio < 1.0:
                firm_hi *= max(0.5, accel_ratio)

    return float(min(1.0, max(0.0, firm_hi)))


def sustainable_learn_from_outcome(firm: "FirmAgent", strategy: str, profit_change: float):
    if strategy not in firm.strategy_weights:
        return
    scores = compute_stakeholder_scores(firm)
    current_floor = scores['floor']
    # Track floor history and compute firm HI
    firm.floor_history.append(current_floor)
    firm.horizon_index = compute_firm_horizon_index(firm)
    if not hasattr(firm, '_prev_floor'):
        firm._prev_floor = current_floor
        firm._prev_scores = scores
    floor_change = current_floor - firm._prev_floor
    # Firm HI modulates learning: declining firms get dampened signal
    use_firm_hi = getattr(firm.model, 'use_firm_hi', False)
    if use_firm_hi:
        effective_signal = floor_change * firm.horizon_index
    else:
        effective_signal = floor_change
    adj = 0.02 * np.tanh(effective_signal / max(current_floor + 0.1, 0.1))
    cur = firm.strategy_weights[strategy]
    firm.strategy_weights[strategy] = float(np.clip(cur + adj, 0.01, 0.99))
    if firm.profit < 0:
        firm._consecutive_losses += 1
    else:
        firm._consecutive_losses = 0
    firm._prev_floor = current_floor
    firm._prev_scores = scores


def sustainable_choose_strategy(firm: "FirmAgent") -> str:
    rng = firm.model.rng
    n_workers = len(firm.workers)
    profitable = firm.profit > 0

    scores = compute_stakeholder_scores(firm)
    floor_dim = min(scores, key=lambda k: scores[k] if k != 'floor' else 999)

    context = {
        "invest_capital":  (1.0 if profitable else 0.2) * min(firm.wealth / 200, 1),
        "raise_wages":     (0.3 if n_workers < 3 else 0.1) * (1.0 if profitable else 0.3),
        "cut_wages":       (0.8 if not profitable else 0.1) * (1.0 if n_workers > 0 else 0.0),
        "hire":            (0.5 if n_workers < 10 else 0.1) * (1.0 if profitable else 0.2),
        "downsize":        (0.8 if firm._consecutive_losses > 2 else 0.1),
        "acquire":         min(firm.wealth / 500, 1) * firm.market_share * 5,
        "form_cartel":     0.3 if firm.cartel_id is None else 0.0,
        "capture_media":   min(firm.wealth / 1000, 1) * (0.5 if firm.cartel_id else 0.1),
        "pollute_more":    0.3 if not profitable else 0.1,
        "clean_up":        firm.model.planner.policy.get("pollution_tax", 0) * 0.5,
        # Innovation: profitable firms with capital and workers should innovate
        "innovate":        (0.5 if profitable else 0.1) * min(firm.wealth / 100, 1) * (0.8 if n_workers > 0 else 0.2),
    }

    # Bottleneck-driven strategy boosting
    if floor_dim == 'E':
        context["raise_wages"] *= 3.0
        context["hire"] *= 2.0
        context["cut_wages"] *= 0.1
        context["downsize"] *= 0.1
    elif floor_dim == 'V':
        context["clean_up"] *= 4.0
        context["pollute_more"] *= 0.05
        context["form_cartel"] *= 0.2
        context["innovate"] *= 3.0  # green R&D helps environment
        # Shift green_rd_priority toward 1.0 (green R&D)
        firm.green_rd_priority = min(1.0, firm.green_rd_priority + 0.05)
    elif floor_dim == 'S':
        context["invest_capital"] *= 2.0
        context["innovate"] *= 2.0  # innovation helps shareholder value
        context["raise_wages"] *= 0.5
        # Shift green_rd_priority toward 0.0 (productivity R&D)
        firm.green_rd_priority = max(0.0, firm.green_rd_priority - 0.03)
    elif floor_dim == 'C':
        context["invest_capital"] *= 3.0
        context["innovate"] *= 1.5  # tech readiness helps company health
        context["downsize"] *= 0.3
        context["acquire"] *= 0.1
        # Shift green_rd_priority toward 0.0 (productivity R&D)
        firm.green_rd_priority = max(0.0, firm.green_rd_priority - 0.03)

    # Firm HI intervention: when firm is on unsustainable trajectory,
    # boost floor-raising strategies and suppress extractive ones
    use_firm_hi = getattr(firm.model, 'use_firm_hi', False)
    if use_firm_hi and firm.horizon_index < 0.7:
        context["invest_capital"] *= 2.0
        context["raise_wages"] *= 2.0
        context["cut_wages"] *= 0.3
        context["downsize"] *= 0.3
        context["pollute_more"] *= 0.3

    strategy_scores = {}
    for action, weight in firm.strategy_weights.items():
        strategy_scores[action] = weight * context.get(action, 0.5) + float(rng.normal(0, 0.03))

    return max(strategy_scores, key=strategy_scores.get)

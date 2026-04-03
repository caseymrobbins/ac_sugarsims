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
        firm.capture_ratio = 0.0
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

        # Production-capture ratio: what fraction of per-worker revenue goes to workers?
        # Uses wages_this_step (actual cash paid this step) vs current-step revenue.
        wages_paid = getattr(firm, 'wages_this_step', firm.offered_wage * n_workers)
        total_rev = max(firm.revenue, 1e-9)
        capture_ratio = wages_paid / total_rev
        firm.capture_ratio = float(np.clip(capture_ratio, 0.0, 2.0))

        # Gate: only include capture_score in E when production_aware_E flag is set
        if getattr(firm.model, 'production_aware_E', False):
            # Normalisation: fixed reference (0.3) or EMA-based adaptive reference
            if getattr(firm.model, 'capture_normalization', 'fixed') == 'ema':
                # Update EMA; score = ratio / own-history → rewards improvement trajectory
                ema_alpha = 0.05
                firm.capture_ratio_ema = float(
                    ema_alpha * capture_ratio + (1.0 - ema_alpha) * firm.capture_ratio_ema)
                ref = max(firm.capture_ratio_ema, 0.001)
                capture_score = float(np.clip(capture_ratio / ref, 0.0, 2.0))
            else:
                # Fixed Costco benchmark: 0.3 = adequate
                capture_score = min(capture_ratio / 0.3, 2.0)
            # Geometric mean of four components (equal weight each)
            E = (max(wage_fairness, 1e-6) * max(skill_health, 1e-6)
                 * max(wealth_floor, 1e-6) * max(capture_score, 1e-6)) ** 0.25
        else:
            E = (wage_fairness * 0.35 + skill_health * 0.25
                 + wealth_floor * 0.25 + employment * 0.15)
        E = max(0.01, min(1.0, E))

    # V: Environmental
    if getattr(model, 'v_measures_total_emissions', False) and n_workers > 0 and firm.production_this_step > 0:
        # Two components: intensity (per-unit rate) and scale (total emissions per worker)
        intensity_score = max(0.01, 1.0 - firm.pollution_factor * 2.0)
        emissions_this_step = firm.production_this_step * firm.pollution_factor
        emissions_per_worker = emissions_this_step / max(n_workers, 1)
        # 10 emissions/worker = adequate, above = bad; 50 = zero score
        scale_score = max(0.01, 1.0 - emissions_per_worker / 50.0)
        # Geometric mean: both must be healthy for V to be healthy
        pollution_score = max(0.01, (intensity_score * scale_score) ** 0.5)
    else:
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

    # EMA-reference normalization: score = raw / ema_reference
    # 1.0 = performing at historical average; <1.0 = declining; >1.0 = improving
    # This ensures min() selects genuinely worst dimension relative to its own baseline.
    alpha = 0.05  # slow adaptation
    dims = {'S': S, 'E': E, 'V': V, 'C': C}
    normed = {}
    for d, val in dims.items():
        # Update EMA reference
        firm.sevc_ema_mean[d] = (1 - alpha) * firm.sevc_ema_mean[d] + alpha * val
        ref = max(firm.sevc_ema_mean[d], 0.001)
        normed[d] = val / ref

    S_n, E_n, V_n, C_n = normed['S'], normed['E'], normed['V'], normed['C']
    floor = min(S_n, E_n, V_n, C_n)
    binding = min(normed, key=normed.get)
    # Write back to firm so CEO compensation and other systems can read without recomputing
    firm.sevc_floor = float(floor)
    firm.sevc_binding = binding
    result = {'S': S_n, 'E': E_n, 'V': V_n, 'C': C_n, 'floor': floor, 'binding': binding}
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
    # Headroom stability tracking (Task 13B): CV of S-floor gap over 30-step window
    # Use raw scores so the gap reflects absolute divergence, not normalized trajectory
    scores_raw_s = scores.get('S_raw', 0.5)
    raw_floor = min(scores.get('E_raw', 0.5), scores.get('V_raw', 0.5), scores.get('C_raw', 0.5))
    headroom_gap = scores_raw_s - raw_floor
    firm.headroom_history.append(headroom_gap)
    if len(firm.headroom_history) >= 10:
        hh = np.array(firm.headroom_history)
        mu = float(np.mean(hh)); sigma = float(np.std(hh))
        cv = sigma / max(abs(mu), 0.01)
        firm.headroom_stability = max(0.0, 1.0 - cv)  # 1.0 = perfectly stable
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
    _skip = ('floor', 'binding', 'S_raw', 'E_raw', 'V_raw', 'C_raw')
    floor_dim = min(scores, key=lambda k: scores[k] if k not in _skip else 999)

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

    # Headroom stability bonus: stable firms are in optimal position to innovate (Task 13B)
    hs = getattr(firm, 'headroom_stability', 0.0)
    if hs > 0.5:
        context["innovate"] *= 1.0 + (hs - 0.5) * 2.0  # up to 2x boost at stability=1.0

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

    # CEO compensation tied to SEVC floor (Task 12): CEO has personal financial stake in
    # fixing the binding dimension — amplify the bottleneck boosts further.
    if getattr(firm.model, 'ceo_compensation_tied', False):
        ceo_floor = getattr(firm, 'sevc_floor', 1.0)
        # Urgency inversely proportional to floor: lower floor = stronger CEO signal
        urgency = max(0.0, 1.0 - ceo_floor)  # 0 at floor=1.0, 1.0 at floor=0.0
        if floor_dim == 'E':
            context["raise_wages"] *= (1.0 + 3.0 * urgency)
            context["hire"] *= (1.0 + 1.5 * urgency)
            context["cut_wages"] *= max(0.01, 1.0 - 2.0 * urgency)
        elif floor_dim == 'V':
            context["clean_up"] *= (1.0 + 3.0 * urgency)
            context["pollute_more"] *= max(0.01, 1.0 - 3.0 * urgency)
        elif floor_dim == 'C':
            context["invest_capital"] *= (1.0 + 2.0 * urgency)
        elif floor_dim == 'S':
            context["innovate"] *= (1.0 + 1.5 * urgency)

    strategy_scores = {}
    for action, weight in firm.strategy_weights.items():
        strategy_scores[action] = weight * context.get(action, 0.5) + float(rng.normal(0, 0.03))

    return max(strategy_scores, key=strategy_scores.get)

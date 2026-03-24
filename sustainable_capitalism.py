"""
sustainable_capitalism.py
-------------------------
Floor-raising incentive structure for firms.

Implements the Sustainable Capitalism mechanism from the AC governance
paper: the firm's learning signal is min(S, E, V, C) across four
stakeholder dimensions rather than raw profit.

The firm still pursues self-interest. It still learns through RL.
It still competes. But its self-interest is now structurally aligned
with all stakeholders because the learning signal is the MINIMUM
across all four dimensions. A firm that suppresses wages to boost
profit sees no improvement in its learning signal because the employee
dimension becomes the floor.

This is not regulation. The planner doesn't enforce it.
This is not altruism. The firm acts in pure self-interest.
This is incentive architecture: the firm's own optimization is
redirected from extraction to generation.

Integration: replace FirmAgent._learn_from_outcome with
sustainable_learn_from_outcome, and add compute_stakeholder_scores
to FirmAgent.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from agents import FirmAgent
    from environment import EconomicModel


def compute_stakeholder_scores(firm: "FirmAgent") -> Dict[str, float]:
    """
    Compute the four stakeholder dimension scores for a firm.
    Each score is normalized to roughly [0, 1] range.
    
    S = Shareholder return (profit growth, capital preservation)
    E = Employee wellbeing (wages, skills, employment)
    V = Environmental impact (pollution control)
    C = Company health (capital, stability, sustainability)
    
    Returns dict with 'S', 'E', 'V', 'C' and 'floor' (the min).
    """
    model = firm.model
    
    # ── S: Shareholder score ────────────────────────────────────
    # Profit change (bounded by tanh to prevent extreme values)
    profit_delta = firm.profit - firm.prev_profit
    profit_signal = math.tanh(profit_delta / max(abs(firm.profit) + 1, 1))
    
    # Capital preservation: is the firm maintaining/growing its capital?
    capital_health = min(1.0, firm.capital_stock / max(firm.wealth * 0.3 + 1, 1))
    
    # Combined: weighted toward profit signal but capital matters
    S = 0.5 + 0.3 * profit_signal + 0.2 * capital_health
    S = max(0.01, min(1.0, S))
    
    # ── E: Employee score ───────────────────────────────────────
    n_workers = len(firm.workers)
    
    if n_workers == 0:
        # No employees: employee score is low (you should be hiring)
        E = 0.1
    else:
        # Wage fairness: offered wage relative to revenue per worker
        rev_per_worker = firm.revenue / max(n_workers, 1)
        if rev_per_worker > 0:
            wage_share = firm.offered_wage / rev_per_worker
            # Healthy range: workers get 40-70% of revenue they generate
            wage_fairness = min(1.0, wage_share / 0.5)  # 1.0 at 50%+ share
        else:
            wage_fairness = 0.3
        
        # Worker skill health: are workers growing or stagnating?
        worker_skills = [w.skill for w in firm.workers.values()]
        mean_skill = np.mean(worker_skills) if worker_skills else 0.3
        skill_health = min(1.0, mean_skill / 0.6)  # 1.0 at skill >= 0.6
        
        # Worker wealth: are workers above subsistence?
        worker_wealths = [w.wealth for w in firm.workers.values()]
        min_worker_wealth = min(worker_wealths) if worker_wealths else 0
        wealth_floor = min(1.0, min_worker_wealth / 50.0)  # 1.0 when min >= 50
        
        # Employment: having workers is good (penalize empty firms)
        employment = min(1.0, n_workers / 5.0)  # 1.0 at 5+ workers
        
        E = (wage_fairness * 0.35 + skill_health * 0.25 
             + wealth_floor * 0.25 + employment * 0.15)
        E = max(0.01, min(1.0, E))
    
    # ── V: Environmental score ──────────────────────────────────
    # Lower pollution factor = better environmental score
    # pollution_factor ranges from 0.02 (clean) to 0.60 (dirty)
    pollution_score = max(0.01, 1.0 - firm.pollution_factor * 2.0)
    # Penalty for being in a cartel (cartels increase pollution)
    if firm.cartel_id is not None:
        pollution_score *= 0.7  # 30% penalty for cartel membership
    
    V = max(0.01, min(1.0, pollution_score))
    
    # ── C: Company health score ─────────────────────────────────
    # Capital adequacy: enough capital to sustain operations
    if n_workers > 0:
        capital_per_worker = firm.capital_stock / max(n_workers, 1)
        capital_adequacy = min(1.0, capital_per_worker / 20.0)
    else:
        capital_adequacy = min(1.0, firm.capital_stock / 50.0)
    
    # Stability: not bleeding money
    stability = 1.0 if firm._consecutive_losses == 0 else max(0.1, 1.0 / (1 + firm._consecutive_losses))
    
    # Longevity: survival is health
    longevity = min(1.0, firm.age / 100.0)  # 1.0 after 100 steps
    
    C = (capital_adequacy * 0.4 + stability * 0.4 + longevity * 0.2)
    C = max(0.01, min(1.0, C))
    
    # ── Floor (the min) ─────────────────────────────────────────
    floor = min(S, E, V, C)
    
    return {'S': S, 'E': E, 'V': V, 'C': C, 'floor': floor}


def sustainable_learn_from_outcome(firm: "FirmAgent", strategy: str, profit_change: float):
    """
    Replacement for FirmAgent._learn_from_outcome.
    
    Instead of learning from raw profit_change (extraction signal),
    learns from the floor of the stakeholder scores (generation signal).
    
    The firm's strategy weights shift toward strategies that raise
    the MINIMUM dimension, not strategies that raise profit alone.
    
    A firm that cuts wages to boost profit sees:
      - S goes up (profit increased)
      - E goes down (employee wellbeing decreased)
      - floor = E (employee is now the bottleneck)
      - Learning signal = E_change, which is NEGATIVE
      - Strategy weight for "cut_wages" DECREASES
    
    A firm that invests in capital while maintaining wages sees:
      - S stays stable (profit neutral short term)
      - E stays stable (wages maintained)
      - C goes up (capital invested)
      - floor rises (if C was the bottleneck) or stays (if not)
      - Learning signal = floor_change, which is POSITIVE or NEUTRAL
      - Strategy weight for "invest_capital" INCREASES
    
    This is the Sustainable Capitalism mechanism: self-interest
    operating through min() produces floor-raising behavior.
    """
    if strategy not in firm.strategy_weights:
        return
    
    # Compute current stakeholder scores
    scores = compute_stakeholder_scores(firm)
    current_floor = scores['floor']
    
    # Store previous floor for comparison (use profit as fallback first time)
    if not hasattr(firm, '_prev_floor'):
        firm._prev_floor = current_floor
        firm._prev_scores = scores
    
    # Learning signal: change in the floor
    floor_change = current_floor - firm._prev_floor
    
    # Adjustment: same tanh-bounded update as original, but on floor_change
    adj = 0.02 * np.tanh(floor_change / max(current_floor + 0.1, 0.1))
    cur = firm.strategy_weights[strategy]
    firm.strategy_weights[strategy] = float(np.clip(cur + adj, 0.01, 0.99))
    
    # Track consecutive losses (still useful for survival mechanics)
    if firm.profit < 0:
        firm._consecutive_losses += 1
    else:
        firm._consecutive_losses = 0
    
    # Store for next step
    firm._prev_floor = current_floor
    firm._prev_scores = scores


def sustainable_choose_strategy(firm: "FirmAgent") -> str:
    """
    Replacement for FirmAgent._choose_strategy.
    
    Context scoring is modified: instead of pure profit-seeking context,
    the context factors in which stakeholder dimension is currently
    the floor, and weights strategies that would raise it.
    """
    rng = firm.model.rng
    n_workers = len(firm.workers)
    profitable = firm.profit > 0
    
    # Compute current stakeholder scores to find the bottleneck
    scores = compute_stakeholder_scores(firm)
    floor_dim = min(scores, key=lambda k: scores[k] if k != 'floor' else 999)
    
    # Base context (similar to original but modified by bottleneck)
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
    }
    
    # SUSTAINABLE CAPITALISM MODIFICATION:
    # Boost strategies that address the current bottleneck dimension
    if floor_dim == 'E':  # Employee is the floor
        context["raise_wages"] *= 3.0     # strong incentive to raise wages
        context["hire"] *= 2.0            # hiring improves employee score
        context["cut_wages"] *= 0.1       # cutting wages hurts the floor
        context["downsize"] *= 0.1        # downsizing hurts the floor
    elif floor_dim == 'V':  # Environment is the floor
        context["clean_up"] *= 4.0        # strong incentive to clean up
        context["pollute_more"] *= 0.05   # pollution hurts the floor
        context["form_cartel"] *= 0.2     # cartels increase pollution
    elif floor_dim == 'S':  # Shareholder is the floor
        context["invest_capital"] *= 2.0  # invest to grow returns
        context["raise_wages"] *= 0.5     # don't raise costs when margins thin
    elif floor_dim == 'C':  # Company health is the floor
        context["invest_capital"] *= 3.0  # strengthen the company
        context["downsize"] *= 0.3        # don't shrink when health is low
        context["acquire"] *= 0.1         # don't overextend
    
    # Score strategies using weights * context
    strategy_scores = {}
    for action, weight in firm.strategy_weights.items():
        strategy_scores[action] = weight * context.get(action, 0.5) + float(rng.normal(0, 0.03))
    
    return max(strategy_scores, key=strategy_scores.get)


# ---------------------------------------------------------------------------
# Integration instructions
# ---------------------------------------------------------------------------
"""
TO INTEGRATE INTO agents.py:

1. Import at top of agents.py:
   from sustainable_capitalism import (
       sustainable_learn_from_outcome, 
       sustainable_choose_strategy,
       compute_stakeholder_scores
   )

2. Replace FirmAgent._learn_from_outcome:
   
   BEFORE (in FirmAgent.step()):
       self._learn_from_outcome(strategy, profit_change)
   
   AFTER:
       sustainable_learn_from_outcome(self, strategy, profit_change)

3. Replace FirmAgent._choose_strategy:
   
   BEFORE (in FirmAgent.step()):
       strategy = self._choose_strategy()
   
   AFTER:
       strategy = sustainable_choose_strategy(self)

That's two line changes in FirmAgent.step(). The firm's production,
execution, hiring, firing, and all other mechanics stay exactly the
same. Only the LEARNING SIGNAL and STRATEGY SELECTION change.

The firm still pursues self-interest. It still competes.
Its self-interest now requires raising ALL stakeholder floors.

OPTIONAL - add to metrics.py for tracking:
    from sustainable_capitalism import compute_stakeholder_scores
    
    # In collect_step_metrics():
    firm_floors = []
    for f in model.firms:
        if not f.defunct:
            scores = compute_stakeholder_scores(f)
            firm_floors.append(scores['floor'])
    if firm_floors:
        m["mean_firm_floor"] = float(np.mean(firm_floors))
        m["min_firm_floor"] = float(np.min(firm_floors))
    else:
        m["mean_firm_floor"] = 0.0
        m["min_firm_floor"] = 0.0
"""

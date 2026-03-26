"""
trust.py
--------
Public trust scores for all agents and institutions.

Every agent carries a trust_score in [0.0, 1.0] that moves based on
observable behavior. Other agents use it when deciding whether to
trade, work for, borrow from, invest in, or listen to that entity.

This is the missing immune system. Extractive behavior tanks your
trust score, which cuts off access to cooperation, which limits
further extraction. No planner enforcement required: the consequence
is structural.

Trust is NOT bilateral (A trusts B). It is reputational: each agent
has ONE score that everyone can observe. This matches how reputation
works in real economies: credit ratings, employer reviews, brand
trust, institutional approval ratings.

Architecture:
  - update_trust_scores(model) called once per step from environment.py
  - Each agent type has specific observable behaviors that move trust
  - Trust changes are bounded and gradual (EMA smoothing)
  - Trust scores feed into existing decision points via simple lookups

Integration points (where trust_score gets used):
  - WorkerAgent._seek_employment: won't work for firms with trust < 0.2
  - WorkerAgent._invest_in_firms: weight investment by firm trust
  - Economy.issue_loan: interest rate includes borrower trust discount
  - FirmAgent.hire_worker: prefer higher-trust workers
  - WorkerAgent.receive_information: scale by source trust (replaces
    part of authority_trust)
  - Planner policy response: firms invest less when planner trust is low
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from environment import EconomicModel
    from agents import WorkerAgent, FirmAgent, LandownerAgent
    from information import NewsFirm
    from banking import BankAgent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Starting trust for new agents
DEFAULT_TRUST_WORKER = 0.5
DEFAULT_TRUST_FIRM = 0.5
DEFAULT_TRUST_BANK = 0.6       # banks start slightly higher (institutional)
DEFAULT_TRUST_NEWS = 0.5
DEFAULT_TRUST_PLANNER = 0.5
DEFAULT_TRUST_LANDOWNER = 0.5

# How fast trust moves (exponential moving average weight for new signal)
# Lower = more inertia, harder to game with one good step
TRUST_MOMENTUM = 0.05

# Floor and ceiling (trust never fully dies or maxes)
TRUST_FLOOR = 0.02
TRUST_CEILING = 0.98

# Decay rate: trust slowly decays toward neutral if no signal (use it or lose it)
TRUST_NEUTRAL = 0.5
TRUST_DECAY_RATE = 0.001


# ---------------------------------------------------------------------------
# Trust score initialization
# ---------------------------------------------------------------------------

def init_trust(agent, agent_type: str):
    """
    Initialize trust_score on an agent. Call from agent __init__.
    """
    defaults = {
        "worker": DEFAULT_TRUST_WORKER,
        "firm": DEFAULT_TRUST_FIRM,
        "bank": DEFAULT_TRUST_BANK,
        "news": DEFAULT_TRUST_NEWS,
        "planner": DEFAULT_TRUST_PLANNER,
        "landowner": DEFAULT_TRUST_LANDOWNER,
    }
    agent.trust_score = defaults.get(agent_type, 0.5)


def _ema_update(current: float, signal: float) -> float:
    """
    Exponential moving average trust update.
    Signal should be in [0, 1] representing how trustworthy
    the agent's behavior was this step.
    """
    new = (1 - TRUST_MOMENTUM) * current + TRUST_MOMENTUM * signal
    # Slow decay toward neutral when signal is near current (no news)
    new += TRUST_DECAY_RATE * (TRUST_NEUTRAL - new)
    return float(np.clip(new, TRUST_FLOOR, TRUST_CEILING))


# ---------------------------------------------------------------------------
# Per-type trust signal computation
# ---------------------------------------------------------------------------

def _worker_trust_signal(worker: "WorkerAgent") -> float:
    """
    Worker trust based on:
      - Loan repayment (not defaulting)
      - Employment stability (employed = productive member)
      - Positive income (contributing to economy)

    Returns signal in [0, 1].
    """
    scores = []

    # Loan behavior: no debt or servicing debt = good, default = bad
    if worker.debt > 0:
        # In debt but making payments (wealth > debt interest) = neutral-good
        if worker.wealth > worker.debt * worker.debt_interest:
            scores.append(0.6)
        else:
            scores.append(0.2)  # struggling to pay
    else:
        if worker.loan_count > 0:
            scores.append(0.8)  # had loans, paid them off
        else:
            scores.append(0.5)  # no credit history

    # Employment: employed workers are more trusted (stable, productive)
    scores.append(0.7 if worker.employed else 0.3)

    # Income: positive income = contributing
    if worker.income_last_step > 0:
        scores.append(min(0.8, 0.4 + worker.income_last_step / 20.0))
    else:
        scores.append(0.3)

    return float(np.mean(scores))


def _firm_trust_signal(firm: "FirmAgent") -> float:
    """
    Firm trust based on:
      - Wage payment (paying workers consistently)
      - Employment (having and keeping workers)
      - Profitability (not about to collapse)
      - Cartel membership (cartels are extractive)
      - Pollution (environmental responsibility)

    Returns signal in [0, 1].
    """
    scores = []
    n_workers = len(firm.workers)

    # Wage reliability: paying wages and not cutting them
    if n_workers > 0:
        if firm.offered_wage >= 2.0 and firm.wealth > firm.offered_wage * n_workers:
            scores.append(0.8)  # can afford and paying decent wages
        elif firm.wealth > 0:
            scores.append(0.5)  # paying but stretched
        else:
            scores.append(0.15)  # might not make payroll
    else:
        scores.append(0.3)  # no workers, less to evaluate

    # Employment stability: firms that fire people lose trust
    if firm._last_strategy == "downsize":
        scores.append(0.2)
    elif firm._last_strategy == "hire" or firm._last_strategy == "raise_wages":
        scores.append(0.8)
    else:
        scores.append(0.5)

    # Profitability: profitable firms are more trusted (won't vanish)
    if firm.profit > 0:
        scores.append(0.7)
    elif firm._consecutive_losses == 0:
        scores.append(0.5)
    else:
        scores.append(max(0.1, 0.5 - firm._consecutive_losses * 0.1))

    # Cartel membership: extractive, trust penalty
    if firm.cartel_id is not None:
        scores.append(0.15)
    else:
        scores.append(0.6)

    # Pollution: clean firms are more trusted
    # pollution_factor ranges 0.02 (clean) to 0.60 (dirty)
    pollution_trust = max(0.1, 1.0 - firm.pollution_factor * 1.5)
    scores.append(pollution_trust)

    return float(np.mean(scores))


def _bank_trust_signal(bank: "BankAgent") -> float:
    """
    Bank trust based on:
      - Solvency (capital > liabilities)
      - Default rate on loans (low = good underwriting)
      - Profit (sustainable operation)
      - Fair rates (not predatory)

    Returns signal in [0, 1].
    """
    scores = []

    # Solvency
    total_deposits = getattr(bank, 'total_deposits', 0)
    if bank.wealth > total_deposits * 0.1:
        scores.append(0.8)  # well capitalized
    elif bank.wealth > 0:
        scores.append(0.4)  # thin but solvent
    else:
        scores.append(0.05)  # insolvent

    # Default rate (if bank tracks it)
    default_rate = getattr(bank, 'default_rate', 0.0)
    if default_rate < 0.05:
        scores.append(0.8)
    elif default_rate < 0.15:
        scores.append(0.5)
    else:
        scores.append(0.2)  # high defaults = bad underwriting or predatory

    # Profitability
    profit = getattr(bank, 'profit', 0)
    if profit > 0:
        scores.append(0.7)
    else:
        scores.append(0.3)

    return float(np.mean(scores))


def _news_trust_signal(news_firm: "NewsFirm") -> float:
    """
    News firm trust based on:
      - Accuracy (editorial quality)
      - Capture status (captured = biased = untrustworthy)
      - Audience size (proxy for perceived value)
      - Financial health (can afford journalism)

    Returns signal in [0, 1].
    """
    scores = []

    # Accuracy is the core of news trust
    scores.append(news_firm.accuracy)

    # Captured firms are untrustworthy (but agents don't know this directly,
    # they learn it through outcomes. The trust signal reflects the ACTUAL
    # quality of output, which capture degrades.)
    if news_firm.captured_by_cartel is not None:
        # Captured firms produce worse signals, which hurts outcomes,
        # which agents eventually notice via authority_trust decay.
        # Trust signal reflects actual output quality.
        scores.append(0.2)
    else:
        scores.append(0.7)

    # Financial health: can this firm sustain journalism?
    if news_firm.wealth > 50:
        scores.append(0.7)
    elif news_firm.wealth > 0:
        scores.append(0.4)
    else:
        scores.append(0.1)

    return float(np.mean(scores))


def _planner_trust_signal(model: "EconomicModel") -> float:
    """
    Planner trust based on observable outcomes:
      - Population stability (not collapsing)
      - Floor wealth (poorest aren't dying)
      - Employment (people can find work)
      - Policy stability (not whipsawing)
      - Inequality trajectory (not exploding)

    This is the "approval rating." When it drops, firms invest less,
    workers save more (precautionary), and the economy contracts.
    """
    scores = []
    planner = model.planner
    history = model.metrics_history

    # Population: stable or growing = trustworthy governance
    pop_ratio = len(model.workers) / max(model.n_workers_initial, 1)
    scores.append(min(1.0, pop_ratio))

    # Floor wealth: poorest above survival = good
    if model.workers:
        worker_w = np.array([w.wealth for w in model.workers])
        floor = float(np.min(worker_w))
        scores.append(min(1.0, floor / 20.0))  # 1.0 when floor >= 20
    else:
        scores.append(0.0)

    # Employment
    if model.workers:
        emp_rate = sum(1 for w in model.workers if w.employed) / max(len(model.workers), 1)
        scores.append(emp_rate)
    else:
        scores.append(0.0)

    # Policy stability: how much did policy instruments change?
    # Large swings = uncertainty = trust loss
    if len(history) >= 2:
        prev = history[-2]
        curr = history[-1]
        policy_keys = [
            "planner_tax_worker", "planner_tax_firm", "planner_ubi",
            "planner_min_wage",
        ]
        changes = []
        for k in policy_keys:
            p = prev.get(k, 0)
            c = curr.get(k, 0)
            if isinstance(p, (int, float)) and isinstance(c, (int, float)):
                scale = max(abs(p), abs(c), 1.0)
                changes.append(abs(c - p) / scale)
        if changes:
            mean_change = float(np.mean(changes))
            # Low change = stable = trustworthy
            stability = max(0.1, 1.0 - mean_change * 10)
            scores.append(stability)
        else:
            scores.append(0.5)
    else:
        scores.append(0.5)

    # Inequality: Gini not exploding
    if history:
        gini = history[-1].get("all_gini", 0.5)
        if isinstance(gini, (int, float)) and np.isfinite(gini):
            # Low Gini = more trust. High Gini = system feels rigged.
            scores.append(max(0.1, 1.0 - gini))
        else:
            scores.append(0.5)
    else:
        scores.append(0.5)

    return float(np.mean(scores))


def _landowner_trust_signal(landowner: "LandownerAgent") -> float:
    """
    Landowner trust based on:
      - Rent rate (fair vs extractive)
      - Territory size (not monopolizing land)

    Returns signal in [0, 1].
    """
    scores = []

    # Rent fairness: low rent = trusted landlord
    # rent_rate ranges 0.02 to 0.50
    rent_trust = max(0.1, 1.0 - landowner.rent_rate * 2.0)
    scores.append(rent_trust)

    # Territory: moderate holdings are fine, huge = monopolist
    n_cells = len(landowner.controlled_cells)
    if n_cells < 20:
        scores.append(0.6)
    elif n_cells < 40:
        scores.append(0.4)
    else:
        scores.append(0.2)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main update function (called once per step from environment.py)
# ---------------------------------------------------------------------------

def update_trust_scores(model: "EconomicModel"):
    """
    Update trust_score for every agent in the model.
    Call this once per step, after all agents have acted.
    """
    # Workers
    for w in model.workers:
        if not hasattr(w, 'trust_score'):
            init_trust(w, "worker")
        signal = _worker_trust_signal(w)
        w.trust_score = _ema_update(w.trust_score, signal)

    # Firms
    for f in model.firms:
        if f.defunct:
            continue
        if not hasattr(f, 'trust_score'):
            init_trust(f, "firm")
        signal = _firm_trust_signal(f)
        f.trust_score = _ema_update(f.trust_score, signal)

    # Banks
    for b in getattr(model, 'banks', []):
        if getattr(b, 'defunct', False):
            continue
        if not hasattr(b, 'trust_score'):
            init_trust(b, "bank")
        signal = _bank_trust_signal(b)
        b.trust_score = _ema_update(b.trust_score, signal)

    # News firms
    for nf in getattr(model, 'news_firms', []):
        if getattr(nf, 'defunct', False):
            continue
        if not hasattr(nf, 'trust_score'):
            init_trust(nf, "news")
        signal = _news_trust_signal(nf)
        nf.trust_score = _ema_update(nf.trust_score, signal)

    # Landowners
    for lo in model.landowners:
        if not hasattr(lo, 'trust_score'):
            init_trust(lo, "landowner")
        signal = _landowner_trust_signal(lo)
        lo.trust_score = _ema_update(lo.trust_score, signal)

    # Planner (trust stored on the planner agent)
    if not hasattr(model.planner, 'trust_score'):
        init_trust(model.planner, "planner")
    signal = _planner_trust_signal(model)
    model.planner.trust_score = _ema_update(model.planner.trust_score, signal)


# ---------------------------------------------------------------------------
# Trust metrics (for metrics.py integration)
# ---------------------------------------------------------------------------

def compute_trust_metrics(model: "EconomicModel") -> Dict[str, float]:
    """
    Compute trust-related metrics for the current step.
    Call from metrics.py collect_step_metrics().
    """
    metrics = {}

    # Worker trust distribution
    if model.workers:
        worker_trusts = np.array([
            getattr(w, 'trust_score', 0.5) for w in model.workers
        ])
        metrics["trust_worker_mean"] = float(np.mean(worker_trusts))
        metrics["trust_worker_min"] = float(np.min(worker_trusts))
        metrics["trust_worker_std"] = float(np.std(worker_trusts))
    else:
        metrics["trust_worker_mean"] = 0.0
        metrics["trust_worker_min"] = 0.0
        metrics["trust_worker_std"] = 0.0

    # Firm trust distribution
    active_firms = [f for f in model.firms if not f.defunct]
    if active_firms:
        firm_trusts = np.array([
            getattr(f, 'trust_score', 0.5) for f in active_firms
        ])
        metrics["trust_firm_mean"] = float(np.mean(firm_trusts))
        metrics["trust_firm_min"] = float(np.min(firm_trusts))
    else:
        metrics["trust_firm_mean"] = 0.0
        metrics["trust_firm_min"] = 0.0

    # Bank trust
    banks = [b for b in getattr(model, 'banks', [])
             if not getattr(b, 'defunct', False)]
    if banks:
        bank_trusts = np.array([
            getattr(b, 'trust_score', 0.5) for b in banks
        ])
        metrics["trust_bank_mean"] = float(np.mean(bank_trusts))
    else:
        metrics["trust_bank_mean"] = 0.0

    # News firm trust
    news = [nf for nf in getattr(model, 'news_firms', [])
            if not getattr(nf, 'defunct', False)]
    if news:
        news_trusts = np.array([
            getattr(nf, 'trust_score', 0.5) for nf in news
        ])
        metrics["trust_news_mean"] = float(np.mean(news_trusts))
        # Captured vs uncaptured trust gap
        captured = [nf for nf in news if nf.captured_by_cartel is not None]
        free = [nf for nf in news if nf.captured_by_cartel is None]
        if captured and free:
            cap_trust = np.mean([getattr(nf, 'trust_score', 0.5) for nf in captured])
            free_trust = np.mean([getattr(nf, 'trust_score', 0.5) for nf in free])
            metrics["trust_news_capture_gap"] = float(free_trust - cap_trust)
        else:
            metrics["trust_news_capture_gap"] = 0.0
    else:
        metrics["trust_news_mean"] = 0.0
        metrics["trust_news_capture_gap"] = 0.0

    # Landowner trust
    if model.landowners:
        lo_trusts = np.array([
            getattr(lo, 'trust_score', 0.5) for lo in model.landowners
        ])
        metrics["trust_landowner_mean"] = float(np.mean(lo_trusts))
    else:
        metrics["trust_landowner_mean"] = 0.0

    # Planner trust (the "approval rating")
    metrics["trust_planner"] = float(
        getattr(model.planner, 'trust_score', 0.5)
    )

    # System-wide trust: geometric mean of all institutional trust scores
    # This is the overall "institutional trust" of the economy
    institutional = []
    if active_firms:
        institutional.append(metrics["trust_firm_mean"])
    if banks:
        institutional.append(metrics["trust_bank_mean"])
    if news:
        institutional.append(metrics["trust_news_mean"])
    institutional.append(metrics["trust_planner"])
    if model.landowners:
        institutional.append(metrics["trust_landowner_mean"])

    if institutional:
        # Geometric mean: one failing institution drags everything down
        log_sum = sum(math.log(max(t, 0.01)) for t in institutional)
        metrics["trust_institutional"] = float(
            math.exp(log_sum / len(institutional))
        )
    else:
        metrics["trust_institutional"] = 0.0

    return metrics

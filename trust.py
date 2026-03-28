"""
Reputation-based trust system for the economic simulation.

Key properties of this architecture:

1. Trust accumulates slowly but collapses quickly.
2. Agents with longer histories have more credible reputations.
3. Catastrophic failures impose heavy penalties.
4. Signals are weighted by importance rather than averaged equally.
5. Trust decays slowly toward neutral if no information arrives.

Trust remains a *public reputation score* observable by all agents.
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


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

DEFAULT_TRUST_WORKER = 0.5
DEFAULT_TRUST_FIRM = 0.5
DEFAULT_TRUST_BANK = 0.6
DEFAULT_TRUST_NEWS = 0.5
DEFAULT_TRUST_PLANNER = 0.5
DEFAULT_TRUST_LANDOWNER = 0.5

TRUST_NEUTRAL = 0.5
TRUST_FLOOR = 0.02
TRUST_CEILING = 0.98

# asymmetric dynamics
TRUST_GROWTH_RATE = 0.02
TRUST_PENALTY_RATE = 0.15

# reputation maturity
CONFIDENCE_SCALE = 5

# slow drift toward neutral
TRUST_DECAY = 0.001


# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------

def init_trust(agent, agent_type: str):

    defaults = {
        "worker": DEFAULT_TRUST_WORKER,
        "firm": DEFAULT_TRUST_FIRM,
        "bank": DEFAULT_TRUST_BANK,
        "news": DEFAULT_TRUST_NEWS,
        "planner": DEFAULT_TRUST_PLANNER,
        "landowner": DEFAULT_TRUST_LANDOWNER,
    }

    agent.trust_score = defaults.get(agent_type, 0.5)
    agent.trust_observations = 0


# ---------------------------------------------------------
# Core update logic
# ---------------------------------------------------------

def _trust_update(agent, signal: float):

    trust = agent.trust_score

    if signal > trust:
        trust += TRUST_GROWTH_RATE * (signal - trust)
    else:
        trust += TRUST_PENALTY_RATE * (signal - trust)

    # confidence weighting
    confidence = agent.trust_observations
    weight = confidence / (confidence + CONFIDENCE_SCALE)

    trust = trust * weight + TRUST_NEUTRAL * (1 - weight)

    # small decay toward neutral
    trust += TRUST_DECAY * (TRUST_NEUTRAL - trust)

    trust = float(np.clip(trust, TRUST_FLOOR, TRUST_CEILING))

    agent.trust_score = trust
    agent.trust_observations += 1


# ---------------------------------------------------------
# Worker trust signal
# ---------------------------------------------------------

def _worker_trust_signal(worker: "WorkerAgent") -> float:

    loan_score = 0.5
    if worker.debt > 0:
        if worker.wealth > worker.debt * worker.debt_interest:
            loan_score = 0.7
        else:
            loan_score = 0.2

    employment_score = 0.7 if worker.employed else 0.3

    income_score = 0.3
    if worker.income_last_step > 0:
        income_score = min(0.8, 0.4 + worker.income_last_step / 20)

    # weights
    return (
        0.4 * loan_score +
        0.4 * employment_score +
        0.2 * income_score
    )


# ---------------------------------------------------------
# Firm trust signal
# ---------------------------------------------------------

def _firm_trust_signal(firm: "FirmAgent") -> float:

    wage_score = 0.3
    n_workers = len(firm.workers)

    if n_workers > 0:
        if firm.wealth > firm.offered_wage * n_workers:
            wage_score = 0.8
        else:
            wage_score = 0.2

    employment_score = 0.5
    if firm._last_strategy in ("hire", "raise_wages"):
        employment_score = 0.8
    elif firm._last_strategy == "downsize":
        employment_score = 0.2

    profit_score = 0.5
    if firm.profit > 0:
        profit_score = 0.7
    elif firm._consecutive_losses > 2:
        profit_score = 0.2

    cartel_score = 0.6 if firm.cartel_id is None else 0.1

    pollution_score = max(0.1, 1 - firm.pollution_factor * 1.5)

    return (
        0.35 * wage_score +
        0.2 * employment_score +
        0.15 * profit_score +
        0.2 * cartel_score +
        0.1 * pollution_score
    )


# ---------------------------------------------------------
# Bank trust signal
# ---------------------------------------------------------

def _bank_trust_signal(bank: "BankAgent") -> float:

    solvency_score = 0.3
    deposits = getattr(bank, "total_deposits", 0)

    if bank.wealth > deposits * 0.1:
        solvency_score = 0.8
    elif bank.wealth > 0:
        solvency_score = 0.4

    default_rate = getattr(bank, "default_rate", 0)

    credit_score = 0.5
    if default_rate < 0.05:
        credit_score = 0.8
    elif default_rate > 0.2:
        credit_score = 0.2

    profit_score = 0.7 if getattr(bank, "profit", 0) > 0 else 0.3

    return (
        0.4 * solvency_score +
        0.35 * credit_score +
        0.25 * profit_score
    )


# ---------------------------------------------------------
# News trust signal
# ---------------------------------------------------------

def _news_trust_signal(news: "NewsFirm") -> float:

    accuracy = news.accuracy

    capture_penalty = 0.2 if news.captured_by_cartel else 0.7

    financial = 0.3
    if news.wealth > 50:
        financial = 0.7
    elif news.wealth > 0:
        financial = 0.4

    return (
        0.6 * accuracy +
        0.25 * capture_penalty +
        0.15 * financial
    )


# ---------------------------------------------------------
# Landowner signal
# ---------------------------------------------------------

def _landowner_trust_signal(lo: "LandownerAgent") -> float:

    rent_score = max(0.1, 1 - lo.rent_rate * 2)

    territory = len(lo.controlled_cells)

    territory_score = 0.6
    if territory > 40:
        territory_score = 0.2
    elif territory > 20:
        territory_score = 0.4

    return 0.7 * rent_score + 0.3 * territory_score


# ---------------------------------------------------------
# Planner signal
# ---------------------------------------------------------

def _planner_trust_signal(model: "EconomicModel") -> float:

    history = model.metrics_history

    pop_ratio = len(model.workers) / max(model.n_workers_initial, 1)

    employment = sum(
        1 for w in model.workers if w.employed
    ) / max(len(model.workers), 1)

    gini = history[-1].get("all_gini", 0.5) if history else 0.5

    floor = min(w.wealth for w in model.workers) if model.workers else 0

    floor_score = min(1.0, floor / 20)

    inequality_score = max(0.1, 1 - gini)

    return (
        0.3 * pop_ratio +
        0.25 * employment +
        0.25 * floor_score +
        0.2 * inequality_score
    )


# ---------------------------------------------------------
# Catastrophic penalties
# ---------------------------------------------------------

def _apply_catastrophic_penalties(agent):

    # firm payroll failure
    if hasattr(agent, "workers") and hasattr(agent, "wealth"):
        if len(agent.workers) > 0 and agent.wealth <= 0:
            agent.trust_score *= 0.5

    # bank insolvency
    if hasattr(agent, "total_deposits"):
        if agent.wealth < 0:
            agent.trust_score *= 0.4


# ---------------------------------------------------------
# Main update loop
# ---------------------------------------------------------

def update_trust_scores(model: "EconomicModel"):

    for w in model.workers:
        if not hasattr(w, "trust_score"):
            init_trust(w, "worker")
        signal = _worker_trust_signal(w)
        _trust_update(w, signal)

    for f in model.firms:
        if getattr(f, "defunct", False):
            continue
        if not hasattr(f, "trust_score"):
            init_trust(f, "firm")
        signal = _firm_trust_signal(f)
        _trust_update(f, signal)
        _apply_catastrophic_penalties(f)

    for b in getattr(model, "banks", []):
        if getattr(b, "defunct", False):
            continue
        if not hasattr(b, "trust_score"):
            init_trust(b, "bank")
        signal = _bank_trust_signal(b)
        _trust_update(b, signal)
        _apply_catastrophic_penalties(b)

    for nf in getattr(model, "news_firms", []):
        if getattr(nf, "defunct", False):
            continue
        if not hasattr(nf, "trust_score"):
            init_trust(nf, "news")
        signal = _news_trust_signal(nf)
        _trust_update(nf, signal)

    for lo in model.landowners:
        if not hasattr(lo, "trust_score"):
            init_trust(lo, "landowner")
        signal = _landowner_trust_signal(lo)
        _trust_update(lo, signal)

    if not hasattr(model.planner, "trust_score"):
        init_trust(model.planner, "planner")

    signal = _planner_trust_signal(model)
    _trust_update(model.planner, signal)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def compute_trust_metrics(model: "EconomicModel") -> Dict[str, float]:

    metrics = {}

    workers = np.array([w.trust_score for w in model.workers]) if model.workers else np.array([0])

    metrics["trust_worker_mean"] = float(np.mean(workers))
    metrics["trust_worker_min"] = float(np.min(workers))

    firms = [f for f in model.firms if not f.defunct]

    if firms:
        firm_trust = np.array([f.trust_score for f in firms])
        metrics["trust_firm_mean"] = float(np.mean(firm_trust))
    else:
        metrics["trust_firm_mean"] = 0

    metrics["trust_planner"] = float(model.planner.trust_score)

    institutional = [
        metrics["trust_firm_mean"],
        metrics["trust_planner"]
    ]

    log_sum = sum(math.log(max(t, 0.01)) for t in institutional)
    metrics["trust_institutional"] = float(math.exp(log_sum / len(institutional)))

    return metrics
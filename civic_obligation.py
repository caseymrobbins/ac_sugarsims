"""
civic_obligation.py
-------------------
Structural accountability: wealth carries civic obligation.

The core insight: concentration isn't the problem. Concentration WITHOUT
accountability is the problem. Feudal lords had obligations to their serfs.
Modern wealth has detached ownership from obligation.

This module re-attaches obligation to wealth. Not as a tax that flows
through the planner (which mesa optimizers can game by capturing the
planner's policy), but as a DIRECT structural transfer from wealth
to public goods that bypasses all policy instruments.

The mechanism:
  - Every agent pays a progressive civic obligation each step
  - The rate scales with log(wealth), so it's negligible at subsistence
    and meaningful at high wealth
  - Payment flows DIRECTLY to infrastructure, education, healthcare
  - The planner cannot redirect, reduce, or block this flow
  - Mesa optimizers cannot game it because it's not a policy

The effect:
  - Firm accumulates wealth -> obligation funds public goods -> floor rises
  - Cartel concentrates wealth -> obligation increases -> public goods increase
  - Investor compounds returns -> obligation compounds -> infrastructure grows
  - The MORE concentration occurs, the MORE public goods are funded
  - Mesa optimizer self-interest IS the mechanism that raises the floor

This is topology shaping through mechanism design: don't fight the
gradient, attach public goods production to the gradient itself.
"""

from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from environment import EconomicModel


# ---------------------------------------------------------------------------
# Obligation rate schedule
# ---------------------------------------------------------------------------

# Threshold below which no obligation applies (subsistence protection)
OBLIGATION_THRESHOLD = 50.0

# Rate scaling: obligation_rate = RATE_SCALE * log(wealth / THRESHOLD)
# At wealth 500:    rate = 0.023  (2.3%)
# At wealth 5000:   rate = 0.046  (4.6%)
# At wealth 50000:  rate = 0.069  (6.9%)
# At wealth 500000: rate = 0.092  (9.2%)
RATE_SCALE = 0.01

# Maximum obligation rate (cap to prevent wealth destruction)
MAX_OBLIGATION_RATE = 0.12

# How obligation payments are allocated to public goods
# These bypass the planner entirely
INFRA_SHARE = 0.40      # 40% to infrastructure
EDUCATION_SHARE = 0.30  # 30% to education
HEALTHCARE_SHARE = 0.20 # 20% to healthcare
CLEANUP_SHARE = 0.10    # 10% to pollution cleanup

# Conversion factors: how much public good per unit of obligation payment
# These are tuned so that a wealthy economy with concentration
# produces meaningful public good improvements
INFRA_CONVERSION = 0.0002      # payment * this -> infrastructure_level delta
EDUCATION_CONVERSION = 0.0001  # payment * this -> education_quality delta
HEALTHCARE_CONVERSION = 0.0001 # payment * this -> healthcare_bonus delta
CLEANUP_CONVERSION = 0.00005   # payment * this -> pollution reduction fraction


def civic_obligation_rate(wealth: float) -> float:
    """
    Compute the civic obligation rate for a given wealth level.
    
    Progressive: zero below threshold, log-scaling above.
    Gentle enough that accumulation is still rewarded.
    Strong enough that concentration funds public goods.
    """
    if wealth <= OBLIGATION_THRESHOLD:
        return 0.0
    return min(MAX_OBLIGATION_RATE,
               RATE_SCALE * math.log(wealth / OBLIGATION_THRESHOLD))


def apply_civic_obligation(agent, model: "EconomicModel") -> float:
    """
    Apply civic obligation to an agent. Returns the payment amount.
    
    Call this in WorkerAgent.step() and FirmAgent.step() AFTER
    production and income, BEFORE tax application.
    
    The payment flows DIRECTLY to model public good variables,
    bypassing the planner's tax_revenue pool entirely.
    """
    wealth = getattr(agent, 'wealth', 0.0)
    if wealth <= OBLIGATION_THRESHOLD:
        return 0.0
    
    rate = civic_obligation_rate(wealth)
    payment = wealth * rate
    
    # Don't let obligation push agent below threshold
    if wealth - payment < OBLIGATION_THRESHOLD * 0.5:
        payment = max(0.0, wealth - OBLIGATION_THRESHOLD * 0.5)
    
    if payment <= 0:
        return 0.0
    
    # Deduct from agent
    agent.wealth -= payment
    
    # Distribute DIRECTLY to public goods (bypasses planner)
    infra_payment = payment * INFRA_SHARE
    edu_payment = payment * EDUCATION_SHARE
    health_payment = payment * HEALTHCARE_SHARE
    cleanup_payment = payment * CLEANUP_SHARE
    
    # Infrastructure: bounded growth
    model._infrastructure_level = min(
        3.0,  # cap
        model._infrastructure_level + infra_payment * INFRA_CONVERSION
    )
    
    # Education: bounded growth
    model._education_quality = min(
        3.0,  # cap
        model._education_quality + edu_payment * EDUCATION_CONVERSION
    )
    
    # Healthcare: bounded growth
    model._healthcare_bonus = min(
        0.50,  # cap (50% metabolism reduction max)
        model._healthcare_bonus + health_payment * HEALTHCARE_CONVERSION
    )
    
    # Pollution cleanup: reduce pollution grid proportionally
    if cleanup_payment > 0:
        cleanup_fraction = min(0.05, cleanup_payment * CLEANUP_CONVERSION)
        model.pollution_grid *= max(0.0, 1.0 - cleanup_fraction)
    
    return payment


# ---------------------------------------------------------------------------
# Integration instructions
# ---------------------------------------------------------------------------
"""
TO ADD TO agents.py:

1. Import at top:
   from civic_obligation import apply_civic_obligation

2. In WorkerAgent.step(), after self.model.planner.apply_tax(self):
   apply_civic_obligation(self, self.model)

3. In FirmAgent.step(), after self.model.planner.apply_tax(self):
   apply_civic_obligation(self, self.model)

4. In LandownerAgent.step(), after self.model.planner.apply_tax(self):
   apply_civic_obligation(self, self.model)

That's it. Three lines of integration. The obligation is structural,
automatic, and bypasses all planner policy instruments.

TO ADD TO metrics.py (optional, for tracking):

In collect_step_metrics(), add:
    from civic_obligation import civic_obligation_rate
    obligation_payments = sum(
        w.wealth * civic_obligation_rate(w.wealth)
        for w in model.workers
    )
    firm_obligations = sum(
        f.wealth * civic_obligation_rate(f.wealth)
        for f in model.firms if not f.defunct
    )
    m["total_civic_obligation"] = obligation_payments + firm_obligations
"""

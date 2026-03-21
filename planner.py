"""
planner.py
----------
The PlannerAgent acts as the policy authority.

It sets:
  - Tax rates (progressive or flat)
  - Redistribution policies (UBI, targeted transfers)
  - Resource regulation (harvest limits)
  - Labour rules (minimum wage floor)

The planner is optimised under one of three objective functions:
  SUM  : Maximise total wealth  → R = sum(wealth_i)
  NASH : Maximise Nash welfare  → R = sum(log(wealth_i + ε))
  JAM  : Maximise agency floor  → R = log(min(agency_i))

Policy parameters are adjusted via gradient-free hill-climbing every
POLICY_UPDATE_INTERVAL steps, ensuring the planner's interventions
emerge from the objective rather than being hard-coded.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Any

import numpy as np
from mesa import Agent

if TYPE_CHECKING:
    from environment import EconomicModel
    from agents import WorkerAgent, FirmAgent, LandownerAgent


POLICY_UPDATE_INTERVAL = 25   # steps between policy adjustments
EPSILON = 1e-6                 # log-safety


class PlannerAgent(Agent):
    """
    Policy authority agent.

    Policy instruments:
      tax_rate_worker    : fraction of income taxed from workers
      tax_rate_firm      : fraction of profit taxed from firms
      tax_rate_landowner : fraction of rent income taxed
      ubi_payment        : unconditional basic income per worker per step
      min_wage           : minimum wage floor (enforced on firms)
      harvest_limit      : maximum extraction per cell per step
    """

    def __init__(self, model: "EconomicModel"):
        super().__init__(model)  # Mesa 3.x: only model arg
        self.objective = model.objective

        # Policy instruments (start at neutral values)
        self.policy: Dict[str, float] = {
            "tax_rate_worker": 0.05,
            "tax_rate_firm": 0.10,
            "tax_rate_landowner": 0.08,
            "ubi_payment": 0.0,
            "min_wage": 1.0,
            "harvest_limit": 10.0,
        }

        # Revenue pool for redistribution
        self.tax_revenue: float = 0.0

        # Objective tracking
        self.last_objective_value: float = -math.inf
        self.objective_history: list = []

        # Perturbation for hill climbing
        self._perturbation_scale = 0.02
        self._steps_since_update = 0

    # ------------------------------------------------------------------
    # Mesa step
    # ------------------------------------------------------------------

    def step(self):
        self._steps_since_update += 1

        # Redistribute collected tax revenue
        self._redistribute()

        # Periodically update policy
        if self._steps_since_update >= POLICY_UPDATE_INTERVAL:
            self._update_policy()
            self._steps_since_update = 0

    # ------------------------------------------------------------------
    # Tax application (called by each agent in its own step)
    # ------------------------------------------------------------------

    def apply_tax(self, agent):
        """Deduct taxes from agent and add to revenue pool."""
        from agents import WorkerAgent, FirmAgent, LandownerAgent

        if isinstance(agent, WorkerAgent):
            rate = self.policy["tax_rate_worker"]
            taxable = max(0, agent.income_last_step)
            tax = taxable * rate
            # Enforce min wage: if employed, ensure wage >= floor
            if agent.employed:
                floor = self.policy["min_wage"]
                if agent.wage < floor:
                    shortfall = floor - agent.wage
                    agent.wealth += shortfall  # top up from planner reserve
                    self.tax_revenue -= shortfall

        elif isinstance(agent, FirmAgent):
            rate = self.policy["tax_rate_firm"]
            taxable = max(0, agent.profit)
            tax = taxable * rate
            # Enforce minimum wage on firm's offer
            if agent.offered_wage < self.policy["min_wage"]:
                agent.offered_wage = self.policy["min_wage"]

        elif isinstance(agent, LandownerAgent):
            rate = self.policy["tax_rate_landowner"]
            taxable = max(0, agent.total_rent_collected * 0.01)  # flow proxy
            tax = taxable * rate

        else:
            return

        if agent.wealth >= tax:
            agent.wealth -= tax
            self.tax_revenue += tax

    # ------------------------------------------------------------------
    # Redistribution
    # ------------------------------------------------------------------

    def _redistribute(self):
        """Distribute UBI and targeted transfers."""
        workers = self.model.workers
        if not workers:
            return

        ubi = self.policy["ubi_payment"]
        total_ubi = ubi * len(workers)

        if total_ubi <= 0:
            return

        if self.tax_revenue >= total_ubi:
            for w in workers:
                w.wealth += ubi
            self.tax_revenue -= total_ubi
        else:
            # Partial distribution
            per_worker = self.tax_revenue / len(workers)
            for w in workers:
                w.wealth += per_worker
            self.tax_revenue = 0.0

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------

    def compute_objective(self) -> float:
        """Compute the current objective value."""
        if self.objective == "SUM":
            return self._objective_sum()
        elif self.objective == "NASH":
            return self._objective_nash()
        elif self.objective == "JAM":
            return self._objective_jam()
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def _objective_sum(self) -> float:
        """R = sum(wealth_i) — utilitarian aggregate."""
        return float(np.sum(self.model.get_all_agent_wealths()))

    def _objective_nash(self) -> float:
        """R = sum(log(wealth_i + ε)) — Nash social welfare."""
        wealths = self.model.get_all_agent_wealths()
        return float(np.sum(np.log(np.maximum(wealths, EPSILON))))

    def _objective_jam(self) -> float:
        """R = log(min(agency_i)) — maximise the agency floor."""
        workers = self.model.workers
        if not workers:
            return -math.inf
        agencies = [w.compute_agency() for w in workers]
        floor = min(agencies)
        return math.log(max(floor, EPSILON))

    # ------------------------------------------------------------------
    # Policy update (hill climbing)
    # ------------------------------------------------------------------

    def _update_policy(self):
        """
        Gradient-free hill climbing:
        1. Evaluate current objective.
        2. Perturb one instrument.
        3. Keep perturbation if it improves objective.
        """
        current_obj = self.compute_objective()
        self.objective_history.append(current_obj)

        # Pick instrument to perturb
        instruments = list(self.policy.keys())
        key = self.model.rng.choice(instruments)
        old_val = self.policy[key]

        # Perturb
        delta = self.model.rng.normal(0, self._perturbation_scale * abs(old_val + 0.01))
        new_val = old_val + delta
        new_val = self._clip_policy(key, new_val)
        self.policy[key] = new_val

        # Re-evaluate (approximate: compare against last)
        new_obj = self.compute_objective()

        if new_obj >= current_obj:
            # Accept
            self.last_objective_value = new_obj
        else:
            # Reject — revert
            self.policy[key] = old_val
            self.last_objective_value = current_obj

    def _clip_policy(self, key: str, val: float) -> float:
        """Enforce valid ranges for each policy instrument."""
        limits = {
            "tax_rate_worker": (0.0, 0.80),
            "tax_rate_firm": (0.0, 0.90),
            "tax_rate_landowner": (0.0, 0.95),
            "ubi_payment": (0.0, 10.0),
            "min_wage": (0.0, 15.0),
            "harvest_limit": (1.0, 20.0),
        }
        lo, hi = limits.get(key, (0.0, 1.0))
        return float(np.clip(val, lo, hi))

    # ------------------------------------------------------------------
    # Accessors for metrics
    # ------------------------------------------------------------------

    def get_policy_snapshot(self) -> Dict[str, Any]:
        snap = dict(self.policy)
        snap["objective_value"] = self.last_objective_value
        snap["tax_revenue_pool"] = self.tax_revenue
        return snap

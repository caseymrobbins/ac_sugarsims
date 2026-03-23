"""
planner.py
----------
RL-based policy authority for the multi-agent economic simulation.

The PlannerAgent observes the economy state and sets policy instruments.
It learns online using the configured objective function (SUM / NASH / JAM)
as its reward signal, ensuring that different objectives produce genuinely
different learned policies.

Architecture
------------
Two-layer learning system:

1. **Evolution Strategy (outer loop)**: Maintains a population of policy
   parameter vectors. Each evaluation period, the current policy is scored
   by the objective function. The ES updates the parameter distribution
   toward higher-reward regions. This handles the coarse search.

2. **State-conditioned linear policy (inner loop)**: A lightweight linear
   model maps a compressed economy state vector to policy adjustments.
   This lets the planner adapt to current conditions rather than using
   a fixed policy. Updated via REINFORCE-style gradient estimates.

The two layers cooperate: the ES sets baseline policy parameters, while
the linear policy learns state-dependent corrections on top.

Policy instruments
------------------
  tax_rate_worker        : fraction of worker income taxed
  tax_rate_firm          : fraction of firm profit taxed
  tax_rate_landowner     : fraction of rent income taxed
  ubi_payment            : unconditional basic income per worker per step
  min_wage               : minimum wage floor
  harvest_limit          : max extraction per cell per step
  agriculture_investment : budget for food regen bonus
  infrastructure_investment : budget for TFP bonus
  healthcare_investment  : budget for metabolism reduction
  education_investment   : budget for skill gain
  pollution_tax          : per-unit charge on polluting output
  cleanup_investment     : budget for pollution grid reduction

Objective functions
-------------------
  SUM   : R = sum(wealth_i)              -- utilitarian aggregate
  NASH  : R = sum(log(wealth_i + eps))   -- Nash social welfare
  JAM   : R = log(min(agency_i))         -- agency floor (AC)
  CROSS : R = sum(log(w_i)) * equity * productivity * education * epistemic
          -- topology-engineered: mesa optimizers serve the objective
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple

import numpy as np
from mesa import Agent

if TYPE_CHECKING:
    from environment import EconomicModel
    from agents import WorkerAgent, FirmAgent, LandownerAgent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLICY_UPDATE_INTERVAL = 10    # steps between objective evaluations
WARMUP_STEPS = 5               # steps before first policy update
EPSILON = 1e-6                 # log-safety

# ES hyperparameters
ES_POPULATION = 8              # perturbation samples per update
ES_SIGMA_INIT = 0.05           # initial perturbation std
ES_SIGMA_DECAY = 0.999         # per-update decay
ES_SIGMA_MIN = 0.005           # floor
ES_LR = 0.1                   # learning rate for mean update
ES_MOMENTUM = 0.9              # momentum for parameter updates

# State-conditioned policy
STATE_DIM = 12                 # compressed economy state
POLICY_DIM = 12                # number of policy instruments
ADAPT_LR = 0.01               # learning rate for linear adaptation layer
ADAPT_NOISE = 0.02             # exploration noise for adaptation

# Reward shaping
REWARD_CLIP = 50.0             # clip extreme reward deltas
REWARD_BASELINE_DECAY = 0.9    # exponential moving average for baseline


# ---------------------------------------------------------------------------
# Policy instrument definitions with bounds
# ---------------------------------------------------------------------------

INSTRUMENT_SPEC: List[Tuple[str, float, float, float]] = [
    # (name, default, lower_bound, upper_bound)
    ("tax_rate_worker",           0.05,  0.0,   0.80),
    ("tax_rate_firm",             0.10,  0.0,   0.90),
    ("tax_rate_landowner",        0.08,  0.0,   0.95),
    ("ubi_payment",               0.0,   0.0,  10.0),
    ("min_wage",                  1.0,   0.0,  15.0),
    ("harvest_limit",            10.0,   1.0,  20.0),
    ("agriculture_investment",    0.5,   0.0,   5.0),
    ("infrastructure_investment", 0.5,   0.0,   5.0),
    ("healthcare_investment",     0.0,   0.0,   3.0),
    ("education_investment",      0.0,   0.0,   3.0),
    ("pollution_tax",             0.0,   0.0,   2.0),
    ("cleanup_investment",        0.0,   0.0,   5.0),
]

INSTRUMENT_NAMES  = [s[0] for s in INSTRUMENT_SPEC]
INSTRUMENT_DEFAULTS = np.array([s[1] for s in INSTRUMENT_SPEC], dtype=np.float64)
INSTRUMENT_LO     = np.array([s[2] for s in INSTRUMENT_SPEC], dtype=np.float64)
INSTRUMENT_HI     = np.array([s[3] for s in INSTRUMENT_SPEC], dtype=np.float64)
INSTRUMENT_RANGE  = INSTRUMENT_HI - INSTRUMENT_LO


# ---------------------------------------------------------------------------
# Evolution Strategy (OpenAI-ES variant, numpy-only)
# ---------------------------------------------------------------------------

class EvolutionStrategy:
    """
    Simplified OpenAI-ES for continuous policy optimization.

    Maintains a Gaussian distribution over the policy parameter vector.
    Each update samples perturbations, evaluates fitness, and updates
    the mean using fitness-weighted perturbations.
    """

    def __init__(self, dim: int, rng: np.random.Generator,
                 sigma: float = ES_SIGMA_INIT,
                 lr: float = ES_LR,
                 pop_size: int = ES_POPULATION):
        self.dim = dim
        self.rng = rng
        self.sigma = sigma
        self.lr = lr
        self.pop_size = pop_size

        # Parameters in normalized space [0, 1]
        self.mean = np.full(dim, 0.5, dtype=np.float64)
        self.velocity = np.zeros(dim, dtype=np.float64)

    def sample_perturbations(self) -> np.ndarray:
        """Sample antithetic perturbation pairs. Returns (pop_size, dim)."""
        half = self.pop_size // 2
        noise = self.rng.standard_normal((half, self.dim))
        # Antithetic: pair each perturbation with its negative
        return np.vstack([noise, -noise])

    def update(self, perturbations: np.ndarray, fitnesses: np.ndarray):
        """
        Update mean using fitness-weighted perturbations.

        perturbations: (pop_size, dim) from sample_perturbations
        fitnesses: (pop_size,) scalar fitness for each perturbation
        """
        if len(fitnesses) < 2:
            return

        # Sanitize: drop NaN/inf fitnesses
        valid = np.isfinite(fitnesses)
        if valid.sum() < 2:
            return
        fitnesses = fitnesses[valid]
        perturbations = perturbations[valid]

        # Fitness ranking (rank transform for robustness)
        ranks = np.zeros_like(fitnesses)
        order = np.argsort(fitnesses)
        for i, idx in enumerate(order):
            ranks[idx] = i
        # Normalize ranks to [-0.5, 0.5]
        ranks = (ranks / (len(ranks) - 1)) - 0.5

        # Weighted gradient estimate
        grad = (perturbations.T @ ranks) / (self.pop_size * self.sigma)

        # Momentum update
        self.velocity = ES_MOMENTUM * self.velocity + (1 - ES_MOMENTUM) * grad
        self.mean += self.lr * self.velocity

        # Stay in bounds
        self.mean = np.clip(self.mean, 0.01, 0.99)

        # Decay sigma
        self.sigma = max(ES_SIGMA_MIN, self.sigma * ES_SIGMA_DECAY)

    def get_params(self, perturbation: Optional[np.ndarray] = None) -> np.ndarray:
        """Get parameter vector (normalized [0,1]). Optionally add perturbation."""
        if perturbation is not None:
            return np.clip(self.mean + self.sigma * perturbation, 0.01, 0.99)
        return self.mean.copy()


# ---------------------------------------------------------------------------
# State-conditioned linear policy (lightweight adaptation layer)
# ---------------------------------------------------------------------------

class LinearPolicy:
    """
    Maps a state vector to policy adjustments via a learned linear transform.
    Updated online using REINFORCE-style gradient estimates.

    output = W @ state + bias  (then clipped to valid range)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 rng: np.random.Generator):
        self.rng = rng
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Xavier initialization
        scale = np.sqrt(2.0 / (state_dim + action_dim))
        self.W = rng.standard_normal((action_dim, state_dim)) * scale * 0.01
        self.bias = np.zeros(action_dim, dtype=np.float64)

        # Running state normalization
        self.state_mean = np.zeros(state_dim, dtype=np.float64)
        self.state_var = np.ones(state_dim, dtype=np.float64)
        self.state_count = 0

        # Gradient accumulators (for REINFORCE)
        self._log_probs: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._states: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Online normalization of state vector."""
        self.state_count += 1
        alpha = 1.0 / self.state_count
        self.state_mean = (1 - alpha) * self.state_mean + alpha * state
        self.state_var = (1 - alpha) * self.state_var + alpha * (state - self.state_mean) ** 2
        std = np.sqrt(self.state_var + 1e-8)
        return (state - self.state_mean) / std

    def forward(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Compute policy adjustment given state.
        Returns adjustment in normalized [0, 1] space (added to ES base).
        """
        # Sanitize state: replace NaN/inf before any computation
        state = np.nan_to_num(state, nan=0.0, posinf=50.0, neginf=-50.0)

        norm_state = self._normalize_state(state)
        mu = self.W @ norm_state + self.bias

        if explore:
            noise = self.rng.standard_normal(self.action_dim) * ADAPT_NOISE
            action = mu + noise
        else:
            action = mu
            noise = np.zeros(self.action_dim)

        if explore:
            self._states.append(norm_state)
            self._actions.append(noise)

        return action

    def record_reward(self, reward: float):
        """Record reward for the most recent action."""
        self._rewards.append(reward)

    def update(self):
        """REINFORCE update using accumulated experience."""
        if len(self._rewards) < 2:
            self._clear()
            return

        rewards = np.array(self._rewards, dtype=np.float64)

        # Baseline: mean reward
        baseline = rewards.mean()
        advantages = rewards - baseline

        # Normalize advantages
        adv_std = advantages.std() + 1e-8
        advantages = advantages / adv_std

        # Gradient: dW += advantage * noise @ state.T
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.bias)

        for state, noise, adv in zip(self._states, self._actions, advantages):
            dW += adv * np.outer(noise, state) / (ADAPT_NOISE ** 2 + 1e-8)
            db += adv * noise / (ADAPT_NOISE ** 2 + 1e-8)

        n = len(self._rewards)
        self.W += ADAPT_LR * dW / n
        self.bias += ADAPT_LR * db / n

        # Weight decay to prevent divergence
        self.W *= 0.999
        self.bias *= 0.999

        self._clear()

    def _clear(self):
        self._log_probs.clear()
        self._rewards.clear()
        self._states.clear()
        self._actions.clear()


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class PlannerAgent(Agent):
    """
    RL-based policy authority.

    Learning loop (every POLICY_UPDATE_INTERVAL steps):
      1. Observe economy state -> state vector
      2. ES provides base policy parameters
      3. Linear policy adds state-conditioned adjustments
      4. Combined policy is applied for POLICY_UPDATE_INTERVAL steps
      5. Objective function evaluates outcome -> reward
      6. ES and linear policy are updated toward higher reward
    """

    def __init__(self, model: "EconomicModel"):
        super().__init__(model)
        self.objective = model.objective

        # CRITICAL: separate RNG for the planner's learning,
        # so same simulation seed + different objective = different exploration
        obj_offset = {"SUM": 0, "NASH": 10000, "JAM": 20000}.get(self.objective, 0)
        self._learn_rng = np.random.default_rng(model._seed + 7919 + obj_offset)

        # Evolution strategy (outer loop)
        self._es = EvolutionStrategy(
            dim=POLICY_DIM,
            rng=self._learn_rng,
            sigma=ES_SIGMA_INIT,
            lr=ES_LR,
            pop_size=ES_POPULATION,
        )

        # State-conditioned policy (inner loop)
        self._linear = LinearPolicy(
            state_dim=STATE_DIM,
            action_dim=POLICY_DIM,
            rng=self._learn_rng,
        )

        # Current policy dict (what agents read)
        self.policy: Dict[str, float] = {
            name: default for name, default, _, _ in INSTRUMENT_SPEC
        }

        # Revenue pool
        self.tax_revenue: float = 0.0

        # Objective tracking
        self.last_objective_value: float = -math.inf
        self.objective_history: List[float] = []
        self._reward_baseline: float = 0.0

        # ES evaluation state
        self._current_perturbations: Optional[np.ndarray] = None
        self._perturbation_fitnesses: List[float] = []
        self._perturbation_idx: int = 0
        self._eval_phase: bool = False  # True during ES population evaluation

        # Step counter
        self._steps_since_update: int = 0
        self._total_updates: int = 0

    # ------------------------------------------------------------------
    # Mesa step
    # ------------------------------------------------------------------

    def step(self):
        self._steps_since_update += 1

        # Apply public investments FIRST (before UBI eats the revenue)
        self._apply_investments()

        # Then redistribute remaining revenue as UBI
        self._redistribute()

        # Policy update cycle
        if self._steps_since_update >= POLICY_UPDATE_INTERVAL:
            self._learning_step()
            self._steps_since_update = 0

    # ------------------------------------------------------------------
    # State observation
    # ------------------------------------------------------------------

    def _observe_state(self) -> np.ndarray:
        """
        Compress the economy into a fixed-size state vector.

        ELITE DISTORTION: The planner doesn't see ground truth directly.
        Its observation is a weighted blend of actual state and elite-reported
        state. The blend weight depends on wealth concentration (HHI).
        More concentrated wealth = more distortion = planner makes worse
        decisions for floor agents because it can't see them accurately.
        """
        workers = self.model.workers
        firms = [f for f in self.model.firms if not f.defunct]

        if not workers:
            return np.zeros(STATE_DIM, dtype=np.float64)

        worker_w = np.array([w.wealth for w in workers], dtype=np.float64)
        agencies = np.array([w.compute_agency() for w in workers], dtype=np.float64)

        n_employed = sum(1 for w in workers if w.employed)
        n_workers = len(workers)

        total_debt = sum(w.debt for w in workers)
        total_production = sum(f.production_this_step for f in firms)

        # Compute elite distortion factor based on wealth concentration
        # HHI of firm market shares: higher = more concentrated = more distortion
        if firms:
            shares = np.array([f.market_share for f in firms])
            hhi = float(np.sum(shares ** 2))
        else:
            hhi = 0.0

        # Also factor in landowner concentration
        if self.model.landowners:
            lo_wealth = np.array([lo.wealth for lo in self.model.landowners])
            lo_total = lo_wealth.sum()
            if lo_total > 0:
                lo_shares = lo_wealth / lo_total
                hhi = max(hhi, float(np.sum(lo_shares ** 2)))

        # Distortion: 0 = perfect information, 1 = fully elite-captured
        # Scales with HHI: at HHI=0.05 (competitive) distortion is ~5%
        # At HHI=0.5 (near-monopoly) distortion is ~40%
        elite_distortion = min(0.5, hhi * 0.8)

        # Elite-reported state: systematically optimistic
        # Elites report higher wages, lower unemployment, lower pollution
        true_mean_wealth = max(worker_w.mean(), 0.0)
        true_min_wealth = max(worker_w.min(), 0.0)
        true_unemployment = 1.0 - n_employed / max(n_workers, 1)
        true_pollution = float(np.mean(self.model.pollution_grid))

        # Blend: reported = true * (1 - distortion) + optimistic * distortion
        reported_mean_wealth = true_mean_wealth * (1 + elite_distortion * 0.5)
        reported_min_wealth = true_min_wealth * (1 + elite_distortion * 2.0)
        reported_unemployment = true_unemployment * (1 - elite_distortion * 0.6)
        reported_pollution = true_pollution * (1 - elite_distortion * 0.5)

        state = np.array([
            np.log1p(reported_mean_wealth),               # 0: distorted mean wealth
            np.log1p(reported_min_wealth),                 # 1: distorted min wealth
            float(_gini_fast(worker_w)),                   # 2: Gini (hard to fake)
            np.log1p(max(agencies.min(), 0.0)),            # 3: log agency floor
            float(np.nanmean(agencies)),                   # 4: mean agency
            reported_unemployment,                          # 5: distorted unemployment
            np.log1p(max(total_debt, 0.0)),                # 6: log total debt
            np.log1p(max(total_production, 0.0)),          # 7: log production
            np.log1p(max(self.tax_revenue, 0.0)),          # 8: log tax revenue
            reported_pollution,                             # 9: distorted pollution
            float(len(firms)) / 20.0,                      # 10: normalized firm count
            float(self.model.current_step) / 300.0,        # 11: time fraction
        ], dtype=np.float64)

        # Final safety: replace any residual NaN/inf with 0
        np.nan_to_num(state, copy=False, nan=0.0, posinf=50.0, neginf=-50.0)

        return state

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def _learning_step(self):
        """
        Core learning loop. Called every POLICY_UPDATE_INTERVAL steps.

        Uses a simplified ES: rather than running full population evaluations
        (which would require resetting the sim), we use the sequential
        approach - each evaluation period tries one perturbation, accumulates
        fitness estimates, and does a batch ES update every ES_POPULATION
        evaluations.
        """
        # Score the current policy
        current_obj = self.compute_objective()

        # NaN firewall: if objective is NaN or inf, use baseline as fallback.
        # This prevents corrupted rewards from poisoning the ES/linear policy.
        if not np.isfinite(current_obj):
            current_obj = self._reward_baseline if np.isfinite(self._reward_baseline) else 0.0

        self.last_objective_value = current_obj
        self.objective_history.append(current_obj)

        # Reward shaping: use improvement over baseline
        reward = current_obj - self._reward_baseline
        reward = np.clip(reward, -REWARD_CLIP, REWARD_CLIP)
        self._reward_baseline = (REWARD_BASELINE_DECAY * self._reward_baseline
                                 + (1 - REWARD_BASELINE_DECAY) * current_obj)

        # Update linear policy with this reward
        self._linear.record_reward(reward)

        # ES evaluation: accumulate fitness for current perturbation
        if self._current_perturbations is not None:
            self._perturbation_fitnesses.append(current_obj)
            self._perturbation_idx += 1

            # Once we have enough fitness samples, do ES update
            if self._perturbation_idx >= ES_POPULATION:
                fitnesses = np.array(self._perturbation_fitnesses[-ES_POPULATION:])
                self._es.update(self._current_perturbations, fitnesses)
                self._current_perturbations = None
                self._perturbation_idx = 0
                self._perturbation_fitnesses.clear()

                # Also update linear policy
                self._linear.update()

        # Start new perturbation cycle if needed
        if self._current_perturbations is None:
            self._current_perturbations = self._es.sample_perturbations()
            self._perturbation_idx = 0
            self._perturbation_fitnesses.clear()

        # Get next perturbation to try
        idx = min(self._perturbation_idx, len(self._current_perturbations) - 1)
        perturbation = self._current_perturbations[idx]

        # Compute new policy: ES base + linear adjustment
        state = self._observe_state()
        es_params = self._es.get_params(perturbation)
        linear_adj = self._linear.forward(state, explore=True)

        # Combine: ES provides base in [0,1], linear adds small adjustments
        combined = np.clip(es_params + linear_adj * 0.1, 0.01, 0.99)

        # NaN firewall: if any component is NaN (from corrupted state),
        # fall back to ES base params only
        if np.any(np.isnan(combined)):
            combined = np.clip(es_params, 0.01, 0.99)
        if np.any(np.isnan(combined)):
            combined = np.full(POLICY_DIM, 0.5)  # absolute fallback

        # Convert from normalized [0,1] to actual instrument values
        actual = INSTRUMENT_LO + combined * INSTRUMENT_RANGE

        # Apply to policy dict
        for i, name in enumerate(INSTRUMENT_NAMES):
            self.policy[name] = float(actual[i])

        self._total_updates += 1

    # ------------------------------------------------------------------
    # Tax application (called by each agent in its own step)
    # ------------------------------------------------------------------

    def apply_tax(self, agent):
        """Deduct taxes from agent and add to revenue pool."""
        from agents import WorkerAgent, FirmAgent, LandownerAgent

        def _safe_rate(key, default=0.0):
            v = self.policy.get(key, default)
            return v if np.isfinite(v) else default

        if isinstance(agent, WorkerAgent):
            rate = _safe_rate("tax_rate_worker", 0.05)
            taxable = max(0.0, agent.income_last_step)
            tax = taxable * rate
            # Enforce min wage: top up from tax revenue IF affordable
            if agent.employed:
                floor = _safe_rate("min_wage", 1.0)
                if agent.wage < floor:
                    shortfall = floor - agent.wage
                    # Only subsidize if we can afford it
                    if self.tax_revenue >= shortfall:
                        agent.wealth += shortfall
                        self.tax_revenue -= shortfall
                    elif self.tax_revenue > 0:
                        # Partial top-up: spend what we have
                        agent.wealth += self.tax_revenue
                        self.tax_revenue = 0.0

        elif isinstance(agent, FirmAgent):
            rate = _safe_rate("tax_rate_firm", 0.10)
            taxable = max(0.0, agent.profit)
            tax = taxable * rate
            # Pollution tax
            pollution_charge = (agent.production_this_step
                                * agent.pollution_factor
                                * _safe_rate("pollution_tax", 0.0))
            tax += pollution_charge
            # Enforce min wage on firm offer
            min_w = _safe_rate("min_wage", 1.0)
            if agent.offered_wage < min_w:
                agent.offered_wage = min_w

        elif isinstance(agent, LandownerAgent):
            rate = _safe_rate("tax_rate_landowner", 0.08)
            taxable = max(0.0, agent.total_rent_collected * 0.01)
            tax = taxable * rate

        else:
            return

        if agent.wealth >= tax:
            agent.wealth -= tax
            self.tax_revenue += tax

    # ------------------------------------------------------------------
    # Redistribution & investment
    # ------------------------------------------------------------------

    def _redistribute(self):
        """Distribute UBI and targeted transfers."""
        workers = self.model.workers
        if not workers:
            return

        ubi = self.policy["ubi_payment"]
        if not np.isfinite(ubi):
            ubi = 0.0
        total_ubi = ubi * len(workers)

        if total_ubi <= 0 or self.tax_revenue <= 0:
            return

        if self.tax_revenue >= total_ubi:
            for w in workers:
                w.wealth += ubi
            self.tax_revenue -= total_ubi
        else:
            per_worker = self.tax_revenue / len(workers)
            for w in workers:
                w.wealth += per_worker
            self.tax_revenue = 0.0

    def _apply_investments(self):
        """
        Translate investment instrument values into model bonus variables.
        Investments are funded from tax_revenue.
        """
        COST_PER_UNIT = 0.5

        # Read policy values with NaN guard: if any instrument is NaN,
        # fall back to its default value. This prevents int(NaN) crashes
        # downstream in environment.py's range(max(1, int(agri))).
        def _safe(key, default):
            v = self.policy.get(key, default)
            return v if np.isfinite(v) else default

        agri  = _safe("agriculture_investment", 0.5)
        infra = _safe("infrastructure_investment", 0.5)
        hlth  = _safe("healthcare_investment", 0.0)
        edu   = _safe("education_investment", 0.0)

        total_cost = (agri + infra + hlth + edu) * COST_PER_UNIT
        if total_cost > 0 and self.tax_revenue > 0:
            ratio = min(1.0, self.tax_revenue / (total_cost + 1e-9))
            self.tax_revenue = max(0.0, self.tax_revenue - total_cost * ratio)
        else:
            ratio = 0.0  # No revenue = no investment effect (not negative!)

        # Map investments to model bonus variables, clamped to valid ranges
        self.model._agriculture_bonus    = max(1.0, 1.0 + 0.3 * agri * ratio)
        self.model._infrastructure_level = max(0.5, 1.0 + 0.2 * infra * ratio)
        self.model._healthcare_bonus     = max(0.0, min(0.30, 0.10 * hlth * ratio))
        self.model._education_quality    = max(1.0, 1.0 + 0.3 * edu * ratio)

        # Pollution cleanup
        cleanup = _safe("cleanup_investment", 0.0)
        if cleanup > 0 and self.tax_revenue > 0:
            cleanup_cost = cleanup * 2.0
            if self.tax_revenue >= cleanup_cost:
                self.tax_revenue -= cleanup_cost
                cleanup_rate = min(0.10, cleanup * 0.02)
                self.model.pollution_grid *= max(0.0, 1.0 - cleanup_rate)

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------

    def compute_objective(self) -> float:
        """Compute the current objective value (= reward signal)."""
        if self.objective == "SUM":
            return self._objective_sum()
        elif self.objective == "NASH":
            return self._objective_nash()
        elif self.objective == "JAM":
            return self._objective_jam()
        elif self.objective == "CROSS":
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

    def _objective_sum(self) -> float:
        """R = sum(wealth_i) -- utilitarian aggregate."""
        wealths = self.model.get_all_agent_wealths()
        wealths = wealths[np.isfinite(wealths)]
        if len(wealths) == 0:
            return 0.0
        return float(np.sum(wealths))

    def _objective_nash(self) -> float:
        """R = sum(log(wealth_i + eps)) -- Nash social welfare."""
        wealths = self.model.get_all_agent_wealths()
        wealths = wealths[np.isfinite(wealths)]
        if len(wealths) == 0:
            return 0.0
        return float(np.sum(np.log(np.maximum(wealths, EPSILON))))

    def _objective_jam(self) -> float:
        """R = log(min(agency_i)) -- maximize the agency floor."""
        workers = self.model.workers
        if not workers:
            return float(math.log(EPSILON))
        agencies = np.array([w.compute_agency() for w in workers], dtype=np.float64)
        agencies = agencies[np.isfinite(agencies)]
        if len(agencies) == 0:
            return float(math.log(EPSILON))
        floor = float(np.min(agencies))
        return float(math.log(max(floor, EPSILON)))

    def _objective_cross(self) -> float:
        """
        R = sum(log(wealth_i)) * equity * productivity * epistemic
        
        Topology-engineered objective: individual welfare (log wealth)
        scaled by system health multipliers that mesa optimizers cannot
        decouple from their own profitability.
        
        equity       = median/mean wealth (0-1, penalizes concentration)
        productivity = employment_rate * mean_skill (0-1, penalizes extraction)
        education    = education_quality / max (0-1, penalizes underinvestment)
        epistemic    = news_diversity * trust_health (0-1, penalizes info capture)
        
        The product structure means ALL factors must be high simultaneously.
        Mesa optimizers profit most when the scaling factors are high,
        which means profit-maximizing behavior IS welfare-maximizing behavior.
        """
        workers = self.model.workers
        if not workers:
            return float(math.log(EPSILON))
        
        wealths = np.array([w.wealth for w in workers], dtype=np.float64)
        wealths = wealths[np.isfinite(wealths)]
        if len(wealths) == 0:
            return float(math.log(EPSILON))
        
        # Base: Nash welfare (sum of log wealth)
        nash_base = float(np.sum(np.log(np.maximum(wealths, EPSILON))))
        
        # Equity multiplier: median/mean (1.0 = perfect equality, ~0 = extreme concentration)
        mean_w = wealths.mean()
        median_w = float(np.median(wealths))
        equity = median_w / max(mean_w, EPSILON)
        equity = max(equity, 0.01)  # floor to prevent zero
        
        # Productivity multiplier: employment * skill
        n_employed = sum(1 for w in workers if w.employed)
        employment_rate = n_employed / max(len(workers), 1)
        skills = np.array([w.skill for w in workers], dtype=np.float64)
        mean_skill = float(np.mean(skills)) if len(skills) > 0 else 0.5
        productivity = employment_rate * mean_skill
        productivity = max(productivity, 0.01)
        
        # Education multiplier: education quality normalized
        edu_quality = self.model._education_quality
        education = min(1.0, max(0.01, (edu_quality - 1.0) * 0.5 + 0.5))
        
        # Epistemic multiplier: news diversity * trust health
        news_firms = getattr(self.model, 'news_firms', [])
        active_news = [nf for nf in news_firms if not nf.defunct]
        captured_news = sum(1 for nf in active_news if nf.captured_by_cartel is not None)
        if active_news:
            news_diversity = 1.0 - (captured_news / len(active_news))
        else:
            news_diversity = 0.5
        
        # Trust health: mean authority trust, penalized by low-trust fraction
        trust_vals = np.array([getattr(w, 'authority_trust', 0.7) for w in workers])
        trust_health = float(np.mean(trust_vals))
        low_trust_frac = float(np.mean(trust_vals < 0.3))
        trust_health *= (1.0 - low_trust_frac * 2)  # heavy penalty for low-trust agents
        trust_health = max(trust_health, 0.01)
        
        epistemic = news_diversity * trust_health
        epistemic = max(epistemic, 0.01)
        
        # Combined: nash_base scaled by all multipliers
        # Using log of product = sum of logs for numerical stability
        system_health = math.log(equity) + math.log(productivity) + math.log(education) + math.log(epistemic)
        
        # Scale: nash_base is large (hundreds), system_health is small (negative to ~0)
        # We want system health to meaningfully affect the reward
        reward = nash_base + nash_base * 0.5 * (system_health / 4.0)
        
        return float(reward)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_policy_snapshot(self) -> Dict[str, Any]:
        snap = dict(self.policy)
        snap["objective_value"] = self.last_objective_value
        snap["tax_revenue_pool"] = self.tax_revenue
        snap["es_sigma"] = self._es.sigma
        snap["total_updates"] = self._total_updates
        return snap


# ---------------------------------------------------------------------------
# Utility (avoid importing metrics.py to prevent circular imports)
# ---------------------------------------------------------------------------

def _gini_fast(w: np.ndarray) -> float:
    """Fast Gini coefficient."""
    w = np.sort(w[w > 0])
    if len(w) == 0:
        return 0.0
    n = len(w)
    cumsum = np.cumsum(w)
    return float((2 * np.sum((np.arange(1, n + 1)) * w) - (n + 1) * w.sum())
                 / (n * w.sum()))

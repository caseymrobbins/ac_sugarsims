"""
planner.py - RL policy authority with ideal baseline, inheritance tax,
             and horizon index sustainability multiplier.

Changes from original:
  - INSTRUMENT_SPEC includes inheritance_tax (planner-controlled)
  - POLICY_DIM = 18 (was 15): added surveillance, enforcement_budget, propaganda
  - _learning_step uses ideal baseline instead of moving average
  - _compute_ideal_score provides fixed reference for each objective
  - All objective functions multiplied by horizon_index
  - All objective functions preserved (SUM, NASH, JAM, CROSS, TOPO, TARGET)
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple
import numpy as np
from mesa import Agent
if TYPE_CHECKING:
    from environment import EconomicModel
    from agents import WorkerAgent, FirmAgent, LandownerAgent

POLICY_UPDATE_INTERVAL = 10
WARMUP_STEPS = 5
EPSILON = 1e-6
ES_POPULATION = 8
ES_SIGMA_INIT = 0.05
ES_SIGMA_DECAY = 0.999
ES_SIGMA_MIN = 0.005
ES_LR = 0.1
ES_MOMENTUM = 0.9
STATE_DIM = 16
POLICY_DIM = 18
ADAPT_LR = 0.01
ADAPT_NOISE = 0.02
REWARD_CLIP = 50.0
REWARD_BASELINE_DECAY = 0.9

# ---------------------------------------------------------------------------
# Government types (Task 8)
# ---------------------------------------------------------------------------

ELECTION_CYCLE = 100  # steps between elections

# Election platforms: each maps to instrument floor constraints
PLATFORM_CONSTRAINTS = {
    "redistribution": {"ubi_payment": 1.0, "tax_rate_firm": 0.10, "inheritance_tax": 0.10},
    "growth":         {"tax_rate_worker": 0.0, "tax_rate_firm": 0.0, "infrastructure_investment": 1.0},
    "education":      {"education_investment": 1.0, "min_wage": 3.0},
    "environment":    {"pollution_tax": 0.5, "cleanup_investment": 1.0},
    "security":       {"healthcare_investment": 1.0, "agriculture_investment": 1.0},
}

INSTRUMENT_SPEC: List[Tuple[str, float, float, float]] = [
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
    ("inheritance_tax",           0.0,   0.0,   0.80),
    ("media_funding",             0.0,   0.0,   5.0),
    ("antitrust_enforcement",     0.0,   0.0,   3.0),
    ("surveillance_level",        0.0,   0.0,   1.0),
    ("enforcement_budget",        0.3,   0.0,   3.0),
    ("propaganda_budget",         0.0,   0.0,   3.0),
]

INSTRUMENT_NAMES = [s[0] for s in INSTRUMENT_SPEC]
INSTRUMENT_DEFAULTS = np.array([s[1] for s in INSTRUMENT_SPEC], dtype=np.float64)
INSTRUMENT_LO = np.array([s[2] for s in INSTRUMENT_SPEC], dtype=np.float64)
INSTRUMENT_HI = np.array([s[3] for s in INSTRUMENT_SPEC], dtype=np.float64)
INSTRUMENT_RANGE = INSTRUMENT_HI - INSTRUMENT_LO

class EvolutionStrategy:
    def __init__(self, dim, rng, sigma=ES_SIGMA_INIT, lr=ES_LR, pop_size=ES_POPULATION):
        self.dim = dim; self.rng = rng; self.sigma = sigma; self.lr = lr; self.pop_size = pop_size
        self.mean = np.full(dim, 0.5, dtype=np.float64); self.velocity = np.zeros(dim, dtype=np.float64)
    def sample_perturbations(self):
        half = self.pop_size // 2; noise = self.rng.standard_normal((half, self.dim))
        return np.vstack([noise, -noise])
    def update(self, perturbations, fitnesses):
        if len(fitnesses) < 2: return
        valid = np.isfinite(fitnesses)
        if valid.sum() < 2: return
        fitnesses = fitnesses[valid]; perturbations = perturbations[valid]
        ranks = np.zeros_like(fitnesses); order = np.argsort(fitnesses)
        for i, idx in enumerate(order): ranks[idx] = i
        ranks = (ranks / (len(ranks) - 1)) - 0.5
        grad = (perturbations.T @ ranks) / (self.pop_size * self.sigma)
        self.velocity = ES_MOMENTUM * self.velocity + (1 - ES_MOMENTUM) * grad
        self.mean += self.lr * self.velocity; self.mean = np.clip(self.mean, 0.01, 0.99)
        self.sigma = max(ES_SIGMA_MIN, self.sigma * ES_SIGMA_DECAY)
    def get_params(self, perturbation=None):
        if perturbation is not None: return np.clip(self.mean + self.sigma * perturbation, 0.01, 0.99)
        return self.mean.copy()

class LinearPolicy:
    def __init__(self, state_dim, action_dim, rng):
        self.rng = rng; self.state_dim = state_dim; self.action_dim = action_dim
        scale = np.sqrt(2.0 / (state_dim + action_dim))
        self.W = rng.standard_normal((action_dim, state_dim)) * scale * 0.01
        self.bias = np.zeros(action_dim, dtype=np.float64)
        self.state_mean = np.zeros(state_dim, dtype=np.float64)
        self.state_var = np.ones(state_dim, dtype=np.float64); self.state_count = 0
        self._log_probs = []; self._rewards = []; self._states = []; self._actions = []
    def _normalize_state(self, state):
        self.state_count += 1; alpha = 1.0 / self.state_count
        self.state_mean = (1 - alpha) * self.state_mean + alpha * state
        self.state_var = (1 - alpha) * self.state_var + alpha * (state - self.state_mean) ** 2
        return (state - self.state_mean) / np.sqrt(self.state_var + 1e-8)
    def forward(self, state, explore=True):
        state = np.nan_to_num(state, nan=0.0, posinf=50.0, neginf=-50.0)
        norm_state = self._normalize_state(state); mu = self.W @ norm_state + self.bias
        if explore:
            noise = self.rng.standard_normal(self.action_dim) * ADAPT_NOISE; action = mu + noise
            self._states.append(norm_state); self._actions.append(noise)
        else: action = mu
        return action
    def record_reward(self, reward): self._rewards.append(reward)
    def update(self):
        if len(self._rewards) < 2: self._clear(); return
        rewards = np.array(self._rewards, dtype=np.float64)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        dW = np.zeros_like(self.W); db = np.zeros_like(self.bias)
        for state, noise, adv in zip(self._states, self._actions, advantages):
            dW += adv * np.outer(noise, state) / (ADAPT_NOISE ** 2 + 1e-8)
            db += adv * noise / (ADAPT_NOISE ** 2 + 1e-8)
        n = len(self._rewards); self.W += ADAPT_LR * dW / n; self.bias += ADAPT_LR * db / n
        self.W *= 0.999; self.bias *= 0.999; self._clear()
    def _clear(self): self._log_probs.clear(); self._rewards.clear(); self._states.clear(); self._actions.clear()


# ---------------------------------------------------------------------------
# Horizon Index helper
# ---------------------------------------------------------------------------

def _get_horizon_index(model: "EconomicModel") -> float:
    """
    Get the current horizon index. Returns 1.0 (no effect) if not
    enough history has accumulated or if the module is unavailable.
    """
    try:
        from horizon_index import compute_horizon_index
        return compute_horizon_index(model)
    except ImportError:
        return 1.0


class PlannerAgent(Agent):
    def __init__(self, model):
        super().__init__(model); self.objective = model.objective
        obj_offset = {"SUM":0,"NASH":10000,"JAM":20000,"CROSS":30000,"TOPO":40000,"TARGET":50000,
                      "TOPO_X":60000,"TOPO_MIN":70000,"NASH_MIN":80000,"SUM_RAW":90000}.get(self.objective, 0)
        self._learn_rng = np.random.default_rng(model._seed + 7919 + obj_offset)
        self._es = EvolutionStrategy(dim=POLICY_DIM, rng=self._learn_rng)
        self._linear = LinearPolicy(state_dim=STATE_DIM, action_dim=POLICY_DIM, rng=self._learn_rng)
        self.policy = {name: default for name, default, _, _ in INSTRUMENT_SPEC}
        self.tax_revenue = 0.0; self.last_objective_value = -math.inf
        self.trust_score = 0.5
        self.objective_history = []; self._reward_baseline = 0.0
        self._current_perturbations = None; self._perturbation_fitnesses = []
        self._perturbation_idx = 0; self._eval_phase = False
        self._steps_since_update = 0; self._total_updates = 0
        # Democratic government state (Task 8)
        self._last_election_winner = "none"
        self._vote_shares = {}
        self._active_constraints = {}  # instrument floors from election
        self._steps_since_election = 0

    def step(self):
        self._steps_since_update += 1; self._steps_since_election += 1
        gov = getattr(self.model, 'gov_type', 'authoritarian')
        if gov in ('democratic', 'demo_captured') and self._steps_since_election >= ELECTION_CYCLE:
            self._run_election(); self._steps_since_election = 0
        self._apply_investments(); self._redistribute()
        if self._steps_since_update >= POLICY_UPDATE_INTERVAL: self._learning_step(); self._steps_since_update = 0

    def _observe_state(self):
        workers = self.model.workers; firms = [f for f in self.model.firms if not f.defunct]
        if not workers: return np.zeros(STATE_DIM, dtype=np.float64)
        worker_w = np.array([w.wealth for w in workers], dtype=np.float64)
        agencies = np.array([w.compute_agency() for w in workers], dtype=np.float64)
        n_employed = sum(1 for w in workers if w.employed); n_workers = len(workers)
        total_debt = sum(w.debt for w in workers)
        total_production = sum(f.production_this_step for f in firms)
        if firms:
            shares = np.array([f.market_share for f in firms]); hhi = float(np.sum(shares ** 2))
        else: hhi = 0.0
        if self.model.landowners:
            lo_w = np.array([lo.wealth for lo in self.model.landowners]); lo_t = lo_w.sum()
            if lo_t > 0: hhi = max(hhi, float(np.sum((lo_w / lo_t) ** 2)))
        # Elite distortion: authoritarian captured planner sees distorted reality
        gov = getattr(self.model, 'gov_type', 'authoritarian')
        if gov == 'auth_captured':
            ed = min(0.5, hhi * 0.8)
        else:
            ed = 0.0
        true_mw = max(worker_w.mean(), 0.0); true_minw = max(worker_w.min(), 0.0)
        true_unemp = 1.0 - n_employed / max(n_workers, 1)
        true_poll = float(np.mean(self.model.pollution_grid))
        hi = _get_horizon_index(self.model)
        mean_conflict = float(np.mean(self.model.conflict_grid)) if hasattr(self.model, 'conflict_grid') else 0.0
        mean_legitimacy = float(np.mean(self.model.legitimacy_grid)) if hasattr(self.model, 'legitimacy_grid') else 0.7
        state = np.array([
            np.log1p(true_mw * (1 + ed * 0.5)), np.log1p(true_minw * (1 + ed * 2.0)),
            float(_gini_fast(worker_w)), np.log1p(max(agencies.min(), 0.0)),
            float(np.nanmean(agencies)), true_unemp * (1 - ed * 0.6),
            np.log1p(max(total_debt, 0.0)), np.log1p(max(total_production, 0.0)),
            np.log1p(max(self.tax_revenue, 0.0)), true_poll * (1 - ed * 0.5),
            float(len(firms)) / 20.0, float(self.model.current_step) / 300.0,
            self.policy.get("inheritance_tax", 0.0),
            hi,
            mean_conflict * (1 - ed * 0.3),  # auth_captured sees less conflict
            mean_legitimacy,
        ], dtype=np.float64)
        np.nan_to_num(state, copy=False, nan=0.0, posinf=50.0, neginf=-50.0)
        return state

    def _compute_ideal_score(self):
        n = max(len(self.model.workers), 1)
        if self.objective == "SUM": return 200.0 * n
        elif self.objective == "SUM_RAW": return 200.0 * n
        elif self.objective == "NASH": return float(np.sum(np.log(np.full(n, 200.0))))
        elif self.objective == "NASH_MIN": return float(np.sum(np.log(np.full(n, 200.0))))
        elif self.objective == "TOPO": return float(np.sum(np.log(np.full(n, 200.0)))) * 1.0
        elif self.objective == "TOPO_X": return float(np.sum(np.log(np.full(n, 200.0)))) * 1.0
        elif self.objective == "TOPO_MIN": return float(np.sum(np.log(np.full(n, 200.0)))) * 1.0
        elif self.objective == "TARGET": return 1.0 * math.log(max(n, 1)) * 100
        elif self.objective == "CROSS": return float(np.sum(np.log(np.full(n, 200.0)))) * 0.5
        elif self.objective == "JAM": return math.log(5.0)
        else: return 0.0

    def _learning_step(self):
        current_obj = self.compute_objective()
        if not np.isfinite(current_obj):
            current_obj = self._reward_baseline if np.isfinite(self._reward_baseline) else 0.0
        self.last_objective_value = current_obj; self.objective_history.append(current_obj)
        ideal = self._compute_ideal_score()
        if abs(ideal) > 1e-6: reward = (current_obj - ideal) / abs(ideal)
        else: reward = current_obj
        reward = np.clip(reward, -REWARD_CLIP, REWARD_CLIP)
        self._reward_baseline = REWARD_BASELINE_DECAY * self._reward_baseline + (1 - REWARD_BASELINE_DECAY) * current_obj
        self._linear.record_reward(reward)
        if self._current_perturbations is not None:
            self._perturbation_fitnesses.append(current_obj); self._perturbation_idx += 1
            if self._perturbation_idx >= ES_POPULATION:
                fitnesses = np.array(self._perturbation_fitnesses[-ES_POPULATION:])
                self._es.update(self._current_perturbations, fitnesses)
                self._current_perturbations = None; self._perturbation_idx = 0
                self._perturbation_fitnesses.clear(); self._linear.update()
        if self._current_perturbations is None:
            self._current_perturbations = self._es.sample_perturbations()
            self._perturbation_idx = 0; self._perturbation_fitnesses.clear()
        idx = min(self._perturbation_idx, len(self._current_perturbations) - 1)
        perturbation = self._current_perturbations[idx]
        state = self._observe_state(); es_params = self._es.get_params(perturbation)
        linear_adj = self._linear.forward(state, explore=True)
        combined = np.clip(es_params + linear_adj * 0.1, 0.01, 0.99)
        if np.any(np.isnan(combined)): combined = np.clip(es_params, 0.01, 0.99)
        if np.any(np.isnan(combined)): combined = np.full(POLICY_DIM, 0.5)
        actual = INSTRUMENT_LO + combined * INSTRUMENT_RANGE
        for i, name in enumerate(INSTRUMENT_NAMES): self.policy[name] = float(actual[i])
        # Apply democratic election constraints as floors
        for instr, floor in self._active_constraints.items():
            if instr in self.policy:
                self.policy[instr] = max(self.policy[instr], floor)
        self._total_updates += 1

    def _run_election(self):
        """Workers vote on policy direction based on their bottleneck."""
        workers = self.model.workers
        if not workers:
            return
        gov = getattr(self.model, 'gov_type', 'authoritarian')
        votes = {"redistribution": 0, "growth": 0, "education": 0, "environment": 0, "security": 0}
        rng = self.model.rng
        for w in workers:
            # Compute worker bottleneck (POLI dimensions)
            p_score = min(1.0, w.wealth / 100.0)  # P: resources
            o_score = w.skill  # O: options/skills
            l_score = (float(w.employed) + float(w.debt < 100) + float(len(w.network_connections) > 0)) / 3.0  # L: levers
            i_score = min(1.0, max(0.0, (w.income_last_step - w.income_prev_step + 5.0) / 10.0))  # I: impact

            bottleneck_scores = {"redistribution": p_score, "education": o_score,
                                 "growth": l_score, "security": i_score}
            # Environment vote: triggered by local pollution
            if w.pos is not None:
                local_poll = float(self.model.pollution_grid[int(w.pos[0]), int(w.pos[1])])
                env_score = max(0.0, 1.0 - local_poll * 0.1)
                bottleneck_scores["environment"] = env_score

            # Find worst dimension
            platform = min(bottleneck_scores, key=bottleneck_scores.get)

            # Captured democracy: misidentify bottleneck based on epistemic drift
            if gov == "demo_captured":
                # Workers with drifted weights may misidentify their bottleneck
                # Use distance from neutral weights (0.5) as drift proxy
                weight_vals = list(w.decision_weights.values())
                drift = float(np.mean([abs(v - 0.5) for v in weight_vals]))
                # Higher drift = more likely to vote for wrong platform
                if rng.random() < drift * 2.0:
                    platform = list(votes.keys())[rng.integers(len(votes))]

            votes[platform] += 1

        # Tally and determine winner
        total = max(sum(votes.values()), 1)
        self._vote_shares = {k: v / total for k, v in votes.items()}
        winner = max(votes, key=votes.get)
        self._last_election_winner = winner
        self._active_constraints = dict(PLATFORM_CONSTRAINTS.get(winner, {}))

    def apply_tax(self, agent):
        from agents import WorkerAgent, FirmAgent, LandownerAgent
        def _safe(key, default=0.0):
            v = self.policy.get(key, default); return v if np.isfinite(v) else default
        if isinstance(agent, WorkerAgent):
            rate = _safe("tax_rate_worker", 0.05); tax = max(0.0, agent.income_last_step) * rate
            if agent.employed:
                floor = _safe("min_wage", 1.0)
                if agent.wage < floor:
                    shortfall = floor - agent.wage
                    if self.tax_revenue >= shortfall: agent.wealth += shortfall; self.tax_revenue -= shortfall
                    elif self.tax_revenue > 0: agent.wealth += self.tax_revenue; self.tax_revenue = 0.0
        elif isinstance(agent, FirmAgent):
            rate = _safe("tax_rate_firm", 0.10); tax = max(0.0, agent.profit) * rate
            tax += agent.production_this_step * agent.pollution_factor * _safe("pollution_tax", 0.0)
            min_w = _safe("min_wage", 1.0)
            if agent.offered_wage < min_w: agent.offered_wage = min_w
        elif isinstance(agent, LandownerAgent):
            rate = _safe("tax_rate_landowner", 0.08); tax = max(0.0, agent.total_rent_collected * 0.01) * rate
        else: return
        if agent.wealth >= tax: agent.wealth -= tax; self.tax_revenue += tax

    def _redistribute(self):
        workers = self.model.workers
        if not workers: return
        ubi = self.policy["ubi_payment"]
        if not np.isfinite(ubi): ubi = 0.0
        total_ubi = ubi * len(workers)
        if total_ubi <= 0 or self.tax_revenue <= 0: return
        if self.tax_revenue >= total_ubi:
            for w in workers: w.wealth += ubi
            self.tax_revenue -= total_ubi
        else:
            per = self.tax_revenue / len(workers)
            for w in workers: w.wealth += per
            self.tax_revenue = 0.0

    def _apply_investments(self):
        COST_PER_UNIT = 0.5
        def _safe(key, default):
            v = self.policy.get(key, default); return v if np.isfinite(v) else default
        agri = _safe("agriculture_investment", 0.5); infra = _safe("infrastructure_investment", 0.5)
        hlth = _safe("healthcare_investment", 0.0); edu = _safe("education_investment", 0.0)
        total_cost = (agri + infra + hlth + edu) * COST_PER_UNIT
        if total_cost > 0 and self.tax_revenue > 0:
            ratio = min(1.0, self.tax_revenue / (total_cost + 1e-9))
            self.tax_revenue = max(0.0, self.tax_revenue - total_cost * ratio)
        else: ratio = 0.0
        self.model._agriculture_bonus = max(1.0, 1.0 + 0.3 * agri * ratio)
        self.model._infrastructure_level = max(0.5, 1.0 + 0.2 * infra * ratio)
        self.model._healthcare_bonus = max(0.0, min(0.30, 0.10 * hlth * ratio))
        self.model._education_quality = max(1.0, 1.0 + 0.3 * edu * ratio)
        cleanup = _safe("cleanup_investment", 0.0)
        if cleanup > 0 and self.tax_revenue > 0:
            cc = cleanup * 2.0
            if self.tax_revenue >= cc:
                self.tax_revenue -= cc
                self.model.pollution_grid *= max(0.0, 1.0 - min(0.10, cleanup * 0.02))

        # Media funding: public-good subsidy for news firms
        media_fund = _safe("media_funding", 0.0)
        if media_fund > 0 and self.tax_revenue > 0:
            active_nf = [nf for nf in self.model.news_firms if not nf.defunct]
            if active_nf:
                budget = min(media_fund * 2.0, self.tax_revenue)
                per_firm = budget / len(active_nf)
                for nf in active_nf:
                    nf.wealth += per_firm
                    nf._received_public_funding = True
                self.tax_revenue -= budget

        # Antitrust enforcement: probabilistic cartel breakup + fines
        antitrust = _safe("antitrust_enforcement", 0.0)
        if antitrust > 0 and self.model.active_cartels:
            breakup_prob = min(0.30, antitrust * 0.10)
            fine_rate = min(0.15, antitrust * 0.05)
            for cid in list(self.model.active_cartels.keys()):
                members = self.model.active_cartels.get(cid, set())
                if len(members) < 2:
                    continue
                if self.model.rng.random() < breakup_prob:
                    for mid in list(members):
                        m = self.model.get_agent_by_id(mid)
                        if m is None:
                            continue
                        fine = m.wealth * fine_rate
                        m.wealth -= fine
                        self.tax_revenue += fine
                        m.cartel_id = None
                        m.cartel_partners = []
                        m.pollution_factor = max(0.05, m.pollution_factor * 0.85)
                    del self.model.active_cartels[cid]

        # Surveillance: set model-wide surveillance level for conflict system
        surv = _safe("surveillance_level", 0.0)
        self.model._surveillance_level = float(np.clip(surv, 0.0, 1.0))

        # Enforcement budget: scale enforcer effectiveness
        enf_budget = _safe("enforcement_budget", 0.3)
        if enf_budget > 0 and self.tax_revenue > 0:
            enf_cost = enf_budget * 1.0
            paid = min(enf_cost, self.tax_revenue); self.tax_revenue -= paid
            ratio_e = paid / max(enf_cost, 1e-9)
            for e in self.model.enforcers:
                e.force = float(np.clip(0.3 + 0.7 * enf_budget * ratio_e, 0.1, 1.5))

        # Propaganda: reduces conflict but increases scapegoating risk
        prop = _safe("propaganda_budget", 0.0)
        self.model._propaganda_budget = prop  # expose for media scapegoating
        if prop > 0 and self.tax_revenue > 0:
            prop_cost = prop * 1.5
            paid_p = min(prop_cost, self.tax_revenue); self.tax_revenue -= paid_p
            ratio_p = paid_p / max(prop_cost, 1e-9)
            # Directly suppress conflict grid
            if hasattr(self.model, 'conflict_grid'):
                suppress = min(0.05, prop * 0.01 * ratio_p)
                self.model.conflict_grid *= max(0.9, 1.0 - suppress)
            # Boost legitimacy slightly
            if hasattr(self.model, 'legitimacy_grid'):
                self.model.legitimacy_grid += prop * 0.002 * ratio_p
                np.clip(self.model.legitimacy_grid, 0.0, 1.0, out=self.model.legitimacy_grid)

    # ------------------------------------------------------------------
    # Objective functions
    # ------------------------------------------------------------------
    # Every objective is multiplied by the horizon index.
    # Unsustainable policies (sugar rushes) get their reward crushed
    # proportionally. A policy with hi=0.15 keeps only 15% of its
    # base reward. A sustainable path with hi=0.85 keeps 85%.
    # ------------------------------------------------------------------

    def compute_objective(self):
        if self.objective == "SUM": return self._objective_sum()
        elif self.objective == "SUM_RAW": return self._objective_sum_raw()
        elif self.objective == "NASH": return self._objective_nash()
        elif self.objective == "NASH_MIN": return self._objective_nash_min()
        elif self.objective == "JAM": return self._objective_jam()
        elif self.objective == "CROSS": return self._objective_cross()
        elif self.objective == "TOPO": return self._objective_topo()
        elif self.objective == "TOPO_X": return self._objective_topo_x()
        elif self.objective == "TOPO_MIN": return self._objective_topo_min()
        elif self.objective == "TARGET": return self._objective_target()
        else: raise ValueError(f"Unknown objective: {self.objective}")

    def _objective_sum_raw(self):
        """SUM without horizon index. Pure aggregate baseline."""
        w = self.model.get_all_agent_wealths(); w = w[np.isfinite(w)]
        return float(np.sum(w)) if len(w) > 0 else 0.0

    def _objective_sum(self):
        w = self.model.get_all_agent_wealths(); w = w[np.isfinite(w)]
        base = float(np.sum(w)) if len(w) > 0 else 0.0
        return base * _get_horizon_index(self.model)

    def _objective_nash(self):
        w = self.model.get_all_agent_wealths(); w = w[np.isfinite(w)]
        base = float(np.sum(np.log(np.maximum(w, EPSILON)))) if len(w) > 0 else 0.0
        return base * _get_horizon_index(self.model)

    def _objective_nash_min(self):
        """
        NASH with min(base, HI) instead of base * HI.
        Emergency brake: when trajectory is unsustainable, the horizon
        index IS the reward regardless of how good the base looks.
        When trajectory is healthy, base reward flows through normally.
        Normalized to handle scale mismatch between NASH (~hundreds) and HI ([0,1]).
        """
        w = self.model.get_all_agent_wealths(); w = w[np.isfinite(w)]
        base = float(np.sum(np.log(np.maximum(w, EPSILON)))) if len(w) > 0 else 0.0
        hi = _get_horizon_index(self.model)
        ideal = self._compute_ideal_score()
        if abs(ideal) < 1e-6:
            return base * hi  # fallback to multiplier if no ideal
        norm_base = base / abs(ideal)  # map to roughly [0, 1]
        gated = min(norm_base, hi)     # HI dominates when trajectory is bad
        return gated * abs(ideal)      # scale back to reward space

    def _objective_jam(self):
        workers = self.model.workers
        if not workers: return float(math.log(EPSILON))
        a = np.array([w.compute_agency() for w in workers], dtype=np.float64)
        a = a[np.isfinite(a)]
        base = float(math.log(max(float(np.min(a)), EPSILON))) if len(a) > 0 else float(math.log(EPSILON))
        return base * _get_horizon_index(self.model)

    def _objective_cross(self):
        workers = self.model.workers
        if not workers: return float(math.log(EPSILON))
        wealths = np.array([w.wealth for w in workers], dtype=np.float64); wealths = wealths[np.isfinite(wealths)]
        if len(wealths) == 0: return float(math.log(EPSILON))
        nash_base = float(np.sum(np.log(np.maximum(wealths, EPSILON))))
        all_w = self.model.get_all_agent_wealths(); all_w = all_w[np.isfinite(all_w) & (all_w > 0)]
        if len(all_w) > 1:
            s = np.sort(all_w); n = len(s)
            all_gini = float((2*np.sum(np.arange(1,n+1)*s)-(n+1)*s.sum())/(n*s.sum()))
        else: all_gini = 0.0
        gini_gate = max(0.001, (1.0 - all_gini)**2)
        alpha_health = 1.0
        if len(all_w) >= 20:
            thr = np.percentile(all_w, 90); tail = all_w[all_w >= thr]
            if len(tail) >= 5 and thr > 0:
                ls = np.sum(np.log(tail/thr))
                if ls > EPSILON: alpha_health = min(1.0, max(0.1, (len(tail)/ls)/3.0))
        n_emp = sum(1 for w in workers if w.employed)
        productivity = max(0.01, (n_emp/max(len(workers),1)) * float(np.mean([w.skill for w in workers])))
        edu = min(1.0, max(0.01, (self.model._education_quality - 1.0)*0.5 + 0.5))
        nf = getattr(self.model, 'news_firms', []); anf = [f for f in nf if not f.defunct]
        cn = sum(1 for f in anf if f.captured_by_cartel is not None)
        nd = 1.0 - (cn/len(anf)) if anf else 0.5
        tv = np.array([getattr(w,'authority_trust',0.7) for w in workers])
        th = max(0.01, float(np.mean(tv)) * (1.0 - float(np.mean(tv < 0.3))*2))
        base = float(nash_base * max(gini_gate * alpha_health * productivity * edu * max(0.01, nd*th), 1e-10))
        return base * _get_horizon_index(self.model)

    def _objective_topo(self):
        workers = self.model.workers
        if not workers: return float(math.log(EPSILON))
        wealths = np.array([w.wealth for w in workers], dtype=np.float64); wealths = wealths[np.isfinite(wealths)]
        if len(wealths) == 0: return float(math.log(EPSILON))
        nash_base = float(np.sum(np.log(np.maximum(wealths, EPSILON))))
        def gs(v, c, wi): return math.exp(-((v-c)/max(wi,0.01))**2)
        all_w = self.model.get_all_agent_wealths(); all_w = all_w[np.isfinite(all_w) & (all_w > 0)]
        if len(all_w) > 1:
            s = np.sort(all_w); n = len(s)
            all_gini = float((2*np.sum(np.arange(1,n+1)*s)-(n+1)*s.sum())/(n*s.sum()))
        else: all_gini = 0.5
        equity = gs(all_gini, 0.325, 0.15)
        n_emp = sum(1 for w in workers if w.employed)
        employment_health = gs(1.0 - n_emp/max(len(workers),1), 0.08, 0.12)
        skills = np.array([w.skill for w in workers]); skill_health = gs(float(np.mean(skills)) if len(skills)>0 else 0.3, 0.7, 0.25)
        mc = float(np.mean([len(w.network_connections) for w in workers])) if workers else 0
        tv = np.array([getattr(w,'authority_trust',0.7) for w in workers])
        r0 = (mc*0.05*float(np.mean(tv))*0.1)/0.02; epistemic = gs(r0, 1.2, 1.0)
        firms = [f for f in self.model.firms if not f.defunct]
        if firms: hhi = float(np.sum(np.array([f.market_share for f in firms])**2))
        else: hhi = 1.0
        competition = gs(hhi, 0.05, 0.08)
        sustainability = min(1.0, len(workers)/max(self.model.n_workers_initial, 1))
        system_health = equity * employment_health * skill_health * epistemic * competition * sustainability
        base = float(nash_base * max(system_health, 1e-10))
        return base * _get_horizon_index(self.model)

    def _objective_topo_base(self):
        """
        Shared TOPO computation. Returns (base_reward, system_health_components).
        Both TOPO_X and TOPO_MIN use the same landscape shaping,
        they differ only in how they gate with the horizon index.
        """
        workers = self.model.workers
        if not workers: return float(math.log(EPSILON)), {}
        wealths = np.array([w.wealth for w in workers], dtype=np.float64); wealths = wealths[np.isfinite(wealths)]
        if len(wealths) == 0: return float(math.log(EPSILON)), {}
        nash_base = float(np.sum(np.log(np.maximum(wealths, EPSILON))))
        def gs(v, c, wi): return math.exp(-((v-c)/max(wi,0.01))**2)
        all_w = self.model.get_all_agent_wealths(); all_w = all_w[np.isfinite(all_w) & (all_w > 0)]
        if len(all_w) > 1:
            s = np.sort(all_w); n = len(s)
            all_gini = float((2*np.sum(np.arange(1,n+1)*s)-(n+1)*s.sum())/(n*s.sum()))
        else: all_gini = 0.5
        equity = gs(all_gini, 0.325, 0.15)
        n_emp = sum(1 for w in workers if w.employed)
        employment_health = gs(1.0 - n_emp/max(len(workers),1), 0.08, 0.12)
        skills = np.array([w.skill for w in workers]); skill_health = gs(float(np.mean(skills)) if len(skills)>0 else 0.3, 0.7, 0.25)
        mc = float(np.mean([len(w.network_connections) for w in workers])) if workers else 0
        tv = np.array([getattr(w,'authority_trust',0.7) for w in workers])
        r0 = (mc*0.05*float(np.mean(tv))*0.1)/0.02; epistemic = gs(r0, 1.2, 1.0)
        firms = [f for f in self.model.firms if not f.defunct]
        if firms: hhi = float(np.sum(np.array([f.market_share for f in firms])**2))
        else: hhi = 1.0
        competition = gs(hhi, 0.05, 0.08)
        sustainability = min(1.0, len(workers)/max(self.model.n_workers_initial, 1))
        system_health = equity * employment_health * skill_health * epistemic * competition * sustainability
        base = float(nash_base * max(system_health, 1e-10))
        return base, {"equity": equity, "employment": employment_health, "skill": skill_health,
                      "epistemic": epistemic, "competition": competition, "sustainability": sustainability}

    def _objective_topo_x(self):
        """
        TOPO_X: Topology shaping with HI multiplier.
        Same as TOPO. Unsustainable trajectories get reward proportionally reduced.
        """
        base, _ = self._objective_topo_base()
        return base * _get_horizon_index(self.model)

    def _objective_topo_min(self):
        """
        TOPO_MIN: Topology shaping with HI emergency brake.

        min(normalized_base, HI) means:
        - When trajectory is sustainable (HI > normalized_base): base reward flows through.
          The planner optimizes the landscape normally.
        - When trajectory is unsustainable (HI < normalized_base): HI IS the reward.
          Nothing the planner does matters until the trajectory is fixed.

        This is "plan for the future unless there is an emergency."
        The emergency brake forces the planner to fix sustainability FIRST.
        """
        base, _ = self._objective_topo_base()
        hi = _get_horizon_index(self.model)
        ideal = self._compute_ideal_score()
        if abs(ideal) < 1e-6:
            return base * hi
        norm_base = base / abs(ideal)
        gated = min(norm_base, hi)
        return gated * abs(ideal)

    def _objective_target(self):
        workers = self.model.workers
        if not workers: return -100.0
        def rs(v, lo, hi, decay=5.0):
            if lo <= v <= hi: return 1.0
            if v < lo: return math.exp(-decay*(lo-v)/max(hi-lo,0.01))
            return math.exp(-decay*(v-hi)/max(hi-lo,0.01))
        all_w = self.model.get_all_agent_wealths(); all_w = all_w[np.isfinite(all_w) & (all_w > 0)]
        if len(all_w) > 1:
            s = np.sort(all_w); n = len(s)
            all_gini = float((2*np.sum(np.arange(1,n+1)*s)-(n+1)*s.sum())/(n*s.sum()))
        else: all_gini = 0.5
        alpha = 5.0
        if len(all_w) >= 20:
            thr = np.percentile(all_w, 90); tail = all_w[all_w >= thr]
            if len(tail) >= 5 and thr > 0:
                ls = np.sum(np.log(tail/thr))
                if ls > EPSILON: alpha = len(tail)/ls
        n_emp = sum(1 for w in workers if w.employed); unemp = 1.0 - n_emp/max(len(workers),1)
        skills = np.array([w.skill for w in workers]); ms = float(np.mean(skills)) if len(skills)>0 else 0.3
        mc = float(np.mean([len(w.network_connections) for w in workers])) if workers else 0
        tv = np.array([getattr(w,'authority_trust',0.7) for w in workers])
        r0 = (mc*0.05*float(np.mean(tv))*0.1)/0.02
        wm = np.array([[w.decision_weights.get(a,0.5) for a in ["harvest","seek_work","trade","migrate","save","invest","found_firm"]] for w in workers[:200]])
        pol = float(np.mean(np.sqrt(np.sum((wm - wm.mean(axis=0))**2, axis=1)))) if len(wm)>1 else 0.1
        firms = [f for f in self.model.firms if not f.defunct]
        if firms:
            hhi = float(np.sum(np.array([f.market_share for f in firms])**2))
            cpct = sum(1 for cid, m in self.model.active_cartels.items() if len(m)>=2)/max(len(firms),1)
        else: hhi = 1.0; cpct = 0.0
        df = float(np.mean([w.debt > 0 for w in workers])) if workers else 0
        ww = np.array([w.wealth for w in workers]); wmin = float(np.min(ww)) if len(ww)>0 else 0
        pr = len(workers)/max(self.model.n_workers_initial, 1)
        sc = {'gini':rs(all_gini,0.25,0.40),'alpha':min(1.0,alpha/3.0),
              'top10':rs(all_w[all_w>=np.percentile(all_w,90)].sum()/max(all_w.sum(),1) if len(all_w)>10 else 0.3,0.15,0.35),
              'unemployment':rs(unemp,0.03,0.15),'skill':rs(ms,0.50,1.00),'info_r0':rs(r0,0.50,2.00),
              'trust':rs(float(np.mean(tv)),0.50,0.80),'polarization':rs(pol,0.05,0.20),
              'hhi':rs(hhi,0.0,0.10),'cartels':rs(cpct,0.0,0.05),'debt':rs(df,0.0,0.20),
              'floor':min(1.0,wmin/40.0),'population':min(1.0,pr)}
        wts = {'gini':2.0,'alpha':1.0,'top10':1.5,'unemployment':2.0,'skill':1.0,'info_r0':1.5,
               'trust':1.0,'polarization':0.5,'hhi':1.0,'cartels':0.5,'debt':0.5,'floor':2.0,'population':2.0}
        total = sum(sc[k]*wts[k] for k in sc); mx = sum(wts.values())
        base = float((total/mx) * math.log(max(len(workers),1)) * 100)
        return base * _get_horizon_index(self.model)

    def get_policy_snapshot(self):
        snap = dict(self.policy); snap["objective_value"] = self.last_objective_value
        snap["tax_revenue_pool"] = self.tax_revenue; snap["es_sigma"] = self._es.sigma
        snap["total_updates"] = self._total_updates; return snap

def _gini_fast(w):
    w = np.sort(w[w > 0])
    if len(w) == 0: return 0.0
    n = len(w)
    return float((2*np.sum(np.arange(1,n+1)*w)-(n+1)*w.sum())/(n*w.sum()))

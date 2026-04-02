"""
agents.py - Sustainable Capitalism + Inheritance Tax + Trust + Innovation
Trust scores affect employment, investment, news absorption, and lending.
Innovation: firms can invest in R&D, tech_level multiplies production.
"""
from __future__ import annotations
import contextlib
from collections import deque
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Dict
from sustainable_capitalism import sustainable_learn_from_outcome, sustainable_choose_strategy, compute_stakeholder_scores
from innovation import (init_firm_tech, firm_rd_invest, apply_tech_to_production,
                         apply_tech_skill_effects, transfer_tech_on_hire)
import numpy as np
from mesa import Agent
if TYPE_CHECKING:
    from environment import EconomicModel

class AgentType(Enum):
    WORKER = auto()
    FIRM = auto()
    LANDOWNER = auto()
    PLANNER = auto()
    MEDIA = auto()
    ENFORCER = auto()

SURVIVAL_THRESHOLD = 1.0
REPRODUCTION_COST = 50.0
REPRODUCTION_THRESHOLD = 200.0
TARGET_POPULATION = 1500
BASE_REPRO_RATE = 0.001
MAX_POPULATION = 1600
ENTREPRENEURSHIP_THRESHOLD = 300.0
ENTREPRENEURSHIP_PROB = 0.003


# ---------------------------------------------------------------------------
# Diversity / identity helpers
# ---------------------------------------------------------------------------
IDENTITY_TRAITS = ("A", "B", "C")
IDENTITY_TYPES = (
    ("A", "A"),
    ("A", "B"),
    ("A", "C"),
    ("B", "B"),
    ("B", "C"),
    ("C", "C"),
)

def _normalize_identity(identity):
    if identity is None:
        return None
    if isinstance(identity, str):
        if len(identity) == 2 and identity[0] in IDENTITY_TRAITS and identity[1] in IDENTITY_TRAITS:
            return tuple(sorted((identity[0], identity[1])))
        return None
    try:
        a, b = identity
        if a in IDENTITY_TRAITS and b in IDENTITY_TRAITS:
            return tuple(sorted((str(a), str(b))))
    except Exception:
        return None
    return None

def _random_legacy_identity(rng):
    # Baseline population starts as AA / BB only.
    return ("A", "A") if rng.random() < 0.5 else ("B", "B")

def _identity_similarity(id1, id2):
    id1 = _normalize_identity(id1)
    id2 = _normalize_identity(id2)
    if id1 is None or id2 is None:
        return 1.0
    shared = len(set(id1).intersection(id2))
    return shared / 2.0

def _identity_trust_multiplier(agent_a, agent_b):
    """
    Trust modifier from the design doc:
        Trust = Trust_base * (0.5 + 0.5 * similarity)
    """
    sim = _identity_similarity(getattr(agent_a, "identity", None), getattr(agent_b, "identity", None))
    return 0.5 + 0.5 * sim

def _mix_identity(parent_a, parent_b, rng):
    """
    Reproduction / social mixing rule.
    """
    pa = _normalize_identity(parent_a) or ("A", "A")
    pb = _normalize_identity(parent_b) or ("B", "B")
    child = tuple(sorted((rng.choice(pa), rng.choice(pb))))
    if child not in IDENTITY_TYPES:
        child = tuple(sorted((pa[0], pb[0])))
    return child

class WorkerAgent(Agent):
    agent_type = AgentType.WORKER
    def __init__(self, model, wealth=None, skill=None, metabolism=None, risk_tolerance=None, mobility=None):
        super().__init__(model)
        rng = model.rng
        self.wealth = wealth if wealth is not None else float(rng.lognormal(mean=3.5, sigma=1.2))
        self.skill = skill if skill is not None else float(np.clip(rng.beta(2, 2), 0.05, 1.0))
        self.metabolism = metabolism if metabolism is not None else float(rng.uniform(0.5, 2.5))
        self.risk_tolerance = risk_tolerance if risk_tolerance is not None else float(rng.beta(2, 5))
        self.mobility = mobility if mobility is not None else float(rng.beta(1, 4))
        self.employed = False
        self.employer_id = None
        self.wage = 0.0
        self.consecutive_unemployed_steps = 0
        self.debt = 0.0
        self.debt_interest = 0.0
        self.loan_count = 0
        self.network_connections = []
        from information import init_decision_weights
        self.decision_weights = init_decision_weights(rng)
        self.authority_trust = float(rng.uniform(0.5, 0.9))
        self._last_action = "harvest"
        self._last_action_outcome = 0.0
        self.investments = {}
        self.dividend_income = 0.0
        self.age = 0
        self.income_last_step = 0.0
        self.income_prev_step = 0.0
        self.lifetime_harvested = 0.0
        self.lifetime_wages = 0.0
        self.harvested_this_step = 0.0
        self.trust_score = 0.5
        self.identity = _random_legacy_identity(rng)
        self._prev_employer_id = None  # for tech transfer tracking
        # Aggression / conflict system
        self.aggression = float(rng.beta(2, 5))  # 0=content, 0.5=friction, 1.0=rebellion
        self._crime_events = 0  # per-step counter, reset each step
        self._riot_events = 0
        # Per-agent epistemic health: rolling window of received signal metadata
        # Each entry: (is_captured_source: bool, source_accuracy: float, source_id: int)
        from information import SIGNAL_WINDOW_SIZE
        self._signal_window: deque = deque(maxlen=SIGNAL_WINDOW_SIZE)
        # EH four-variable attributes (updated in receive_information)
        self.misinformation_exposure: float = 0.0   # M_i: fraction of signals from captured sources
        self.viewpoint_entropy: float = 0.5          # VE_i: source diversity (HHI-based)
        self.claim_integrity: float = 0.5            # CI_i: weighted mean source accuracy
        self.contestation_quality: float = 0.5       # tau_c_i: experience vs captured signal ratio

    def step(self):
        self.age += 1
        self.income_prev_step = self.income_last_step
        self.income_last_step = 0.0
        self.harvested_this_step = 0.0
        self._crime_events = 0; self._riot_events = 0
        local_pollution = 0.0
        if self.pos is not None:
            local_pollution = float(self.model.pollution_grid[int(self.pos[0]), int(self.pos[1])])
        # Aggression update from stress factors
        self._update_aggression(local_pollution)
        effective_metabolism = self.metabolism * max(0.0, 1.0 - self.model._healthcare_bonus) + local_pollution * 0.05
        self.wealth -= effective_metabolism
        if self.wealth <= SURVIVAL_THRESHOLD:
            self._die(); return
        if self.debt > 0:
            interest_payment = self.debt * self.debt_interest
            if self.wealth >= interest_payment:
                self.wealth -= interest_payment; self.debt *= (1 + self.debt_interest)
            else:
                self.model.economy.handle_default(self)
        if self.wealth > 100 and self.model.firms: self._invest_in_firms()
        # Skill update: base + tech learning bonus
        if not self.employed:
            self.skill = max(0.05, self.skill * 0.998)
        else:
            self.skill = min(1.0, self.skill * 1.001)
            apply_tech_skill_effects(self)
        if self.employed and self.employer_id is not None:
            expected_wage = self.skill * 5.0
            if self.wage < expected_wage * 0.5 and self.model.rng.random() < 0.05:
                firm = self.model.get_agent_by_id(self.employer_id)
                if firm and isinstance(firm, FirmAgent): firm.fire_worker(self.unique_id)
        action = self._choose_action()
        if action == "harvest": self._harvest()
        elif action == "seek_work": self._seek_employment()
        elif action == "trade": self._trade()
        if self.employed and self.employer_id is not None:
            firm = self.model.get_agent_by_id(self.employer_id)
            if firm is not None and isinstance(firm, FirmAgent) and not firm.defunct:
                wage_paid = firm.pay_worker(self); self.wage = wage_paid
                self.wealth += wage_paid; self.income_last_step += wage_paid; self.lifetime_wages += wage_paid
            else: self.employed = False; self.employer_id = None; self.wage = 0.0
        if self.model.rng.random() < self.mobility: self._migrate()
        self._pay_rent()
        if self.wealth >= REPRODUCTION_THRESHOLD:
            population = len(self.model.workers)
            if population >= MAX_POPULATION:
                return
            pressure = max(0.0, 1.0 - population / TARGET_POPULATION)
            repro_prob = BASE_REPRO_RATE * pressure * max(0.1, 1.0 - local_pollution * 0.02)
            if self.model.rng.random() < repro_prob:
                self._reproduce()
        if self.wealth >= ENTREPRENEURSHIP_THRESHOLD and not self.employed and self.model.rng.random() < ENTREPRENEURSHIP_PROB:
            self._found_firm()
        if self.wealth < 10.0 and self.debt < 50.0 and self.model.rng.random() < self.risk_tolerance:
            self.model.economy.issue_loan(self, amount=20.0)
        self.model.planner.apply_tax(self)
        if hasattr(self.model, 'conflict_grid'): self._attempt_aggressive_action()
        if self.model.rng.random() < 0.02: self._discover_neighbor()

    def _tpos(self):
        if self.pos is None: return None
        return (int(self.pos[0]), int(self.pos[1]))

    def _safe_pos(self, pos=None):
        """Ensure Mesa always receives tuple positions."""
        if pos is None:
            pos = self.pos
        if pos is None:
            return None
        if isinstance(pos, tuple):
            return pos
        return (int(pos[0]), int(pos[1]))

    def _choose_action(self):
        from information import choose_action_from_weights, compute_action_context, update_weights_from_experience
        if self._last_action:
            outcome = self.income_last_step - self.income_prev_step
            update_weights_from_experience(self.decision_weights, self._last_action, outcome)
        if not self.employed: self.consecutive_unemployed_steps += 1
        else: self.consecutive_unemployed_steps = 0
        context = compute_action_context(self, self.model)
        action = choose_action_from_weights(self.decision_weights, context, self.model.rng)
        if action == "save": action = "harvest"
        elif action == "invest" and self.wealth >= 50: action = "harvest"
        elif action == "found_firm":
            if self.wealth >= ENTREPRENEURSHIP_THRESHOLD and not self.employed:
                self._last_action = action; return action
            action = "harvest"
        self._last_action = action; return action

    def receive_information(self, signal):
        from information import NEWS_ABSORPTION_RATE, ACTIONS
        # Trust gate: blend authority_trust with source's trust_score and identity similarity.
        source = self.model.get_agent_by_id(signal.source_id)
        source_trust = getattr(source, 'trust_score', 0.5) if source else 0.5
        identity_modifier = _identity_trust_multiplier(self, source) if source is not None else 1.0
        effective_trust = self.authority_trust * signal.trust * (0.5 + 0.5 * source_trust) * identity_modifier
        if effective_trust < 0.01:
            return
        for action in ACTIONS:
            delta = signal.weight_deltas.get(action, 0.0)
            current = self.decision_weights.get(action, 0.5)
            self.decision_weights[action] = float(np.clip(current + NEWS_ABSORPTION_RATE * effective_trust * delta, 0.01, 0.99))
        signal_avg = np.mean(list(signal.weight_deltas.values()))
        ed = np.sign(self._last_action_outcome)
        sd = np.sign(signal_avg)
        if ed != 0 and sd != 0:
            if ed == sd:
                self.authority_trust = min(0.95, self.authority_trust + 0.005)
            else:
                self.authority_trust = max(0.05, self.authority_trust - 0.01)
        self._last_action_outcome = self.income_last_step - self.income_prev_step
        # Process scapegoating: if signal blames a different identity, increase hostility
        scapegoat = getattr(signal, 'scapegoat_identity', None)
        if scapegoat is not None and _identity_similarity(self.identity, scapegoat) < 0.5:
            # Worker absorbs blame narrative: increase aggression toward scapegoat group
            self.aggression = min(1.0, self.aggression + effective_trust * 0.02)

        # Record signal metadata in rolling window for per-agent EH computation
        is_captured = getattr(signal, 'is_captured_source', False)
        src_accuracy = getattr(signal, 'source_accuracy', 0.5)
        self._signal_window.append((is_captured, src_accuracy, signal.source_id))
        self._update_eh_attributes()

    def _update_eh_attributes(self):
        """Recompute M_i, VE_i, CI_i, tau_c_i from the rolling signal window."""
        window = list(self._signal_window)
        n = len(window)
        if n == 0:
            return

        # --- M_i: misinformation exposure ---
        # Fraction of signals that came from captured sources
        n_captured = sum(1 for (cap, _, _) in window if cap)
        self.misinformation_exposure = float(n_captured) / n

        # --- VE_i: viewpoint entropy via HHI ---
        # Count how many signals came from each distinct source
        source_counts: Dict[int, int] = {}
        for (_, _, sid) in window:
            source_counts[sid] = source_counts.get(sid, 0) + 1
        n_sources = len(source_counts)
        if n_sources <= 1:
            self.viewpoint_entropy = 0.0
        else:
            shares = [c / n for c in source_counts.values()]
            hhi = sum(s * s for s in shares)
            # VE = 1 - HHI, normalized to [0,1] via (1-HHI)/(1-1/n_sources)
            max_diversity = 1.0 - 1.0 / n_sources
            self.viewpoint_entropy = float(np.clip((1.0 - hhi) / max_diversity, 0.0, 1.0))

        # --- CI_i: claim integrity (weighted mean source accuracy) ---
        # Equal weight per signal; source_accuracy is effective_accuracy from producer
        self.claim_integrity = float(np.mean([acc for (_, acc, _) in window]))

        # --- tau_c_i: contestation quality ---
        # Fraction of agent's information update driven by experience vs captured signals
        # experience_correction_rate ≈ weight-update learning rate * |outcome|
        experience_rate = 0.02 * (abs(self._last_action_outcome) + 1e-9)
        captured_rate = self.misinformation_exposure * len(window) / max(n, 1)
        denom = experience_rate + captured_rate
        if denom > 0:
            raw_tau = experience_rate / denom
        else:
            raw_tau = 0.5
        # Weight correction by mean accuracy of non-captured sources in the window
        non_captured_acc = [acc for (cap, acc, _) in window if not cap]
        quality_weight = float(np.mean(non_captured_acc)) if non_captured_acc else 0.0
        self.contestation_quality = float(np.clip(raw_tau * quality_weight, 0.0, 1.0))

    def _harvest(self):
        pos = self._safe_pos()
        if pos is None: return
        x, y = pos
        food_val = float(self.model.food_grid[x, y]); raw_val = float(self.model.raw_grid[x, y])
        water_val = float(self.model.water_grid[x, y]); pollution_val = float(self.model.pollution_grid[x, y])
        tfp = self.model.infrastructure_bonus
        water_bonus = 1.0 + 0.1 * min(water_val / max(self.model.water_grid.max(), 1.0), 1.0)
        pollution_penalty = max(0.5, 1.0 - pollution_val * 0.01)
        amount = min(food_val + raw_val * 0.5, self.skill * 5.0 * tfp * water_bonus * pollution_penalty * (1 + self.model.rng.exponential(0.1)))
        amount = max(0, amount); take_food = min(food_val, amount * 0.7); take_raw = min(raw_val, amount * 0.3)
        self.model.food_grid[x, y] = max(0.0, food_val - take_food); self.model.raw_grid[x, y] = max(0.0, raw_val - take_raw)
        value = (take_food + take_raw) * self.model.economy.prices["food"]
        self.wealth += value; self.income_last_step += value; self.harvested_this_step = value; self.lifetime_harvested += value

    def _seek_employment(self):
        pos = self._safe_pos()
        if pos is None:
            return
        neighbours = self.model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=4)
        firms_nearby = [a for cell in neighbours for a in self.model.grid.get_cell_list_contents([cell]) if isinstance(a, FirmAgent) and not a.defunct]
        if not firms_nearby:
            return

        # Trust filter: skip firms with very low trust.
        trusted_firms = [f for f in firms_nearby if getattr(f, 'trust_score', 0.5) >= 0.15]
        if not trusted_firms:
            trusted_firms = firms_nearby

        def firm_attractiveness(f):
            identity_bonus = _identity_trust_multiplier(self, f)
            return f.offered_wage * (0.5 + 0.5 * getattr(f, 'trust_score', 0.5)) * identity_bonus

        best = max(trusted_firms, key=firm_attractiveness, default=None)
        current_employer = self.model.get_agent_by_id(self.employer_id) if self.employer_id else None
        current_trust = getattr(current_employer, 'trust_score', 0.5) if current_employer else 0.5
        current_score = self.wage * (0.5 + 0.5 * current_trust) * (
            _identity_trust_multiplier(self, current_employer) if current_employer is not None else 1.0
        )
        if best and firm_attractiveness(best) > current_score:
            if self.employer_id is not None:
                old_firm = self.model.get_agent_by_id(self.employer_id)
                if old_firm and isinstance(old_firm, FirmAgent):
                    old_firm.fire_worker(self.unique_id)
            best.hire_worker(self)

    def _trade(self):
        if self.network_connections:
            partner_id = self.model.rng.choice(self.network_connections)
            partner = self.model.get_agent_by_id(partner_id)
            if partner is None or not isinstance(partner, WorkerAgent): self.network_connections.remove(partner_id)
            else: self.model.economy.bilateral_trade(self, partner); return
        self._discover_and_trade()

    def _discover_neighbor(self):
        pos = self._safe_pos()
        if pos is None: return
        neighbours = self.model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=3)
        nearby = [a for cell in neighbours for a in self.model.grid.get_cell_list_contents([cell]) if isinstance(a, WorkerAgent) and a.unique_id != self.unique_id]
        if nearby:
            p = self.model.rng.choice(nearby)
            if p.unique_id not in self.network_connections: self.network_connections.append(p.unique_id)
            if self.unique_id not in p.network_connections: p.network_connections.append(self.unique_id)

    def _update_aggression(self, local_pollution):
        """Update aggression from economic stress, trust, pollution, and minority status."""
        stress = 0.0
        if self.wealth < self.metabolism * 5: stress += 0.3
        if not self.employed: stress += 0.2
        if self.trust_score < 0.3: stress += 0.2
        stress += min(local_pollution, 5.0) * 0.1
        # Minority identity stress: fraction of nearby agents with different identity
        pos = self._safe_pos()
        if pos is not None:
            neighbors = self.model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=2)
            nearby_workers = [a for cell in neighbors for a in self.model.grid.get_cell_list_contents([cell]) if isinstance(a, WorkerAgent)]
            if nearby_workers:
                same = sum(1 for w in nearby_workers if _identity_similarity(self.identity, w.identity) > 0.5)
                minority_stress = 1.0 - same / len(nearby_workers)
                stress += minority_stress * 0.2
        # Read local conflict field feedback
        if pos is not None and hasattr(self.model, 'conflict_grid'):
            local_conflict = float(self.model.conflict_grid[pos[0], pos[1]])
            self.aggression += local_conflict * 0.01
        # Surveillance suppression
        surv = getattr(self.model, '_surveillance_level', 0.0)
        effective_agg = self.aggression * (1.0 - surv)
        # Apply stress update
        self.aggression = float(np.clip(self.aggression + stress * 0.02 - 0.01, 0.0, 1.0))
        # Contribute to conflict grid
        if pos is not None and hasattr(self.model, 'conflict_grid'):
            identity_tension = 0.0
            if nearby_workers:
                avg_sim = sum(_identity_similarity(self.identity, w.identity) for w in nearby_workers) / len(nearby_workers)
                identity_tension = 1.0 - avg_sim
            contribution = effective_agg * (1.0 - self.trust_score) * (1.0 + identity_tension) * 0.05
            self.model.conflict_grid[pos[0], pos[1]] = min(1.0, self.model.conflict_grid[pos[0], pos[1]] + contribution)
        # Surveillance resentment
        if surv > 0:
            self.aggression += surv * (1.0 - self.trust_score) * 0.01
            self.aggression = min(1.0, self.aggression)

    def _attempt_aggressive_action(self):
        """Attempt theft or riot based on aggression level using blame model targeting."""
        rng = self.model.rng; pos = self._safe_pos()
        if pos is None: return
        surv = getattr(self.model, '_surveillance_level', 0.0)
        effective_agg = self.aggression * (1.0 - surv)
        local_conflict = float(self.model.conflict_grid[pos[0], pos[1]]) if hasattr(self.model, 'conflict_grid') else 0.0
        crime_prob = effective_agg * max(local_conflict, 0.2)
        # Theft: aggression > 0.4
        if effective_agg > 0.4 and rng.random() < crime_prob * 0.3:
            target = self._select_blame_target(pos)
            if target is not None:
                stolen = target.wealth * 0.01 * self.aggression
                if stolen > 0 and target.wealth > stolen:
                    target.wealth -= stolen; self.wealth += stolen
                    self._crime_events += 1
        # Riot: aggression > 0.7 and local conflict high
        if effective_agg > 0.7 and local_conflict > 0.5 and rng.random() < effective_agg * 0.1:
            self._riot_events += 1
            # Reduce nearby firm production and increase pollution
            neighbors = self.model.grid.get_neighborhood(pos, moore=True, include_center=True, radius=2)
            for cell in neighbors:
                for a in self.model.grid.get_cell_list_contents([cell]):
                    if isinstance(a, FirmAgent) and not a.defunct:
                        a.wealth -= a.wealth * 0.02 * self.aggression
                        from hardware import POLLUTION_CAP as _PC
                        self.model.pollution_grid[cell[0], cell[1]] = min(_PC, self.model.pollution_grid[cell[0], cell[1]] + 0.5)
            # Check rebellion threshold
            if hasattr(self.model, 'legitimacy_grid'):
                leg = float(self.model.legitimacy_grid[pos[0], pos[1]])
                if local_conflict > 0.9 and leg < 0.3:
                    self.model._rebellion_events = getattr(self.model, '_rebellion_events', 0) + 1

    def _select_blame_target(self, pos):
        """Select target using viewpoint-weighted hostility (blame model)."""
        neighbors = self.model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=3)
        candidates = []
        for cell in neighbors:
            for a in self.model.grid.get_cell_list_contents([cell]):
                if a.unique_id == self.unique_id: continue
                if isinstance(a, (FirmAgent, LandownerAgent, WorkerAgent)):
                    candidates.append(a)
        if not candidates: return None
        # Compute hostility scores using viewpoints, trust, and identity distance
        best_target = None; best_hostility = -1.0
        for t in candidates[:20]:  # cap scan for performance
            trust_to = getattr(t, 'trust_score', 0.5)
            id_dist = 1.0 - _identity_similarity(self.identity, getattr(t, 'identity', self.identity))
            # Viewpoint weight: bias toward blaming wealthier targets
            vw = 0.5
            if isinstance(t, FirmAgent): vw = self.decision_weights.get("invest", 0.5)
            elif isinstance(t, LandownerAgent): vw = 0.6
            elif isinstance(t, WorkerAgent) and t.wealth > self.wealth * 2: vw = 0.3
            hostility = vw * (1.0 - trust_to) * (1.0 + id_dist)
            if hostility > best_hostility:
                best_hostility = hostility; best_target = t
        return best_target

    def _discover_and_trade(self):
        pos = self._safe_pos()
        if pos is None: return
        neighbours = self.model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=3)
        nearby = [a for cell in neighbours for a in self.model.grid.get_cell_list_contents([cell]) if isinstance(a, WorkerAgent) and a.unique_id != self.unique_id]
        if not nearby: return
        p = self.model.rng.choice(nearby)
        if p.unique_id not in self.network_connections: self.network_connections.append(p.unique_id)
        if self.unique_id not in p.network_connections: p.network_connections.append(self.unique_id)
        self.model.economy.bilateral_trade(self, p)

    def _invest_in_firms(self):
        if self.model.current_step % 10 != 0:
            self._collect_dividends()
            return
        surplus = self.wealth - 100
        if surplus <= 0:
            return
        invest_amount = surplus * 0.05
        active_firms = [f for f in self.model.firms if not f.defunct and f.profit > 0]
        if not active_firms:
            return

        # Trust-weighted investment: prefer trustworthy, profitable, and identity-similar firms.
        if self.wealth > 500 and len(active_firms) > 1:
            def invest_score(f):
                return f.profit * (0.3 + 0.7 * getattr(f, 'trust_score', 0.5)) * _identity_trust_multiplier(self, f)
            target = max(active_firms, key=invest_score)
        else:
            target = self.model.rng.choice(active_firms)

        self.wealth -= invest_amount
        target.capital_stock += invest_amount * 0.9
        target.wealth += invest_amount * 0.1
        self.investments[target.unique_id] = self.investments.get(target.unique_id, 0) + invest_amount
        self._collect_dividends()

    def _collect_dividends(self):
        self.dividend_income = 0.0; dead = []
        for fid, amount in self.investments.items():
            firm = self.model.get_agent_by_id(fid)
            if firm is None or not isinstance(firm, FirmAgent) or firm.defunct: dead.append(fid); continue
            if firm.profit > 0 and firm.capital_stock > 0:
                share = amount / firm.capital_stock; div = min(firm.profit * 0.2 * share, firm.wealth * 0.1)
                if div > 0: self.wealth += div; firm.wealth -= div; self.dividend_income += div; self.income_last_step += div
        for fid in dead: self.wealth += self.investments.pop(fid) * 0.2

    def _migrate(self):
        pos = self._safe_pos()
        if pos is None:
            return
        neighbourhood = self.model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=3)
        empty = [c for c in neighbourhood if self.model.grid.is_cell_empty(c)]
        if not empty:
            return

        def cell_value(p):
            return float(self.model.food_grid[p[0], p[1]]) + float(self.model.raw_grid[p[0], p[1]]) + float(self.model.capital_grid[p[0], p[1]]) + float(self.model.water_grid[p[0], p[1]]) * 0.5

        self.model.grid.move_agent(self, max(empty, key=cell_value))

    def _pay_rent(self):
        if self.pos is None:
            return
        lo = self.model.get_landowner_at(self._tpos())
        if lo is not None:
            rent = lo.compute_rent(self)
            lo_trust = getattr(lo, 'trust_score', 0.5)
            rent *= _identity_trust_multiplier(self, lo)
            if lo_trust < 0.3 and self.model.rng.random() < 0.1:
                return
            if self.wealth >= rent:
                self.wealth -= rent
                lo.wealth += rent
                lo.total_rent_collected += rent

    def _reproduce(self):
        if self.wealth < REPRODUCTION_THRESHOLD:
            return
        self.wealth -= REPRODUCTION_COST
        is_elite = False
        if self.employed and self.employer_id is not None:
            firm = self.model.get_agent_by_id(self.employer_id)
            if firm and hasattr(firm, 'wealth') and firm.wealth > 500:
                is_elite = True
        if not is_elite and self.wealth > 300:
            is_elite = True
        edu_boost = 0.08 if is_elite else (self.model._education_quality - 1.0) * 0.05
        child_skill = float(np.clip(self.skill + edu_boost + self.model.rng.normal(0, 0.05), 0.05, 1.0))
        child = WorkerAgent(
            model=self.model,
            wealth=REPRODUCTION_COST * 0.8,
            skill=child_skill,
            metabolism=self.metabolism * float(self.model.rng.uniform(0.9, 1.1)),
            risk_tolerance=self.risk_tolerance,
            mobility=self.mobility,
        )
        # Hybrid identity formation from parent identities.
        parent_b = self.model.get_agent_by_id(self.employer_id) if self.employer_id is not None else None
        child.identity = _mix_identity(self.identity, getattr(parent_b, "identity", self.identity), self.model.rng)

        pos = self._safe_pos()
        if pos is None:
            return
        neighbourhood = self.model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=2)
        empty = [c for c in neighbourhood if self.model.grid.is_cell_empty(c)]
        if empty:
            self.model.grid.place_agent(child, tuple(self.model.rng.choice(empty)))
            self.model.workers.append(child)
            self.model._id_cache[child.unique_id] = child
            for a, w in self.decision_weights.items():
                child.decision_weights[a] = float(np.clip(w + self.model.rng.normal(0, 0.05), 0.01, 0.99))
            child.authority_trust = float(np.clip(self.authority_trust + self.model.rng.normal(0, 0.05), 0.05, 0.95))
            child.aggression = float(np.clip(self.aggression * 0.5 + self.model.rng.normal(0, 0.05), 0.0, 0.3))

    def _choose_governance(self):
        """Choose firm governance type based on observable performance."""
        # If worker has a current or previous employer, strongly weight that experience
        prev_employer = self.model.get_agent_by_id(self.employer_id) if self.employer_id else None
        if prev_employer is None and hasattr(self, '_prev_employer_id'):
            prev_employer = self.model.get_agent_by_id(self._prev_employer_id)
        if prev_employer and hasattr(prev_employer, 'is_sevc'):
            if prev_employer.profit > 0:
                return prev_employer.is_sevc
        # Otherwise: observe market and compare SEVC vs vanilla performance
        active_firms = [f for f in self.model.firms if not f.defunct]
        if not active_firms:
            return True
        sevc_profits = [f.profit for f in active_firms if getattr(f, 'is_sevc', True)]
        vanilla_profits = [f.profit for f in active_firms if not getattr(f, 'is_sevc', True)]
        sevc_mean = float(np.mean(sevc_profits)) if sevc_profits else 0.0
        vanilla_mean = float(np.mean(vanilla_profits)) if vanilla_profits else 0.0
        sevc_prob = (max(sevc_mean, 0) + 1.0) / (max(sevc_mean, 0) + max(vanilla_mean, 0) + 2.0)
        return bool(self.model.rng.random() < sevc_prob)

    def _found_firm(self):
        capital = self.wealth * 0.4; self.wealth -= capital
        firm = FirmAgent(model=self.model, capital=capital)
        firm.is_sevc = self._choose_governance()
        if not firm.is_sevc:
            for k in firm.strategy_weights:
                firm.strategy_weights[k] = 0.2
        pos = self._safe_pos()
        if pos is None: return
        neighbourhood = self.model.grid.get_neighborhood(pos, moore=True, include_center=True, radius=2)
        empty = [c for c in neighbourhood if self.model.grid.is_cell_empty(c)]
        place = tuple(self.model.rng.choice(empty)) if empty else pos
        self.model.grid.place_agent(firm, place); self.model.firms.append(firm); self.model._id_cache[firm.unique_id] = firm
        firm.hire_worker(self)

    def _die(self):
        if self.wealth > 5.0 and self.network_connections:
            inh_rate = self.model.planner.policy.get("inheritance_tax", 0.0)
            if not np.isfinite(inh_rate): inh_rate = 0.0
            inh_rate = max(0.0, min(0.80, inh_rate))
            tax_amount = self.wealth * inh_rate
            self.model.planner.tax_revenue += tax_amount
            after_tax = self.wealth - tax_amount
            inheritance = after_tax * 0.8
            recipients = []
            for cid in self.network_connections[:5]:
                heir = self.model.get_agent_by_id(cid)
                if heir and isinstance(heir, WorkerAgent): recipients.append(heir)
            if recipients:
                share = inheritance / len(recipients)
                for heir in recipients: heir.wealth += share
        if self.employed and self.employer_id:
            firm = self.model.get_agent_by_id(self.employer_id)
            if firm and isinstance(firm, FirmAgent): firm.fire_worker(self.unique_id)
        if self.pos is not None: self.model.grid.remove_agent(self)
        if self in self.model.workers: self.model.workers.remove(self)
        self.remove()

    def remove(self):
        if self in self.model.workers: self.model.workers.remove(self)
        super().remove()

    def compute_agency(self):
        # Raw POLI dimensions
        resources = max(self.wealth, 1e-9)                                  # P: material prerequisites
        if self.pos is not None:
            p = (int(self.pos[0]), int(self.pos[1]))
            local_density = (float(self.model.food_grid[p[0],p[1]]) +
                             float(self.model.raw_grid[p[0],p[1]]) +
                             float(self.model.capital_grid[p[0],p[1]]))
        else:
            local_density = 1.0
        options = max(self.skill * (local_density + 1), 1e-9)               # O: available options
        levers = 1.0 + float(self.employed) + float(self.debt < 100) + float(len(self.network_connections) > 0)  # L: action levers
        impact = max(abs(self.income_last_step - self.income_prev_step) + 1e-9, 1e-9)  # I: effective impact

        # POLI-EH separation: each dimension filtered by its phase-specific epistemic variable
        #   P-phase: M degrades perception of own prerequisites
        #   O-phase: VE (viewpoint entropy) constrains which options are visible
        #   L-phase: CI (claim integrity) determines quality of lever selection
        #   I-phase: tau_c (contestation quality) determines if impact is read correctly
        M       = getattr(self, 'misinformation_exposure', 0.0)
        VE      = getattr(self, 'viewpoint_entropy', 0.5)
        CI      = getattr(self, 'claim_integrity', 0.5)
        tau_c   = getattr(self, 'contestation_quality', 0.5)

        P_eff = resources / (1.0 + M)          # high M distorts P-perception
        O_eff = max(options * VE, 1e-9)        # low VE shrinks visible option set
        L_eff = levers * CI                    # low CI corrupts lever selection
        I_eff = max(impact * tau_c, 1e-9)      # low tau_c mutes effective impact

        agency = (P_eff * O_eff * L_eff * I_eff) ** 0.25
        return float(agency)


class FirmAgent(Agent):
    """Sustainable Capitalism + Innovation: firms optimize min(S,E,V,C) and can invest in R&D."""
    agent_type = AgentType.FIRM
    def __init__(self, model, capital=None):
        super().__init__(model); rng = model.rng
        self.wealth = capital if capital is not None else float(rng.lognormal(mean=5.5, sigma=1.5))
        self.capital_stock = self.wealth * 0.4
        self.workers = {}; self.offered_wage = float(rng.uniform(2.0, 5.0))
        self.profit = 0.0; self.prev_profit = 0.0; self.revenue = 0.0; self.production_this_step = 0.0
        self.cartel_id = None; self.cartel_partners = []; self.defunct = False; self.age = 0
        self.market_share = 0.0; self.total_wages_paid = 0.0; self.total_profit_accumulated = 0.0
        self.pollution_factor = float(rng.uniform(0.05, 0.30)); self.total_pollution_emitted = 0.0
        self._consecutive_losses = 0; self.total_dividends_paid = 0.0
        self.trust_score = 0.5
        self.strategy_weights = {"invest_capital": float(rng.beta(3,2)), "raise_wages": float(rng.beta(2,3)),
            "cut_wages": float(rng.beta(2,4)), "hire": float(rng.beta(3,2)), "downsize": float(rng.beta(1.5,4)),
            "acquire": float(rng.beta(1,5)), "form_cartel": float(rng.beta(1.5,4)),
            "capture_media": float(rng.beta(1,6)), "pollute_more": float(rng.beta(2,3)),
            "clean_up": float(rng.beta(1.5,4)), "innovate": float(rng.beta(2,3))}
        self._last_strategy = "invest_capital"
        # SEVC EMA normalization (Task 1)
        self.sevc_ema_mean = {'S': 0.5, 'E': 0.5, 'V': 0.5, 'C': 0.5}
        # Green R&D priority slider (Task 2): 0.0 = pure productivity, 1.0 = pure green
        self.green_rd_priority = 0.5
        # Per-firm SEVC flag (Task 4): default True (SEVC behavior)
        self.is_sevc = True
        # Firm-level Horizon Index (Task 7): ring buffer of floor scores
        self.floor_history = deque(maxlen=100)
        self.horizon_index = 1.0
        # Innovation
        init_firm_tech(self)

    def step(self):
        if self.defunct: return
        self.age += 1; self.prev_profit = self.profit; self.production_this_step = 0.0; self.revenue = 0.0
        self._produce()
        if self.is_sevc:
            strategy = sustainable_choose_strategy(self)
        else:
            # Vanilla: pick randomly from strategy_weights (no SEVC context scoring)
            strategies = list(self.strategy_weights.keys())
            strategy = strategies[self.model.rng.integers(len(strategies))] if strategies else "invest_capital"
        self._execute_strategy(strategy); self._last_strategy = strategy
        if self.is_sevc:
            sustainable_learn_from_outcome(self, strategy, self.profit - self.prev_profit)
        else:
            # Vanilla learning: use raw profit_change as signal
            profit_change = self.profit - self.prev_profit
            if strategy in self.strategy_weights:
                adj = 0.02 * np.tanh(profit_change / max(abs(self.profit) + 1, 1))
                cur = self.strategy_weights[strategy]
                self.strategy_weights[strategy] = float(np.clip(cur + adj, 0.01, 0.99))
            if self.profit < 0: self._consecutive_losses += 1
            else: self._consecutive_losses = 0
        self.capital_stock *= 0.998
        if self.wealth < -200 and len(self.workers) == 0: self._go_bankrupt(); return
        self._consider_mitosis()
        self.model.planner.apply_tax(self)

    def _execute_strategy(self, strategy):
        if strategy == "invest_capital":
            if self.profit > 0 and self.wealth > 50:
                rate = 0.10 + 0.03 * min(self.market_share * 10, 1.0)
                invest = min(self.wealth * rate, self.profit * 0.5); self.capital_stock += invest; self.wealth -= invest
        elif strategy == "raise_wages":
            n = max(len(self.workers), 1); rev_per = self.revenue / n if self.revenue > 0 else 0
            self.offered_wage = max(self.offered_wage, min(20.0, rev_per * 0.65))
        elif strategy == "cut_wages": self.offered_wage = max(1.0, self.offered_wage * 0.92)
        elif strategy == "downsize": self._manage_workforce()
        elif strategy == "acquire": self._consider_acquisition()
        elif strategy == "form_cartel": self._consider_cartel()
        elif strategy == "capture_media": self._attempt_media_capture()
        elif strategy == "pollute_more": self.pollution_factor = min(0.60, self.pollution_factor * 1.05)
        elif strategy == "clean_up": self.pollution_factor = max(0.02, self.pollution_factor * 0.90)
        elif strategy == "innovate": firm_rd_invest(self)
        self._maintain_wages()

    def _maintain_wages(self):
        n = max(len(self.workers), 1); rev_per = self.revenue / n if self.revenue > 0 else 0
        if rev_per > 0: self.offered_wage = min(self.offered_wage, rev_per * 0.85)
        self.offered_wage = max(1.0, min(20.0, self.offered_wage))

    def _attempt_media_capture(self):
        if self.wealth < 200: return
        news_firms = getattr(self.model, 'news_firms', [])
        targets = [nf for nf in news_firms if not nf.defunct and nf.captured_by_cartel is None and nf.wealth < 100]
        if not targets or self.model.rng.random() > 0.05: return
        target = self.model.rng.choice(targets); inv = min(self.wealth * 0.05, 100)
        self.wealth -= inv; target.wealth += inv; target.captured_by_cartel = self.cartel_id or -self.unique_id
        target.bias_direction = {"harvest":0.0,"seek_work":-0.03,"trade":0.0,"migrate":-0.02,"save":0.02,"invest":-0.02,"found_firm":-0.04}

    def _produce(self):
        n_workers = len(self.workers)
        if n_workers == 0: return
        scale_bonus = 1.0 + 0.1 * np.log1p(n_workers)
        A = 3.0 * self.model.infrastructure_bonus * scale_bonus
        K = max(self.capital_stock, 1.0); L = max(sum(w.skill for w in self.workers.values()), 0.01)
        output = A * (K ** 0.35) * (L ** 0.65)
        # Apply technology multiplier
        output = apply_tech_to_production(self, output)
        price = self.model.economy.prices.get("goods", 1.0)
        self.revenue = output * price; self.production_this_step = output
        self.profit = self.revenue - self.offered_wage * n_workers; self.wealth += self.profit
        self.total_profit_accumulated += max(self.profit, 0)
        if self.profit < 0: self._consecutive_losses += 1
        else: self._consecutive_losses = 0
        if self.pos is not None:
            fx, fy = int(self.pos[0]), int(self.pos[1])
            from hardware import POLLUTION_CAP
            emission = output * self.pollution_factor
            self.model.pollution_grid[fx, fy] = min(POLLUTION_CAP, self.model.pollution_grid[fx, fy] + emission)
            self.total_pollution_emitted += emission

    def _manage_workforce(self):
        if self._consecutive_losses >= 3 and len(self.workers) > 1 and self.workers:
            worst_id = min(self.workers.keys(), key=lambda wid: self.workers[wid].skill)
            self.fire_worker(worst_id); self._consecutive_losses = 0

    def _consider_cartel(self):
        if self.cartel_id is not None:
            if self.profit > 20 and self.model.rng.random() < 0.05: self._leave_cartel(); return
            if self.model.rng.random() < 0.01: self._leave_cartel(); return
            return
        if self.model.rng.random() > 0.01 or self.pos is None: return
        pos_t = (int(self.pos[0]), int(self.pos[1]))
        neighbours = self.model.grid.get_neighborhood(pos_t, moore=True, include_center=False, radius=10)
        nearby = [a for cell in neighbours for a in self.model.grid.get_cell_list_contents([cell]) if isinstance(a, FirmAgent) and a.unique_id != self.unique_id and not a.defunct]
        if len(nearby) < 2: return
        candidate = max(nearby, key=lambda f: f.wealth, default=None)
        if candidate and candidate.cartel_id is None:
            cid = self.model.next_cartel_id(); self.cartel_id = cid; candidate.cartel_id = cid
            self.cartel_partners = [candidate.unique_id]; candidate.cartel_partners = [self.unique_id]
            self.model.active_cartels[cid] = {self.unique_id, candidate.unique_id}
            self.pollution_factor = min(0.60, self.pollution_factor * 1.4); candidate.pollution_factor = min(0.60, candidate.pollution_factor * 1.4)

    def _leave_cartel(self):
        if self.cartel_id is not None and self.cartel_id in self.model.active_cartels:
            self.model.active_cartels[self.cartel_id].discard(self.unique_id)
            remaining = self.model.active_cartels.get(self.cartel_id, set())
            if len(remaining) < 2:
                for mid in list(remaining):
                    m = self.model.get_agent_by_id(mid)
                    if m and isinstance(m, FirmAgent): m.cartel_id = None; m.cartel_partners = []
                if self.cartel_id in self.model.active_cartels: del self.model.active_cartels[self.cartel_id]
        self.cartel_id = None; self.cartel_partners = []; self.pollution_factor = max(0.05, self.pollution_factor * 0.85)

    def _consider_acquisition(self):
        if self.wealth < 500 or self.market_share < 0.05 or self.model.rng.random() > 0.01 or self.pos is None: return
        pos_t = (int(self.pos[0]), int(self.pos[1]))
        neighbours = self.model.grid.get_neighborhood(pos_t, moore=True, include_center=False, radius=15)
        nearby = [a for cell in neighbours for a in self.model.grid.get_cell_list_contents([cell]) if isinstance(a, FirmAgent) and a.unique_id != self.unique_id and not a.defunct and a.wealth < self.wealth * 0.5]
        if not nearby: return
        target = min(nearby, key=lambda f: f.wealth); cost = max(target.wealth * 0.5, 50)
        if self.wealth < cost: return
        self.wealth -= cost; self.capital_stock += target.capital_stock
        # Acquire target's technology (take the max)
        if hasattr(target, 'tech_level') and hasattr(self, 'tech_level'):
            self.tech_level = max(self.tech_level, target.tech_level)
        for wid, worker in list(target.workers.items()): target.fire_worker(wid); self.hire_worker(worker)
        if target.cartel_id is not None and self.cartel_id is None:
            self.cartel_id = target.cartel_id
            if target.cartel_id in self.model.active_cartels:
                self.model.active_cartels[target.cartel_id].discard(target.unique_id); self.model.active_cartels[target.cartel_id].add(self.unique_id)
        target._go_bankrupt()

    def hire_worker(self, worker):
        if worker.unique_id in self.workers: return
        if len(self.workers) >= max(5, int(self.capital_stock ** 0.6)): return
        A = 1.5 * self.model.infrastructure_bonus; K = max(self.capital_stock, 1.0)
        L = max(sum(w.skill for w in self.workers.values()) + worker.skill, 0.01)
        mp = A * 0.65 * (K ** 0.35) * (L ** -0.35) * self.model.economy.prices.get("goods", 1.0)
        if mp < self.offered_wage * 0.5: return
        # Record previous employer for tech transfer
        worker._prev_employer_id = worker.employer_id
        self.workers[worker.unique_id] = worker; worker.employed = True; worker.employer_id = self.unique_id; worker.wage = self.offered_wage
        # Technology transfer: worker carries knowledge from old firm
        transfer_tech_on_hire(worker, self)

    def fire_worker(self, worker_id):
        if worker_id in self.workers:
            w = self.workers.pop(worker_id); w.employed = False; w.employer_id = None; w.wage = 0.0

    def pay_worker(self, worker):
        wage = self.offered_wage
        if self.wealth < wage: wage = max(0, self.wealth * 0.5)
        self.wealth -= wage; self.total_wages_paid += wage; return wage

    def _consider_mitosis(self):
        """Split large, mature, dominant firms into two (Task 3)."""
        if len(self.workers) < 15: return
        if self.market_share < 0.10: return
        if self.age < 50: return
        if self.model.rng.random() >= 0.02: return
        # Create daughter firm with 40% of capital
        daughter = FirmAgent(model=self.model, capital=self.capital_stock * 0.4)
        self.capital_stock *= 0.6
        # Daughter inherits key attributes
        daughter.tech_level = self.tech_level
        daughter.pollution_factor = self.pollution_factor
        daughter.green_rd_priority = self.green_rd_priority
        daughter.trust_score = self.trust_score
        daughter.strategy_weights = dict(self.strategy_weights)
        daughter.is_sevc = self.is_sevc
        daughter.age = 0
        # Place daughter near parent
        if self.pos is not None:
            pos_t = (int(self.pos[0]), int(self.pos[1]))
            neighbours = self.model.grid.get_neighborhood(pos_t, moore=True, include_center=False, radius=5)
            empty = [c for c in neighbours if self.model.grid.is_cell_empty(c)]
            if empty:
                place = empty[int(self.model.rng.integers(0, len(empty)))]
            else:
                place = pos_t
            self.model.grid.place_agent(daughter, place)
        else:
            place = self.model._random_cell()
            self.model.grid.place_agent(daughter, place)
        # Split workers: every other worker goes to daughter
        worker_ids = list(self.workers.keys())
        for i, wid in enumerate(worker_ids):
            if i % 2 == 1:
                w = self.workers.pop(wid)
                daughter.workers[wid] = w
                w.employer_id = daughter.unique_id
        # Register daughter
        self.model.firms.append(daughter)
        self.model._id_cache[daughter.unique_id] = daughter
        # Reset parent market_share (recalculated next step)
        self.market_share = 0.0

    def _go_bankrupt(self):
        for wid in list(self.workers.keys()): self.fire_worker(wid)
        if self.cartel_id is not None and self.cartel_id in self.model.active_cartels: self.model.active_cartels[self.cartel_id].discard(self.unique_id)
        if self.pos is not None: self.model.grid.remove_agent(self)
        self.defunct = True
        if self in self.model.firms: self.model.firms.remove(self)
        self.remove()

    def remove(self):
        if self in self.model.firms: self.model.firms.remove(self)
        super().remove()


class LandownerAgent(Agent):
    agent_type = AgentType.LANDOWNER
    def __init__(self, model, wealth=None):
        super().__init__(model); rng = model.rng
        self.wealth = wealth if wealth is not None else float(rng.lognormal(mean=6.0, sigma=1.0))
        self.controlled_cells = []; self.rent_rate = float(rng.uniform(0.05, 0.25))
        self.total_rent_collected = 0.0; self.age = 0
        self.trust_score = 0.5
        self.identity = _random_legacy_identity(rng)

    def step(self):
        self.age += 1
        if self.model.rng.random() < 0.05 and self.wealth > 100: self._expand()
        self._adjust_rent(); self.model.planner.apply_tax(self)

    def compute_rent(self, worker):
        base_rent = self.rent_rate * max(worker.income_last_step, 0.5)
        pos = worker._tpos()
        if pos is not None:
            x, y = pos
            local_value = float(self.model.food_grid[x,y]) + float(self.model.raw_grid[x,y]) + float(self.model.capital_grid[x,y])
            nearby_firms = 0
            try:
                neighbours = self.model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=3)
                for cell in neighbours[:12]:
                    for a in self.model.grid.get_cell_list_contents([cell]):
                        if isinstance(a, FirmAgent) and not a.defunct: nearby_firms += 1
            except: pass
            return base_rent * (1.0 + 0.02 * local_value + 0.05 * nearby_firms)
        return base_rent

    def _expand(self):
        if not self.controlled_cells or self.pos is None: return
        candidates = []
        for cx, cy in self.controlled_cells:
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < self.model.grid_width and 0 <= ny < self.model.grid_height and (nx,ny) not in self.controlled_cells:
                        candidates.append((nx,ny))
        if not candidates: return
        pos = candidates[int(self.model.rng.integers(0, len(candidates)))]
        if self.model.get_landowner_at(pos) is None:
            self.controlled_cells.append(pos); self.model.cell_ownership[pos] = self.unique_id; self.wealth -= 10.0

    def _adjust_rent(self):
        if not self.controlled_cells: return
        occupied = sum(1 for pos in self.controlled_cells if any(isinstance(a, WorkerAgent) for a in self.model.grid.get_cell_list_contents([pos])))
        rate = occupied / max(len(self.controlled_cells), 1)
        if rate > 0.7: self.rent_rate = min(0.5, self.rent_rate * 1.01)
        elif rate < 0.2: self.rent_rate = max(0.02, self.rent_rate * 0.99)

    def remove(self):
        if self in self.model.landowners: self.model.landowners.remove(self)
        super().remove()


# ---------------------------------------------------------------------------
# Enforcement agent
# ---------------------------------------------------------------------------

class EnforcerAgent(Agent):
    """
    State enforcement agent that patrols, detects aggressive workers,
    and applies punishment. Harsh enforcement generates resentment.
    """
    agent_type = AgentType.ENFORCER

    def __init__(self, model):
        super().__init__(model); rng = model.rng
        self.aggression = float(rng.beta(2, 2))       # policing intensity / brutality
        self.force = float(rng.uniform(0.5, 1.0))     # enforcement strength
        self.legitimacy_bias = float(rng.normal(0, 0.1))  # identity targeting bias
        self.trust_score = 0.5
        self.arrests = 0
        self.defunct = False

    def step(self):
        if self.pos is None: return
        pos = (int(self.pos[0]), int(self.pos[1]))
        surv = getattr(self.model, '_surveillance_level', 0.0)
        neighbors = self.model.grid.get_neighborhood(pos, moore=True, include_center=True, radius=3)
        for cell in neighbors:
            for a in self.model.grid.get_cell_list_contents([cell]):
                if not isinstance(a, WorkerAgent): continue
                # Detection probability scales with worker aggression and surveillance
                detect_prob = a.aggression * max(surv, 0.1)
                # Identity bias: slightly more likely to detect dissimilar identity
                id_sim = _identity_similarity(getattr(a, 'identity', ("A","A")), ("A","A"))
                detect_prob *= (1.0 + abs(self.legitimacy_bias) * (1.0 - id_sim))
                if self.model.rng.random() < detect_prob and a.aggression > 0.3:
                    # Punishment
                    punishment = self.force * a.aggression
                    fine = a.wealth * 0.02 * punishment
                    if fine > 0 and a.wealth > fine:
                        a.wealth -= fine; self.arrests += 1
                    # Suppress aggression
                    a.aggression *= max(0.3, 1.0 - punishment)
                    # But harsh enforcement breeds resentment
                    a.aggression += self.aggression * 0.05
                    a.aggression = min(1.0, a.aggression)
                    # Reduce local legitimacy from brutal enforcement
                    if hasattr(self.model, 'legitimacy_grid'):
                        cx, cy = int(cell[0]), int(cell[1])
                        self.model.legitimacy_grid[cx, cy] -= self.aggression * 0.01
                        self.model.legitimacy_grid[cx, cy] = max(0.0, self.model.legitimacy_grid[cx, cy])

    def remove(self):
        if self in self.model.enforcers: self.model.enforcers.remove(self)
        super().remove()

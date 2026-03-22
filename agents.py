"""
agents.py
---------
Heterogeneous agent classes for the multi-agent economic simulation.

Compatible with Mesa 3.x (no RandomActivation, no unique_id constructor arg).

Agent types:
  - WorkerAgent
  - FirmAgent
  - LandownerAgent

Changes from original:
  - Workers get initial network connections (fixes trade deadlock)
  - _choose_action allows trade attempts with nearby strangers
  - Workers can found firms (entrepreneurship)
  - Lower reproduction threshold with population floor via immigration
  - Firms have survival mechanics (cost cutting, downsizing)
"""

from __future__ import annotations

import contextlib
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Dict

import numpy as np
from mesa import Agent

if TYPE_CHECKING:
    from environment import EconomicModel


# ---------------------------------------------------------------------------
# Shared enumerations and constants
# ---------------------------------------------------------------------------

class AgentType(Enum):
    WORKER = auto()
    FIRM = auto()
    LANDOWNER = auto()
    PLANNER = auto()
    MEDIA = auto()


SURVIVAL_THRESHOLD = 1.0       # agents with wealth <= this die
REPRODUCTION_COST = 50.0       # wealth deducted when reproducing
REPRODUCTION_THRESHOLD = 200.0 # minimum wealth to reproduce
ENTREPRENEURSHIP_THRESHOLD = 300.0  # minimum wealth to found a firm
ENTREPRENEURSHIP_PROB = 0.003       # probability per step of founding


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------

class WorkerAgent(Agent):
    """
    Worker agents harvest resources, seek employment, trade, migrate,
    borrow money, and reproduce or die based on their wealth.
    """

    agent_type = AgentType.WORKER

    def __init__(self, model: "EconomicModel",
                 wealth: float = None,
                 skill: float = None,
                 metabolism: float = None,
                 risk_tolerance: float = None,
                 mobility: float = None):
        super().__init__(model)  # Mesa 3.x: only model arg

        rng = model.rng

        # Economic state
        self.wealth: float = wealth if wealth is not None else float(rng.lognormal(mean=3.5, sigma=1.2))
        self.skill: float = skill if skill is not None else float(np.clip(rng.beta(2, 2), 0.05, 1.0))
        self.metabolism: float = metabolism if metabolism is not None else float(rng.uniform(0.5, 2.5))
        self.risk_tolerance: float = risk_tolerance if risk_tolerance is not None else float(rng.beta(2, 5))
        self.mobility: float = mobility if mobility is not None else float(rng.beta(1, 4))

        # Employment state
        self.employed: bool = False
        self.employer_id: Optional[int] = None
        self.wage: float = 0.0
        self.consecutive_unemployed_steps: int = 0

        # Credit state
        self.debt: float = 0.0
        self.debt_interest: float = 0.0
        self.loan_count: int = 0

        # Social network (seeded with a few random connections at init)
        self.network_connections: List[int] = []

        # Decision matrix intelligence
        from information import init_decision_weights
        self.decision_weights: Dict[str, float] = init_decision_weights(rng)
        self.authority_trust: float = float(rng.uniform(0.5, 0.9))  # trust in institutional signals
        self._last_action: str = "harvest"
        self._last_action_outcome: float = 0.0

        # Investment portfolio: {firm_unique_id: amount_invested}
        self.investments: Dict[int, float] = {}
        self.dividend_income: float = 0.0

        # Tracking
        self.age: int = 0
        self.income_last_step: float = 0.0
        self.income_prev_step: float = 0.0
        self.lifetime_harvested: float = 0.0
        self.lifetime_wages: float = 0.0
        self.harvested_this_step: float = 0.0

    # ------------------------------------------------------------------
    # Step logic
    # ------------------------------------------------------------------

    def step(self):
        self.age += 1
        self.income_prev_step = self.income_last_step
        self.income_last_step = 0.0
        self.harvested_this_step = 0.0

        # Pay metabolism cost (reduced by healthcare, increased by local pollution)
        local_pollution = 0.0
        if self.pos is not None:
            local_pollution = float(self.model.pollution_grid[int(self.pos[0]), int(self.pos[1])])
        effective_metabolism = (self.metabolism * max(0.0, 1.0 - self.model._healthcare_bonus)
                                + local_pollution * 0.05)
        self.wealth -= effective_metabolism
        if self.wealth <= SURVIVAL_THRESHOLD:
            self._die()
            return

        # Pay debt interest
        if self.debt > 0:
            interest_payment = self.debt * self.debt_interest
            if self.wealth >= interest_payment:
                self.wealth -= interest_payment
                self.debt *= (1 + self.debt_interest)
            else:
                self.model.economy.handle_default(self)

        # Investment returns: wealthy workers invest in firms (stock market)
        # Returns come from actual firm profits, not created from nothing
        if self.wealth > 100 and self.model.firms:
            self._invest_in_firms()

        # Skill dynamics: meaningful experience gain, real decay
        if not self.employed:
            # Unemployed: skills decay (~15% loss over 100 steps)
            self.skill = max(0.05, self.skill * 0.998)
        else:
            # Employed: learning by doing (~10% gain over 100 steps)
            self.skill = min(1.0, self.skill * 1.001)

        # Quit check: skilled workers leave underpaying firms
        if self.employed and self.employer_id is not None:
            expected_wage = self.skill * 5.0  # what this worker is "worth"
            if self.wage < expected_wage * 0.5 and self.model.rng.random() < 0.05:
                firm = self.model.get_agent_by_id(self.employer_id)
                if firm and isinstance(firm, FirmAgent):
                    firm.fire_worker(self.unique_id)  # quit = fire from worker side

        # Decide action
        action = self._choose_action()
        if action == "harvest":
            self._harvest()
        elif action == "seek_work":
            self._seek_employment()
        elif action == "trade":
            self._trade()

        # Collect wage if employed
        if self.employed and self.employer_id is not None:
            firm = self.model.get_agent_by_id(self.employer_id)
            if firm is not None and isinstance(firm, FirmAgent) and not firm.defunct:
                wage_paid = firm.pay_worker(self)
                self.wage = wage_paid
                self.wealth += wage_paid
                self.income_last_step += wage_paid
                self.lifetime_wages += wage_paid
            else:
                self.employed = False
                self.employer_id = None
                self.wage = 0.0

        # Possibly migrate
        if self.model.rng.random() < self.mobility:
            self._migrate()

        # Pay rent for occupied cell
        self._pay_rent()

        # Possibly reproduce — harder when population is high (carrying capacity)
        if self.wealth >= REPRODUCTION_THRESHOLD:
            # Base prob scaled down by population pressure
            pop_ratio = len(self.model.workers) / self.model.n_workers_initial
            carrying_pressure = max(0.1, 1.0 / pop_ratio)  # halves at 2x pop
            repro_prob = 0.003 * carrying_pressure * max(0.1, 1.0 - local_pollution * 0.02)
            if self.model.rng.random() < repro_prob:
                self._reproduce()

        # Possibly found a firm (entrepreneurship)
        if (self.wealth >= ENTREPRENEURSHIP_THRESHOLD
                and not self.employed
                and self.model.rng.random() < ENTREPRENEURSHIP_PROB):
            self._found_firm()

        # Possibly borrow
        if self.wealth < 10.0 and self.debt < 50.0 and self.model.rng.random() < self.risk_tolerance:
            self.model.economy.issue_loan(self, amount=20.0)

        # Apply tax
        self.model.planner.apply_tax(self)

        # Occasionally discover new trade partners (breaks chicken-and-egg)
        if self.model.rng.random() < 0.02:
            self._discover_neighbor()

    def _tpos(self):
        """Return pos as a plain Python tuple (needed for Mesa 3.x numpy pos)."""
        if self.pos is None:
            return None
        return (int(self.pos[0]), int(self.pos[1]))

    def _choose_action(self) -> str:
        """Decision matrix: score actions using weights × situational context."""
        from information import choose_action_from_weights, compute_action_context, update_weights_from_experience

        # Learn from last step's outcome before choosing new action
        if self._last_action:
            # Outcome = income change (positive = good decision)
            outcome = self.income_last_step - self.income_prev_step
            update_weights_from_experience(
                self.decision_weights, self._last_action, outcome)

        # Update unemployment tracking
        if not self.employed:
            self.consecutive_unemployed_steps += 1
        else:
            self.consecutive_unemployed_steps = 0

        # Compute context and choose
        context = compute_action_context(self, self.model)
        action = choose_action_from_weights(
            self.decision_weights, context, self.model.rng)

        # Map non-standard actions to available methods
        if action == "save":
            action = "harvest"  # save = don't spend, just harvest minimally
        elif action == "invest" and self.wealth >= 50:
            action = "harvest"  # invest through capital accumulation
        elif action == "found_firm":
            if (self.wealth >= ENTREPRENEURSHIP_THRESHOLD
                    and not self.employed):
                self._last_action = action
                return action  # handled in step() already
            action = "harvest"

        self._last_action = action
        return action

    def receive_information(self, signal):
        """
        Receive an information signal and blend it into decision weights.
        Trust-weighted: high authority_trust = accept more of the signal.
        """
        from information import NEWS_ABSORPTION_RATE, ACTIONS

        effective_trust = self.authority_trust * signal.trust
        if effective_trust < 0.01:
            return  # below perception threshold

        for action in ACTIONS:
            delta = signal.weight_deltas.get(action, 0.0)
            current = self.decision_weights.get(action, 0.5)
            adjustment = NEWS_ABSORPTION_RATE * effective_trust * delta
            self.decision_weights[action] = float(
                np.clip(current + adjustment, 0.01, 0.99))

        # Update authority trust based on whether signal matches experience
        # If signal says "seek_work" but agent just failed to find work,
        # trust decreases. Simplified: compare signal direction to last outcome.
        signal_avg = np.mean(list(signal.weight_deltas.values()))
        experience_direction = np.sign(self._last_action_outcome)
        signal_direction = np.sign(signal_avg)
        if experience_direction != 0 and signal_direction != 0:
            if experience_direction == signal_direction:
                self.authority_trust = min(0.95, self.authority_trust + 0.005)
            else:
                self.authority_trust = max(0.05, self.authority_trust - 0.01)

        self._last_action_outcome = self.income_last_step - self.income_prev_step

    def _harvest(self):
        pos = self._tpos()
        if pos is None:
            return
        x, y = pos
        food_val      = float(self.model.food_grid[x, y])
        raw_val       = float(self.model.raw_grid[x, y])
        water_val     = float(self.model.water_grid[x, y])
        pollution_val = float(self.model.pollution_grid[x, y])
        # Infrastructure TFP and water availability boost harvest; pollution degrades it
        tfp = self.model.infrastructure_bonus
        water_bonus      = 1.0 + 0.1 * min(water_val / max(self.model.water_grid.max(), 1.0), 1.0)
        pollution_penalty = max(0.5, 1.0 - pollution_val * 0.01)   # up to -50% at cap
        amount = min(food_val + raw_val * 0.5,
                     self.skill * 5.0 * tfp * water_bonus * pollution_penalty
                     * (1 + self.model.rng.exponential(0.1)))
        amount = max(0, amount)
        take_food = min(food_val, amount * 0.7)
        take_raw  = min(raw_val,  amount * 0.3)
        self.model.food_grid[x, y] = max(0.0, food_val - take_food)
        self.model.raw_grid[x, y]  = max(0.0, raw_val  - take_raw)
        value = (take_food + take_raw) * self.model.economy.prices["food"]
        self.wealth += value
        self.income_last_step += value
        self.harvested_this_step = value
        self.lifetime_harvested += value

    def _seek_employment(self):
        pos = self._tpos()
        if pos is None:
            return
        neighbours = self.model.grid.get_neighborhood(
            pos, moore=True, include_center=False, radius=5)
        firms_nearby = [
            a for cell in neighbours
            for a in self.model.grid.get_cell_list_contents([cell])
            if isinstance(a, FirmAgent) and not a.defunct
        ]
        if not firms_nearby:
            return
        best = max(firms_nearby, key=lambda f: f.offered_wage, default=None)
        if best and best.offered_wage > self.wage:
            if self.employer_id is not None:
                old_firm = self.model.get_agent_by_id(self.employer_id)
                if old_firm and isinstance(old_firm, FirmAgent):
                    old_firm.fire_worker(self.unique_id)
            best.hire_worker(self)

    def _trade(self):
        """Trade with a known partner or discover a new one nearby."""
        # Try existing connections first
        if self.network_connections:
            partner_id = self.model.rng.choice(self.network_connections)
            partner = self.model.get_agent_by_id(partner_id)
            if partner is None or not isinstance(partner, WorkerAgent):
                self.network_connections.remove(partner_id)
            else:
                self.model.economy.bilateral_trade(self, partner)
                return

        # No connections or failed: find a nearby worker to trade with
        self._discover_and_trade()

    def _discover_neighbor(self):
        """Find a nearby worker and add them as a trade connection."""
        pos = self._tpos()
        if pos is None:
            return
        neighbours = self.model.grid.get_neighborhood(
            pos, moore=True, include_center=False, radius=3)
        nearby_workers = [
            a for cell in neighbours
            for a in self.model.grid.get_cell_list_contents([cell])
            if isinstance(a, WorkerAgent) and a.unique_id != self.unique_id
        ]
        if nearby_workers:
            partner = self.model.rng.choice(nearby_workers)
            if partner.unique_id not in self.network_connections:
                self.network_connections.append(partner.unique_id)
            if self.unique_id not in partner.network_connections:
                partner.network_connections.append(self.unique_id)

    def _discover_and_trade(self):
        """Find a nearby worker, form a connection, and trade."""
        pos = self._tpos()
        if pos is None:
            return
        neighbours = self.model.grid.get_neighborhood(
            pos, moore=True, include_center=False, radius=3)
        nearby_workers = [
            a for cell in neighbours
            for a in self.model.grid.get_cell_list_contents([cell])
            if isinstance(a, WorkerAgent) and a.unique_id != self.unique_id
        ]
        if not nearby_workers:
            return
        partner = self.model.rng.choice(nearby_workers)
        # Form connection
        if partner.unique_id not in self.network_connections:
            self.network_connections.append(partner.unique_id)
        if self.unique_id not in partner.network_connections:
            partner.network_connections.append(self.unique_id)
        # Trade
        self.model.economy.bilateral_trade(self, partner)

    def _invest_in_firms(self):
        """
        Simple stock market: invest surplus wealth in firms.
        Returns come from firm profits (dividends), not created from nothing.
        Wealthier agents invest more and in better firms (information advantage).
        """
        # Investment budget: fraction of wealth above threshold
        surplus = self.wealth - 100
        if surplus <= 0:
            return
        invest_amount = surplus * 0.02  # invest 2% of surplus per step

        # Pick a firm to invest in (biased toward profitable ones)
        active_firms = [f for f in self.model.firms if not f.defunct and f.profit > 0]
        if not active_firms:
            return

        # Wealthier workers pick better firms (information advantage)
        if self.wealth > 500 and len(active_firms) > 1:
            # Rich: invest in the most profitable firm
            target = max(active_firms, key=lambda f: f.profit)
        else:
            # Others: random profitable firm
            target = self.model.rng.choice(active_firms)

        # Transfer wealth to firm as capital investment
        self.wealth -= invest_amount
        target.capital_stock += invest_amount * 0.9  # 10% transaction cost
        target.wealth += invest_amount * 0.1  # firm keeps a fee

        # Record investment
        fid = target.unique_id
        self.investments[fid] = self.investments.get(fid, 0) + invest_amount

        # Collect dividends from existing investments
        self.dividend_income = 0.0
        dead_investments = []
        for fid, amount in self.investments.items():
            firm = self.model.get_agent_by_id(fid)
            if firm is None or not isinstance(firm, FirmAgent) or firm.defunct:
                dead_investments.append(fid)
                continue
            if firm.profit > 0:
                # Dividend: proportional to investment relative to total capital
                if firm.capital_stock > 0:
                    ownership_share = amount / firm.capital_stock
                    dividend = firm.profit * 0.2 * ownership_share  # 20% of profit to investors
                    dividend = min(dividend, firm.wealth * 0.1)  # cap at 10% of firm wealth
                    if dividend > 0:
                        self.wealth += dividend
                        firm.wealth -= dividend
                        self.dividend_income += dividend
                        self.income_last_step += dividend

        # Clean up dead investments (partial loss)
        for fid in dead_investments:
            lost = self.investments.pop(fid)
            # Lose 80% of investment in bankrupt firms
            recovery = lost * 0.2
            self.wealth += recovery

    def _migrate(self):
        grid = self.model.grid
        pos = self._tpos()
        if pos is None:
            return
        neighbourhood = grid.get_neighborhood(pos, moore=True, include_center=False, radius=3)
        empty_cells = [c for c in neighbourhood if grid.is_cell_empty(c)]
        if not empty_cells:
            return
        def cell_value(pos):
            return (float(self.model.food_grid[pos[0], pos[1]])
                    + float(self.model.raw_grid[pos[0], pos[1]])
                    + float(self.model.capital_grid[pos[0], pos[1]])
                    + float(self.model.water_grid[pos[0], pos[1]]) * 0.5)
        best_cell = max(empty_cells, key=cell_value)
        grid.move_agent(self, best_cell)

    def _pay_rent(self):
        if self.pos is None:
            return
        landowner = self.model.get_landowner_at(self._tpos())
        if landowner is not None:
            rent = landowner.compute_rent(self)
            if self.wealth >= rent:
                self.wealth -= rent
                landowner.wealth += rent
                landowner.total_rent_collected += rent

    def _reproduce(self):
        if self.wealth < REPRODUCTION_THRESHOLD:
            return
        self.wealth -= REPRODUCTION_COST

        # Tiered education: elite vs public
        # Elite education: workers employed by wealthy firms or renting from
        # landowners get a private education bonus regardless of public funding.
        # Public education: everyone else gets the planner's education quality.
        is_elite = False
        if self.employed and self.employer_id is not None:
            firm = self.model.get_agent_by_id(self.employer_id)
            if firm and hasattr(firm, 'wealth') and firm.wealth > 500:
                is_elite = True
        if not is_elite and self.wealth > 300:
            is_elite = True  # wealthy enough for private education

        if is_elite:
            # Private education: high quality, independent of public investment
            edu_boost = 0.08  # fixed high bonus
        else:
            # Public education: depends on planner investment
            edu_boost = (self.model._education_quality - 1.0) * 0.05

        child_skill = float(np.clip(
            self.skill + edu_boost + self.model.rng.normal(0, 0.05), 0.05, 1.0))
        child = WorkerAgent(
            model=self.model,
            wealth=REPRODUCTION_COST * 0.8,
            skill=child_skill,
            metabolism=self.metabolism * float(self.model.rng.uniform(0.9, 1.1)),
            risk_tolerance=self.risk_tolerance,
            mobility=self.mobility,
        )
        pos = self._tpos()
        if pos is None:
            return
        neighbourhood = self.model.grid.get_neighborhood(
            pos, moore=True, include_center=False, radius=2)
        empty_cells = [c for c in neighbourhood if self.model.grid.is_cell_empty(c)]
        if empty_cells:
            place_pos = self.model.rng.choice(empty_cells)
            self.model.grid.place_agent(child, place_pos)
            self.model.workers.append(child)
            self.model._id_cache[child.unique_id] = child
            # Cultural transmission: child inherits parent's decision weights with noise
            for action, w in self.decision_weights.items():
                child.decision_weights[action] = float(np.clip(
                    w + self.model.rng.normal(0, 0.05), 0.01, 0.99))
            child.authority_trust = float(np.clip(
                self.authority_trust + self.model.rng.normal(0, 0.05), 0.05, 0.95))

    def _found_firm(self):
        """Wealthy unemployed worker founds a new firm."""
        capital = self.wealth * 0.4
        self.wealth -= capital
        firm = FirmAgent(model=self.model, capital=capital)
        pos = self._tpos()
        if pos is None:
            return
        # Place firm near the founder
        neighbourhood = self.model.grid.get_neighborhood(
            pos, moore=True, include_center=True, radius=2)
        empty_cells = [c for c in neighbourhood if self.model.grid.is_cell_empty(c)]
        if empty_cells:
            place_pos = self.model.rng.choice(empty_cells)
        else:
            place_pos = pos  # MultiGrid allows stacking
        self.model.grid.place_agent(firm, place_pos)
        self.model.firms.append(firm)
        self.model._id_cache[firm.unique_id] = firm
        # Founder becomes first employee
        firm.hire_worker(self)

    def _die(self):
        # Inheritance: distribute wealth to network connections
        if self.wealth > 5.0 and self.network_connections:
            inheritance = self.wealth * 0.8  # 80% transferred, 20% lost (estate costs)
            # Split among connections (prioritizes stronger connections)
            recipients = []
            for cid in self.network_connections[:5]:  # max 5 heirs
                heir = self.model.get_agent_by_id(cid)
                if heir and isinstance(heir, WorkerAgent):
                    recipients.append(heir)
            if recipients:
                share = inheritance / len(recipients)
                for heir in recipients:
                    heir.wealth += share

        if self.employed and self.employer_id:
            firm = self.model.get_agent_by_id(self.employer_id)
            if firm and isinstance(firm, FirmAgent):
                firm.fire_worker(self.unique_id)
        if self.pos is not None:
            self.model.grid.remove_agent(self)
        if self in self.model.workers:
            self.model.workers.remove(self)
        self.remove()  # Mesa 3.x deregistration

    def remove(self):
        """Override to clean up from workers list."""
        if self in self.model.workers:
            self.model.workers.remove(self)
        super().remove()

    # ------------------------------------------------------------------
    # Agency metric
    # ------------------------------------------------------------------

    def compute_agency(self) -> float:
        """Agency proxy = geometric mean(resources, options, levers, impact) * EH filter."""
        resources = max(self.wealth, 1e-9)
        if self.pos is not None:
            p = (int(self.pos[0]), int(self.pos[1]))
            local_density = (float(self.model.food_grid[p[0], p[1]])
                             + float(self.model.raw_grid[p[0], p[1]])
                             + float(self.model.capital_grid[p[0], p[1]]))
        else:
            local_density = 1.0
        options = max(self.skill * (local_density + 1), 1e-9)
        levers = 1.0 + float(self.employed) + float(self.debt < 100) + float(len(self.network_connections) > 0)
        delta_income = abs(self.income_last_step - self.income_prev_step) + 1e-9
        impact = max(delta_income, 1e-9)

        raw_agency = (resources * options * levers * impact) ** 0.25

        # Epistemic health filter: agents with degraded information
        # environment have reduced effective agency
        eh_filter = 0.5 + 0.5 * self.authority_trust  # ranges 0.525 to 0.975

        return float(raw_agency * eh_filter)


# ---------------------------------------------------------------------------
# FirmAgent
# ---------------------------------------------------------------------------

class FirmAgent(Agent):
    """
    Firms hire workers, produce goods, set wages, invest, and accumulate profit.
    Firms can naturally form cartels.

    Changes from original:
      - Firms downsize (fire workers) when unprofitable instead of bleeding to death
      - Bankruptcy threshold raised (less fragile)
      - Wage setting is more responsive to profitability
      - Capital stock depreciates but can be reinvested
    """

    agent_type = AgentType.FIRM

    def __init__(self, model: "EconomicModel", capital: float = None):
        super().__init__(model)
        rng = model.rng
        self.wealth: float = capital if capital is not None else float(rng.lognormal(mean=5.5, sigma=1.5))
        self.capital_stock: float = self.wealth * 0.4
        self.workers: Dict[int, "WorkerAgent"] = {}
        self.offered_wage: float = float(rng.uniform(2.0, 5.0))
        self.profit: float = 0.0
        self.prev_profit: float = 0.0
        self.revenue: float = 0.0
        self.production_this_step: float = 0.0
        self.cartel_id: Optional[int] = None
        self.cartel_partners: List[int] = []
        self.defunct: bool = False
        self.age: int = 0
        self.market_share: float = 0.0
        self.total_wages_paid: float = 0.0
        self.total_profit_accumulated: float = 0.0
        self.pollution_factor: float = float(rng.uniform(0.05, 0.30))
        self.total_pollution_emitted: float = 0.0
        self._consecutive_losses: int = 0
        self.total_dividends_paid: float = 0.0

        # Firm decision intelligence: learned strategy weights
        self.strategy_weights: Dict[str, float] = {
            "invest_capital":  float(rng.beta(3, 2)),
            "raise_wages":     float(rng.beta(2, 3)),
            "cut_wages":       float(rng.beta(2, 4)),
            "hire":            float(rng.beta(3, 2)),
            "downsize":        float(rng.beta(1.5, 4)),
            "acquire":         float(rng.beta(1, 5)),
            "form_cartel":     float(rng.beta(1.5, 4)),
            "capture_media":   float(rng.beta(1, 6)),
            "pollute_more":    float(rng.beta(2, 3)),
            "clean_up":        float(rng.beta(1.5, 4)),
        }
        self._last_strategy: str = "invest_capital"

    def step(self):
        if self.defunct:
            return
        self.age += 1
        self.prev_profit = self.profit
        self.production_this_step = 0.0
        self.revenue = 0.0

        self._produce()

        # Strategic decision via learned weights
        strategy = self._choose_strategy()
        self._execute_strategy(strategy)
        self._last_strategy = strategy

        # Learn from outcome
        profit_change = self.profit - self.prev_profit
        self._learn_from_outcome(strategy, profit_change)

        self.capital_stock *= 0.998

        if self.wealth < -200 and len(self.workers) == 0:
            self._go_bankrupt()
            return

        self.model.planner.apply_tax(self)

    def _choose_strategy(self) -> str:
        rng = self.model.rng
        n_workers = len(self.workers)
        profitable = self.profit > 0
        context = {
            "invest_capital":  (1.0 if profitable else 0.2) * min(self.wealth / 200, 1),
            "raise_wages":     (0.3 if n_workers < 3 else 0.1) * (1.0 if profitable else 0.3),
            "cut_wages":       (0.8 if not profitable else 0.1) * (1.0 if n_workers > 0 else 0.0),
            "hire":            (0.5 if n_workers < 10 else 0.1) * (1.0 if profitable else 0.2),
            "downsize":        (0.8 if self._consecutive_losses > 2 else 0.1),
            "acquire":         min(self.wealth / 500, 1) * self.market_share * 5,
            "form_cartel":     0.3 if self.cartel_id is None else 0.0,
            "capture_media":   min(self.wealth / 1000, 1) * (0.5 if self.cartel_id else 0.1),
            "pollute_more":    0.3 if not profitable else 0.1,
            "clean_up":        self.model.planner.policy.get("pollution_tax", 0) * 0.5,
        }
        scores = {}
        for action, weight in self.strategy_weights.items():
            scores[action] = weight * context.get(action, 0.5) + float(rng.normal(0, 0.03))
        return max(scores, key=scores.get)

    def _execute_strategy(self, strategy: str):
        if strategy == "invest_capital":
            if self.profit > 0 and self.wealth > 50:
                rate = 0.10 + 0.03 * min(self.market_share * 10, 1.0)
                invest = min(self.wealth * rate, self.profit * 0.5)
                self.capital_stock += invest
                self.wealth -= invest
        elif strategy == "raise_wages":
            n = max(len(self.workers), 1)
            rev_per = self.revenue / n if self.revenue > 0 else 0
            self.offered_wage = max(self.offered_wage, min(20.0, rev_per * 0.65))
        elif strategy == "cut_wages":
            self.offered_wage = max(1.0, self.offered_wage * 0.92)
        elif strategy == "downsize":
            self._manage_workforce()
        elif strategy == "acquire":
            self._consider_acquisition()
        elif strategy == "form_cartel":
            self._consider_cartel()
        elif strategy == "capture_media":
            self._attempt_media_capture()
        elif strategy == "pollute_more":
            self.pollution_factor = min(0.60, self.pollution_factor * 1.05)
        elif strategy == "clean_up":
            self.pollution_factor = max(0.02, self.pollution_factor * 0.90)
        self._maintain_wages()

    def _learn_from_outcome(self, strategy: str, profit_change: float):
        if strategy in self.strategy_weights:
            adj = 0.02 * np.tanh(profit_change / max(abs(self.profit) + 1, 1))
            cur = self.strategy_weights[strategy]
            self.strategy_weights[strategy] = float(np.clip(cur + adj, 0.01, 0.99))
        if self.profit < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def _maintain_wages(self):
        n = max(len(self.workers), 1)
        rev_per = self.revenue / n if self.revenue > 0 else 0
        if rev_per > 0:
            self.offered_wage = min(self.offered_wage, rev_per * 0.85)
        self.offered_wage = max(1.0, min(20.0, self.offered_wage))

    def _attempt_media_capture(self):
        if self.wealth < 200:
            return
        news_firms = getattr(self.model, 'news_firms', [])
        targets = [nf for nf in news_firms if not nf.defunct
                    and nf.captured_by_cartel is None and nf.wealth < 100]
        if not targets or self.model.rng.random() > 0.05:
            return
        target = self.model.rng.choice(targets)
        investment = min(self.wealth * 0.05, 100)
        self.wealth -= investment
        target.wealth += investment
        target.captured_by_cartel = self.cartel_id or -self.unique_id
        target.bias_direction = {
            "harvest": 0.0, "seek_work": -0.03, "trade": 0.0,
            "migrate": -0.02, "save": 0.02, "invest": -0.02, "found_firm": -0.04,
        }

    def _produce(self):
        n_workers = len(self.workers)
        if n_workers == 0:
            return
        alpha = 0.35
        # Economies of scale: larger firms get a productivity bonus
        scale_bonus = 1.0 + 0.1 * np.log1p(n_workers)  # ~1.0 at 1, ~1.3 at 20
        A = 3.0 * self.model.infrastructure_bonus * scale_bonus
        K = max(self.capital_stock, 1.0)
        L = max(sum(w.skill for w in self.workers.values()), 0.01)
        output = A * (K ** alpha) * (L ** (1 - alpha))
        price = self.model.economy.prices.get("goods", 1.0)
        self.revenue = output * price
        self.production_this_step = output
        wage_bill = self.offered_wage * n_workers
        self.profit = self.revenue - wage_bill
        self.wealth += self.profit
        self.total_profit_accumulated += max(self.profit, 0)

        # Track losses for workforce management
        if self.profit < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Emit pollution at this firm's cell (proportional to output)
        if self.pos is not None:
            fx, fy = int(self.pos[0]), int(self.pos[1])
            emission = output * self.pollution_factor
            from hardware import POLLUTION_CAP
            self.model.pollution_grid[fx, fy] = min(
                POLLUTION_CAP,
                self.model.pollution_grid[fx, fy] + emission)
            self.total_pollution_emitted += emission

    def _manage_workforce(self):
        """Downsize if consistently unprofitable instead of bleeding to bankruptcy."""
        if self._consecutive_losses >= 3 and len(self.workers) > 1:
            # Fire the least skilled worker to cut costs
            if self.workers:
                worst_id = min(self.workers.keys(),
                               key=lambda wid: self.workers[wid].skill)
                self.fire_worker(worst_id)
                self._consecutive_losses = 0  # Reset after action

        # If no workers and wealth is positive, keep firm alive to rehire
        # (do nothing, firm persists as an empty shell)

    def _consider_cartel(self):
        # Existing cartel: check for dissolution
        if self.cartel_id is not None:
            # Defection: profitable firms leave cartels (cartel suppresses wages,
            # but if firm is profitable enough it doesn't need the cartel)
            if self.profit > 20 and self.model.rng.random() < 0.05:
                self._leave_cartel()
                return
            # Random antitrust breakup
            if self.model.rng.random() < 0.01:
                self._leave_cartel()
                return
            return

        # Formation: lower probability, requires multiple nearby firms
        if self.model.rng.random() > 0.01:
            return
        if self.pos is None:
            return
        pos_t = (int(self.pos[0]), int(self.pos[1]))
        neighbours = self.model.grid.get_neighborhood(
            pos_t, moore=True, include_center=False, radius=10)
        nearby_firms = [
            a for cell in neighbours
            for a in self.model.grid.get_cell_list_contents([cell])
            if isinstance(a, FirmAgent) and a.unique_id != self.unique_id and not a.defunct
        ]
        if len(nearby_firms) < 2:
            return
        candidate = max(nearby_firms, key=lambda f: f.wealth, default=None)
        if candidate and candidate.cartel_id is None:
            cartel_id = self.model.next_cartel_id()
            self.cartel_id = cartel_id
            candidate.cartel_id = cartel_id
            self.cartel_partners = [candidate.unique_id]
            candidate.cartel_partners = [self.unique_id]
            self.model.active_cartels[cartel_id] = {self.unique_id, candidate.unique_id}
            # Cartel coordination reduces regulatory compliance: both firms pollute more
            self.pollution_factor = min(0.60, self.pollution_factor * 1.4)
            candidate.pollution_factor = min(0.60, candidate.pollution_factor * 1.4)

    def _leave_cartel(self):
        """Exit current cartel."""
        if self.cartel_id is not None and self.cartel_id in self.model.active_cartels:
            self.model.active_cartels[self.cartel_id].discard(self.unique_id)
            # If only one member left, dissolve the cartel
            remaining = self.model.active_cartels.get(self.cartel_id, set())
            if len(remaining) < 2:
                for member_id in list(remaining):
                    member = self.model.get_agent_by_id(member_id)
                    if member and isinstance(member, FirmAgent):
                        member.cartel_id = None
                        member.cartel_partners = []
                if self.cartel_id in self.model.active_cartels:
                    del self.model.active_cartels[self.cartel_id]
        self.cartel_id = None
        self.cartel_partners = []
        # Pollution factor slowly recovers when leaving cartel
        self.pollution_factor = max(0.05, self.pollution_factor * 0.85)

    def _consider_acquisition(self):
        """Large firms attempt to acquire smaller nearby competitors (M&A)."""
        if self.wealth < 500 or self.market_share < 0.05:
            return
        if self.model.rng.random() > 0.01:
            return
        if self.pos is None:
            return

        pos_t = (int(self.pos[0]), int(self.pos[1]))
        neighbours = self.model.grid.get_neighborhood(
            pos_t, moore=True, include_center=False, radius=15)
        nearby_firms = [
            a for cell in neighbours
            for a in self.model.grid.get_cell_list_contents([cell])
            if isinstance(a, FirmAgent) and a.unique_id != self.unique_id
            and not a.defunct and a.wealth < self.wealth * 0.5
        ]
        if not nearby_firms:
            return

        target = min(nearby_firms, key=lambda f: f.wealth)
        acquisition_cost = max(target.wealth * 0.5, 50)

        if self.wealth < acquisition_cost:
            return

        # Execute acquisition
        self.wealth -= acquisition_cost
        self.capital_stock += target.capital_stock

        # Absorb target's workers
        for wid, worker in list(target.workers.items()):
            target.fire_worker(wid)
            self.hire_worker(worker)

        # Absorb cartel position
        if target.cartel_id is not None and self.cartel_id is None:
            self.cartel_id = target.cartel_id
            if target.cartel_id in self.model.active_cartels:
                self.model.active_cartels[target.cartel_id].discard(target.unique_id)
                self.model.active_cartels[target.cartel_id].add(self.unique_id)

        target._go_bankrupt()

    def hire_worker(self, worker: "WorkerAgent"):
        """Hire worker only if the marginal product justifies the wage."""
        if worker.unique_id in self.workers:
            return
        # Check marginal product: would adding this worker produce enough?
        n_current = len(self.workers)
        max_workers = max(5, int(self.capital_stock ** 0.6))  # steeper scaling with capital
        if n_current >= max_workers:
            return  # at capacity
        # Estimate marginal product (Cobb-Douglas derivative w.r.t. L)
        alpha = 0.35
        A = 1.5 * self.model.infrastructure_bonus
        K = max(self.capital_stock, 1.0)
        L = max(sum(w.skill for w in self.workers.values()) + worker.skill, 0.01)
        marginal_product = A * (1 - alpha) * (K ** alpha) * (L ** (-alpha))
        price = self.model.economy.prices.get("goods", 1.0)
        if marginal_product * price < self.offered_wage * 0.5:
            return  # not worth hiring

        self.workers[worker.unique_id] = worker
        worker.employed = True
        worker.employer_id = self.unique_id
        worker.wage = self.offered_wage

    def fire_worker(self, worker_id: int):
        if worker_id in self.workers:
            w = self.workers.pop(worker_id)
            w.employed = False
            w.employer_id = None
            w.wage = 0.0

    def pay_worker(self, worker: "WorkerAgent") -> float:
        wage = self.offered_wage
        if self.wealth < wage:
            wage = max(0, self.wealth * 0.5)
        self.wealth -= wage
        self.total_wages_paid += wage
        return wage

    def _go_bankrupt(self):
        for wid in list(self.workers.keys()):
            self.fire_worker(wid)
        if self.cartel_id is not None and self.cartel_id in self.model.active_cartels:
            self.model.active_cartels[self.cartel_id].discard(self.unique_id)
        if self.pos is not None:
            self.model.grid.remove_agent(self)
        self.defunct = True
        if self in self.model.firms:
            self.model.firms.remove(self)
        self.remove()

    def remove(self):
        if self in self.model.firms:
            self.model.firms.remove(self)
        super().remove()


# ---------------------------------------------------------------------------
# LandownerAgent
# ---------------------------------------------------------------------------

class LandownerAgent(Agent):
    """
    Landowners control territory, extract rent, expand holdings.
    """

    agent_type = AgentType.LANDOWNER

    def __init__(self, model: "EconomicModel", wealth: float = None):
        super().__init__(model)
        rng = model.rng
        self.wealth: float = wealth if wealth is not None else float(rng.lognormal(mean=6.0, sigma=1.0))
        self.controlled_cells: List[tuple] = []
        self.rent_rate: float = float(rng.uniform(0.05, 0.25))
        self.total_rent_collected: float = 0.0
        self.age: int = 0

    def step(self):
        self.age += 1
        if self.model.rng.random() < 0.05 and self.wealth > 100:
            self._expand()
        self._adjust_rent()
        self.model.planner.apply_tax(self)

    def compute_rent(self, worker: "WorkerAgent") -> float:
        """Rent based on location quality, not just worker income.
        Good locations (near resources and firms) cost more."""
        base_rent = self.rent_rate * max(worker.income_last_step, 0.5)

        # Location premium: cells near resources and firms are worth more
        pos = worker._tpos()
        if pos is not None:
            x, y = pos
            local_value = (float(self.model.food_grid[x, y])
                          + float(self.model.raw_grid[x, y])
                          + float(self.model.capital_grid[x, y]))
            # Check for nearby firms (proximity to jobs increases rent)
            nearby_firms = 0
            try:
                neighbours = self.model.grid.get_neighborhood(
                    pos, moore=True, include_center=False, radius=3)
                for cell in neighbours[:12]:  # sample for speed
                    for a in self.model.grid.get_cell_list_contents([cell]):
                        if isinstance(a, FirmAgent) and not a.defunct:
                            nearby_firms += 1
            except Exception:
                pass
            location_premium = 1.0 + 0.02 * local_value + 0.05 * nearby_firms
            return base_rent * location_premium

        return base_rent

    def _expand(self):
        if not self.controlled_cells or self.pos is None:
            return
        candidates = []
        for cx, cy in self.controlled_cells:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < self.model.grid_width and 0 <= ny < self.model.grid_height:
                        if (nx, ny) not in self.controlled_cells:
                            candidates.append((nx, ny))
        if not candidates:
            return
        idx = int(self.model.rng.integers(0, len(candidates)))
        pos = candidates[idx]
        if self.model.get_landowner_at(pos) is None:
            self.controlled_cells.append(pos)
            self.model.cell_ownership[pos] = self.unique_id
            self.wealth -= 10.0

    def _adjust_rent(self):
        if not self.controlled_cells:
            return
        occupied = sum(
            1 for pos in self.controlled_cells
            if any(isinstance(a, WorkerAgent)
                   for a in self.model.grid.get_cell_list_contents([pos]))
        )
        occupancy_rate = occupied / max(len(self.controlled_cells), 1)
        if occupancy_rate > 0.7:
            self.rent_rate = min(0.5, self.rent_rate * 1.01)
        elif occupancy_rate < 0.2:
            self.rent_rate = max(0.02, self.rent_rate * 0.99)

    def remove(self):
        if self in self.model.landowners:
            self.model.landowners.remove(self)
        super().remove()

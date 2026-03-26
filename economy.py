"""
economy.py
----------
Economic sub-system: loan market, decentralised trade, price dynamics,
and trade network management.

Changes:
  - issue_loan uses borrower trust_score in credit scoring
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from environment import EconomicModel
    from agents import WorkerAgent, FirmAgent


# ---------------------------------------------------------------------------
# Loan record
# ---------------------------------------------------------------------------

class Loan:
    def __init__(self, borrower_id: int, principal: float, interest_rate: float,
                 duration: int, issued_at: int):
        self.borrower_id = borrower_id
        self.principal = principal
        self.outstanding = principal
        self.interest_rate = interest_rate
        self.duration = duration
        self.issued_at = issued_at
        self.defaulted = False


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------

class Economy:
    BASE_PRICES: Dict[str, float] = {
        "food": 1.0, "goods": 1.2, "raw_materials": 0.8,
        "capital": 2.0, "land": 3.0, "water": 0.5,
    }

    def __init__(self, model: "EconomicModel"):
        self.model = model
        self.prices: Dict[str, float] = dict(self.BASE_PRICES)
        self._supply: Dict[str, float] = {k: 0.0 for k in self.BASE_PRICES}
        self._demand: Dict[str, float] = {k: 0.0 for k in self.BASE_PRICES}
        self.loans: List[Loan] = []
        self.total_defaults: int = 0
        self.total_debt_outstanding: float = 0.0
        self.trade_network: nx.DiGraph = nx.DiGraph()
        self.trade_volume_this_step: float = 0.0
        self.base_interest_rate: float = 0.05
        self.risk_premium_slope: float = 0.5

    def update_prices(self):
        for good in self.prices:
            supply = self._supply.get(good, 0.0) + 1e-6
            demand = self._demand.get(good, 0.0) + 1e-6
            ratio = demand / supply
            adjustment = 0.05 * math.log(ratio)
            noise = self.model.rng.normal(0, 0.01)
            self.prices[good] = max(0.1, self.prices[good] * math.exp(adjustment + noise))
        self._supply = {k: 0.0 for k in self.prices}
        self._demand = {k: 0.0 for k in self.prices}

    def record_supply(self, good: str, amount: float):
        self._supply[good] = self._supply.get(good, 0.0) + amount

    def record_demand(self, good: str, amount: float):
        self._demand[good] = self._demand.get(good, 0.0) + amount

    def issue_loan(self, borrower: "WorkerAgent", amount: float):
        """
        Issue a loan with trust-adjusted interest rate.
        Credit score = blend of wealth proxy and trust_score.
        High-trust borrowers get better rates. Low-trust get denied or pay more.
        """
        # Credit score: 60% wealth-based, 40% trust-based
        wealth_score = min(1.0, borrower.wealth / 100.0)
        trust_component = getattr(borrower, 'trust_score', 0.5)
        credit_score = 0.6 * wealth_score + 0.4 * trust_component

        # Very low trust = loan denied
        if credit_score < 0.1:
            return

        interest_rate = self.base_interest_rate + self.risk_premium_slope * (1 - credit_score)
        interest_rate = max(0.01, min(0.5, interest_rate))

        loan = Loan(
            borrower_id=borrower.unique_id,
            principal=amount,
            interest_rate=interest_rate,
            duration=50,
            issued_at=self.model.current_step,
        )
        self.loans.append(loan)
        borrower.debt += amount
        borrower.debt_interest = interest_rate
        borrower.loan_count += 1
        borrower.wealth += amount
        self.total_debt_outstanding += amount

    def handle_default(self, borrower: "WorkerAgent"):
        borrower.debt = 0.0
        borrower.debt_interest = 0.0
        self.total_defaults += 1
        borrower.skill = max(0.05, borrower.skill * 0.95)

    def service_loans(self):
        active = []
        for loan in self.loans:
            if loan.defaulted:
                continue
            age = self.model.current_step - loan.issued_at
            if age >= loan.duration:
                borrower = self.model.get_agent_by_id(loan.borrower_id)
                if borrower and hasattr(borrower, "wealth"):
                    if borrower.wealth >= loan.outstanding:
                        borrower.wealth -= loan.outstanding
                        borrower.debt = max(0, borrower.debt - loan.outstanding)
                    else:
                        self.handle_default(borrower)
                        loan.defaulted = True
            else:
                loan.outstanding *= (1 + loan.interest_rate / 52)
                active.append(loan)
        self.loans = active
        self.total_debt_outstanding = sum(l.outstanding for l in self.loans)

    def bilateral_trade(self, agent_a: "WorkerAgent", agent_b: "WorkerAgent"):
        if agent_a.pos is None or agent_b.pos is None:
            return
        ax, ay = int(agent_a.pos[0]), int(agent_a.pos[1])
        bx, by = int(agent_b.pos[0]), int(agent_b.pos[1])
        surplus_food_a = float(self.model.food_grid[ax, ay]) - 5.0
        surplus_food_b = float(self.model.food_grid[bx, by]) - 5.0
        if surplus_food_a > 0 and surplus_food_b < 0:
            trade_amount = min(surplus_food_a, abs(surplus_food_b))
            value = trade_amount * self.prices["food"]
            self.model.food_grid[ax, ay] -= trade_amount
            self.model.food_grid[bx, by] += trade_amount * 0.9
            agent_a.wealth += value * 0.9
            agent_b.wealth -= value
            self._record_trade(agent_a.unique_id, agent_b.unique_id, value)
            self.trade_volume_this_step += value
            if agent_b.unique_id not in agent_a.network_connections:
                agent_a.network_connections.append(agent_b.unique_id)
            if agent_a.unique_id not in agent_b.network_connections:
                agent_b.network_connections.append(agent_a.unique_id)

    def _record_trade(self, from_id: int, to_id: int, value: float):
        if not self.trade_network.has_node(from_id):
            self.trade_network.add_node(from_id)
        if not self.trade_network.has_node(to_id):
            self.trade_network.add_node(to_id)
        if self.trade_network.has_edge(from_id, to_id):
            self.trade_network[from_id][to_id]["weight"] += value
        else:
            self.trade_network.add_edge(from_id, to_id, weight=value)

    def update_market_sentiment(self):
        pass

    def refresh_trade_network(self):
        self.trade_volume_this_step = 0.0
        edges_to_remove = []
        for u, v, data in self.trade_network.edges(data=True):
            data["weight"] *= 0.95
            if data["weight"] < 0.01:
                edges_to_remove.append((u, v))
        self.trade_network.remove_edges_from(edges_to_remove)

    def get_trade_centrality(self) -> Dict[int, float]:
        if self.trade_network.number_of_nodes() == 0:
            return {}
        try:
            return nx.degree_centrality(self.trade_network)
        except Exception:
            return {}

    def get_network_stats(self) -> Dict:
        G = self.trade_network
        if G.number_of_nodes() < 2:
            return {"nodes": 0, "edges": 0, "density": 0.0, "max_centrality": 0.0}
        centrality = self.get_trade_centrality()
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "max_centrality": max(centrality.values()) if centrality else 0.0,
        }

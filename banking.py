"""
banking.py
----------
Banking system with trust integration.

Changes:
  - BankAgent has trust_score attribute
  - _attract_deposits uses bank trust_score (not authority_trust)
  - _make_loans uses borrower trust_score in credit decision
  - _check_bank_run uses bank trust_score as the run trigger
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from mesa import Agent

if TYPE_CHECKING:
    from environment import EconomicModel
    from agents import WorkerAgent


BASE_DEPOSIT_RATE = 0.0005
BASE_LENDING_RATE = 0.003
RISK_PREMIUM_SLOPE = 0.005
MAX_LEVERAGE_RATIO = 10.0
RESERVE_REQUIREMENT = 0.10
DEPOSIT_THRESHOLD = 50.0
LOAN_AMOUNT_BASE = 30.0
BANK_RUN_TRUST_THRESHOLD = 0.25


class BankAgent(Agent):
    def __init__(self, model: "EconomicModel", capital: float = None):
        super().__init__(model)
        rng = model.rng
        self.wealth: float = capital if capital is not None else float(rng.lognormal(mean=5.0, sigma=1.0))
        self.capital: float = self.wealth
        self.deposits: Dict[int, float] = {}
        self.total_deposits: float = 0.0
        self.loans: Dict[int, Dict] = {}
        self.total_loans: float = 0.0
        self.interest_income: float = 0.0
        self.interest_expense: float = 0.0
        self.profit: float = 0.0
        self.defaults_this_step: int = 0
        self.total_defaults: int = 0
        self.default_rate: float = 0.0
        self.defunct: bool = False
        self.age: int = 0
        self.lending_aggression: float = float(rng.uniform(0.3, 0.8))
        self._consecutive_losses: int = 0
        self.trust_score: float = 0.6

    def step(self):
        if self.defunct:
            return
        self.age += 1
        self.interest_income = 0.0
        self.interest_expense = 0.0
        self.defaults_this_step = 0
        self._collect_loan_payments()
        self._pay_deposit_interest()
        self._attract_deposits()
        self._make_loans()
        self._check_bank_run()
        self.profit = self.interest_income - self.interest_expense
        self.wealth += self.profit
        # Update running default rate
        n_loans = max(len(self.loans), 1)
        self.default_rate = 0.95 * self.default_rate + 0.05 * (self.defaults_this_step / n_loans)
        if self.profit > 0:
            self.lending_aggression = min(0.95, self.lending_aggression * 1.01)
            self._consecutive_losses = 0
        else:
            self.lending_aggression = max(0.1, self.lending_aggression * 0.95)
            self._consecutive_losses += 1
        if self.wealth < -self.total_deposits * 0.1:
            self._fail()

    def _collect_loan_payments(self):
        dead_loans = []
        for wid, loan in self.loans.items():
            worker = self.model.get_agent_by_id(wid)
            if worker is None or not hasattr(worker, 'wealth'):
                dead_loans.append(wid); continue
            interest = loan['outstanding'] * loan['rate']
            principal_payment = loan['outstanding'] * 0.01
            total_payment = interest + principal_payment
            if worker.wealth >= total_payment:
                worker.wealth -= total_payment
                self.interest_income += interest
                loan['outstanding'] -= principal_payment
                self.total_loans -= principal_payment
                if loan['outstanding'] <= 1.0:
                    worker.debt = max(0, worker.debt - loan.get('principal', 0))
                    dead_loans.append(wid)
            elif worker.wealth >= interest:
                worker.wealth -= interest
                self.interest_income += interest
            else:
                self._handle_default(wid, worker, loan)
                dead_loans.append(wid)
        for wid in dead_loans:
            if wid in self.loans:
                remaining = max(0, self.loans[wid]['outstanding'])
                self.total_loans = max(0, self.total_loans - remaining)
                del self.loans[wid]

    def _handle_default(self, wid: int, worker, loan: Dict):
        loss = loan['outstanding']
        self.wealth -= loss * 0.5
        self.defaults_this_step += 1
        self.total_defaults += 1
        worker.debt = max(0, worker.debt - loan.get('principal', 0))
        worker.skill = max(0.05, worker.skill * 0.95)

    def _pay_deposit_interest(self):
        dead_deposits = []
        for wid, amount in self.deposits.items():
            worker = self.model.get_agent_by_id(wid)
            if worker is None or not hasattr(worker, 'wealth'):
                dead_deposits.append(wid); continue
            interest = amount * BASE_DEPOSIT_RATE
            if self.wealth >= interest:
                worker.wealth += interest
                worker.income_last_step += interest
                self.interest_expense += interest
                self.wealth -= interest
            self.deposits[wid] = amount * (1 + BASE_DEPOSIT_RATE * 0.5)
        for wid in dead_deposits:
            self.total_deposits -= self.deposits.get(wid, 0)
            del self.deposits[wid]
        self.total_deposits = sum(self.deposits.values())

    def _attract_deposits(self):
        if self.defunct or not self.model.workers:
            return
        n_sample = min(20, len(self.model.workers))
        candidates = self.model.rng.choice(self.model.workers, size=n_sample, replace=False)
        for worker in candidates:
            if worker.wealth < DEPOSIT_THRESHOLD:
                continue
            if worker.unique_id in self.deposits:
                continue
            # Use BANK trust_score: workers deposit at trusted banks
            bank_trust = self.trust_score
            deposit_prob = 0.05 * bank_trust * min(worker.wealth / 200, 1.0)
            if self.model.rng.random() < deposit_prob:
                deposit_amount = (worker.wealth - DEPOSIT_THRESHOLD) * 0.1
                if deposit_amount > 1.0:
                    worker.wealth -= deposit_amount
                    self.deposits[worker.unique_id] = (
                        self.deposits.get(worker.unique_id, 0) + deposit_amount)
                    self.total_deposits += deposit_amount

    def _make_loans(self):
        if self.defunct or not self.model.workers:
            return
        available = (self.total_deposits * (1 - RESERVE_REQUIREMENT)
                    + self.capital * MAX_LEVERAGE_RATIO
                    - self.total_loans)
        if available < LOAN_AMOUNT_BASE:
            return
        n_sample = min(20, len(self.model.workers))
        candidates = self.model.rng.choice(self.model.workers, size=n_sample, replace=False)
        for worker in candidates:
            if available < LOAN_AMOUNT_BASE:
                break
            if worker.unique_id in self.loans:
                continue
            if worker.wealth > 100:
                continue
            if worker.debt > 100:
                continue
            # Credit scoring: blend wealth and trust
            wealth_score = min(1.0, worker.wealth / 50.0)
            borrower_trust = getattr(worker, 'trust_score', 0.5)
            credit_score = 0.6 * wealth_score + 0.4 * borrower_trust
            # Deny very low credit scores
            if credit_score < 0.1:
                continue
            loan_rate = BASE_LENDING_RATE + RISK_PREMIUM_SLOPE * (1 - credit_score)
            if self.model.rng.random() < self.lending_aggression:
                amount = LOAN_AMOUNT_BASE * (0.5 + credit_score)
                self.loans[worker.unique_id] = {
                    'principal': amount,
                    'outstanding': amount,
                    'rate': loan_rate,
                    'issued_step': self.model.current_step,
                }
                worker.wealth += amount
                worker.debt += amount
                worker.debt_interest = loan_rate
                self.total_loans += amount
                available -= amount

    def _check_bank_run(self):
        if not self.deposits:
            return
        bank_health = self.wealth / max(self.total_deposits, 1)
        deposit_ids = list(self.deposits.keys())
        if bank_health > 0.1:
            n_check = min(10, len(deposit_ids))
            check_ids = list(self.model.rng.choice(deposit_ids, size=n_check, replace=False))
        else:
            check_ids = deposit_ids
        withdrawals = []
        for wid in check_ids:
            worker = self.model.get_agent_by_id(wid)
            if worker is None:
                withdrawals.append(wid); continue
            # Bank run trigger: bank's own trust_score, not worker's authority_trust
            if self.trust_score < BANK_RUN_TRUST_THRESHOLD or bank_health < 0.05:
                if self.model.rng.random() < 0.1:
                    withdrawals.append(wid)
        for wid in withdrawals:
            amount = self.deposits.get(wid, 0)
            worker = self.model.get_agent_by_id(wid)
            if worker and hasattr(worker, 'wealth'):
                if self.wealth >= amount:
                    worker.wealth += amount
                    self.wealth -= amount
                else:
                    partial = max(0, self.wealth * 0.5)
                    worker.wealth += partial
                    self.wealth -= partial
            if wid in self.deposits:
                self.total_deposits -= self.deposits[wid]
                del self.deposits[wid]

    def _fail(self):
        if self.total_deposits > 0:
            recovery_rate = max(0, self.wealth / self.total_deposits)
        else:
            recovery_rate = 0
        for wid, amount in self.deposits.items():
            worker = self.model.get_agent_by_id(wid)
            if worker and hasattr(worker, 'wealth'):
                recovered = amount * min(recovery_rate, 0.8)
                worker.wealth += recovered
        self.loans.clear()
        self.deposits.clear()
        self.total_deposits = 0.0
        self.total_loans = 0.0
        self.defunct = True
        if self.pos is not None:
            self.model.grid.remove_agent(self)
        self.remove()

    def remove(self):
        if hasattr(self.model, 'banks') and self in self.model.banks:
            self.model.banks.remove(self)
        super().remove()


def compute_banking_metrics(model: "EconomicModel") -> Dict[str, float]:
    banks = getattr(model, 'banks', [])
    active_banks = [b for b in banks if not b.defunct]
    if not active_banks:
        return {
            "n_banks": 0, "total_bank_deposits": 0.0, "total_bank_loans": 0.0,
            "bank_leverage_ratio": 0.0, "bank_default_rate": 0.0,
            "bank_profit": 0.0, "deposit_concentration": 0.0,
        }
    total_deposits = sum(b.total_deposits for b in active_banks)
    total_loans = sum(b.total_loans for b in active_banks)
    total_capital = sum(b.capital for b in active_banks)
    total_defaults = sum(b.defaults_this_step for b in active_banks)
    total_profit = sum(b.profit for b in active_banks)
    n_loans = sum(len(b.loans) for b in active_banks)
    if total_deposits > 0:
        dep_shares = np.array([b.total_deposits / total_deposits for b in active_banks])
        deposit_hhi = float(np.sum(dep_shares ** 2))
    else:
        deposit_hhi = 0.0
    return {
        "n_banks": len(active_banks),
        "total_bank_deposits": total_deposits,
        "total_bank_loans": total_loans,
        "bank_leverage_ratio": total_loans / max(total_capital, 1) if total_capital > 0 else 0,
        "bank_default_rate": total_defaults / max(n_loans, 1),
        "bank_profit": total_profit,
        "deposit_concentration": deposit_hhi,
    }

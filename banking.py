"""
banking.py
----------
Banking system for the multi-agent economic simulation.

BankAgent takes deposits from wealthy workers (paying interest),
lends to poor workers (charging higher interest), and profits on the spread.

Emergent dynamics:
  - Predatory lending (highest-interest loans to poorest borrowers)
  - Bank runs (deposit withdrawals cascade when trust drops)
  - Too-big-to-fail (concentrated deposits create systemic risk)
  - Inherited debt (heirs inherit net position)
  - Credit cycles (overlending -> defaults -> credit contraction)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
from mesa import Agent

if TYPE_CHECKING:
    from environment import EconomicModel
    from agents import WorkerAgent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DEPOSIT_RATE = 0.0005      # interest paid to depositors per step (~0.6%/yr)
BASE_LENDING_RATE = 0.003       # interest charged to borrowers per step (~3.6%/yr)
RISK_PREMIUM_SLOPE = 0.005      # extra rate per unit of credit risk
MAX_LEVERAGE_RATIO = 10.0       # max loans / capital
RESERVE_REQUIREMENT = 0.10      # fraction of deposits held as reserves
DEPOSIT_THRESHOLD = 50.0        # min wealth to deposit
LOAN_AMOUNT_BASE = 30.0         # base loan size
BANK_RUN_TRUST_THRESHOLD = 0.3  # authority_trust below this triggers withdrawal


# ---------------------------------------------------------------------------
# BankAgent
# ---------------------------------------------------------------------------

class BankAgent(Agent):
    """
    Bank: takes deposits, makes loans, profits on spread.
    Fails when losses exceed capital, cascading to depositors.
    """

    def __init__(self, model: "EconomicModel", capital: float = None):
        super().__init__(model)
        rng = model.rng

        self.wealth: float = capital if capital is not None else float(rng.lognormal(mean=5.0, sigma=1.0))
        self.capital: float = self.wealth

        # Deposits: {worker_id: amount}
        self.deposits: Dict[int, float] = {}
        self.total_deposits: float = 0.0

        # Loans: {worker_id: {principal, outstanding, rate, issued_step}}
        self.loans: Dict[int, Dict] = {}
        self.total_loans: float = 0.0

        # Performance
        self.interest_income: float = 0.0
        self.interest_expense: float = 0.0
        self.profit: float = 0.0
        self.defaults_this_step: int = 0
        self.total_defaults: int = 0
        self.defunct: bool = False
        self.age: int = 0

        # Decision weights (firm-like intelligence)
        self.lending_aggression: float = float(rng.uniform(0.3, 0.8))
        self._consecutive_losses: int = 0

    def step(self):
        if self.defunct:
            return
        self.age += 1
        self.interest_income = 0.0
        self.interest_expense = 0.0
        self.defaults_this_step = 0

        # Collect interest on loans
        self._collect_loan_payments()

        # Pay interest on deposits
        self._pay_deposit_interest()

        # Seek new deposits
        self._attract_deposits()

        # Make new loans
        self._make_loans()

        # Handle bank run conditions
        self._check_bank_run()

        # Profit
        self.profit = self.interest_income - self.interest_expense
        self.wealth += self.profit

        # Learn: adjust lending aggression based on results
        if self.profit > 0:
            self.lending_aggression = min(0.95, self.lending_aggression * 1.01)
            self._consecutive_losses = 0
        else:
            self.lending_aggression = max(0.1, self.lending_aggression * 0.95)
            self._consecutive_losses += 1

        # Bank failure
        if self.wealth < -self.total_deposits * 0.1:
            self._fail()

    def _collect_loan_payments(self):
        """Collect interest and principal payments, handle defaults."""
        dead_loans = []
        for wid, loan in self.loans.items():
            worker = self.model.get_agent_by_id(wid)
            if worker is None or not hasattr(worker, 'wealth'):
                dead_loans.append(wid)
                continue

            # Payment = interest + small principal repayment
            interest = loan['outstanding'] * loan['rate']
            principal_payment = loan['outstanding'] * 0.01  # 1% principal per step
            total_payment = interest + principal_payment

            if worker.wealth >= total_payment:
                worker.wealth -= total_payment
                self.interest_income += interest
                loan['outstanding'] -= principal_payment  # principal goes down
                self.total_loans -= principal_payment
                # Loan fully repaid?
                if loan['outstanding'] <= 1.0:
                    worker.debt = max(0, worker.debt - loan.get('principal', 0))
                    dead_loans.append(wid)
            elif worker.wealth >= interest:
                # Can pay interest but not principal (interest-only)
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
        """Worker defaults on loan. Bank absorbs loss, worker's credit impaired."""
        loss = loan['outstanding']
        self.wealth -= loss * 0.5  # bank absorbs partial loss
        self.defaults_this_step += 1
        self.total_defaults += 1

        # Worker consequences
        worker.debt = max(0, worker.debt - loan.get('principal', 0))
        worker.skill = max(0.05, worker.skill * 0.95)  # credit impairment

    def _pay_deposit_interest(self):
        """Pay interest to depositors."""
        dead_deposits = []
        for wid, amount in self.deposits.items():
            worker = self.model.get_agent_by_id(wid)
            if worker is None or not hasattr(worker, 'wealth'):
                dead_deposits.append(wid)
                continue

            interest = amount * BASE_DEPOSIT_RATE
            if self.wealth >= interest:
                worker.wealth += interest
                worker.income_last_step += interest
                self.interest_expense += interest
                self.wealth -= interest
            # Compound deposits slightly
            self.deposits[wid] = amount * (1 + BASE_DEPOSIT_RATE * 0.5)

        for wid in dead_deposits:
            self.total_deposits -= self.deposits.get(wid, 0)
            del self.deposits[wid]

        self.total_deposits = sum(self.deposits.values())

    def _attract_deposits(self):
        """Workers with surplus wealth deposit at the bank. Samples for performance."""
        if self.defunct or not self.model.workers:
            return

        # Sample a small subset each step instead of scanning everyone
        n_sample = min(20, len(self.model.workers))
        candidates = self.model.rng.choice(self.model.workers, size=n_sample, replace=False)

        for worker in candidates:
            if worker.wealth < DEPOSIT_THRESHOLD:
                continue
            if worker.unique_id in self.deposits:
                continue
            trust = getattr(worker, 'authority_trust', 0.7)
            deposit_prob = 0.05 * trust * min(worker.wealth / 200, 1.0)
            if self.model.rng.random() < deposit_prob:
                deposit_amount = (worker.wealth - DEPOSIT_THRESHOLD) * 0.1
                if deposit_amount > 1.0:
                    worker.wealth -= deposit_amount
                    self.deposits[worker.unique_id] = (
                        self.deposits.get(worker.unique_id, 0) + deposit_amount)
                    self.total_deposits += deposit_amount

    def _make_loans(self):
        """Lend to workers who need credit. Samples for performance."""
        if self.defunct or not self.model.workers:
            return

        available = (self.total_deposits * (1 - RESERVE_REQUIREMENT)
                    + self.capital * MAX_LEVERAGE_RATIO
                    - self.total_loans)
        if available < LOAN_AMOUNT_BASE:
            return

        # Sample a small subset instead of scanning everyone
        n_sample = min(20, len(self.model.workers))
        candidates = self.model.rng.choice(self.model.workers, size=n_sample, replace=False)

        for worker in candidates:
            if available < LOAN_AMOUNT_BASE:
                break
            if worker.unique_id in self.loans:
                continue  # already has a loan
            if worker.wealth > 100:
                continue  # doesn't need a loan
            if worker.debt > 100:
                continue  # too indebted

            # Credit scoring: poorer = higher rate (predatory lending emerges)
            credit_score = min(1.0, worker.wealth / 50.0)
            loan_rate = BASE_LENDING_RATE + RISK_PREMIUM_SLOPE * (1 - credit_score)

            # Lending decision based on aggression
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
        """Depositors with low trust withdraw funds. Sample for performance."""
        if not self.deposits:
            return

        bank_health = self.wealth / max(self.total_deposits, 1)

        # Only check a sample unless bank is stressed
        deposit_ids = list(self.deposits.keys())
        if bank_health > 0.1:
            # Bank is healthy: spot-check 10 depositors
            n_check = min(10, len(deposit_ids))
            check_ids = list(self.model.rng.choice(deposit_ids, size=n_check, replace=False))
        else:
            # Bank is stressed: check all (bank run cascades)
            check_ids = deposit_ids

        withdrawals = []
        for wid in check_ids:
            worker = self.model.get_agent_by_id(wid)
            if worker is None:
                withdrawals.append(wid)
                continue

            trust = getattr(worker, 'authority_trust', 0.7)
            # Low trust OR bank looks weak -> withdraw
            bank_health = self.wealth / max(self.total_deposits, 1)
            if trust < BANK_RUN_TRUST_THRESHOLD or bank_health < 0.05:
                if self.model.rng.random() < 0.1:  # 10% chance per step
                    withdrawals.append(wid)

        for wid in withdrawals:
            amount = self.deposits.get(wid, 0)
            worker = self.model.get_agent_by_id(wid)
            if worker and hasattr(worker, 'wealth'):
                # Can the bank pay?
                if self.wealth >= amount:
                    worker.wealth += amount
                    self.wealth -= amount
                else:
                    # Partial withdrawal (bank is illiquid)
                    partial = max(0, self.wealth * 0.5)
                    worker.wealth += partial
                    self.wealth -= partial
                    # Worker loses the rest (bank can't pay)
            if wid in self.deposits:
                self.total_deposits -= self.deposits[wid]
                del self.deposits[wid]

    def _fail(self):
        """Bank failure: depositors lose a fraction of deposits."""
        # Return what we can to depositors
        if self.total_deposits > 0:
            recovery_rate = max(0, self.wealth / self.total_deposits)
        else:
            recovery_rate = 0

        for wid, amount in self.deposits.items():
            worker = self.model.get_agent_by_id(wid)
            if worker and hasattr(worker, 'wealth'):
                recovered = amount * min(recovery_rate, 0.8)
                worker.wealth += recovered

        # Write off all loans (borrowers keep the money, bank eats the loss)
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


# ---------------------------------------------------------------------------
# Banking metrics
# ---------------------------------------------------------------------------

def compute_banking_metrics(model: "EconomicModel") -> Dict[str, float]:
    """Compute banking system metrics."""
    banks = getattr(model, 'banks', [])
    active_banks = [b for b in banks if not b.defunct]

    if not active_banks:
        return {
            "n_banks": 0,
            "total_bank_deposits": 0.0,
            "total_bank_loans": 0.0,
            "bank_leverage_ratio": 0.0,
            "bank_default_rate": 0.0,
            "bank_profit": 0.0,
            "deposit_concentration": 0.0,
        }

    total_deposits = sum(b.total_deposits for b in active_banks)
    total_loans = sum(b.total_loans for b in active_banks)
    total_capital = sum(b.capital for b in active_banks)
    total_defaults = sum(b.defaults_this_step for b in active_banks)
    total_profit = sum(b.profit for b in active_banks)
    n_loans = sum(len(b.loans) for b in active_banks)

    # Deposit concentration (HHI)
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

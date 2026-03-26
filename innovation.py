"""
innovation.py
-------------
Technology and innovation system for the economic simulation.

Firms invest in R&D, which produces technology improvements that raise
their productivity. Innovation diffuses through the economy via worker
mobility (employees carry knowledge) and proximity (technology spillovers).

Key dynamics produced:
  - Schumpeterian creative destruction: innovative firms displace stagnant ones
  - Skill-biased technical change: high-tech firms need skilled workers
  - Technology diffusion: innovation spreads through labor mobility and proximity
  - Innovation inequality: R&D requires capital, so wealthy firms innovate more
  - Productivity frontiers: economy-wide TFP grows through cumulative innovation

Architecture:
  - Each firm has a tech_level (starts at 1.0, grows through R&D)
  - R&D investment has diminishing returns and stochastic breakthroughs
  - Tech level multiplies production output (compounds with infrastructure)
  - Workers at high-tech firms gain skill faster (learning by doing)
  - Technology spills over to nearby firms (knowledge externality)
  - The economy tracks a technology frontier (max tech across all firms)

Integration:
  - FirmAgent gets tech_level attribute and R&D strategy option
  - Production function multiplied by tech_level
  - Worker skill growth modulated by employer tech_level
  - Called from environment.py step() for diffusion
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from environment import EconomicModel
    from agents import FirmAgent, WorkerAgent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# R&D investment parameters
RD_BASE_COST = 0.05          # fraction of wealth invested when choosing R&D
RD_DIMINISHING_RATE = 0.5    # exponent on investment (sqrt = diminishing returns)
RD_BREAKTHROUGH_PROB = 0.03  # chance of a large jump per R&D step
RD_BREAKTHROUGH_SIZE = 0.15  # size of breakthrough (added to tech_level)
RD_INCREMENTAL_SIZE = 0.005  # base size of incremental improvement

# Technology diffusion
TECH_SPILLOVER_RATE = 0.002  # fraction of tech gap that spills to nearby firms
TECH_SPILLOVER_RADIUS = 8    # grid cells for spillover
WORKER_TECH_TRANSFER = 0.01  # tech knowledge workers carry when changing firms

# Skill-biased technical change
TECH_SKILL_BOOST = 0.003     # extra skill growth per unit of employer tech_level above 1.0
TECH_SKILL_THRESHOLD = 1.5   # tech_level above which low-skill workers struggle

# Technology depreciation (knowledge becomes obsolete)
TECH_DEPRECIATION = 0.001    # fraction lost per step

# Production bonus
# Effective TFP = infrastructure_bonus * tech_level
# So tech_level=1.5 means 50% more productive than baseline


# ---------------------------------------------------------------------------
# Firm innovation functions
# ---------------------------------------------------------------------------

def init_firm_tech(firm: "FirmAgent"):
    """Initialize technology attributes on a firm. Call from FirmAgent.__init__."""
    firm.tech_level = 1.0
    firm.rd_investment_total = 0.0
    firm.innovations_count = 0
    firm.tech_level_at_founding = 1.0


def firm_rd_invest(firm: "FirmAgent") -> float:
    """
    Firm invests in R&D. Returns the tech_level increase.

    Called when the firm chooses the "innovate" strategy.
    Investment has diminishing returns: doubling R&D spending
    does not double innovation. Stochastic breakthroughs
    create the possibility of large jumps.
    """
    rng = firm.model.rng

    # Investment amount: fraction of wealth
    invest = firm.wealth * RD_BASE_COST
    if invest < 1.0 or firm.wealth < invest:
        return 0.0

    firm.wealth -= invest
    firm.rd_investment_total += invest

    # Diminishing returns: sqrt of investment relative to current tech
    # Higher tech = harder to improve further
    effective_invest = invest ** RD_DIMINISHING_RATE
    base_improvement = RD_INCREMENTAL_SIZE * effective_invest / max(firm.tech_level, 1.0)

    # Skill of workers matters: smarter workers = better R&D
    if firm.workers:
        mean_skill = float(np.mean([w.skill for w in firm.workers.values()]))
        skill_multiplier = 0.5 + mean_skill  # range 0.55 to 1.5
    else:
        skill_multiplier = 0.5

    improvement = base_improvement * skill_multiplier

    # Stochastic breakthrough
    if rng.random() < RD_BREAKTHROUGH_PROB:
        improvement += RD_BREAKTHROUGH_SIZE * rng.uniform(0.5, 1.5)
        firm.innovations_count += 1

    firm.tech_level += improvement
    return improvement


def apply_tech_to_production(firm: "FirmAgent", base_output: float) -> float:
    """
    Apply technology multiplier to production output.

    Call this in FirmAgent._produce() to multiply the base output:
        output = apply_tech_to_production(self, output)

    Returns modified output.
    """
    tech = getattr(firm, 'tech_level', 1.0)
    return base_output * max(tech, 0.5)


def apply_tech_skill_effects(worker: "WorkerAgent"):
    """
    Workers at high-tech firms learn faster.
    Workers at very high-tech firms without sufficient skill struggle.

    Call this in WorkerAgent.step() during the skill update section.
    Replaces or supplements the existing skill *= 1.001 line.
    """
    if not worker.employed or worker.employer_id is None:
        return

    firm = worker.model.get_agent_by_id(worker.employer_id)
    if firm is None or not hasattr(firm, 'tech_level'):
        return

    tech = firm.tech_level

    # Learning by doing: faster skill growth at high-tech firms
    if tech > 1.0:
        bonus = TECH_SKILL_BOOST * (tech - 1.0)
        worker.skill = min(1.0, worker.skill + bonus)

    # Skill-biased friction: very high tech + low skill = productivity loss
    # (worker can't operate the equipment, but they're learning)
    # This is already handled by the production function since L uses skill


# ---------------------------------------------------------------------------
# Technology diffusion (called from environment.py)
# ---------------------------------------------------------------------------

def diffuse_technology(model: "EconomicModel"):
    """
    Technology spillovers: firms near high-tech firms learn from them.
    Workers who change jobs carry technology knowledge.

    Call once per step from environment.py, after agent stepping.
    """
    active_firms = [f for f in model.firms if not f.defunct]
    if len(active_firms) < 2:
        return

    # Spatial spillover: nearby firms absorb a fraction of the tech gap
    for firm in active_firms:
        if firm.pos is None or not hasattr(firm, 'tech_level'):
            continue

        pos = (int(firm.pos[0]), int(firm.pos[1]))
        try:
            neighbours = model.grid.get_neighborhood(
                pos, moore=True, include_center=False,
                radius=TECH_SPILLOVER_RADIUS
            )
            nearby_firms = []
            for cell in neighbours:
                for a in model.grid.get_cell_list_contents([cell]):
                    if (hasattr(a, 'tech_level') and hasattr(a, 'defunct')
                            and not a.defunct and a.unique_id != firm.unique_id):
                        nearby_firms.append(a)

            if nearby_firms:
                max_nearby_tech = max(f.tech_level for f in nearby_firms)
                if max_nearby_tech > firm.tech_level:
                    gap = max_nearby_tech - firm.tech_level
                    spillover = gap * TECH_SPILLOVER_RATE
                    firm.tech_level += spillover
        except Exception:
            pass

    # Technology depreciation: knowledge becomes obsolete
    for firm in active_firms:
        if hasattr(firm, 'tech_level') and firm.tech_level > 1.0:
            firm.tech_level = max(1.0, firm.tech_level * (1 - TECH_DEPRECIATION))


def transfer_tech_on_hire(worker: "WorkerAgent", new_firm: "FirmAgent"):
    """
    When a worker moves to a new firm, they carry technology knowledge.
    The new firm's tech_level gets a small boost based on the worker's
    previous employer's tech level.

    Call from FirmAgent.hire_worker() after the worker is added.
    """
    if not hasattr(new_firm, 'tech_level'):
        return

    # Worker carries knowledge from previous employer
    old_firm_id = getattr(worker, '_prev_employer_id', None)
    if old_firm_id is not None:
        old_firm = worker.model.get_agent_by_id(old_firm_id)
        if old_firm and hasattr(old_firm, 'tech_level'):
            tech_gap = old_firm.tech_level - new_firm.tech_level
            if tech_gap > 0:
                transfer = tech_gap * WORKER_TECH_TRANSFER * worker.skill
                new_firm.tech_level += transfer


# ---------------------------------------------------------------------------
# Innovation metrics
# ---------------------------------------------------------------------------

def compute_innovation_metrics(model: "EconomicModel") -> Dict[str, float]:
    """Compute innovation-related metrics for the current step."""
    active_firms = [f for f in model.firms if not f.defunct]

    if not active_firms:
        return {
            "tech_frontier": 1.0,
            "tech_mean": 1.0,
            "tech_min": 1.0,
            "tech_std": 0.0,
            "tech_gini": 0.0,
            "n_innovators": 0,
            "total_rd_investment": 0.0,
            "total_innovations": 0,
        }

    tech_levels = np.array([
        getattr(f, 'tech_level', 1.0) for f in active_firms
    ])

    # Technology Gini: inequality in innovation
    tech_sorted = np.sort(tech_levels)
    n = len(tech_sorted)
    if n > 1 and tech_sorted.sum() > 0:
        tech_gini = float(
            (2 * np.sum(np.arange(1, n + 1) * tech_sorted)
             - (n + 1) * tech_sorted.sum())
            / (n * tech_sorted.sum())
        )
    else:
        tech_gini = 0.0

    # Innovators: firms with tech > 1.1 (above baseline)
    n_innovators = int(np.sum(tech_levels > 1.1))

    return {
        "tech_frontier": float(np.max(tech_levels)),
        "tech_mean": float(np.mean(tech_levels)),
        "tech_min": float(np.min(tech_levels)),
        "tech_std": float(np.std(tech_levels)),
        "tech_gini": tech_gini,
        "n_innovators": n_innovators,
        "total_rd_investment": float(sum(
            getattr(f, 'rd_investment_total', 0) for f in active_firms
        )),
        "total_innovations": int(sum(
            getattr(f, 'innovations_count', 0) for f in active_firms
        )),
    }

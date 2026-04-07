"""
run_parallel.py
---------------
Unified experiment runner for the multi-agent economic simulation.

Launches each run as a separate subprocess for genuine process isolation
and multi-core parallelism. Supports all experiment presets via the
Condition dataclass.

Usage:
    python run_parallel.py                          # default conditions
    python run_parallel.py --preset test2            # vanilla vs topo
    python run_parallel.py --preset arch             # architecture staircase C1-C8
    python run_parallel.py --preset resp             # C12-C15 responsiveness
    python run_parallel.py --preset pa               # C16-C18 production-aware
    python run_parallel.py --preset mitosis          # C21-C22 mitosis
    python run_parallel.py --preset eh               # C23-C24 epistemic health
    python run_parallel.py --preset structural       # C25-C26 structural fixes
    python run_parallel.py --preset comparison       # SUM/NASH/TOPO objective comparison
    python run_parallel.py --workers 6               # explicit worker count
    python run_parallel.py --only C1_baseline        # single condition
    python run_parallel.py --animate                 # generate HTML animations
"""

from __future__ import annotations

import subprocess
import sys
import os
import time
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields

import numpy as np


# ── Condition dataclass ─────────────────────────────────────────

@dataclass
class Condition:
    """A single experimental condition with all feature flags."""
    name: str
    label: str
    objective: str
    use_sevc: bool           # sustainable capitalism (SEVC scoring, innovation)
    use_trust: bool          # trust system active
    trust_noise: float       # noise on trust reads (0 = perfect, 0.1 = fuzzy)
    use_horizon_index: bool  # HI coupled to planner objective
    use_firm_hi: bool        # per-firm horizon index tracking
    gov_type: str            # authoritarian | auth_captured | democratic | demo_captured
    mixed_sevc_ratio: float = 1.0  # fraction of firms that are SEVC (1.0 = all)
    election_weight: float = 0.0   # democratic responsiveness weight
    media_captured: bool = False
    production_aware_E: bool = False
    production_aware_S_pop: bool = False
    ceo_compensation_tied: bool = False
    ceo_base_equals_floor: bool = False
    ceo_equity_tied: bool = False
    capture_normalization: str = "fixed"  # "fixed" | "ema"
    use_capacity_mitosis: bool = True
    government_broadcaster: bool = False
    eh_formula: str = "legacy"            # "legacy" | "paper"
    entrepreneurship_requires_innovation: bool = False
    zombie_firm_cleanup: bool = False
    v_measures_total_emissions: bool = False
    worker_ownership: bool = False        # Task 17: 51/49 worker-majority ownership
    worker_ownership_share: float = 0.51  # fraction owned by workers
    mitosis_trigger: str = "standard"     # "standard" | "dilution"


# ── All conditions ──────────────────────────────────────────────

# Architecture staircase (C1-C8)
C1  = Condition("C1_baseline",       "Vanilla baseline",        "SUM_RAW", False, False, 0.0, False, False, "authoritarian")
C2  = Condition("C2_sevc",           "SEVC only",               "SUM_RAW", True,  False, 0.0, False, False, "authoritarian")
C3  = Condition("C3_sevc_hi",        "SEVC + HI + FirmHI",      "SUM_RAW", True,  False, 0.0, True,  True,  "authoritarian")
C4  = Condition("C4_full_auth",      "Full stack, auth gov",    "SUM_RAW", True,  True,  0.1, True,  True,  "authoritarian")
C5  = Condition("C5_demo_captured",  "Full + captured demo",    "SUM_RAW", True,  True,  0.1, True,  True,  "demo_captured")
C6  = Condition("C6_auth_captured",  "Full + captured auth",    "SUM_RAW", True,  True,  0.1, True,  True,  "auth_captured")
C7  = Condition("C7_democratic",     "Full + clean democracy",  "SUM_RAW", True,  True,  0.1, True,  True,  "democratic")
C8  = Condition("C8_mixed",          "Mixed 50/50 + democracy", "SUM_RAW", True,  True,  0.1, True,  True,  "democratic", mixed_sevc_ratio=0.5)
ARCH_CONDITIONS = [C1, C2, C3, C4, C5, C6, C7, C8]

# Planner SEVC objectives (C9-C11)
C9  = Condition("C9_planner_sevc_democratic",    "Planner SEVC + democracy",     "PLANNER_SEVC", True, True, 0.1, True, True, "democratic")
C10 = Condition("C10_planner_sevc_auth",         "Planner SEVC + authoritarian", "PLANNER_SEVC", True, True, 0.1, True, True, "authoritarian")
C11 = Condition("C11_planner_sevc_demo_captured","Planner SEVC + captured demo", "PLANNER_SEVC", True, True, 0.1, True, True, "demo_captured")

# Responsiveness (C12-C15)
C12 = Condition("C12_responsive_democratic",      "Responsive SEVC democracy",  "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",    election_weight=2.0)
C13 = Condition("C13_responsive_demo_captured",   "Responsive SEVC captured",   "PLANNER_SEVC", True, True, 0.1, True, True, "demo_captured", election_weight=2.0, media_captured=True)
C14 = Condition("C14_pure_technocrat_democratic",  "Technocrat SEVC democracy",  "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",    election_weight=0.0)
C15 = Condition("C15_pure_technocrat_auth",        "Technocrat SEVC auth",       "PLANNER_SEVC", True, True, 0.1, True, True, "authoritarian", election_weight=0.0)
RESP_CONDITIONS = [C12, C13, C14, C15]

# Production-Aware Capital (C16-C18)
C16 = Condition("C16_production_aware_democratic", "Production-aware SEVC + demo", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",    election_weight=2.0, production_aware_E=True, production_aware_S_pop=True)
C17 = Condition("C17_production_aware_no_sevc",    "PA planner + vanilla firms",   "PLANNER_SEVC", False, True, 0.1, True, True, "democratic",   election_weight=2.0, production_aware_S_pop=True)
C18 = Condition("C18_production_aware_captured",   "Production-aware + captured",  "PLANNER_SEVC", True, True, 0.1, True, True, "demo_captured", election_weight=2.0, media_captured=True, production_aware_E=True, production_aware_S_pop=True)
PA_CONDITIONS = [C16, C17, C18]

# CEO Compensation (C19-C20)
C19 = Condition("C19_ceo_tied_democratic", "CEO tied + clean democracy", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",    election_weight=2.0, production_aware_E=True, production_aware_S_pop=True, ceo_compensation_tied=True, ceo_base_equals_floor=True, ceo_equity_tied=True, capture_normalization="ema")
C20 = Condition("C20_ceo_tied_captured",   "CEO tied + captured media",  "PLANNER_SEVC", True, True, 0.1, True, True, "demo_captured", election_weight=2.0, media_captured=True, production_aware_E=True, production_aware_S_pop=True, ceo_compensation_tied=True, ceo_base_equals_floor=True, ceo_equity_tied=True, capture_normalization="ema")

# Mitosis (C21-C22)
C21 = Condition("C21_mitosis_democratic",   "Capacity mitosis + democracy",   "PLANNER_SEVC", True, True, 0.1, True, True, "democratic", election_weight=1.0)
C22 = Condition("C22_no_mitosis_democratic","No mitosis baseline + democracy", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic", election_weight=1.0, use_capacity_mitosis=False)
MITOSIS_CONDITIONS = [C21, C22]

# Epistemic Health (C23-C24)
C23 = Condition("C23_full_structural", "EH overhaul, clean democracy", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",
                election_weight=2.0, production_aware_E=True, production_aware_S_pop=True,
                ceo_compensation_tied=True, ceo_base_equals_floor=True, ceo_equity_tied=True,
                capture_normalization="ema", government_broadcaster=True, eh_formula="paper")
C24 = Condition("C24_full_captured", "EH overhaul, captured demo", "PLANNER_SEVC", True, True, 0.1, True, True, "demo_captured",
                election_weight=2.0, media_captured=True, production_aware_E=True, production_aware_S_pop=True,
                ceo_compensation_tied=True, ceo_base_equals_floor=True, ceo_equity_tied=True,
                capture_normalization="ema", government_broadcaster=True, eh_formula="paper")
EH_CONDITIONS = [C23, C24]

# Structural Fixes (C25-C26)
C25 = Condition("C25_structural_fixes", "Structural fixes, full stack", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",
                election_weight=1.0, production_aware_E=True, production_aware_S_pop=True,
                ceo_compensation_tied=True, ceo_base_equals_floor=True, ceo_equity_tied=True,
                capture_normalization="ema", government_broadcaster=True, eh_formula="paper",
                entrepreneurship_requires_innovation=True, zombie_firm_cleanup=True, v_measures_total_emissions=True)
C26 = Condition("C26_structural_no_ceo", "Structural fixes, no CEO", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",
                election_weight=1.0, production_aware_E=True, production_aware_S_pop=True,
                capture_normalization="ema", government_broadcaster=True, eh_formula="paper",
                entrepreneurship_requires_innovation=True, zombie_firm_cleanup=True, v_measures_total_emissions=True)
STRUCTURAL_CONDITIONS = [C25, C26]

# Worker-Majority Ownership (C27-C28)
C27 = Condition("C27_worker_owned_democratic", "Worker-owned 51/49, clean democracy", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",
                election_weight=1.0, production_aware_S_pop=True,
                capture_normalization="ema", government_broadcaster=True, eh_formula="paper",
                entrepreneurship_requires_innovation=True, zombie_firm_cleanup=True, v_measures_total_emissions=True,
                worker_ownership=True, worker_ownership_share=0.51,
                mitosis_trigger="dilution")
C28 = Condition("C28_worker_owned_captured", "Worker-owned 51/49, captured media", "PLANNER_SEVC", True, True, 0.1, True, True, "demo_captured",
                election_weight=1.0, media_captured=True, production_aware_S_pop=True,
                capture_normalization="ema", government_broadcaster=True, eh_formula="paper",
                entrepreneurship_requires_innovation=True, zombie_firm_cleanup=True, v_measures_total_emissions=True,
                worker_ownership=True, worker_ownership_share=0.51,
                mitosis_trigger="dilution")
WORKER_OWNED_CONDITIONS = [C27, C28]

# Worker Ownership Dose-Response (C29-C30)
C29 = Condition("C29_worker_owned_25_democratic", "Worker-owned 25% minority, clean democracy", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",
                election_weight=1.0, production_aware_S_pop=True,
                capture_normalization="ema", government_broadcaster=True, eh_formula="paper",
                entrepreneurship_requires_innovation=True, zombie_firm_cleanup=True, v_measures_total_emissions=True,
                worker_ownership=True, worker_ownership_share=0.25,
                mitosis_trigger="dilution")
C30 = Condition("C30_worker_owned_75_democratic", "Worker-owned 75% supermajority, clean democracy", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",
                election_weight=1.0, production_aware_S_pop=True,
                capture_normalization="ema", government_broadcaster=True, eh_formula="paper",
                entrepreneurship_requires_innovation=True, zombie_firm_cleanup=True, v_measures_total_emissions=True,
                worker_ownership=True, worker_ownership_share=0.75,
                mitosis_trigger="dilution")
DOSE_RESPONSE_CONDITIONS = [C29, C27, C30]  # ordered 25% → 51% → 75% for dose-response

# Full default set (latest structural conditions)
DEFAULT_CONDITIONS = [C23, C24, C25, C26]

# All conditions combined
ALL_CONDITIONS = ARCH_CONDITIONS + [C9, C10, C11] + RESP_CONDITIONS + PA_CONDITIONS + [C19, C20] + MITOSIS_CONDITIONS + EH_CONDITIONS + STRUCTURAL_CONDITIONS + WORKER_OWNED_CONDITIONS + [C29, C30]

# Test2: vanilla vs full stack
TEST2_CONDITIONS = [
    Condition("vanilla_sum",  "Vanilla SUM (no features)", "SUM_RAW", False, True, 0.15, False, False, "democratic"),
    Condition("topo_sevc_hi", "TOPO + SEVC + HI + FirmHI", "TOPO_X",  True,  True, 0.1,  True,  True,  "democratic"),
]

# Comparison: objective functions
COMPARISON_CONDITIONS = [
    Condition("SUM_RAW",  "Pure aggregate sum",           "SUM_RAW",  True, True, 0.1, False, False, "democratic"),
    Condition("NASH_MIN", "Nash welfare + min-gate HI",   "NASH_MIN", True, True, 0.1, True,  True,  "democratic"),
    Condition("TOPO_X",   "Topology + HI multiplier",     "TOPO_X",   True, True, 0.1, True,  True,  "democratic"),
    Condition("TOPO_MIN", "Topology + min-gate HI",       "TOPO_MIN", True, True, 0.1, True,  True,  "democratic"),
]


# ── Preset definitions ──────────────────────────────────────────

PRESETS = {
    "full":        {"conditions": DEFAULT_CONDITIONS,     "seeds": [42, 101, 137, 202, 256, 303, 404, 505], "steps": 2000, "output_dir": "results/architecture"},
    "all":         {"conditions": ALL_CONDITIONS,         "seeds": [42, 137, 2024],                         "steps": 3000, "output_dir": "results/architecture"},
    "arch":        {"conditions": ARCH_CONDITIONS,        "seeds": [42, 137, 2024],                         "steps": 3000, "output_dir": "results/architecture"},
    "test2":       {"conditions": TEST2_CONDITIONS,       "seeds": [7, 23, 59, 101, 233, 347, 461, 587, 719, 853], "steps": 500,  "output_dir": "results/test_conditions"},
    "resp":        {"conditions": RESP_CONDITIONS,        "seeds": [42, 137, 256, 389, 501, 623, 777, 888], "steps": 2000, "output_dir": "results/responsiveness"},
    "pa":          {"conditions": PA_CONDITIONS,          "seeds": [42, 137, 256, 389, 501, 623, 777, 888], "steps": 2000, "output_dir": "results/production_aware"},
    "mitosis":     {"conditions": MITOSIS_CONDITIONS,     "seeds": [42, 137, 256, 389, 501, 623, 777, 888], "steps": 2000, "output_dir": "results/mitosis"},
    "eh":          {"conditions": EH_CONDITIONS,          "seeds": [42, 137, 256, 389, 501, 623, 777, 888], "steps": 3000, "output_dir": "results/epistemic_health"},
    "structural":  {"conditions": STRUCTURAL_CONDITIONS,  "seeds": [42, 137, 256, 389, 501, 623, 777, 888], "steps": 3000, "output_dir": "results/structural"},
    "worker_owned": {"conditions": WORKER_OWNED_CONDITIONS, "seeds": [42, 137, 256, 389, 501, 623, 777, 888], "steps": 3000, "output_dir": "results/worker_owned"},
    "dose_response": {"conditions": DOSE_RESPONSE_CONDITIONS, "seeds": [42, 137, 256, 389, 501, 623, 777, 888], "steps": 3000, "output_dir": "results/dose_response"},
    "comparison":  {"conditions": COMPARISON_CONDITIONS,  "seeds": [42, 137, 2024],                         "steps": 3000, "output_dir": "results/comparison"},
}


# ── Feature flag injection (for notebook / in-process use) ──────

def configure_model(model, condition: Condition):
    """Inject feature flags into the model after construction."""
    model.use_sevc = condition.use_sevc
    model.use_trust = condition.use_trust
    model.trust_noise = condition.trust_noise
    model.use_horizon_index = condition.use_horizon_index
    model.use_firm_hi = condition.use_firm_hi
    model.gov_type = condition.gov_type
    model._trust_frozen = not condition.use_trust
    model.election_weight = condition.election_weight
    model.use_capacity_mitosis = condition.use_capacity_mitosis
    model.production_aware_E = condition.production_aware_E
    model.production_aware_S_pop = condition.production_aware_S_pop
    model.ceo_compensation_tied = condition.ceo_compensation_tied
    model.ceo_base_equals_floor = condition.ceo_base_equals_floor
    model.ceo_equity_tied = condition.ceo_equity_tied
    model.capture_normalization = condition.capture_normalization
    model.use_government_broadcaster = condition.government_broadcaster
    model.eh_formula = condition.eh_formula
    model.entrepreneurship_requires_innovation = condition.entrepreneurship_requires_innovation
    model.zombie_firm_cleanup = condition.zombie_firm_cleanup
    model.v_measures_total_emissions = condition.v_measures_total_emissions
    model.worker_ownership = condition.worker_ownership
    model.worker_ownership_share = condition.worker_ownership_share
    model.mitosis_trigger = condition.mitosis_trigger

    # Apply worker ownership to all existing firms
    if condition.worker_ownership:
        for firm in model.firms:
            firm.worker_ownership_share = condition.worker_ownership_share
            firm.investor_ownership_share = 1.0 - condition.worker_ownership_share

    if not condition.use_sevc:
        for firm in model.firms:
            firm.is_sevc = False
            if hasattr(firm, 'strategy_weights'):
                for k in firm.strategy_weights:
                    firm.strategy_weights[k] = 0.2
                if 'innovate' in firm.strategy_weights:
                    del firm.strategy_weights['innovate']
            if hasattr(firm, 'tech_level'):
                firm.tech_level = 1.0
    else:
        for firm in model.firms:
            if not hasattr(firm, 'floor_history'):
                from collections import deque
                firm.floor_history = deque(maxlen=100)
                firm.horizon_index = 1.0

    if condition.mixed_sevc_ratio < 1.0 and condition.use_sevc:
        n_vanilla = int(len(model.firms) * (1.0 - condition.mixed_sevc_ratio))
        if n_vanilla > 0:
            indices = list(range(len(model.firms)))
            model.rng.shuffle(indices)
            for i in indices[:n_vanilla]:
                firm = model.firms[i]
                firm.is_sevc = False
                if hasattr(firm, 'strategy_weights'):
                    for k in firm.strategy_weights:
                        firm.strategy_weights[k] = 0.2

    if condition.gov_type == 'demo_captured' or condition.media_captured:
        if hasattr(model, 'news_firms'):
            for nf in model.news_firms:
                if hasattr(nf, 'accuracy') and nf.accuracy > 0.5:
                    nf.accuracy = 0.3
                    nf.audience_capture = 0.6
                    break

    if not condition.use_trust:
        for agent in (model.schedule.agents if hasattr(model, 'schedule') else []):
            if hasattr(agent, 'trust_score'):
                agent.trust_score = 0.5

    return model


_patches_applied = False

def apply_patches():
    """Apply runtime patches that check model feature flags."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    try:
        import sustainable_capitalism as sc_module
        _original_choose = sc_module.sustainable_choose_strategy
        _original_learn = sc_module.sustainable_learn_from_outcome

        def patched_choose(firm):
            if not getattr(firm, 'is_sevc', True) or not getattr(firm.model, 'use_sevc', True):
                strategies = list(firm.strategy_weights.keys())
                if strategies:
                    rng = firm.model.rng if hasattr(firm.model, 'rng') else np.random.default_rng()
                    return strategies[rng.integers(len(strategies))]
                return "expand"
            return _original_choose(firm)

        def patched_learn(firm, strategy, reward):
            if not getattr(firm, 'is_sevc', True) or not getattr(firm.model, 'use_sevc', True):
                return
            return _original_learn(firm, strategy, reward)

        sc_module.sustainable_choose_strategy = patched_choose
        sc_module.sustainable_learn_from_outcome = patched_learn
    except ImportError:
        pass

    try:
        import innovation as inno_module
        _original_rd = inno_module.firm_rd_invest
        _original_diffuse = inno_module.diffuse_technology

        def patched_rd(firm):
            if not getattr(firm.model, 'use_sevc', True) and not getattr(firm, 'is_sevc', True):
                return
            return _original_rd(firm)

        def patched_diffuse(model):
            if not getattr(model, 'use_sevc', True):
                return
            return _original_diffuse(model)

        inno_module.firm_rd_invest = patched_rd
        inno_module.diffuse_technology = patched_diffuse
    except ImportError:
        pass

    try:
        import planner as planner_module
        if hasattr(planner_module, '_get_horizon_index'):
            _original_hi = planner_module._get_horizon_index

            def patched_hi(model):
                if not getattr(model, 'use_horizon_index', True):
                    return 1.0
                return _original_hi(model)

            planner_module._get_horizon_index = patched_hi
    except ImportError:
        pass


# ── Subprocess script template ──────────────────────────────────

SCRIPT_TEMPLATE = r'''
import sys, os, time
import numpy as np
import pandas as pd

os.chdir("@@CWD@@")
sys.path.insert(0, "@@CWD@@")

from environment import EconomicModel

# -- Patches --

import sustainable_capitalism as sc_module
_oc = sc_module.sustainable_choose_strategy
_ol = sc_module.sustainable_learn_from_outcome

def pc(firm):
    if not getattr(firm.model, 'use_sevc', True):
        strats = list(firm.strategy_weights.keys())
        return strats[firm.model.rng.integers(len(strats))] if strats else "expand"
    return _oc(firm)

def pl(firm, strategy, reward):
    if not getattr(firm.model, 'use_sevc', True):
        return
    return _ol(firm, strategy, reward)

sc_module.sustainable_choose_strategy = pc
sc_module.sustainable_learn_from_outcome = pl

import innovation as inno
_ord = inno.firm_rd_invest
_odf = inno.diffuse_technology

def prd(firm):
    if not getattr(firm.model, 'use_innovation', True):
        return
    return _ord(firm)

def pdf(model):
    if not getattr(model, 'use_innovation', True):
        return
    return _odf(model)

inno.firm_rd_invest = prd
inno.diffuse_technology = pdf

import trust as trust_mod
_ot = trust_mod.update_trust_scores

def pt(model):
    if getattr(model, '_trust_frozen', False):
        return
    _ot(model)
    noise = getattr(model, 'trust_noise', 0.0)
    if noise > 0:
        rng = model.rng
        for agent in (model.schedule.agents if hasattr(model, 'schedule') else []):
            if hasattr(agent, 'trust_score'):
                agent.trust_score = float(np.clip(
                    agent.trust_score + rng.normal(0, noise), 0.0, 1.0))

trust_mod.update_trust_scores = pt

import planner as planner_mod
if hasattr(planner_mod, '_get_horizon_index'):
    _ohi = planner_mod._get_horizon_index
    def phi(model):
        if not getattr(model, 'use_horizon_index', True):
            return 1.0
        return _ohi(model)
    planner_mod._get_horizon_index = phi

# -- Config (injected) --

COND_NAME = "@@NAME@@"
OBJECTIVE = "@@OBJECTIVE@@"
USE_SEVC = @@USE_SEVC@@
USE_INNOVATION = @@USE_INNOVATION@@
USE_TRUST = @@USE_TRUST@@
TRUST_NOISE = @@TRUST_NOISE@@
USE_HI = @@USE_HI@@
USE_FIRM_HI = @@USE_FIRM_HI@@
GOV_TYPE = "@@GOV_TYPE@@"
MIXED_SEVC_RATIO = @@MIXED_SEVC_RATIO@@
ELECTION_WEIGHT = @@ELECTION_WEIGHT@@
MEDIA_CAPTURED = @@MEDIA_CAPTURED@@
PRODUCTION_AWARE_E    = @@PRODUCTION_AWARE_E@@
PRODUCTION_AWARE_S_POP = @@PRODUCTION_AWARE_S_POP@@
CEO_COMPENSATION_TIED = @@CEO_COMPENSATION_TIED@@
CEO_BASE_EQUALS_FLOOR = @@CEO_BASE_EQUALS_FLOOR@@
CEO_EQUITY_TIED       = @@CEO_EQUITY_TIED@@
CAPTURE_NORMALIZATION = "@@CAPTURE_NORMALIZATION@@"
USE_CAPACITY_MITOSIS = @@USE_CAPACITY_MITOSIS@@
GOVERNMENT_BROADCASTER = @@GOVERNMENT_BROADCASTER@@
EH_FORMULA = "@@EH_FORMULA@@"
ENTREPRENEURSHIP_REQUIRES_INNOVATION = @@ENTREPRENEURSHIP_REQUIRES_INNOVATION@@
ZOMBIE_FIRM_CLEANUP = @@ZOMBIE_FIRM_CLEANUP@@
V_MEASURES_TOTAL_EMISSIONS = @@V_MEASURES_TOTAL_EMISSIONS@@
SEED = @@SEED@@
N_STEPS = @@N_STEPS@@
ANIMATE = @@ANIMATE@@
ANIM_SUBSAMPLE = @@ANIM_SUBSAMPLE@@
OUTPUT_DIR = "@@OUTPUT_DIR@@"
GRID_SIZE = @@GRID_SIZE@@
N_WORKERS_SIM = @@N_WORKERS_SIM@@
N_FIRMS = @@N_FIRMS@@
N_LANDOWNERS = @@N_LANDOWNERS@@

# -- Run --

model = EconomicModel(
    seed=SEED, grid_width=GRID_SIZE, grid_height=GRID_SIZE,
    n_workers=N_WORKERS_SIM, n_firms=N_FIRMS, n_landowners=N_LANDOWNERS,
    objective=OBJECTIVE,
)

model._collect_animation = ANIMATE
model.use_sevc = USE_SEVC
model.use_innovation = USE_INNOVATION
model.use_trust = USE_TRUST
model.trust_noise = TRUST_NOISE
model.use_horizon_index = USE_HI
model.use_firm_hi = USE_FIRM_HI
model.gov_type = GOV_TYPE
model.election_weight = ELECTION_WEIGHT
model.production_aware_E     = PRODUCTION_AWARE_E
model.production_aware_S_pop = PRODUCTION_AWARE_S_POP
model.ceo_compensation_tied  = CEO_COMPENSATION_TIED
model.ceo_base_equals_floor  = CEO_BASE_EQUALS_FLOOR
model.ceo_equity_tied        = CEO_EQUITY_TIED
model.capture_normalization  = CAPTURE_NORMALIZATION
model.use_capacity_mitosis = USE_CAPACITY_MITOSIS
model.use_government_broadcaster = GOVERNMENT_BROADCASTER
model.eh_formula = EH_FORMULA
model.entrepreneurship_requires_innovation = ENTREPRENEURSHIP_REQUIRES_INNOVATION
model.zombie_firm_cleanup = ZOMBIE_FIRM_CLEANUP
model.v_measures_total_emissions = V_MEASURES_TOTAL_EMISSIONS

if not USE_SEVC:
    for firm in model.firms:
        firm.is_sevc = False
        if hasattr(firm, 'strategy_weights'):
            for k in firm.strategy_weights:
                firm.strategy_weights[k] = 0.2
else:
    from collections import deque as _dq
    for firm in model.firms:
        if not hasattr(firm, 'floor_history'):
            firm.floor_history = _dq(maxlen=100)
            firm.horizon_index = 1.0

if MIXED_SEVC_RATIO < 1.0 and USE_SEVC:
    n_vanilla = int(len(model.firms) * (1.0 - MIXED_SEVC_RATIO))
    if n_vanilla > 0:
        indices = list(range(len(model.firms)))
        model.rng.shuffle(indices)
        for i in indices[:n_vanilla]:
            firm = model.firms[i]
            firm.is_sevc = False
            if hasattr(firm, 'strategy_weights'):
                for k in firm.strategy_weights:
                    firm.strategy_weights[k] = 0.2

if (GOV_TYPE == "demo_captured" or MEDIA_CAPTURED) and hasattr(model, 'news_firms'):
    for nf in model.news_firms:
        if hasattr(nf, 'accuracy') and nf.accuracy > 0.5:
            nf.accuracy = 0.3; nf.audience_capture = 0.6
            break

if not USE_TRUST:
    model._trust_frozen = True
    for agent in (model.schedule.agents if hasattr(model, 'schedule') else []):
        if hasattr(agent, 'trust_score'):
            agent.trust_score = 0.5
else:
    model._trust_frozen = False

t0 = time.time()
for step in range(N_STEPS):
    model.step()
    if (step + 1) % 500 == 0:
        elapsed_so_far = time.time() - t0
        print(str(step + 1) + " (" + "{:.0f}".format(elapsed_so_far) + "s)", end=" ", flush=True)

elapsed = time.time() - t0
print("done " + "{:.1f}".format(elapsed) + "s")

os.makedirs(OUTPUT_DIR + "/raw_data", exist_ok=True)
df = pd.DataFrame(model.metrics_history)
df["condition"] = COND_NAME
df["objective"] = OBJECTIVE
df["use_sevc"] = USE_SEVC
df["use_innovation"] = USE_INNOVATION
df["use_trust"] = USE_TRUST
df["trust_noise"] = TRUST_NOISE
df["use_hi"] = USE_HI
df["use_firm_hi"] = USE_FIRM_HI
df["gov_type"] = GOV_TYPE
df["seed"] = SEED
outpath = OUTPUT_DIR + "/raw_data/" + COND_NAME + "_seed" + str(SEED) + ".parquet"
df.to_parquet(outpath, index=False)
print("Saved: " + outpath)

if ANIMATE and model.animation_frames:
    anim_dir = OUTPUT_DIR + "/animations"
    os.makedirs(anim_dir, exist_ok=True)
    try:
        from animate import generate_animation_html
        anim_path = anim_dir + "/" + COND_NAME + "_seed" + str(SEED) + ".html"
        generate_animation_html(
            model.animation_frames,
            output_path=anim_path,
            grid_size=GRID_SIZE,
            title=COND_NAME + " (seed=" + str(SEED) + ")",
            subsample=ANIM_SUBSAMPLE,
        )
        fsize = os.path.getsize(anim_path) / (1024 * 1024)
        print("Animation: " + anim_path + " (" + "{:.1f}".format(fsize) + " MB)")
    except Exception as e:
        print("Animation failed: " + str(e))
'''


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_script(condition: Condition, seed: int, n_steps: int,
                animate: bool = False, anim_subsample: int = 2,
                output_dir: str = "results/architecture",
                grid_size: int = 80, n_workers_sim: int = 400,
                n_firms: int = 20, n_landowners: int = 15) -> str:
    """Generate a self-contained run script from a Condition object."""
    s = SCRIPT_TEMPLATE
    s = s.replace("@@CWD@@", _SCRIPT_DIR)
    s = s.replace("@@NAME@@", condition.name)
    s = s.replace("@@OBJECTIVE@@", condition.objective)
    s = s.replace("@@USE_SEVC@@", str(condition.use_sevc))
    s = s.replace("@@USE_INNOVATION@@", str(condition.use_sevc))  # innovation tracks SEVC
    s = s.replace("@@USE_TRUST@@", str(condition.use_trust))
    s = s.replace("@@TRUST_NOISE@@", str(condition.trust_noise))
    s = s.replace("@@USE_HI@@", str(condition.use_horizon_index))
    s = s.replace("@@USE_FIRM_HI@@", str(condition.use_firm_hi))
    s = s.replace("@@GOV_TYPE@@", condition.gov_type)
    s = s.replace("@@MIXED_SEVC_RATIO@@", str(condition.mixed_sevc_ratio))
    s = s.replace("@@ELECTION_WEIGHT@@", str(condition.election_weight))
    s = s.replace("@@MEDIA_CAPTURED@@", str(condition.media_captured))
    s = s.replace("@@PRODUCTION_AWARE_E@@", str(condition.production_aware_E))
    s = s.replace("@@PRODUCTION_AWARE_S_POP@@", str(condition.production_aware_S_pop))
    s = s.replace("@@CEO_COMPENSATION_TIED@@", str(condition.ceo_compensation_tied))
    s = s.replace("@@CEO_BASE_EQUALS_FLOOR@@", str(condition.ceo_base_equals_floor))
    s = s.replace("@@CEO_EQUITY_TIED@@", str(condition.ceo_equity_tied))
    s = s.replace("@@CAPTURE_NORMALIZATION@@", condition.capture_normalization)
    s = s.replace("@@USE_CAPACITY_MITOSIS@@", str(condition.use_capacity_mitosis))
    s = s.replace("@@GOVERNMENT_BROADCASTER@@", str(condition.government_broadcaster))
    s = s.replace("@@EH_FORMULA@@", condition.eh_formula)
    s = s.replace("@@ENTREPRENEURSHIP_REQUIRES_INNOVATION@@", str(condition.entrepreneurship_requires_innovation))
    s = s.replace("@@ZOMBIE_FIRM_CLEANUP@@", str(condition.zombie_firm_cleanup))
    s = s.replace("@@V_MEASURES_TOTAL_EMISSIONS@@", str(condition.v_measures_total_emissions))
    s = s.replace("@@SEED@@", str(seed))
    s = s.replace("@@N_STEPS@@", str(n_steps))
    s = s.replace("@@ANIMATE@@", str(animate))
    s = s.replace("@@ANIM_SUBSAMPLE@@", str(anim_subsample))
    s = s.replace("@@OUTPUT_DIR@@", output_dir)
    s = s.replace("@@GRID_SIZE@@", str(grid_size))
    s = s.replace("@@N_WORKERS_SIM@@", str(n_workers_sim))
    s = s.replace("@@N_FIRMS@@", str(n_firms))
    s = s.replace("@@N_LANDOWNERS@@", str(n_landowners))
    return s


def run_one(job):
    """Write script, run as subprocess, return result."""
    condition, seed, n_steps, animate, anim_subsample, output_dir, grid_size, n_workers_sim, n_firms, n_landowners = job
    label = condition.name + "/seed" + str(seed)

    script = make_script(condition, seed, n_steps,
                         animate=animate, anim_subsample=anim_subsample,
                         output_dir=output_dir, grid_size=grid_size,
                         n_workers_sim=n_workers_sim, n_firms=n_firms,
                         n_landowners=n_landowners)

    script_path = "/tmp/run_" + condition.name + "_s" + str(seed) + ".py"
    with open(script_path, "w") as f:
        f.write(script)

    print("  START: " + label, flush=True)
    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=7200,
            cwd=os.getcwd(),
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print("  FAIL:  " + label + " (" + str(int(elapsed)) + "s)")
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-5:]:
                print("    " + line)
            return {"name": condition.name, "seed": seed, "status": "FAIL",
                    "elapsed": elapsed, "error": result.stderr[-300:]}
        else:
            print("  DONE:  " + label + " (" + str(int(elapsed)) + "s)")
            return {"name": condition.name, "seed": seed, "status": "OK",
                    "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        print("  TIMEOUT: " + label)
        return {"name": condition.name, "seed": seed, "status": "TIMEOUT", "elapsed": 7200}


# ── Comparison reporting ────────────────────────────────────────

def print_comparison(output_dir):
    """Load raw data and print head-to-head comparison."""
    import pandas as pd
    raw_dir = output_dir + "/raw_data"
    if not os.path.isdir(raw_dir):
        return
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".parquet"))
    if not files:
        return
    all_data = pd.concat([pd.read_parquet(raw_dir + "/" + f) for f in files], ignore_index=True)

    max_step = all_data["step"].max()
    tail = all_data[all_data["step"] >= max_step * 0.8]
    conditions = sorted(tail["condition"].unique())

    metrics = [
        ("worker_min",            "Floor Wealth",     ".1f",  "higher"),
        ("worker_mean",           "Worker Mean",      ".1f",  "higher"),
        ("worker_gini",           "Worker Gini",      ".3f",  "lower"),
        ("unemployment_rate",     "Unemployment",     ".1%",  "lower"),
        ("agency_floor",          "Agency Floor",     ".2f",  "higher"),
        ("horizon_index",         "Horizon Index",    ".3f",  "higher"),
        ("mean_firm_floor",       "Firm SEVC Floor",  ".3f",  "higher"),
        ("mean_firm_floor_raw",   "Firm Floor Raw",   ".3f",  "higher"),
        ("mean_firm_floor_norm",  "Firm Floor Norm",  ".3f",  "higher"),
        ("mean_firm_hi",          "Firm HI",          ".3f",  "higher"),
        ("n_firms",               "Firms",            ".1f",  "higher"),
        ("total_production",      "Production",       ".0f",  "higher"),
        ("total_pollution",       "Pollution",        ".0f",  "lower"),
        ("legitimacy_mean",       "Legitimacy",       ".3f",  "higher"),
        ("mean_aggression",       "Aggression",       ".3f",  "lower"),
        ("mean_conflict",         "Conflict",         ".4f",  "lower"),
        ("planner_sevc_score",    "Planner SEVC",     ".3f",  "higher"),
        ("planner_S_pop",         "Planner S",        ".3f",  "higher"),
        ("planner_E_pop",         "Planner E",        ".3f",  "higher"),
        ("planner_V_pop",         "Planner V",        ".3f",  "higher"),
        ("planner_C_pop",         "Planner C",        ".3f",  "higher"),
        ("epistemic_health_mean", "EH Mean",          ".3f",  "higher"),
        ("epistemic_health_floor","EH Floor",         ".3f",  "higher"),
        ("trust_planner",         "Planner Trust",    ".3f",  "higher"),
        ("trust_institutional",   "Inst. Trust",      ".3f",  "higher"),
        ("crime_events",          "Crime/step",       ".1f",  "lower"),
    ]

    print("\n" + "=" * 80)
    print("  HEAD-TO-HEAD COMPARISON (steady-state mean +/- std across seeds)")
    print("=" * 80)

    header = "{:<20}".format("Metric")
    for c in conditions:
        header += "  {:>26}".format(c)
    print(header)
    print("-" * len(header))

    for key, label, fmt, direction in metrics:
        if key not in tail.columns:
            continue
        row = "{:<20}".format(label)
        for c in conditions:
            subset = tail[tail["condition"] == c][key]
            mean_v = subset.mean()
            std_v = subset.std()
            row += "  {:>12{fmt}} +/- {:>7{fmt}}".format(mean_v, std_v, fmt=fmt)
        print(row)

    print()
    summary_dir = output_dir + "/summary"
    os.makedirs(summary_dir, exist_ok=True)
    all_data.to_parquet(summary_dir + "/all_data.parquet", index=False)
    print("Saved: " + summary_dir + "/all_data.parquet")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified parallel experiment runner for the economic simulation.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = auto)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this condition name")
    parser.add_argument("--steps", type=int, default=0,
                        help="Steps per run (0 = use preset default)")
    parser.add_argument("--animate", action="store_true",
                        help="Generate HTML animation for each run")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Animation frame subsample rate (default: 2)")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(PRESETS.keys()),
                        help="Experiment preset (default: 'full')")
    parser.add_argument("--grid-size", type=int, default=80,
                        help="Grid dimension (default: 80)")
    parser.add_argument("--n-workers-sim", type=int, default=400,
                        help="Number of worker agents (default: 400)")
    parser.add_argument("--n-firms", type=int, default=20,
                        help="Number of firms (default: 20)")
    parser.add_argument("--n-landowners", type=int, default=15,
                        help="Number of landowners (default: 15)")
    args = parser.parse_args()

    preset_name = args.preset or "full"
    preset = PRESETS[preset_name]
    conditions = preset["conditions"]
    seeds = preset["seeds"]
    n_steps = args.steps if args.steps > 0 else preset["steps"]
    output_dir = preset["output_dir"]

    n_workers = args.workers
    if n_workers <= 0:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    # Build jobs
    jobs = []
    for cond in conditions:
        if args.only and cond.name != args.only:
            continue
        for seed in seeds:
            jobs.append((cond, seed, n_steps, args.animate, args.subsample,
                         output_dir, args.grid_size, args.n_workers_sim,
                         args.n_firms, args.n_landowners))

    print("=" * 70)
    print("  PARALLEL EXPERIMENT RUNNER")
    print("  Preset: " + preset_name)
    print("  Jobs: " + str(len(jobs)))
    print("  Workers: " + str(n_workers))
    print("  Steps: " + str(n_steps))
    print("  Grid: " + str(args.grid_size) + "x" + str(args.grid_size))
    print("  Animate: " + str(args.animate))
    print("  Output: " + output_dir)
    print("=" * 70)
    for cond in conditions:
        if args.only and cond.name != args.only:
            continue
        flags = []
        if cond.use_sevc: flags.append("SEVC")
        if cond.use_trust: flags.append("Trust(" + str(cond.trust_noise) + ")")
        if cond.use_horizon_index: flags.append("HI")
        if cond.use_firm_hi: flags.append("FirmHI")
        flags.append("gov=" + cond.gov_type)
        if cond.mixed_sevc_ratio < 1.0: flags.append("Mix(" + str(cond.mixed_sevc_ratio) + ")")
        if cond.election_weight > 0: flags.append("Resp(" + str(cond.election_weight) + ")")
        if cond.media_captured: flags.append("MediaCap")
        if cond.production_aware_E or cond.production_aware_S_pop: flags.append("PA")
        if cond.ceo_compensation_tied: flags.append("CEO(norm=" + cond.capture_normalization + ")")
        if cond.government_broadcaster: flags.append("GovBC")
        if cond.eh_formula != "legacy": flags.append("EH=" + cond.eh_formula)
        if cond.entrepreneurship_requires_innovation: flags.append("InnoReq")
        if cond.zombie_firm_cleanup: flags.append("Zombie")
        if cond.v_measures_total_emissions: flags.append("VTotal")
        print("  " + cond.name + ": " + cond.objective + " [" + ", ".join(flags) + "]")
    print()

    os.makedirs(output_dir + "/raw_data", exist_ok=True)

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                job = futures[future]
                print("  EXCEPTION: " + str(job[0].name) + "/seed" + str(job[1]) + ": " + str(e))

    elapsed = time.time() - t0
    ok = sum(1 for r in results if r.get("status") == "OK")
    fail = len(results) - ok

    print()
    print("=" * 70)
    print("  Complete: " + str(len(results)) + " runs in " +
          str(int(elapsed)) + "s (" + "{:.1f}".format(elapsed / 60) + " min)")
    print("  OK: " + str(ok) + "  FAIL: " + str(fail))
    print("=" * 70)

    import pandas as pd
    pd.DataFrame(results).to_csv(output_dir + "/run_log.csv", index=False)

    print_comparison(output_dir)


if __name__ == "__main__":
    main()

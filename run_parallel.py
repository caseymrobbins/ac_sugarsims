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

try:
    import numpy as np
except ImportError:
    import types as _np_types, sys as _np_sys
    np = _np_types.ModuleType("numpy")
    class _NpRng:
        def shuffle(self, seq): return None
    class _NpRandom:
        @staticmethod
        def default_rng(_seed=None): return _NpRng()
    np.random = _NpRandom()
    _np_sys.modules["numpy"] = np


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
    instrument_caps: str = "standard"     # Task 19: "standard" | "uncapped"
    deficit_spending: bool = False        # Task 19: allow debt-financed planner spending
    bottleneck_policy: str = "off"        # off | enabled | aggressive
    bottleneck_dynamic_capture: bool = False


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

# Uncapped Planner Instruments (C31-C32)
_SHARED_PLANNER_KWARGS = dict(
    production_aware_S_pop=False,
    capture_normalization="ema", government_broadcaster=True, eh_formula="paper",
    entrepreneurship_requires_innovation=False, zombie_firm_cleanup=True, v_measures_total_emissions=True,
)
C33 = Condition("C33_nash_uncapped_planner", "Uncapped planner instruments + deficit", "NASH", True, True, 0.1, True, True, "democratic",
                election_weight=1.0, instrument_caps="uncapped", deficit_spending=True, **_SHARED_PLANNER_KWARGS)
C34 = Condition("C34_topo_uncapped_planner", "Uncapped planner instruments + deficit", "TOPO", True, True, 0.1, True, True, "democratic",
                election_weight=1.0, instrument_caps="uncapped", deficit_spending=True, **_SHARED_PLANNER_KWARGS)
UNCAPPED_CONDITIONS = [C33, C34]

# Full default set (latest structural conditions)
DEFAULT_CONDITIONS = [C23, C24, C25, C26]

# All conditions combined
ALL_CONDITIONS = ARCH_CONDITIONS + [C9, C10, C11] + RESP_CONDITIONS + PA_CONDITIONS + [C19, C20] + MITOSIS_CONDITIONS + EH_CONDITIONS + STRUCTURAL_CONDITIONS + WORKER_OWNED_CONDITIONS + [C29, C30] + UNCAPPED_CONDITIONS

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

# Bottleneck Regulation Experiment (Hypothesis test)
B1 = Condition("B1_baseline_no_reg", "Baseline (no bottleneck regulation)", "NASH",
               True, True, 0.1, True, True, "democratic", election_weight=1.0,
               bottleneck_policy="off", instrument_caps="uncapped", deficit_spending=True, **_SHARED_PLANNER_KWARGS)
B2 = Condition("B2_bottleneck_reg", "Bottleneck regulation enabled", "NASH",
               True, True, 0.1, True, True, "democratic", election_weight=1.0,
               bottleneck_policy="enabled", instrument_caps="uncapped", deficit_spending=True, **_SHARED_PLANNER_KWARGS)
B3 = Condition("B3_bottleneck_aggressive", "Aggressive anti-bottleneck policy", "NASH",
               True, True, 0.1, True, True, "democratic", election_weight=1.0,
               bottleneck_policy="aggressive", bottleneck_dynamic_capture=True, instrument_caps="uncapped", deficit_spending=True, **_SHARED_PLANNER_KWARGS)
BOTTLENECK_CONDITIONS = [B1, B2, B3]

# Focused BICF sanity-test condition (single-condition smoke preset)
BICF_TEST_CONDITIONS = [
    Condition(
        "BICF_test_condition",
        "BICF qualification stress test",
        "NASH",
        True,
        True,
        0.1,
        True,
        True,
        "democratic",
        election_weight=1.0,
        bottleneck_policy="enabled",
        bottleneck_dynamic_capture=True,
    )
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
    "uncapped":    {"conditions": UNCAPPED_CONDITIONS,    "seeds": [42, 137, 256, 389, 501, 623, 777, 888], "steps": 3000, "output_dir": "results/uncapped"},
    "comparison":  {"conditions": COMPARISON_CONDITIONS,  "seeds": [42, 137, 2024],                         "steps": 3000, "output_dir": "results/comparison"},
    "bottleneck":  {"conditions": BOTTLENECK_CONDITIONS,  "seeds": [42, 137, 256, 389, 501, 623, 777, 888], "steps": 3000, "output_dir": "results/"},
    "bicf_test":   {"conditions": BICF_TEST_CONDITIONS,   "seeds": [389, 501, 623, 777, 888],                          "steps": 500,  "output_dir": "results/bicf_test"},
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
    model.instrument_caps = condition.instrument_caps
    model.deficit_spending = condition.deficit_spending
    model.bottleneck_regulation_policy = condition.bottleneck_policy
    model.bottleneck_dynamic_capture = condition.bottleneck_dynamic_capture
    if condition.bottleneck_policy == "aggressive":
        model.max_profit_margin_cap = 0.10
        model.bottleneck_open_access_bonus = 0.10
        model.bottleneck_breakup_threshold = 25
    elif condition.bottleneck_policy == "enabled":
        model.max_profit_margin_cap = 0.15
        model.bottleneck_open_access_bonus = 0.05
        model.bottleneck_breakup_threshold = 50

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
COND_LABEL = "@@LABEL@@"
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
WORKER_OWNERSHIP = @@WORKER_OWNERSHIP@@
WORKER_OWNERSHIP_SHARE = @@WORKER_OWNERSHIP_SHARE@@
MITOSIS_TRIGGER = "@@MITOSIS_TRIGGER@@"
INSTRUMENT_CAPS = "@@INSTRUMENT_CAPS@@"
DEFICIT_SPENDING = @@DEFICIT_SPENDING@@
BOTTLENECK_POLICY = "@@BOTTLENECK_POLICY@@"
BOTTLENECK_DYNAMIC_CAPTURE = @@BOTTLENECK_DYNAMIC_CAPTURE@@
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
model.worker_ownership = WORKER_OWNERSHIP
model.worker_ownership_share = WORKER_OWNERSHIP_SHARE
model.mitosis_trigger = MITOSIS_TRIGGER
model.instrument_caps = INSTRUMENT_CAPS
model.deficit_spending = DEFICIT_SPENDING
model.bottleneck_regulation_policy = BOTTLENECK_POLICY
model.bottleneck_dynamic_capture = BOTTLENECK_DYNAMIC_CAPTURE
if BOTTLENECK_POLICY == "aggressive":
    model.max_profit_margin_cap = 0.10
    model.bottleneck_open_access_bonus = 0.10
    model.bottleneck_breakup_threshold = 25
elif BOTTLENECK_POLICY == "enabled":
    model.max_profit_margin_cap = 0.15
    model.bottleneck_open_access_bonus = 0.05
    model.bottleneck_breakup_threshold = 50

if WORKER_OWNERSHIP:
    for firm in model.firms:
        firm.worker_ownership_share = WORKER_OWNERSHIP_SHARE
        firm.investor_ownership_share = 1.0 - WORKER_OWNERSHIP_SHARE

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
df["condition_label"] = COND_LABEL
df["objective"] = OBJECTIVE
df["use_sevc"] = USE_SEVC
df["use_innovation"] = USE_INNOVATION
df["use_trust"] = USE_TRUST
df["trust_noise"] = TRUST_NOISE
df["use_hi"] = USE_HI
df["use_firm_hi"] = USE_FIRM_HI
df["gov_type"] = GOV_TYPE
df["bottleneck_policy"] = BOTTLENECK_POLICY
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
    s = s.replace("@@LABEL@@", condition.label)
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
    s = s.replace("@@WORKER_OWNERSHIP@@", str(condition.worker_ownership))
    s = s.replace("@@WORKER_OWNERSHIP_SHARE@@", str(condition.worker_ownership_share))
    s = s.replace("@@MITOSIS_TRIGGER@@", condition.mitosis_trigger)
    s = s.replace("@@INSTRUMENT_CAPS@@", condition.instrument_caps)
    s = s.replace("@@DEFICIT_SPENDING@@", str(condition.deficit_spending))
    s = s.replace("@@BOTTLENECK_POLICY@@", condition.bottleneck_policy)
    s = s.replace("@@BOTTLENECK_DYNAMIC_CAPTURE@@", str(condition.bottleneck_dynamic_capture))
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


def export_episode_and_overall_summaries(output_dir):
    import pandas as pd
    from metrics import episode_summary

    raw_dir = output_dir + "/raw_data"
    if not os.path.isdir(raw_dir):
        return
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".parquet"))
    if not files:
        return

    episodes = []
    for f in files:
        df = pd.read_parquet(raw_dir + "/" + f)
        if df.empty:
            continue
        cond = str(df["condition"].iloc[0]) if "condition" in df.columns else f
        label = str(df["condition_label"].iloc[0]) if "condition_label" in df.columns else cond
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else -1
        metrics_hist = df.to_dict("records")
        ep = episode_summary(metrics_hist)
        ep["condition"] = cond
        ep["condition_label"] = label
        ep["seed"] = seed
        episodes.append(ep)

    if not episodes:
        return

    ep_df = pd.DataFrame(episodes)
    ep_out = output_dir + "/episode_summaries.csv"
    ep_df.to_csv(ep_out, index=False)
    print("Saved: " + ep_out)

    core = [
        "total_production", "tech_frontier", "mean_wage", "employment_rate", "n_firms",
        "hhi", "top_firm_share", "mean_capture_ratio", "total_rent_collected",
        "all_gini", "worker_gini",
        "mean_trust", "legitimacy_mean", "identity_conflict_index", "epistemic_health_mean",
        "frac_monopoly", "frac_bottleneck"
    ]
    rows = []
    for cond, grp in ep_df.groupby("condition"):
        row = {"condition": cond, "n_seeds": int(len(grp))}
        for k in core:
            if k in grp.columns:
                row[k + "_mean"] = float(grp[k].mean())
                row[k + "_std"] = float(grp[k].std())
        rows.append(row)
    overall = pd.DataFrame(rows).sort_values("condition")
    overall_out = output_dir + "/overall_run_summary.csv"
    overall.to_csv(overall_out, index=False)
    print("Saved: " + overall_out)


def generate_bottleneck_diagnostics(output_dir):
    import pandas as pd
    import matplotlib.pyplot as plt

    raw_dir = output_dir + "/raw_data"
    if not os.path.isdir(raw_dir):
        return
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".parquet"))
    if not files:
        return
    all_data = pd.concat([pd.read_parquet(raw_dir + "/" + f) for f in files], ignore_index=True)
    if all_data.empty:
        return

    plot_dir = output_dir + "/diagnostic_plots"
    os.makedirs(plot_dir, exist_ok=True)

    def _agg(x, y):
        cols = ["condition", x, y]
        d = all_data[cols].dropna()
        if d.empty:
            return None
        return d.groupby("condition", as_index=False)[[x, y]].mean()

    specs = [
        ("hhi", "total_production", "hhi_vs_production.png", "HHI vs Production"),
        ("monopoly_detected", "all_gini", "monopoly_fraction_vs_inequality.png", "Monopoly Fraction vs Inequality"),
        ("n_firms", "all_gini", "firm_count_vs_inequality.png", "Firm Count vs Inequality"),
        ("mean_capture_ratio", "population_growth_rate", "rent_extraction_vs_growth.png", "Rent Extraction vs Growth"),
        ("all_gini", "population_growth_rate", "growth_vs_inequality_phase_map.png", "Growth vs Inequality Phase Map"),
    ]

    for x, y, fname, title in specs:
        if x not in all_data.columns or y not in all_data.columns:
            continue
        d = _agg(x, y)
        if d is None or d.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for _, r in d.iterrows():
            ax.scatter(r[x], r[y], s=80, label=r["condition"])
            ax.annotate(r["condition"], (r[x], r[y]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir + "/" + fname, dpi=150)
        plt.close(fig)
        print("Saved: " + plot_dir + "/" + fname)


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
        if cond.bottleneck_policy != "off": flags.append("Bottleneck=" + cond.bottleneck_policy)
        if cond.bottleneck_dynamic_capture: flags.append("DynCapture")
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
    export_episode_and_overall_summaries(output_dir)
    generate_bottleneck_diagnostics(output_dir)


# ── Inline test suite ───────────────────────────────────────────
# Run with:  python run_parallel.py --test

import sys as _sys
import types as _types
import unittest as _unittest
from types import SimpleNamespace as _SimpleNamespace

# Provide a tiny numpy stub so the suite runs without a real numpy install.
def _ensure_numpy_stub():
    if "numpy" not in _sys.modules or not hasattr(_sys.modules["numpy"], "array"):
        fake = _types.ModuleType("numpy")
        class _Rng:
            def shuffle(self, seq): return None
        class _Random:
            @staticmethod
            def default_rng(_seed=None): return _Rng()
        fake.random = _Random()
        _sys.modules["numpy"] = fake


class _DummyFirm:
    def __init__(self):
        self.worker_ownership_share = 0.0
        self.investor_ownership_share = 1.0
        self.is_sevc = True
        self.strategy_weights = {"expand": 0.6, "innovate": 0.4}
        self.tech_level = 1.2


class _DummyModel:
    def __init__(self, n_firms=3):
        self.firms = [_DummyFirm() for _ in range(n_firms)]
        self.schedule = _SimpleNamespace(agents=[])
        import numpy as _np
        self.rng = _np.random.default_rng(123)


# ── run_parallel condition / preset tests ───────────────────────

class TestRunParallelConditions(_unittest.TestCase):
    def test_bicf_test_preset_registered(self):
        self.assertIn("bicf_test", PRESETS)
        preset = PRESETS["bicf_test"]
        self.assertEqual(len(preset["conditions"]), 1)
        self.assertEqual(preset["conditions"][0].name, "BICF_test_condition")

    def test_bottleneck_preset_registered(self):
        self.assertIn("bottleneck", PRESETS)
        names = [c.name for c in PRESETS["bottleneck"]["conditions"]]
        self.assertEqual(names, ["B1_baseline_no_reg", "B2_bottleneck_reg", "B3_bottleneck_aggressive"])

    def test_aggressive_bottleneck_policy_sets_caps(self):
        # B3 aggressive = tighter margin cap (rent signal) + larger open_access_bonus.
        # "Aggressive" means larger entrant grant and longer vesting, not faster breakup.
        model = _DummyModel()
        m = configure_model(model, B3)
        self.assertEqual(m.bottleneck_regulation_policy, "aggressive")
        self.assertEqual(m.max_profit_margin_cap, 0.10)       # used to size rent signal
        self.assertEqual(m.bottleneck_open_access_bonus, 0.10) # production boost to entrant

    def test_enabled_bottleneck_policy_sets_moderate_caps(self):
        model = _DummyModel()
        m = configure_model(model, B2)
        self.assertEqual(m.bottleneck_regulation_policy, "enabled")
        self.assertEqual(m.max_profit_margin_cap, 0.15)
        self.assertEqual(m.bottleneck_open_access_bonus, 0.05)

    def test_off_policy_sets_no_caps(self):
        model = _DummyModel()
        m = configure_model(model, B1)
        self.assertEqual(m.bottleneck_regulation_policy, "off")

    def test_worker_ownership_applies_to_existing_firms(self):
        model = _DummyModel(n_firms=2)
        cond = Condition(
            "test_worker_owned", "worker owned", "PLANNER_SEVC",
            True, True, 0.1, True, True, "democratic",
            worker_ownership=True, worker_ownership_share=0.75,
        )
        m = configure_model(model, cond)
        for firm in m.firms:
            self.assertEqual(firm.worker_ownership_share, 0.75)
            self.assertEqual(firm.investor_ownership_share, 0.25)

    def test_bicf_test_condition_flags(self):
        cond = PRESETS["bicf_test"]["conditions"][0]
        self.assertEqual(cond.bottleneck_policy, "enabled")
        self.assertTrue(cond.bottleneck_dynamic_capture)
        self.assertTrue(cond.use_sevc)
        self.assertEqual(cond.objective, "NASH")

    def test_all_preset_conditions_have_valid_bottleneck_policy(self):
        valid = {"off", "enabled", "aggressive"}
        for cond in ALL_CONDITIONS + BOTTLENECK_CONDITIONS + BICF_TEST_CONDITIONS:
            self.assertIn(cond.bottleneck_policy, valid, msg=cond.name)


# ── BICF policy-evaluation tests ────────────────────────────────

from bicf import (
    AcquisitionEvent,
    EntrantProfile,
    IndustryProfile,
    aggregate_improvement,
    check_supply_chain_neutrality,
    compute_relative_improvements,
    evaluate_acquisition_safeguard,
    evaluate_bicf_qualification,
    incentive_package,
    innovation_tier,
    is_bottleneck_industry,
    passes_market_participation,
    required_improvement_threshold,
    supplier_retaliation_levy,
)


def _capital_industry(**kw):
    defaults = dict(
        sector_type="capital_intensive",
        high_capital_barriers=True,
        concentrated_market_share=True,
        chokepoint_infrastructure=True,
        persistent_rent_extraction=True,
        low_innovation_rates=False,
    )
    defaults.update(kw)
    return IndustryProfile(**defaults)


def _good_entrant(**kw):
    defaults = dict(
        incumbent_ownership_share=0.10,
        sells_to_end_customers=True,
        price_vs_incumbent_median=0.98,
        incumbent_linked_revenue_share=0.20,
        market_share_after_5y=0.03,
        board_control_by_incumbent=False,
    )
    defaults.update(kw)
    return EntrantProfile(**defaults)


class TestBottleneckClassification(_unittest.TestCase):
    def test_four_signals_qualifies(self):
        self.assertTrue(is_bottleneck_industry(_capital_industry()))

    def test_five_signals_qualifies(self):
        self.assertTrue(is_bottleneck_industry(_capital_industry(low_innovation_rates=True)))

    def test_three_signals_does_not_qualify(self):
        ind = _capital_industry(concentrated_market_share=False, chokepoint_infrastructure=False)
        self.assertFalse(is_bottleneck_industry(ind))


class TestInnovationQualification(_unittest.TestCase):
    def test_threshold_map(self):
        self.assertEqual(required_improvement_threshold("capital_intensive"), 0.15)
        self.assertEqual(required_improvement_threshold("regulated"), 0.20)
        self.assertEqual(required_improvement_threshold("service_technology"), 0.25)

    def test_unknown_sector_raises(self):
        with self.assertRaises(ValueError):
            required_improvement_threshold("unknown")

    def test_aggregate_uses_min(self):
        # min() — must clear the bar on every metric simultaneously
        improvements = {"cost": 0.20, "energy": 0.15, "emissions": 0.30, "throughput": 0.10, "waste": 0.05}
        self.assertEqual(aggregate_improvement(improvements), 0.05)

    def test_aggregate_empty(self):
        self.assertEqual(aggregate_improvement({}), 0.0)

    def test_aggregate_exposes_negatives(self):
        # Negatives are NOT clipped; a regressing metric must fail the entrant
        self.assertEqual(aggregate_improvement({"cost": -0.10, "energy": 0.20}), -0.10)

    def test_aggregate_negative_fails_qualification(self):
        # A single regressing metric collapses the aggregate below any positive threshold
        industry = _capital_industry()
        improvements = {"cost": 0.20, "emissions": 0.25, "throughput": -0.05}
        result = evaluate_bicf_qualification(industry, _good_entrant(), improvements)
        self.assertFalse(result["innovation_pass"])
        self.assertFalse(result["qualified"])

    def test_tier_assignment_anchored_to_sector_floor(self):
        # capital_intensive floor = 0.15: below floor → tier 0, floor → tier 1, ≥0.25 → tier 2
        self.assertEqual(innovation_tier(0.14, "capital_intensive"), 0)
        self.assertEqual(innovation_tier(0.15, "capital_intensive"), 1)
        self.assertEqual(innovation_tier(0.249, "capital_intensive"), 1)
        self.assertEqual(innovation_tier(0.25, "capital_intensive"), 2)

    def test_tier_alignment_no_unqualified_tier_one(self):
        # 12% is below the 15% capital_intensive floor → must be tier 0, not tier 1
        self.assertEqual(innovation_tier(0.12, "capital_intensive"), 0)
        # same gain clears the 10% floor of... wait, min sector threshold is 15%.
        # regulated floor = 20%: 12% is still below → tier 0
        self.assertEqual(innovation_tier(0.12, "regulated"), 0)

    def test_tier_varies_by_sector(self):
        # 18% clears capital_intensive (15%) but not regulated (20%)
        self.assertEqual(innovation_tier(0.18, "capital_intensive"), 1)
        self.assertEqual(innovation_tier(0.18, "regulated"), 0)


class TestMarketParticipationAndIndependence(_unittest.TestCase):
    def test_passes_when_all_criteria_met(self):
        self.assertTrue(passes_market_participation(_good_entrant()))

    def test_fails_excessive_incumbent_ownership(self):
        self.assertFalse(passes_market_participation(_good_entrant(incumbent_ownership_share=0.21)))

    def test_fails_board_control(self):
        self.assertFalse(passes_market_participation(_good_entrant(board_control_by_incumbent=True)))

    def test_fails_no_direct_sales(self):
        self.assertFalse(passes_market_participation(_good_entrant(sells_to_end_customers=False)))

    def test_fails_price_above_median(self):
        self.assertFalse(passes_market_participation(_good_entrant(price_vs_incumbent_median=1.01)))

    def test_fails_excessive_incumbent_revenue(self):
        self.assertFalse(passes_market_participation(_good_entrant(incumbent_linked_revenue_share=0.31)))

    def test_fails_insufficient_market_share(self):
        self.assertFalse(passes_market_participation(_good_entrant(market_share_after_5y=0.019)))

    def test_boundary_market_share_2pct(self):
        self.assertTrue(passes_market_participation(_good_entrant(market_share_after_5y=0.02)))


class TestIncentiveStructure(_unittest.TestCase):
    def test_not_qualified_no_incentives(self):
        self.assertFalse(any(incentive_package(2, False).values()))

    def test_tier_zero_no_incentives(self):
        self.assertFalse(any(incentive_package(0, True).values()))

    def test_tier_one_incentives(self):
        pkg = incentive_package(1, True)
        self.assertTrue(pkg["tax_reduction"])
        self.assertTrue(pkg["regulatory_fast_track"])
        self.assertFalse(pkg["innovation_grant"])
        self.assertFalse(pkg["legal_defense"])

    def test_tier_two_incentives(self):
        pkg = incentive_package(2, True)
        self.assertTrue(pkg["innovation_grant"])
        self.assertTrue(pkg["legal_defense"])


class TestAcquisitionSafeguards(_unittest.TestCase):
    def test_compliant_acquisition(self):
        r = evaluate_acquisition_safeguard(
            AcquisitionEvent(technology_remains_available=True, technology_suppressed=False))
        self.assertTrue(r["compliant"])
        self.assertIsNone(r["remedy"])

    def test_suppressed_triggers_open_license_remedy(self):
        r = evaluate_acquisition_safeguard(
            AcquisitionEvent(technology_remains_available=True, technology_suppressed=True))
        self.assertFalse(r["compliant"])
        self.assertEqual(r["remedy"], "mandatory_open_license")

    def test_unavailable_triggers_open_license_remedy(self):
        r = evaluate_acquisition_safeguard(
            AcquisitionEvent(technology_remains_available=False, technology_suppressed=False))
        self.assertFalse(r["compliant"])
        self.assertEqual(r["remedy"], "mandatory_open_license")

    def test_open_license_required_flag_default(self):
        event = AcquisitionEvent(technology_remains_available=True, technology_suppressed=False)
        self.assertTrue(event.open_license_required)


class TestSupplierRetaliationProtection(_unittest.TestCase):
    def test_retaliatory_cancellation_levied(self):
        r = supplier_retaliation_levy(500_000.0, False)
        self.assertEqual(r["incumbent_tax"], 500_000.0)
        self.assertEqual(r["entrant_transfer"], 500_000.0)

    def test_legitimate_exemption_no_levy(self):
        r = supplier_retaliation_levy(500_000.0, True)
        self.assertEqual(r["incumbent_tax"], 0.0)

    def test_zero_contract_no_levy(self):
        r = supplier_retaliation_levy(0.0, False)
        self.assertEqual(r["incumbent_tax"], 0.0)


class TestSupplyChainNeutrality(_unittest.TestCase):
    def test_critical_supplier_denying_access_violates(self):
        self.assertFalse(check_supply_chain_neutrality(True, True))

    def test_critical_supplier_granting_access_ok(self):
        self.assertTrue(check_supply_chain_neutrality(True, False))

    def test_non_critical_supplier_may_deny(self):
        self.assertTrue(check_supply_chain_neutrality(False, True))


class TestDynamicBenchmarking(_unittest.TestCase):
    def test_lower_is_better(self):
        r = compute_relative_improvements(
            {"cost": 80.0, "emissions": 150.0},
            {"cost": 100.0, "emissions": 200.0},
            lower_is_better={"cost", "emissions"},
        )
        self.assertAlmostEqual(r["cost"], 0.20)
        self.assertAlmostEqual(r["emissions"], 0.25)

    def test_higher_is_better(self):
        r = compute_relative_improvements(
            {"throughput": 120.0}, {"throughput": 100.0}, lower_is_better=set()
        )
        self.assertAlmostEqual(r["throughput"], 0.20)

    def test_default_treats_cost_as_lower_is_better(self):
        # cost is in DEFAULT_LOWER_IS_BETTER → lower entrant value = positive improvement
        r = compute_relative_improvements({"cost": 80.0}, {"cost": 100.0})
        self.assertAlmostEqual(r["cost"], 0.20)

    def test_default_does_not_treat_throughput_as_lower_is_better(self):
        # throughput is NOT in DEFAULT_LOWER_IS_BETTER → higher entrant value = positive improvement
        r = compute_relative_improvements({"throughput": 120.0}, {"throughput": 100.0})
        self.assertAlmostEqual(r["throughput"], 0.20)
        # If throughput were wrongly treated as lower-is-better this would be -0.20

    def test_default_throughput_regression_is_negative(self):
        # Entrant with worse throughput must produce a negative improvement
        r = compute_relative_improvements({"throughput": 80.0}, {"throughput": 100.0})
        self.assertAlmostEqual(r["throughput"], -0.20)

    def test_missing_metric_defaults_to_zero_improvement(self):
        r = compute_relative_improvements({}, {"cost": 100.0})
        self.assertAlmostEqual(r["cost"], 0.0)

    def test_zero_baseline_skipped(self):
        r = compute_relative_improvements({"cost": 10.0}, {"cost": 0.0})
        self.assertNotIn("cost", r)

    def test_dynamic_pipeline_qualifies(self):
        industry = _capital_industry()
        entrant_profile = _good_entrant()
        baseline = {"cost_per_unit": 100.0, "energy_kwh": 500.0, "co2_kg": 200.0}
        actuals  = {"cost_per_unit": 82.0,  "energy_kwh": 410.0, "co2_kg": 158.0}
        # All three are lower-is-better; pass explicitly so direction is unambiguous
        improvements = compute_relative_improvements(
            actuals, baseline,
            lower_is_better={"cost_per_unit", "energy_kwh", "co2_kg"},
        )
        result = evaluate_bicf_qualification(industry, entrant_profile, improvements)
        self.assertTrue(result["qualified"])


class TestEndToEndBICF(_unittest.TestCase):
    def setUp(self):
        self.industry = _capital_industry()

    def test_qualified_tier_one_entrant(self):
        # min() across all metrics = 0.15 (throughput), exactly at the capital_intensive floor
        improvements = {"production_cost": 0.18, "energy_efficiency": 0.22,
                        "emissions": 0.19, "throughput": 0.15, "waste": 0.16}
        result = evaluate_bicf_qualification(self.industry, _good_entrant(), improvements)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["aggregate_improvement"], 0.15)
        self.assertEqual(result["tier"], 1)
        self.assertTrue(result["incentives"]["tax_reduction"])
        self.assertFalse(result["incentives"]["innovation_grant"])

    def test_one_weak_metric_blocks_qualification(self):
        # min() means a single metric below the floor disqualifies the entrant
        improvements = {"production_cost": 0.20, "energy_efficiency": 0.22,
                        "emissions": 0.19, "throughput": 0.14, "waste": 0.18}
        result = evaluate_bicf_qualification(self.industry, _good_entrant(), improvements)
        self.assertFalse(result["innovation_pass"])
        self.assertFalse(result["qualified"])

    def test_high_gain_entrant_gets_full_incentives(self):
        # min() across all metrics = 0.26, which is ≥ 0.25 → tier 2
        improvements = {"cost": 0.30, "energy": 0.28, "emissions": 0.35, "throughput": 0.26}
        result = evaluate_bicf_qualification(
            self.industry,
            _good_entrant(incumbent_ownership_share=0.05, price_vs_incumbent_median=0.90),
            improvements,
        )
        self.assertTrue(result["qualified"])
        self.assertEqual(result["tier"], 2)
        self.assertTrue(result["incentives"]["innovation_grant"])
        self.assertTrue(result["incentives"]["legal_defense"])

    def test_non_bottleneck_does_not_qualify(self):
        weak = _capital_industry(concentrated_market_share=False, chokepoint_infrastructure=False)
        result = evaluate_bicf_qualification(weak, _good_entrant(), {"cost": 0.30})
        self.assertFalse(result["qualified"])


if __name__ == "__main__":
    if len(_sys.argv) > 1 and _sys.argv[1] == "--test":
        _ensure_numpy_stub()
        _sys.argv = [_sys.argv[0]]   # strip --test so unittest doesn't choke
        _unittest.main(verbosity=2)
    else:
        main()

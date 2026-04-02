"""
run_architecture_experiment.py
------------------------------
Tests the contribution of each architectural layer and governance type:

  C1: Vanilla baseline (no SEVC, no trust, authoritarian)
  C2: SEVC only
  C3: SEVC + HI + Firm HI
  C4: Full stack, authoritarian government
  C5: Full stack, captured democracy (demo_captured)
  C6: Full stack, captured authoritarian (auth_captured)
  C7: Full stack, clean democracy
  C8: Mixed 50/50 SEVC/Vanilla + democracy

Each condition x 3 seeds (42, 137, 2024) = 24 runs at 3000 steps.

Staircase isolates each layer:
  C1 vs C2: SEVC effect
  C2 vs C3: HI + Firm HI effect
  C3 vs C4: Trust effect under authoritarian gov
  C4 vs C7: Authoritarian vs democratic
  C4 vs C6: Authoritarian vs auth_captured (elite distortion)
  C7 vs C5: Democratic vs demo_captured (media capture)
  C8: Mixed population competition dynamics

Usage:
    python run_architecture_experiment.py
    python run_architecture_experiment.py --parallel 4

Output:
    results/architecture/
        raw_data/   - per-step metrics as parquet
        summary/    - comparison tables
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import traceback
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ── Experimental conditions ──────────────────────────────────────

@dataclass
class Condition:
    """A single experimental condition."""
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
    election_weight: float = 0.0   # democratic responsiveness weight (0=technocrat, 2=responsive)
    media_captured: bool = False    # force-capture a news firm at start
    production_aware_E: bool = False      # capture ratio as E component (Task 11)
    production_aware_S_pop: bool = False  # economy-wide capture ratio in S_pop (Task 11)
    ceo_compensation_tied: bool = False   # CEO bonus = profit×10%×sevc_floor (Task 12)
    ceo_base_equals_floor: bool = False   # CEO base = lowest worker wage (Task 12)
    ceo_equity_tied: bool = False         # CEO equity mark-to-market at book×sevc_floor (Task 12)
    capture_normalization: str = "fixed"  # "fixed" | "ema" for E capture_score reference (Task 12)


CONDITIONS = [
    Condition("C1_baseline",       "Vanilla baseline",       "SUM_RAW", False, False, 0.0, False, False, "authoritarian"),
    Condition("C2_sevc",           "SEVC only",              "SUM_RAW", True,  False, 0.0, False, False, "authoritarian"),
    Condition("C3_sevc_hi",        "SEVC + HI + FirmHI",     "SUM_RAW", True,  False, 0.0, True,  True,  "authoritarian"),
    Condition("C4_full_auth",      "Full stack, auth gov",   "SUM_RAW", True,  True,  0.1, True,  True,  "authoritarian"),
    Condition("C5_demo_captured",  "Full + captured demo",   "SUM_RAW", True,  True,  0.1, True,  True,  "demo_captured"),
    Condition("C6_auth_captured",  "Full + captured auth",   "SUM_RAW", True,  True,  0.1, True,  True,  "auth_captured"),
    Condition("C7_democratic",     "Full + clean democracy",  "SUM_RAW", True,  True,  0.1, True,  True,  "democratic"),
    Condition("C8_mixed",          "Mixed 50/50 + democracy", "SUM_RAW", True,  True,  0.1, True,  True,  "democratic", mixed_sevc_ratio=0.5),
    Condition("C9_planner_sevc_democratic",   "Planner SEVC + democracy",      "PLANNER_SEVC", True, True, 0.1, True, True, "democratic"),
    Condition("C10_planner_sevc_auth",        "Planner SEVC + authoritarian",  "PLANNER_SEVC", True, True, 0.1, True, True, "authoritarian"),
    Condition("C11_planner_sevc_demo_captured","Planner SEVC + captured demo", "PLANNER_SEVC", True, True, 0.1, True, True, "demo_captured"),
    Condition("C12_responsive_democratic",     "Responsive SEVC democracy",   "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",    election_weight=2.0),
    Condition("C13_responsive_demo_captured",  "Responsive SEVC captured",    "PLANNER_SEVC", True, True, 0.1, True, True, "demo_captured", election_weight=2.0, media_captured=True),
    Condition("C14_pure_technocrat_democratic", "Technocrat SEVC democracy",   "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",    election_weight=0.0),
    Condition("C15_pure_technocrat_auth",       "Technocrat SEVC auth",        "PLANNER_SEVC", True, True, 0.1, True, True, "authoritarian", election_weight=0.0),
    # Task 11: Production-Aware Capital
    Condition("C16_production_aware_democratic", "Production-aware SEVC + demo", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",   election_weight=2.0, production_aware_E=True,  production_aware_S_pop=True),
    Condition("C17_production_aware_no_sevc",    "PA planner + vanilla firms",   "PLANNER_SEVC", False, True, 0.1, True, True, "democratic",  election_weight=2.0, production_aware_E=False, production_aware_S_pop=True),
    Condition("C18_production_aware_captured",   "Production-aware + captured",  "PLANNER_SEVC", True, True, 0.1, True, True, "demo_captured", election_weight=2.0, media_captured=True, production_aware_E=True, production_aware_S_pop=True),
    # Task 12: CEO Compensation Tied to SEVC Floor
    Condition("C19_ceo_tied_democratic", "CEO tied + clean democracy", "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",    election_weight=2.0, production_aware_E=True, production_aware_S_pop=True, ceo_compensation_tied=True, ceo_base_equals_floor=True, ceo_equity_tied=True, capture_normalization="ema"),
    Condition("C20_ceo_tied_captured",   "CEO tied + captured media",  "PLANNER_SEVC", True, True, 0.1, True, True, "demo_captured", election_weight=2.0, media_captured=True, production_aware_E=True, production_aware_S_pop=True, ceo_compensation_tied=True, ceo_base_equals_floor=True, ceo_equity_tied=True, capture_normalization="ema"),
    # Task 13: Capacity-driven mitosis conditions
    Condition("C21_mitosis_democratic",  "Capacity mitosis + democracy",  "PLANNER_SEVC", True, True, 0.1, True, True, "democratic",    election_weight=1.0),
    Condition("C22_no_mitosis_democratic","No mitosis baseline + democracy","PLANNER_SEVC", True, True, 0.1, True, True, "democratic",   election_weight=1.0),
]

# Seeds for Task 11 production-aware conditions (8 seeds as specified)
SEEDS_PA = [42, 137, 256, 389, 501, 623, 777, 888]
N_STEPS_PA = 2000

SEEDS = [42, 137, 2024]
N_STEPS = 3000
N_WORKERS = 400
N_FIRMS = 20
N_LANDOWNERS = 15
GRID_SIZE = 80

OUTPUT_DIR = "results/architecture"


# ── Feature flag injection ───────────────────────────────────────

def configure_model(model, condition: Condition):
    """
    Inject feature flags into the model after construction.

    These flags are checked by agents and subsystems at runtime.
    The model is constructed normally, then we disable features
    by setting flags that the patched code checks.
    """
    # Feature flags on the model object
    model.use_sevc = condition.use_sevc
    model.use_trust = condition.use_trust
    model.trust_noise = condition.trust_noise
    model.use_horizon_index = condition.use_horizon_index
    model.use_firm_hi = condition.use_firm_hi
    model.gov_type = condition.gov_type
    model._trust_frozen = not condition.use_trust
    model.election_weight = getattr(condition, 'election_weight', 0.0)
    # Task 13: capacity mitosis (disabled for C22 no-mitosis baseline)
    model.use_capacity_mitosis = not condition.name.startswith("C22_no_mitosis")
    model.production_aware_E    = getattr(condition, 'production_aware_E', False)
    model.production_aware_S_pop = getattr(condition, 'production_aware_S_pop', False)
    model.ceo_compensation_tied = getattr(condition, 'ceo_compensation_tied', False)
    model.ceo_base_equals_floor = getattr(condition, 'ceo_base_equals_floor', False)
    model.ceo_equity_tied       = getattr(condition, 'ceo_equity_tied', False)
    model.capture_normalization = getattr(condition, 'capture_normalization', 'fixed')

    # If SEVC is disabled, reset all firms to vanilla behavior
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
        # Even vanilla firms in SEVC-enabled runs track firm_hi
        for firm in model.firms:
            if not hasattr(firm, 'floor_history'):
                from collections import deque
                firm.floor_history = deque(maxlen=100)
                firm.horizon_index = 1.0

    # Mixed population: randomly assign some firms as vanilla (Task 4)
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

    # Force-capture a news firm for demo_captured or explicit media_captured
    if (condition.gov_type == 'demo_captured' or getattr(condition, 'media_captured', False)):
        if hasattr(model, 'news_firms'):
            for nf in model.news_firms:
                if hasattr(nf, 'accuracy') and nf.accuracy > 0.5:
                    nf.accuracy = 0.3; nf.audience_capture = 0.6
                    break  # capture just one

    # If trust is disabled, set all trust scores to neutral 0.5
    # and mark them as frozen so update_trust_scores skips them
    if not condition.use_trust:
        for agent in model.schedule.agents if hasattr(model, 'schedule') else []:
            if hasattr(agent, 'trust_score'):
                agent.trust_score = 0.5
    else:
        model._trust_frozen = False

    return model


# ── Monkey-patches for feature flags ─────────────────────────────

_patches_applied = False

def apply_patches():
    """
    Apply runtime patches that check model feature flags.

    This is called once before any runs. The patches wrap existing
    functions to check flags before executing.
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    # ── Patch 1: Trust engine available ──
    try:
        import trust as trust_module
        print("  [patch] Trust engine: OK")
    except ImportError:
        print("  [patch] Trust engine not found, using fallback trust defaults")

    # ── Patch 2: Sustainable capitalism disable ──
    #
    # When use_sevc is False, the firm's step should skip SEVC
    # strategy selection and use simple profit-maximizing behavior.

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
        print("  [patch] Sustainable capitalism toggle: OK")
    except ImportError:
        print("  [patch] Sustainable capitalism module not found, skipping")

    # ── Patch 3: Innovation disable ──

    try:
        import innovation as inno_module

        _original_rd = inno_module.firm_rd_invest
        _original_diffuse = inno_module.diffuse_technology

        def patched_rd(firm):
            if not getattr(firm.model, 'use_sevc', True) and not getattr(firm, 'is_sevc', True):
                return  # No R&D in vanilla mode
            return _original_rd(firm)

        def patched_diffuse(model):
            if not getattr(model, 'use_sevc', True):
                return
            return _original_diffuse(model)

        inno_module.firm_rd_invest = patched_rd
        inno_module.diffuse_technology = patched_diffuse
        print("  [patch] Innovation toggle: OK")
    except ImportError:
        print("  [patch] Innovation module not found, skipping")

    # ── Patch 4: Horizon Index disable ──
    #
    # When use_horizon_index is False, _get_horizon_index in planner
    # always returns 1.0 (no effect on reward).

    try:
        import planner as planner_module

        # Find and patch the HI helper
        if hasattr(planner_module, '_get_horizon_index'):
            _original_hi = planner_module._get_horizon_index

            def patched_hi(model):
                if not getattr(model, 'use_horizon_index', True):
                    return 1.0  # neutral, no effect
                return _original_hi(model)

            planner_module._get_horizon_index = patched_hi
            print("  [patch] Horizon Index toggle: OK")
        else:
            print("  [patch] _get_horizon_index not found in planner module")
    except ImportError:
        print("  [patch] Planner module not found, skipping HI patches")


# ── Single run ───────────────────────────────────────────────────

def run_single(condition: Condition, seed: int) -> Dict[str, Any]:
    """Run a single episode under a condition and return metrics."""
    from environment import EconomicModel
    from metrics import collect_step_metrics

    label = f"{condition.name}/seed{seed}"
    print(f"  [{label}] starting...", end=" ", flush=True)
    t0 = time.time()

    model = EconomicModel(
        seed=seed,
        grid_width=GRID_SIZE,
        grid_height=GRID_SIZE,
        n_workers=N_WORKERS,
        n_firms=N_FIRMS,
        n_landowners=N_LANDOWNERS,
        objective=condition.objective,
    )

    # Inject feature flags
    configure_model(model, condition)

    from trust import update_trust_scores

    for step in range(N_STEPS):
        model.step()
        if condition.use_trust:
            update_trust_scores(model)
        if (step + 1) % 500 == 0:
            print(f"{step+1}", end=" ", flush=True)

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")

    # Save raw metrics
    raw_dir = f"{OUTPUT_DIR}/raw_data"
    os.makedirs(raw_dir, exist_ok=True)

    metrics_df = pd.DataFrame(model.metrics_history)
    metrics_df["condition"] = condition.name
    metrics_df["condition_label"] = condition.label
    metrics_df["objective"] = condition.objective
    metrics_df["use_sevc"] = condition.use_sevc
    metrics_df["use_trust"] = condition.use_trust
    metrics_df["trust_noise"] = condition.trust_noise
    metrics_df["use_hi"] = condition.use_horizon_index
    metrics_df["use_firm_hi"] = condition.use_firm_hi
    metrics_df["gov_type"] = condition.gov_type
    metrics_df["seed"] = seed

    fname = f"{raw_dir}/{condition.name}_seed{seed}.parquet"
    metrics_df.to_parquet(fname, index=False)

    return {
        "condition": condition.name,
        "label": condition.label,
        "objective": condition.objective,
        "seed": seed,
        "elapsed": elapsed,
        "use_sevc": condition.use_sevc,
        "use_trust": condition.use_trust,
        "trust_noise": condition.trust_noise,
        "use_hi": condition.use_horizon_index,
        "use_firm_hi": condition.use_firm_hi,
        "gov_type": condition.gov_type,
        "mixed_sevc_ratio": condition.mixed_sevc_ratio,
    }


def run_single_args(args):
    """Wrapper for multiprocessing (must be top-level function)."""
    condition, seed = args
    try:
        return run_single(condition, seed)
    except Exception as e:
        print(f"  ERROR: {condition.name}/seed{seed}: {e}")
        traceback.print_exc()
        return {"condition": condition.name, "seed": seed, "error": str(e)}


# ── Production-aware experiment runner (Task 11) ──────────────────

def run_production_aware(parallel: int = 1):
    """
    Run the three production-aware conditions (C16/C17/C18) with 8 seeds
    and 2000 steps.  Results go to results/architecture/raw_data/ alongside
    the other architecture conditions.
    """
    pa_conditions = [c for c in CONDITIONS if c.name.startswith(("C16", "C17", "C18"))]
    queue = [(cond, seed) for cond in pa_conditions for seed in SEEDS_PA]

    os.makedirs(f"{OUTPUT_DIR}/raw_data", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/summary", exist_ok=True)

    print("=" * 70)
    print("  TASK 11: PRODUCTION-AWARE CAPITAL EXPERIMENT")
    print(f"  Conditions: {[c.name for c in pa_conditions]}")
    print(f"  Seeds: {SEEDS_PA}")
    print(f"  Steps: {N_STEPS_PA}")
    print(f"  Total runs: {len(queue)}")
    print("=" * 70)

    apply_patches()

    # Temporarily override N_STEPS so run_single_args picks it up
    global N_STEPS
    _orig_steps = N_STEPS
    N_STEPS = N_STEPS_PA

    results = []
    if parallel > 1:
        from multiprocessing import Pool
        with Pool(parallel, initializer=apply_patches) as pool:
            results = pool.map(run_single_args, queue)
    else:
        for args in queue:
            results.append(run_single_args(args))

    N_STEPS = _orig_steps

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/summary/run_log_pa.csv", index=False)
    print(f"\nDone: {len(results)} runs. Results in {OUTPUT_DIR}/raw_data/")
    return results_df


# ── Main ─────────────────────────────────────────────────────────

def main():
    global N_STEPS
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential)")
    parser.add_argument("--steps", type=int, default=N_STEPS,
                        help="Steps per run")
    parser.add_argument("--production-aware", action="store_true",
                        help="Run only the Task 11 production-aware conditions (C16/C17/C18)")
    args = parser.parse_args()

    if args.production_aware:
        run_production_aware(parallel=args.parallel)
        return

    n_steps = args.steps
    N_STEPS = n_steps

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/raw_data", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/summary", exist_ok=True)

    # Build the run queue
    queue = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            queue.append((cond, seed))

    print("=" * 70)
    print("  ARCHITECTURE EXPERIMENT")
    print(f"  Conditions: {len(CONDITIONS)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Steps: {n_steps}")
    print(f"  Total runs: {len(queue)}")
    print(f"  Parallel workers: {args.parallel}")
    print("=" * 70)

    for i, cond in enumerate(CONDITIONS):
        flags = []
        if cond.use_sevc: flags.append("SEVC")
        if cond.use_trust: flags.append(f"Trust(noise={cond.trust_noise})")
        if cond.use_horizon_index: flags.append("HI")
        if cond.use_firm_hi: flags.append("FirmHI")
        flags.append(f"gov={cond.gov_type}")
        print(f"  C{i+1}: {cond.label:30s} obj={cond.objective:8s} [{', '.join(flags) or 'vanilla'}]")

    print()

    # Apply monkey-patches
    print("Applying feature patches...")
    apply_patches()
    print()

    total_t0 = time.time()
    results = []

    if args.parallel > 1:
        # Parallel execution
        from multiprocessing import Pool

        # Note: patches must be applied in each worker process
        # We handle this by having each worker import and patch on first call
        print(f"Running {len(queue)} episodes across {args.parallel} workers...")
        with Pool(args.parallel, initializer=apply_patches) as pool:
            results = pool.map(run_single_args, queue)
    else:
        # Sequential execution
        for cond, seed in queue:
            result = run_single_args((cond, seed))
            results.append(result)

    total_elapsed = time.time() - total_t0

    print(f"\n{'=' * 70}")
    print(f"  Complete: {len(results)} runs in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")

    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUTPUT_DIR}/summary/run_log.csv", index=False)

    # Load all raw data and print comparison
    try:
        raw_files = sorted([f for f in os.listdir(f"{OUTPUT_DIR}/raw_data")
                           if f.endswith(".parquet")])
        if raw_files:
            all_raw = pd.concat([
                pd.read_parquet(f"{OUTPUT_DIR}/raw_data/{f}") for f in raw_files
            ], ignore_index=True)

            all_raw.to_parquet(f"{OUTPUT_DIR}/summary/all_data.parquet", index=False)

            print_comparison(all_raw)
    except Exception as e:
        print(f"  Comparison table failed: {e}")
        traceback.print_exc()


def print_comparison(df: pd.DataFrame):
    """Print formatted comparison across conditions."""

    # Use last 500 steps as steady state
    late = df[df['step'] >= max(df['step'].max() - 500, df['step'].max() // 2)]

    conditions = sorted(late['condition'].unique())

    metrics = [
        ("worker_min",          "Worker Floor",    "higher"),
        ("worker_mean",         "Worker Mean",     "higher"),
        ("worker_gini",         "Worker Gini",     "lower"),
        ("all_gini",            "All Gini",        "lower"),
        ("unemployment_rate",   "Unemployment",    "lower"),
        ("agency_floor",        "Agency Floor",    "higher"),
        ("agency_mean",         "Agency Mean",     "higher"),
        ("n_workers",           "Population",      "higher"),
        ("n_firms",             "Firms",           "higher"),
        ("n_active_cartels",    "Cartels",         "lower"),
        ("horizon_index",       "Horizon Index",   "higher"),
        ("trust_institutional", "Inst Trust",      "higher"),
        ("trust_planner",       "Planner Trust",   "higher"),
        ("total_production",    "Production",      "higher"),
        ("mean_skill",          "Mean Skill",      "higher"),
        ("mean_firm_floor",     "Firm SEVC Floor", "higher"),
        ("total_pollution",     "Pollution",       "lower"),
    ]

    print(f"\n{'=' * 120}")
    print("  ARCHITECTURE COMPARISON (steady-state, mean across seeds)")
    print(f"{'=' * 120}")

    # Header
    header = f"{'Metric':<18}"
    for c in conditions:
        short = c.replace("C1_baseline", "C1:Base").replace("C2_sevc", "C2:SEVC") \
                 .replace("C3_sevc_hi", "C3:+HI").replace("C4_full_auth", "C4:Auth") \
                 .replace("C5_demo_captured", "C5:DmCap").replace("C6_auth_captured", "C6:AuCap") \
                 .replace("C7_democratic", "C7:Demo").replace("C8_mixed", "C8:Mix")
        header += f" {short:>14}"
    print(header)
    print("-" * len(header))

    for key, label, direction in metrics:
        if key not in late.columns:
            continue
        row = f"{label:<18}"
        vals = {}
        for c in conditions:
            v = late[late['condition'] == c][key].mean()
            vals[c] = v

        best_val = min(vals.values()) if direction == "lower" else max(vals.values())
        for c in conditions:
            v = vals[c]
            marker = "*" if abs(v - best_val) < abs(best_val) * 0.01 + 1e-9 else " "
            if abs(v) >= 100000:
                row += f" {v:>13.0f}{marker}"
            elif abs(v) >= 100:
                row += f" {v:>13.1f}{marker}"
            elif abs(v) >= 1:
                row += f" {v:>13.2f}{marker}"
            else:
                row += f" {v:>13.4f}{marker}"
        print(row)

    # Staircase analysis
    print(f"\n{'=' * 120}")
    print("  STAIRCASE ANALYSIS (what each layer adds)")
    print(f"{'=' * 120}")

    staircase_metrics = ['worker_min', 'worker_gini', 'unemployment_rate',
                         'agency_floor', 'n_firms', 'n_active_cartels',
                         'horizon_index', 'trust_institutional', 'mean_firm_hi']

    pairs = [
        ('C1_baseline',      'C1 (baseline)'),
        ('C2_sevc',          'C2 (+SEVC)'),
        ('C3_sevc_hi',       'C3 (+HI+FirmHI)'),
        ('C4_full_auth',     'C4 (+Trust,auth)'),
        ('C5_demo_captured', 'C5 (demo_cap)'),
        ('C6_auth_captured', 'C6 (auth_cap)'),
        ('C7_democratic',    'C7 (democratic)'),
        ('C8_mixed',         'C8 (mixed)'),
    ]

    for key in staircase_metrics:
        if key not in late.columns:
            continue
        print(f"\n  {key}:")
        prev_val = None
        for cond_name, label in pairs:
            subset = late[late['condition'] == cond_name]
            if subset.empty:
                continue
            v = subset[key].mean()
            if prev_val is not None and np.isfinite(v) and np.isfinite(prev_val) and abs(prev_val) > 1e-9:
                delta = (v - prev_val) / abs(prev_val) * 100
                print(f"    {label:<20s} {v:>12.4f}  [{delta:+.1f}%]")
            else:
                print(f"    {label:<20s} {v:>12.4f}")
            prev_val = v

    print()


if __name__ == "__main__":
    main()

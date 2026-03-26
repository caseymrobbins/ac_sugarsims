"""
run_architecture_experiment.py
------------------------------
Tests the contribution of each architectural layer:

  Condition 1: Vanilla firms, no trust, no HI, SUM_RAW
  Condition 2: SEVC firms, no trust, no HI, SUM_RAW
  Condition 3: SEVC firms, no trust, HI, SUM_RAW
  Condition 4: SEVC firms, fuzzy trust (0.1), no HI, SUM_RAW
  Condition 5: SEVC firms, fuzzy trust (0.1), HI, SUM_RAW
  Condition 6: SEVC firms, fuzzy trust (0.1), HI, TOPO_X

Each condition x 3 seeds (42, 137, 2024) = 18 runs at 3000 steps.

The staircase isolates each layer's contribution:
  1 vs 2: What does SEVC do alone?
  2 vs 3: What does HI add to SEVC?
  2 vs 4: What does fuzzy trust add to SEVC?
  4 vs 5: What does HI add under realistic trust?
  5 vs 6: What does a smart planner add to the full realistic stack?

Usage:
    python run_architecture_experiment.py

    # Or with parallelism (recommended):
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


CONDITIONS = [
    Condition("C1_vanilla_sum",     "Vanilla + SUM",         "SUM_RAW", False, False, 0.0,  False),
    Condition("C2_sevc_sum",        "SEVC + SUM",            "SUM_RAW", True,  False, 0.0,  False),
    Condition("C3_sevc_hi_sum",     "SEVC + HI + SUM",       "SUM_RAW", True,  False, 0.0,  True),
    Condition("C4_sevc_trust_sum",  "SEVC + Trust(.1) + SUM","SUM_RAW", True,  True,  0.1,  False),
    Condition("C5_sevc_trust_hi_sum","SEVC+Trust(.1)+HI+SUM","SUM_RAW", True,  True,  0.1,  True),
    Condition("C6_full_topo",       "SEVC+Trust(.1)+HI+TOPO","TOPO_X",  True,  True,  0.1,  True),
]

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

    # If SEVC is disabled, reset firms to vanilla behavior
    if not condition.use_sevc:
        for firm in model.firms:
            # Remove SEVC scoring: set all strategies to equal weight
            if hasattr(firm, 'strategy_weights'):
                for k in firm.strategy_weights:
                    firm.strategy_weights[k] = 0.2
                # Remove "innovate" strategy
                if 'innovate' in firm.strategy_weights:
                    del firm.strategy_weights['innovate']
            # Reset tech level (no innovation module)
            if hasattr(firm, 'tech_level'):
                firm.tech_level = 1.0

    # If trust is disabled, set all trust scores to neutral 0.5
    # and mark them as frozen so update_trust_scores skips them
    if not condition.use_trust:
        model._trust_frozen = True
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

    # ── Patch 1: Trust score reads add noise when trust_noise > 0 ──
    #
    # We monkey-patch getattr calls on trust_score by wrapping the
    # trust module's update function to add noise to the score
    # AFTER computing the true value. Agents read the noisy value.
    #
    # This is cleaner than patching every getattr call in agents.py.

    try:
        import trust as trust_module
        _original_update_trust = trust_module.update_trust_scores

        def patched_update_trust(model):
            # If trust is frozen (disabled), skip entirely
            if getattr(model, '_trust_frozen', False):
                return

            # Run normal trust update
            _original_update_trust(model)

            # If noise > 0, add noise to all trust scores
            # Store true score separately so noise compounds don't drift
            noise = getattr(model, 'trust_noise', 0.0)
            if noise > 0:
                rng = model.rng if hasattr(model, 'rng') else np.random.default_rng()
                for agent in model.schedule.agents if hasattr(model, 'schedule') else []:
                    if hasattr(agent, 'trust_score'):
                        # Store the true score if not already stored
                        if not hasattr(agent, '_true_trust_score'):
                            agent._true_trust_score = agent.trust_score
                        else:
                            agent._true_trust_score = agent.trust_score

                        # Noisy read: other agents see this
                        noisy = agent._true_trust_score + rng.normal(0, noise)
                        agent.trust_score = float(np.clip(noisy, 0.0, 1.0))

        trust_module.update_trust_scores = patched_update_trust
        print("  [patch] Trust noise injection: OK")
    except ImportError:
        print("  [patch] Trust module not found, skipping trust patches")

    # ── Patch 2: Sustainable capitalism disable ──
    #
    # When use_sevc is False, the firm's step should skip SEVC
    # strategy selection and use simple profit-maximizing behavior.

    try:
        import sustainable_capitalism as sc_module

        _original_choose = sc_module.sustainable_choose_strategy
        _original_learn = sc_module.sustainable_learn_from_outcome

        def patched_choose(firm):
            if not getattr(firm.model, 'use_sevc', True):
                # Vanilla: always choose highest-weight strategy
                # (which is just random since we equalized weights)
                strategies = list(firm.strategy_weights.keys())
                if strategies:
                    rng = firm.model.rng if hasattr(firm.model, 'rng') else np.random.default_rng()
                    return strategies[rng.integers(len(strategies))]
                return "expand"
            return _original_choose(firm)

        def patched_learn(firm, strategy, reward):
            if not getattr(firm.model, 'use_sevc', True):
                return  # No learning in vanilla mode
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
            if not getattr(firm.model, 'use_sevc', True):
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

    for step in range(N_STEPS):
        model.step()
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


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel workers (default: 1 = sequential)")
    parser.add_argument("--steps", type=int, default=None,
                        help=f"Steps per run (default: {N_STEPS})")
    args = parser.parse_args()

    n_steps = args.steps if args.steps is not None else N_STEPS

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
        # Abbreviate condition names
        short = c.replace("C1_vanilla_sum", "C1:Van").replace("C2_sevc_sum", "C2:SEVC") \
                 .replace("C3_sevc_hi_sum", "C3:+HI").replace("C4_sevc_trust_sum", "C4:+Trst") \
                 .replace("C5_sevc_trust_hi_sum", "C5:+T+HI").replace("C6_full_topo", "C6:TOPO")
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
                         'horizon_index', 'trust_institutional']

    for key in staircase_metrics:
        if key not in late.columns:
            continue
        c1 = late[late['condition'] == 'C1_vanilla_sum'][key].mean()
        c2 = late[late['condition'] == 'C2_sevc_sum'][key].mean()
        c3 = late[late['condition'] == 'C3_sevc_hi_sum'][key].mean()
        c4 = late[late['condition'] == 'C4_sevc_trust_sum'][key].mean()
        c5 = late[late['condition'] == 'C5_sevc_trust_hi_sum'][key].mean()
        c6 = late[late['condition'] == 'C6_full_topo'][key].mean()

        print(f"\n  {key}:")
        print(f"    C1 (vanilla):     {c1:>12.4f}")
        if np.isfinite(c2) and np.isfinite(c1) and abs(c1) > 1e-9:
            delta = (c2 - c1) / abs(c1) * 100
            print(f"    C2 (+SEVC):       {c2:>12.4f}  [{delta:+.1f}% from C1]")
        else:
            print(f"    C2 (+SEVC):       {c2:>12.4f}")
        if np.isfinite(c3) and np.isfinite(c2) and abs(c2) > 1e-9:
            delta = (c3 - c2) / abs(c2) * 100
            print(f"    C3 (+HI):         {c3:>12.4f}  [{delta:+.1f}% from C2]")
        else:
            print(f"    C3 (+HI):         {c3:>12.4f}")
        if np.isfinite(c4) and np.isfinite(c2) and abs(c2) > 1e-9:
            delta = (c4 - c2) / abs(c2) * 100
            print(f"    C4 (+Trust):      {c4:>12.4f}  [{delta:+.1f}% from C2]")
        else:
            print(f"    C4 (+Trust):      {c4:>12.4f}")
        if np.isfinite(c5) and np.isfinite(c4) and abs(c4) > 1e-9:
            delta = (c5 - c4) / abs(c4) * 100
            print(f"    C5 (+Trust+HI):   {c5:>12.4f}  [{delta:+.1f}% from C4]")
        else:
            print(f"    C5 (+Trust+HI):   {c5:>12.4f}")
        if np.isfinite(c6) and np.isfinite(c5) and abs(c5) > 1e-9:
            delta = (c6 - c5) / abs(c5) * 100
            print(f"    C6 (+TOPO):       {c6:>12.4f}  [{delta:+.1f}% from C5]")
        else:
            print(f"    C6 (+TOPO):       {c6:>12.4f}")

    print()


if __name__ == "__main__":
    main()

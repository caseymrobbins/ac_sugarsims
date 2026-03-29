"""
run_test_conditions.py
----------------------
Run two specific conditions with 10 seeds each and generate animations + summary.

  1) Vanilla SUM: no SEVC, no trust, no HI, SUM_RAW objective
  2) TOPO + SEVC + Firm HI + HI: full sustainable stack with TOPO_X objective

Usage:
    python run_test_conditions.py                  # 500 steps, 10 seeds
    python run_test_conditions.py --steps 1000     # longer runs
    python run_test_conditions.py --animate-seed 1 # which seed to animate (0-9)
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import traceback
from dataclasses import dataclass

import numpy as np
import pandas as pd

from run_architecture_experiment import Condition, configure_model, apply_patches

# Two conditions to compare
CONDITIONS = [
    Condition(
        name="vanilla_sum",
        label="Vanilla SUM (no features)",
        objective="SUM_RAW",
        use_sevc=False, use_trust=False, trust_noise=0.0,
        use_horizon_index=False, use_firm_hi=False,
        gov_type="authoritarian",
    ),
    Condition(
        name="topo_sevc_hi",
        label="TOPO + SEVC + HI + FirmHI",
        objective="TOPO_X",
        use_sevc=True, use_trust=True, trust_noise=0.1,
        use_horizon_index=True, use_firm_hi=True,
        gov_type="democratic",
    ),
]

SEEDS = [7, 23, 59, 101, 233, 347, 461, 587, 719, 853]

GRID_SIZE = 80
N_WORKERS = 400
N_FIRMS = 20
N_LANDOWNERS = 15
OUTPUT_DIR = "results/test_conditions"


def run_one(condition: Condition, seed: int, n_steps: int, collect_animation: bool):
    from environment import EconomicModel
    from trust import update_trust_scores

    label = f"{condition.name}/seed{seed}"
    print(f"  [{label}] ", end="", flush=True)
    t0 = time.time()

    model = EconomicModel(
        seed=seed,
        grid_width=GRID_SIZE, grid_height=GRID_SIZE,
        n_workers=N_WORKERS, n_firms=N_FIRMS, n_landowners=N_LANDOWNERS,
        objective=condition.objective,
    )
    model._collect_animation = collect_animation
    configure_model(model, condition)

    for step in range(n_steps):
        model.step()
        if condition.use_trust:
            update_trust_scores(model)
        if (step + 1) % 100 == 0:
            print(f"{step+1}", end=" ", flush=True)

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")

    # Save raw metrics
    raw_dir = f"{OUTPUT_DIR}/raw_data"
    os.makedirs(raw_dir, exist_ok=True)
    metrics_df = pd.DataFrame(model.metrics_history)
    metrics_df["condition"] = condition.name
    metrics_df["seed"] = seed
    metrics_df.to_parquet(f"{raw_dir}/{condition.name}_seed{seed}.parquet", index=False)

    # Animation
    if collect_animation and model.animation_frames:
        anim_dir = f"{OUTPUT_DIR}/animations"
        os.makedirs(anim_dir, exist_ok=True)
        try:
            from animate import generate_animation_html
            out = f"{anim_dir}/{condition.name}_seed{seed}.html"
            generate_animation_html(
                model.animation_frames,
                output_path=out,
                grid_size=GRID_SIZE,
                title=f"{condition.label} (seed={seed})",
                subsample=2,
            )
            fsize = os.path.getsize(out) / (1024*1024)
            print(f"    animation: {out} ({fsize:.1f} MB)")
        except Exception as e:
            print(f"    animation failed: {e}")

    return {
        "condition": condition.name,
        "label": condition.label,
        "seed": seed,
        "elapsed": elapsed,
        "n_steps": n_steps,
    }


def print_comparison(output_dir):
    """Load all raw data and print head-to-head comparison."""
    raw_dir = f"{output_dir}/raw_data"
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".parquet"))
    if not files:
        return
    all_data = pd.concat([pd.read_parquet(f"{raw_dir}/{f}") for f in files], ignore_index=True)

    # Use last 20% of steps as steady state
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
        ("mean_firm_hi",          "Firm HI",          ".3f",  "higher"),
        ("n_firms",               "Firms",            ".1f",  "higher"),
        ("total_production",      "Production",       ".0f",  "higher"),
        ("total_pollution",       "Pollution",        ".0f",  "lower"),
        ("trust_planner",         "Planner Trust",    ".3f",  "higher"),
        ("trust_institutional",   "Inst. Trust",      ".3f",  "higher"),
        ("mean_aggression",       "Aggression",       ".3f",  "lower"),
        ("mean_conflict",         "Conflict",         ".4f",  "lower"),
        ("legitimacy_mean",       "Legitimacy",       ".3f",  "higher"),
        ("crime_events",          "Crime/step",       ".1f",  "lower"),
        ("identity_conflict_index","ID Conflict",     ".3f",  "lower"),
    ]

    print(f"\n{'='*80}")
    print("  HEAD-TO-HEAD COMPARISON (steady-state mean +/- std across 10 seeds)")
    print(f"{'='*80}")

    header = f"{'Metric':<20}"
    for c in conditions:
        header += f"  {c:>26}"
    print(header)
    print("-" * len(header))

    for key, label, fmt, direction in metrics:
        if key not in tail.columns:
            continue
        row = f"{label:<20}"
        vals = {}
        for c in conditions:
            subset = tail[tail["condition"] == c][key]
            mean_v = subset.mean(); std_v = subset.std()
            vals[c] = mean_v
            row += f"  {mean_v:>12{fmt}} +/- {std_v:>7{fmt}}"
        # Mark winner
        print(row)

    print()

    # Save comparison CSV
    summary_rows = []
    for c in conditions:
        subset = tail[tail["condition"] == c]
        row_data = {"condition": c}
        for key, label, fmt, direction in metrics:
            if key in subset.columns:
                row_data[f"{key}_mean"] = subset[key].mean()
                row_data[f"{key}_std"] = subset[key].std()
        summary_rows.append(row_data)

    summary_dir = f"{output_dir}/summary"
    os.makedirs(summary_dir, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(f"{summary_dir}/comparison.csv", index=False)
    all_data.to_parquet(f"{summary_dir}/all_data.parquet", index=False)
    print(f"Saved: {summary_dir}/comparison.csv")
    print(f"Saved: {summary_dir}/all_data.parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500, help="Steps per run (default: 500)")
    parser.add_argument("--animate-seed", type=int, default=0, help="Which seed index to animate (0-9, default: 0)")
    parser.add_argument("--subsample", type=int, default=2)
    args = parser.parse_args()

    print("=" * 70)
    print("  TEST CONDITIONS: Vanilla SUM vs TOPO+SEVC+HI")
    print(f"  Seeds: {SEEDS}")
    print(f"  Steps: {args.steps}")
    print(f"  Animate seed index: {args.animate_seed} (seed={SEEDS[args.animate_seed]})")
    print(f"  Total runs: {len(CONDITIONS) * len(SEEDS)}")
    print("=" * 70)

    print("\nApplying patches...")
    apply_patches()
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_t0 = time.time()
    results = []

    for cond in CONDITIONS:
        print(f"\n--- {cond.label} ---")
        for i, seed in enumerate(SEEDS):
            try:
                animate = (i == args.animate_seed)
                r = run_one(cond, seed, args.steps, animate)
                results.append(r)
            except Exception as e:
                print(f"  FAIL: {cond.name}/seed{seed}: {e}")
                traceback.print_exc()

    total_elapsed = time.time() - total_t0

    print(f"\n{'='*70}")
    print(f"  Complete: {len(results)} runs in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}")

    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/run_log.csv", index=False)

    print_comparison(OUTPUT_DIR)


if __name__ == "__main__":
    main()

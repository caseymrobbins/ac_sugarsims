"""
run_comparison.py
-----------------
Comparison experiment: 4 objectives, 3 seeds, 3000 steps.

Objectives:
  SUM_RAW   - Pure aggregate sum, no horizon index. Baseline.
  NASH_MIN  - Nash welfare with HI emergency brake (min gate).
  TOPO_X    - Topology shaping with HI multiplier (proportional discount).
  TOPO_MIN  - Topology shaping with HI emergency brake (min gate).

This answers: does the min gate produce different planner behavior
than the multiplier? Does SUM_RAW (no sustainability constraint)
produce the extractive dynamics we predict?

Usage:
    python run_comparison.py

Output:
    results/comparison/
        raw_data/        - per-step metrics as parquet
        summary/         - episode summaries
        animations/      - HTML animation per run (if enabled)
        comparison.csv   - side-by-side summary
"""

from __future__ import annotations

import os
import sys
import time
import json
import traceback
from typing import Dict, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OBJECTIVES = ["SUM_RAW", "NASH_MIN", "TOPO_X", "TOPO_MIN"]
SEEDS = [42, 137, 2024]
N_STEPS = 3000
N_WORKERS = 400
N_FIRMS = 20
N_LANDOWNERS = 15
GRID_SIZE = 80

# Animation: collect for one seed per objective to keep memory reasonable
ANIMATE_SEED = 42

OUTPUT_DIR = "results/comparison"


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_episode(objective: str, seed: int, n_steps: int,
                collect_animation: bool = False) -> Dict:
    """Run a single episode and return the episode summary."""
    from environment import EconomicModel
    from metrics import episode_summary

    print(f"  [{objective}] seed={seed}, steps={n_steps} ...", end=" ", flush=True)
    t0 = time.time()

    model = EconomicModel(
        seed=seed,
        grid_width=GRID_SIZE,
        grid_height=GRID_SIZE,
        n_workers=N_WORKERS,
        n_firms=N_FIRMS,
        n_landowners=N_LANDOWNERS,
        objective=objective,
    )
    model._collect_animation = collect_animation

    for step in range(n_steps):
        model.step()
        if (step + 1) % 500 == 0:
            print(f"{step+1}", end=" ", flush=True)

    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")

    # Episode summary
    summary = episode_summary(model.metrics_history)
    summary["objective"] = objective
    summary["seed"] = seed
    summary["n_steps"] = n_steps
    summary["elapsed_seconds"] = elapsed

    # Save raw metrics
    raw_dir = f"{OUTPUT_DIR}/raw_data"
    os.makedirs(raw_dir, exist_ok=True)
    metrics_df = pd.DataFrame(model.metrics_history)
    metrics_df["objective"] = objective
    metrics_df["seed"] = seed
    metrics_df.to_parquet(f"{raw_dir}/{objective}_seed{seed}.parquet", index=False)

    # Save animation if collected
    if collect_animation and model.animation_frames:
        anim_dir = f"{OUTPUT_DIR}/animations"
        os.makedirs(anim_dir, exist_ok=True)
        try:
            from animate import generate_animation_html
            generate_animation_html(
                model.animation_frames,
                output_path=f"{anim_dir}/{objective}_seed{seed}.html",
                grid_size=GRID_SIZE,
                title=f"{objective} (seed={seed})",
                subsample=2,  # every other frame to keep file size down
            )
            print(f"    Animation saved: {anim_dir}/{objective}_seed{seed}.html")
        except Exception as e:
            print(f"    Animation failed: {e}")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/summary", exist_ok=True)

    print("=" * 60)
    print("  COMPARISON EXPERIMENT")
    print(f"  Objectives: {', '.join(OBJECTIVES)}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Steps: {N_STEPS}")
    print(f"  Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"  Workers: {N_WORKERS}, Firms: {N_FIRMS}, Landowners: {N_LANDOWNERS}")
    print("=" * 60)
    print()

    all_summaries = []
    total_t0 = time.time()

    for objective in OBJECTIVES:
        print(f"\n{'─' * 50}")
        print(f"  Objective: {objective}")
        print(f"{'─' * 50}")

        for seed in SEEDS:
            try:
                animate = (seed == ANIMATE_SEED)
                summary = run_episode(objective, seed, N_STEPS,
                                      collect_animation=animate)
                all_summaries.append(summary)
            except Exception as e:
                print(f"  ERROR: {objective} seed={seed}: {e}")
                traceback.print_exc()

    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 60}")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 60}")

    # Build comparison table
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_df.to_csv(f"{OUTPUT_DIR}/summary/all_summaries.csv", index=False)
        summary_df.to_parquet(f"{OUTPUT_DIR}/summary/all_summaries.parquet", index=False)

        # Print comparison
        _print_comparison(summary_df)

        # Run statistical analysis if we have enough data
        try:
            from analysis import run_analysis
            raw_files = [f for f in os.listdir(f"{OUTPUT_DIR}/raw_data") if f.endswith(".parquet")]
            if raw_files:
                all_raw = pd.concat([
                    pd.read_parquet(f"{OUTPUT_DIR}/raw_data/{f}") for f in raw_files
                ], ignore_index=True)
                run_analysis(all_raw, output_dir=OUTPUT_DIR)
        except Exception as e:
            print(f"  Statistical analysis skipped: {e}")


def _print_comparison(df: pd.DataFrame):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 70}")
    print("  COMPARISON RESULTS (mean across seeds)")
    print(f"{'=' * 70}")

    key_metrics = [
        ("all_gini", "Gini", ".3f", "lower"),
        ("worker_min", "Floor Wealth", ".1f", "higher"),
        ("agency_floor", "Agency Floor", ".2f", "higher"),
        ("unemployment_rate", "Unemployment", ".1%", "lower"),
        ("horizon_index", "Horizon Index", ".3f", "higher"),
        ("mean_firm_floor", "Firm SEVC Floor", ".3f", "higher"),
        ("trust_planner", "Planner Trust", ".3f", "higher"),
        ("trust_institutional", "Inst. Trust", ".3f", "higher"),
        ("tech_frontier", "Tech Frontier", ".2f", "higher"),
        ("tech_mean", "Tech Mean", ".2f", "higher"),
        ("frac_monopoly", "% Monopoly", ".1%", "lower"),
        ("frac_poverty_trap", "% Poverty Trap", ".1%", "lower"),
        ("n_firms", "Active Firms", ".1f", "higher"),
        ("n_workers", "Population", ".0f", "higher"),
        ("epistemic_health", "Epistemic Health", ".3f", "higher"),
        ("total_production", "Production", ".0f", "higher"),
    ]

    objectives = sorted(df["objective"].unique())

    # Header
    header = f"{'Metric':<20}"
    for obj in objectives:
        header += f" {obj:>12}"
    print(header)
    print("-" * len(header))

    for metric_key, label, fmt, direction in key_metrics:
        if metric_key not in df.columns:
            continue
        row = f"{label:<20}"
        values = []
        for obj in objectives:
            subset = df[df["objective"] == obj][metric_key]
            val = subset.mean() if len(subset) > 0 else float("nan")
            values.append(val)
            row += f" {val:>12{fmt}}" if np.isfinite(val) else f" {'N/A':>12}"
        # Highlight best
        print(row)

    print()

    # Terminal values
    print("Terminal values (end of episode):")
    terminal_keys = [
        ("terminal_worker_min", "Floor Wealth"),
        ("terminal_agency_floor", "Agency Floor"),
        ("terminal_all_gini", "Gini"),
        ("terminal_horizon_index", "Horizon Index"),
        ("terminal_trust_planner", "Planner Trust"),
        ("terminal_tech_frontier", "Tech Frontier"),
    ]
    for metric_key, label in terminal_keys:
        if metric_key not in df.columns:
            continue
        row = f"  {label:<18}"
        for obj in objectives:
            subset = df[df["objective"] == obj][metric_key]
            val = subset.mean() if len(subset) > 0 else float("nan")
            row += f" {val:>12.3f}" if np.isfinite(val) else f" {'N/A':>12}"
        print(row)

    print()


if __name__ == "__main__":
    main()

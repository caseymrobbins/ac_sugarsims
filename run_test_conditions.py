"""
run_test_conditions.py
----------------------
Run two specific conditions with multiple seeds and generate animations + summary.

Conditions:
  1) Vanilla SUM: no SEVC, no trust, no HI, SUM_RAW objective
  2) TOPO + SEVC + HI + FirmHI: full sustainable stack with TOPO_X objective

Usage:
    python run_test_conditions.py
    python run_test_conditions.py --workers 6
    python run_test_conditions.py --steps 1000
    python run_test_conditions.py --animate-all
"""

from __future__ import annotations

import os
import time
import argparse
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from run_architecture_experiment import Condition, configure_model, apply_patches


# ─────────────────────────────────────────────
# Experiment configuration
# ─────────────────────────────────────────────

CONDITIONS = [
    Condition(
        name="vanilla_sum",
        label="Vanilla SUM (no features)",
        objective="SUM_RAW",
        use_sevc=False,
        use_trust=true,
        trust_noise=0.15,
        use_horizon_index=False,
        use_firm_hi=False,
        gov_type="democratic",
    ),
    Condition(
        name="topo_sevc_hi",
        label="TOPO + SEVC + HI + FirmHI",
        objective="TOPO_X",
        use_sevc=True,
        use_trust=True,
        trust_noise=0.1,
        use_horizon_index=True,
        use_firm_hi=True,
        gov_type="democratic",
    ),
]

SEEDS = [114, 214, 314, 514, 814]

GRID_SIZE = 100
N_WORKERS = 200
N_FIRMS = 10
N_LANDOWNERS = 10

OUTPUT_DIR = "results/test_conditions"


# ─────────────────────────────────────────────
# Single simulation run
# ─────────────────────────────────────────────

def run_one(condition: Condition, seed: int, n_steps: int, collect_animation: bool):

    from environment import EconomicModel
    from trust import update_trust_scores

    label = f"{condition.name}/seed{seed}"
    print(f"START {label}", flush=True)

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

    model._collect_animation = collect_animation

    configure_model(model, condition)

    for step in range(n_steps):

        model.step()

        if condition.use_trust:
            update_trust_scores(model)

    elapsed = time.time() - t0

    print(f"DONE {label} ({elapsed:.1f}s)", flush=True)

    # ───────────────────────────
    # Save raw metrics
    # ───────────────────────────

    raw_dir = f"{OUTPUT_DIR}/raw_data"
    os.makedirs(raw_dir, exist_ok=True)

    df = pd.DataFrame(model.metrics_history)
    df["condition"] = condition.name
    df["seed"] = seed

    df.to_parquet(f"{raw_dir}/{condition.name}_seed{seed}.parquet", index=False)

    # ───────────────────────────
    # Animation generation
    # ───────────────────────────

    if collect_animation and getattr(model, "animation_frames", None):

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

            print(f"animation saved: {out}")

        except Exception as e:

            print(f"animation failed: {e}")

    return {
        "condition": condition.name,
        "seed": seed,
        "elapsed": elapsed,
        "steps": n_steps,
    }


# ─────────────────────────────────────────────
# Comparison report
# ─────────────────────────────────────────────

def print_comparison(output_dir):

    raw_dir = f"{output_dir}/raw_data"

    files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet")]

    if not files:
        return

    all_data = pd.concat(
        [pd.read_parquet(f"{raw_dir}/{f}") for f in files],
        ignore_index=True,
    )

    max_step = all_data["step"].max()

    tail = all_data[all_data["step"] >= max_step * 0.8]

    conditions = sorted(tail["condition"].unique())

    metrics = [
        ("worker_min", "Floor Wealth"),
        ("worker_mean", "Worker Mean"),
        ("worker_gini", "Worker Gini"),
        ("unemployment_rate", "Unemployment"),
        ("agency_floor", "Agency Floor"),
        ("horizon_index", "Horizon Index"),
        ("mean_firm_floor", "Firm SEVC Floor"),
        ("mean_firm_hi", "Firm HI"),
        ("total_production", "Production"),
        ("total_pollution", "Pollution"),
        ("trust_planner", "Planner Trust"),
        ("trust_institutional", "Inst Trust"),
        ("mean_conflict", "Conflict"),
        ("legitimacy_mean", "Legitimacy"),
        ("crime_events", "Crime"),
    ]

    print("\n" + "=" * 80)
    print("HEAD TO HEAD COMPARISON (steady state)")
    print("=" * 80)

    for key, label in metrics:

        if key not in tail.columns:
            continue

        row = f"{label:20}"

        for c in conditions:

            subset = tail[tail["condition"] == c][key]

            mean = subset.mean()
            std = subset.std()

            row += f"{mean:12.3f} ± {std:8.3f}"

        print(row)

    summary_dir = f"{output_dir}/summary"
    os.makedirs(summary_dir, exist_ok=True)

    all_data.to_parquet(f"{summary_dir}/all_data.parquet", index=False)


# ─────────────────────────────────────────────
# Main experiment runner
# ─────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--animate-seed", type=int, default=0)
    parser.add_argument("--animate-all", action="store_true")

    args = parser.parse_args()

    workers = args.workers
    if workers <= 0:
        workers = max(1, multiprocessing.cpu_count() - 1)

    print("=" * 70)
    print("TEST CONDITIONS: Vanilla SUM vs TOPO+SEVC+HI")
    print(f"Seeds: {SEEDS}")
    print(f"Steps: {args.steps}")
    print(f"Workers: {workers}")
    print("=" * 70)

    print("Applying patches...")
    apply_patches()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jobs = []

    for cond in CONDITIONS:

        for i, seed in enumerate(SEEDS):

            if args.animate_all:
                animate = True
            else:
                animate = i == args.animate_seed

            jobs.append((cond, seed, args.steps, animate))

    results = []

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:

        futures = {executor.submit(run_one, *job): job for job in jobs}

        for future in as_completed(futures):

            cond, seed, *_ = futures[future]

            try:

                results.append(future.result())

            except Exception:

                print(f"FAIL {cond.name}/seed{seed}")
                traceback.print_exc()

    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"COMPLETE {len(results)} runs in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/run_log.csv", index=False)

    print_comparison(OUTPUT_DIR)


# ─────────────────────────────────────────────

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()


"""
run_experiments.py
------------------
Experiment runner for the multi-agent economic simulation.

Full protocol:
  - 3 objective conditions: SUM, NASH, JAM
  - 20 random seeds per condition
  - 100 episodes per seed (not used in this implementation — 1 episode per seed)
  - 1500 timesteps per episode

Test protocol (--test):
  - 3 conditions × 10 seeds × 150 timesteps

Parallelisation:
  - Uses multiprocessing.Pool (compatible with Google Colab)
  - Optional Ray support if available

Data storage:
  - results/raw_data/      : per-episode metrics Parquet
  - results/processed_data/: episode summaries, trajectories
  - results/plots/         : all static plots
  - results/animations/    : GIF animations
  - results/statistical_tests/
  - results/summary_tables/
  - results/logs/          : run logs
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Configure matplotlib before any imports that use it
import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

os.makedirs("results/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("results/logs/run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_single_episode(args: Tuple) -> Optional[Dict]:
    """
    Run one full episode.  Designed to be called from a worker process.

    args = (objective, seed, n_steps, episode_id, output_dir, save_raw, accel_cfg)
    accel_cfg may be None (auto-detect inside worker).
    """
    objective, seed, n_steps, episode_id, output_dir, save_raw, accel_cfg = args

    try:
        from environment import EconomicModel
        from metrics import episode_summary

        model = EconomicModel(
            seed=seed,
            grid_width=80,
            grid_height=80,
            n_workers=400,
            n_firms=20,
            n_landowners=15,
            objective=objective,
            accel_config=accel_cfg,
        )

        # Run steps
        for _ in range(n_steps):
            model.step()

        # Episode summary
        summary = episode_summary(model.metrics_history)
        summary["objective"] = objective
        summary["seed"] = seed
        summary["episode_id"] = episode_id
        summary["n_steps"] = n_steps
        summary["n_workers_final"] = len(model.workers)
        summary["n_firms_final"] = len([f for f in model.firms if not f.defunct])

        # Save raw metrics
        if save_raw:
            raw_df = pd.DataFrame(model.metrics_history)
            raw_df["objective"] = objective
            raw_df["seed"] = seed
            raw_path = f"{output_dir}/raw_data/metrics_{objective}_seed{seed}_ep{episode_id}.parquet"
            os.makedirs(f"{output_dir}/raw_data", exist_ok=True)
            raw_df.to_parquet(raw_path, index=False)

        logger.info(f"  Done: {objective} seed={seed} ep={episode_id} "
                    f"gini={summary.get('all_gini', 'nan'):.3f} "
                    f"floor={summary.get('terminal_worker_min', 'nan'):.2f}")
        return summary

    except Exception as e:
        logger.error(f"FAILED: {objective} seed={seed} ep={episode_id}: {e}")
        logger.debug(traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Experiment builder
# ---------------------------------------------------------------------------

def build_experiment_list(objectives: List[str],
                           seeds: List[int],
                           n_steps: int,
                           output_dir: str,
                           save_raw: bool,
                           accel_cfg=None) -> List[Tuple]:
    """Build the flat list of episode task tuples."""
    tasks = []
    ep = 0
    for obj in objectives:
        for seed in seeds:
            tasks.append((obj, seed, n_steps, ep, output_dir, save_raw, accel_cfg))
            ep += 1
    return tasks


# ---------------------------------------------------------------------------
# Visualisation pass (single process — matplotlib not MP-safe)
# ---------------------------------------------------------------------------

def run_visualisation_pass(results_df: pd.DataFrame,
                            raw_data_dir: str,
                            output_dir: str):
    """
    Load one representative episode per condition and generate all plots.
    """
    from visualizations import (
        generate_all_plots, compare_floor_wealth, compare_inequality,
        compare_wealth_distributions, compare_stability, compare_agency_floor,
        animate_wealth_distribution,
    )
    from analysis import run_analysis

    logger.info("Running analysis and visualisation pass...")

    # Statistical analysis
    analysis_results = run_analysis(results_df, output_dir)

    # Per-condition trajectory averages
    objectives = results_df["objective"].unique().tolist()

    # Load representative raw episodes (first seed for each condition)
    trajectories: Dict[str, pd.DataFrame] = {}
    final_wealth: Dict[str, np.ndarray] = {}

    for obj in objectives:
        # Get first seed for this objective
        seed_row = results_df[results_df["objective"] == obj].iloc[0]
        seed = int(seed_row["seed"])
        ep = int(seed_row["episode_id"])

        raw_path = f"{raw_data_dir}/metrics_{obj}_seed{seed}_ep{ep}.parquet"
        if os.path.exists(raw_path):
            raw_df = pd.read_parquet(raw_path)
            trajectories[obj] = raw_df

            # Use terminal worker wealth stats to approximate distribution
            last = raw_df.iloc[-1]
            # Reconstruct approximate wealth distribution from summary stats
            # (actual agent wealth not stored in parquet — re-run for one episode)
            final_wealth[obj] = _reconstruct_wealth_distribution(
                last.get("worker_mean", 50),
                last.get("worker_std", 30),
                last.get("all_gini", 0.5),
                n=400
            )

            # Single-condition plots
            generate_all_plots(
                metrics_history=raw_df.to_dict("records"),
                condition=obj,
                final_wealth=final_wealth[obj],
                output_dir=output_dir,
            )

            # Animate wealth evolution
            animate_wealth_distribution(
                raw_df.to_dict("records"),
                condition=obj,
                output_dir=f"{output_dir}/animations",
            )

    # Comparative plots
    if trajectories:
        compare_floor_wealth(trajectories, f"{output_dir}/plots")
        compare_inequality(trajectories, f"{output_dir}/plots")
        compare_stability(trajectories, f"{output_dir}/plots")
        compare_agency_floor(trajectories, f"{output_dir}/plots")

    if final_wealth:
        compare_wealth_distributions(final_wealth, f"{output_dir}/plots")

    # Summary bar charts for key metrics
    from visualizations import summary_bar_chart
    if "summary" in analysis_results:
        for metric in ["all_gini", "worker_min", "agency_floor", "unemployment_rate"]:
            try:
                summary_bar_chart(analysis_results["summary"], metric,
                                  f"{output_dir}/plots")
            except Exception:
                pass

    logger.info(f"Visualisation complete. Plots saved to {output_dir}/plots/")


def _reconstruct_wealth_distribution(mean: float, std: float, gini: float,
                                      n: int = 400) -> np.ndarray:
    """
    Approximate reconstruction of wealth distribution from summary statistics
    using a log-normal parameterisation consistent with the Gini coefficient.
    """
    # For lognormal: Gini ≈ 2*Φ(σ/√2) - 1, so σ ≈ √2 * Φ⁻¹((Gini+1)/2)
    from scipy.special import erfinv
    sigma = max(0.1, np.sqrt(2) * erfinv(gini))
    mu = np.log(max(mean, 1.0)) - sigma ** 2 / 2
    rng = np.random.default_rng(0)
    w = rng.lognormal(mean=mu, sigma=sigma, size=n)
    return w.clip(0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run multi-agent economic simulation experiments.")
    parser.add_argument("--test", action="store_true",
                        help="Run test protocol (10 seeds, 150 steps)")
    parser.add_argument("--objectives", nargs="+",
                        default=["SUM", "NASH", "JAM"],
                        choices=["SUM", "NASH", "JAM"],
                        help="Which objective conditions to run")
    parser.add_argument("--seeds", type=int, default=None,
                        help="Number of seeds (overrides protocol default)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Steps per episode (overrides protocol default)")
    parser.add_argument("--workers", type=int,
                        default=None,  # set after hardware detection
                        help="Number of parallel worker processes (default: auto)")
    parser.add_argument("--output-dir", default="results",
                        help="Root output directory")
    parser.add_argument("--no-save-raw", action="store_true",
                        help="Do not save per-step raw data")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualisation pass")
    args = parser.parse_args()

    # Auto-detect hardware and pick acceleration backend
    from hardware import auto_configure
    accel_cfg = auto_configure(verbose=True)

    # Protocol parameters (use hardware-recommended defaults if not overridden)
    if args.test:
        n_seeds = args.seeds or 10
        n_steps = args.steps or 150
        logger.info(f"TEST PROTOCOL: {n_seeds} seeds × {n_steps} steps × {len(args.objectives)} conditions")
    else:
        n_seeds = args.seeds or accel_cfg.recommended_seeds
        n_steps = args.steps or accel_cfg.recommended_steps
        logger.info(f"FULL PROTOCOL: {n_seeds} seeds × {n_steps} steps × {len(args.objectives)} conditions")

    seeds = list(range(n_seeds))
    output_dir = args.output_dir
    save_raw = not args.no_save_raw

    # Create output structure
    for subdir in ["raw_data", "processed_data", "plots", "animations",
                   "statistical_tests", "summary_tables", "logs"]:
        os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)

    # Build task list
    tasks = build_experiment_list(args.objectives, seeds, n_steps,
                                  output_dir, save_raw, accel_cfg)
    total = len(tasks)
    logger.info(f"Total episodes to run: {total}")

    # Run in parallel
    t_start = time.time()
    n_workers_cfg = args.workers if args.workers is not None else accel_cfg.n_workers
    n_workers = min(n_workers_cfg, total)

    results = []
    if n_workers <= 1:
        # Serial execution (debug / Colab)
        for i, task in enumerate(tasks):
            logger.info(f"[{i+1}/{total}] Running {task[0]} seed={task[1]}")
            result = run_single_episode(task)
            if result:
                results.append(result)
    else:
        logger.info(f"Parallel execution: {n_workers} processes")
        with Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(run_single_episode, tasks)):
                if result:
                    results.append(result)
                if (i + 1) % 10 == 0:
                    logger.info(f"  Progress: {i+1}/{total}")

    elapsed = time.time() - t_start
    logger.info(f"All episodes complete in {elapsed:.1f}s. "
                f"Successful: {len(results)}/{total}")

    if not results:
        logger.error("No results collected. Exiting.")
        sys.exit(1)

    # Aggregate results
    results_df = pd.DataFrame(results)
    processed_path = f"{output_dir}/processed_data/episode_summaries.parquet"
    results_df.to_parquet(processed_path, index=False)
    results_df.to_csv(processed_path.replace(".parquet", ".csv"), index=False)
    logger.info(f"Episode summaries saved: {processed_path}")

    # Visualisation pass
    if not args.no_viz:
        try:
            run_visualisation_pass(results_df, f"{output_dir}/raw_data", output_dir)
        except Exception as e:
            logger.error(f"Visualisation pass failed: {e}")
            logger.debug(traceback.format_exc())

    # Print quick stats
    logger.info("\n--- Quick Stats ---")
    for obj in args.objectives:
        sub = results_df[results_df["objective"] == obj]
        if len(sub) == 0:
            continue
        gini = sub["all_gini"].mean()
        floor = sub["terminal_worker_min"].mean()
        agency = sub["terminal_agency_floor"].mean()
        logger.info(f"  {obj}: gini={gini:.3f}, floor={floor:.2f}, agency={agency:.3f}")

    logger.info(f"\nResults in: {output_dir}/")
    return results_df


if __name__ == "__main__":
    main()

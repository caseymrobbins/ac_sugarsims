"""
analysis.py
-----------
Statistical analysis of simulation results.

Produces:
  - Mean outcomes per condition (SUM / NASH / JAM / CROSS / TOPO / TARGET)
  - 95% confidence intervals
  - Mann-Whitney U tests between conditions
  - Effect sizes (rank-biserial correlation)
  - Summary tables (CSV + Parquet)
  - Power-law fitting reports

Changes:
  - COMPARISON_METRICS expanded with horizon_index, firm floors, agency
  - Objectives list includes all six conditions
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Confidence interval helpers
# ---------------------------------------------------------------------------

def ci_mean(data: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return (mean, lower CI, upper CI) using t-distribution."""
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return float("nan"), float("nan"), float("nan")
    n = len(data)
    mean = data.mean()
    se = stats.sem(data)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return mean, mean - t_crit * se, mean + t_crit * se


def bootstrap_ci(data: np.ndarray, n_boot: int = 1000,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Return (lower, upper) percentile bootstrap CI."""
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boots = rng.choice(data, size=(n_boot, len(data)), replace=True).mean(axis=1)
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Mann-Whitney tests
# ---------------------------------------------------------------------------

def mannwhitney_test(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """
    Two-sided Mann-Whitney U test with rank-biserial effect size.

    Returns dict with: U, p_value, effect_size, significant (p<0.05)
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return {"U": float("nan"), "p_value": float("nan"),
                "effect_size": float("nan"), "significant": False}
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    # Rank-biserial correlation: r = 1 - 2U/(n_a * n_b)
    r = 1 - (2 * U) / (len(a) * len(b))
    return {
        "U": float(U),
        "p_value": float(p),
        "effect_size": float(r),
        "significant": bool(p < 0.05),
    }


# ---------------------------------------------------------------------------
# Condition comparison
# ---------------------------------------------------------------------------

COMPARISON_METRICS = [
    "all_gini",
    "worker_min",
    "worker_mean",
    "worker_top10_share",
    "agency_floor",
    "agency_mean",
    "agency_gini",
    "unemployment_rate",
    "hhi",
    "top_firm_share",
    "n_active_cartels",
    "total_rent_collected",
    "total_debt",
    "fraction_in_debt",
    "frac_monopoly",
    "frac_cartel",
    "frac_poverty_trap",
    "terminal_worker_min",
    "terminal_agency_floor",
    "terminal_all_gini",
    "horizon_index",
    "terminal_horizon_index",
    "mean_firm_floor",
    "min_firm_floor",
    "epistemic_health_mean",
    "epistemic_health_floor",
    "system_M",
    "system_VE",
    "system_CI",
    "system_tau_c",
    "mean_skill",
    "total_production",
    "mean_pollution",
    "trust_planner",
    "trust_institutional",
    "trust_firm_mean",
    "trust_worker_mean",
    "trust_news_capture_gap",
]


def condition_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns [objective, seed, <metrics>],
    compute per-condition means and 95% CIs.

    Returns a wide DataFrame with rows = metrics, cols = conditions.
    """
    # Dynamically detect which objectives are present
    objectives = sorted(results_df["objective"].unique().tolist())

    rows = []
    for metric in COMPARISON_METRICS:
        if metric not in results_df.columns:
            continue
        row = {"metric": metric}
        for obj in objectives:
            subset = results_df[results_df["objective"] == obj][metric].dropna().values
            if len(subset) == 0:
                row[f"{obj}_mean"] = float("nan")
                row[f"{obj}_ci_lo"] = float("nan")
                row[f"{obj}_ci_hi"] = float("nan")
            else:
                mean, ci_lo, ci_hi = ci_mean(subset)
                row[f"{obj}_mean"] = mean
                row[f"{obj}_ci_lo"] = ci_lo
                row[f"{obj}_ci_hi"] = ci_hi
        rows.append(row)

    return pd.DataFrame(rows)


def pairwise_tests_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Mann-Whitney tests for all pairs of conditions.
    Returns DataFrame with columns: metric, pair, U, p_value, effect_size, significant
    """
    objectives = sorted(results_df["objective"].unique().tolist())
    pairs = [(a, b) for i, a in enumerate(objectives) for b in objectives[i+1:]]
    rows = []

    for metric in COMPARISON_METRICS:
        if metric not in results_df.columns:
            continue
        for obj_a, obj_b in pairs:
            a = results_df[results_df["objective"] == obj_a][metric].dropna().values
            b = results_df[results_df["objective"] == obj_b][metric].dropna().values
            test = mannwhitney_test(a, b)
            rows.append({
                "metric": metric,
                "pair": f"{obj_a}_vs_{obj_b}",
                **test,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Power-law analysis
# ---------------------------------------------------------------------------

def fit_pareto(w: np.ndarray) -> Dict[str, float]:
    """
    Fit a Pareto distribution to the upper tail using MLE.
    Returns alpha (shape), x_min, and goodness-of-fit p-value.
    """
    w = np.sort(w[w > 0])
    if len(w) < 20:
        return {"alpha": float("nan"), "x_min": float("nan"), "p_value": float("nan")}

    # Estimate x_min via Clauset method (approx: use 80th percentile)
    x_min = np.percentile(w, 80)
    tail = w[w >= x_min]
    if len(tail) < 5:
        return {"alpha": float("nan"), "x_min": float(x_min), "p_value": float("nan")}

    # MLE for Pareto: alpha = n / sum(log(x / x_min))
    alpha = len(tail) / np.sum(np.log(tail / x_min))

    # KS test
    ks, p = stats.kstest(tail, "pareto", args=(alpha, 0, x_min))

    return {
        "alpha": float(alpha),
        "x_min": float(x_min),
        "p_value": float(p),
        "is_power_law": bool(p > 0.05 and alpha < 3.5),
    }


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(results_df: pd.DataFrame, output_dir: str = "results") -> Dict[str, pd.DataFrame]:
    """
    Full analysis pipeline.

    Parameters
    ----------
    results_df : DataFrame with columns [objective, seed, episode, <metrics>]
    output_dir : root output directory

    Returns
    -------
    dict with keys: 'summary', 'pairwise_tests', 'condition_means'
    """
    os.makedirs(f"{output_dir}/statistical_tests", exist_ok=True)
    os.makedirs(f"{output_dir}/summary_tables", exist_ok=True)

    outputs = {}

    # 1. Condition summary
    summary = condition_summary_table(results_df)
    summary.to_csv(f"{output_dir}/summary_tables/condition_summary.csv", index=False)
    summary.to_parquet(f"{output_dir}/summary_tables/condition_summary.parquet", index=False)
    outputs["summary"] = summary

    # 2. Pairwise statistical tests
    pairwise = pairwise_tests_table(results_df)
    pairwise.to_csv(f"{output_dir}/statistical_tests/pairwise_mannwhitney.csv", index=False)
    pairwise.to_parquet(f"{output_dir}/statistical_tests/pairwise_mannwhitney.parquet", index=False)
    outputs["pairwise_tests"] = pairwise

    # 3. Per-condition mean trajectories (if timestep data available)
    if "step" in results_df.columns:
        # Only include metrics that exist in the data
        available_metrics = [m for m in COMPARISON_METRICS if m in results_df.columns]
        if available_metrics:
            os.makedirs(f"{output_dir}/processed_data", exist_ok=True)
            traj = (results_df
                    .groupby(["objective", "step"])[available_metrics]
                    .mean()
                    .reset_index())
            traj.to_csv(f"{output_dir}/processed_data/condition_trajectories.csv", index=False)
            traj.to_parquet(f"{output_dir}/processed_data/condition_trajectories.parquet", index=False)
            outputs["trajectories"] = traj

    # 4. Print summary to console
    _print_summary(summary, pairwise)

    return outputs


def _print_summary(summary: pd.DataFrame, pairwise: pd.DataFrame):
    """Print formatted summary to stdout."""
    print("\n" + "=" * 70)
    print("CONDITION COMPARISON SUMMARY")
    print("=" * 70)
    key_metrics = [
        "all_gini", "worker_min", "agency_floor",
        "frac_monopoly", "frac_poverty_trap", "unemployment_rate",
        "horizon_index", "mean_firm_floor", "epistemic_health_mean",
        "trust_planner", "trust_institutional",
    ]

    # Detect which objectives are present
    obj_cols = [c.replace("_mean", "") for c in summary.columns if c.endswith("_mean")]

    for _, row in summary.iterrows():
        if row["metric"] not in key_metrics:
            continue
        print(f"\n{row['metric']}:")
        for obj in obj_cols:
            if f"{obj}_mean" in row:
                print(f"  {obj}: {row[f'{obj}_mean']:.4f} "
                      f"[{row[f'{obj}_ci_lo']:.4f}, {row[f'{obj}_ci_hi']:.4f}]")

    print("\n" + "=" * 70)
    print("SIGNIFICANT PAIRWISE DIFFERENCES (p < 0.05)")
    print("=" * 70)
    sig = pairwise[pairwise["significant"] == True]
    if len(sig) == 0:
        print("  (none)")
    else:
        for _, row in sig.iterrows():
            print(f"  {row['metric']} | {row['pair']} | "
                  f"p={row['p_value']:.4f} | r={row['effect_size']:.3f}")
    print()

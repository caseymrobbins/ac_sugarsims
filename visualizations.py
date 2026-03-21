"""
visualizations.py
-----------------
All visualisation routines for the multi-agent economic simulation.

Static plots (matplotlib / seaborn):
  - wealth_distribution_histograms
  - lorenz_curves
  - gini_over_time
  - floor_wealth_trajectory
  - top10_wealth_share
  - cartel_network
  - trade_network
  - agent_mobility_heatmap
  - resource_heatmap
  - wealth_powerlaw

Comparative plots:
  - compare_floor_wealth
  - compare_inequality
  - compare_wealth_distributions
  - compare_stability
  - compare_agency_floor

Animations (matplotlib FuncAnimation):
  - animate_wealth_distribution
  - animate_resource_extraction
  - animate_trade_network
  - animate_economic_stratification
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

CONDITION_COLORS = {"SUM": "#E74C3C", "NASH": "#F39C12", "JAM": "#2ECC71"}
CONDITION_LABELS = {"SUM": "Aggregate (SUM)", "NASH": "Nash Welfare", "JAM": "Floor (JAM)"}

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-run / single-condition plots
# ---------------------------------------------------------------------------

def wealth_distribution_histogram(wealth_array: np.ndarray,
                                   condition: str, step: int,
                                   output_dir: str = "results/plots"):
    """Log-scale wealth histogram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    w = wealth_array[wealth_array > 0]
    ax.hist(np.log10(w + 1), bins=40,
            color=CONDITION_COLORS.get(condition, "steelblue"),
            edgecolor="white", alpha=0.85)
    ax.set_xlabel("log₁₀(Wealth + 1)")
    ax.set_ylabel("Count")
    ax.set_title(f"Wealth Distribution — {CONDITION_LABELS.get(condition, condition)} (step {step})")
    _savefig(fig, f"{output_dir}/wealth_dist_{condition}_step{step}.png")


def lorenz_curve(wealth_array: np.ndarray, condition: str,
                 output_dir: str = "results/plots"):
    """Lorenz curve with Gini annotation."""
    fig, ax = plt.subplots(figsize=(5, 5))
    w = np.sort(wealth_array[wealth_array > 0])
    n = len(w)
    if n == 0:
        plt.close(fig)
        return
    cumulative_w = np.cumsum(w) / w.sum()
    cumulative_pop = np.linspace(0, 1, n)
    ax.plot(cumulative_pop, cumulative_w,
            color=CONDITION_COLORS.get(condition, "steelblue"), lw=2,
            label=CONDITION_LABELS.get(condition, condition))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect equality")
    gini = 1 - 2 * np.trapezoid(cumulative_w, cumulative_pop)
    ax.text(0.05, 0.85, f"Gini = {gini:.3f}", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Cumulative population share")
    ax.set_ylabel("Cumulative wealth share")
    ax.set_title(f"Lorenz Curve — {CONDITION_LABELS.get(condition, condition)}")
    ax.legend()
    _savefig(fig, f"{output_dir}/lorenz_{condition}.png")


def gini_over_time(metrics_df: pd.DataFrame, condition: str,
                   output_dir: str = "results/plots"):
    """Gini coefficient trajectory."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics_df["step"], metrics_df["all_gini"],
            color=CONDITION_COLORS.get(condition, "steelblue"), lw=1.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Gini coefficient")
    ax.set_title(f"Gini over Time — {CONDITION_LABELS.get(condition, condition)}")
    ax.set_ylim(0, 1)
    _savefig(fig, f"{output_dir}/gini_trajectory_{condition}.png")


def floor_wealth_trajectory(metrics_df: pd.DataFrame, condition: str,
                             output_dir: str = "results/plots"):
    """Minimum wealth trajectory."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics_df["step"], metrics_df["worker_min"],
            color=CONDITION_COLORS.get(condition, "steelblue"), lw=1.5)
    ax.axhline(y=1.0, color="red", ls="--", lw=1, label="Survival threshold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Minimum worker wealth")
    ax.set_title(f"Floor Wealth — {CONDITION_LABELS.get(condition, condition)}")
    ax.legend()
    _savefig(fig, f"{output_dir}/floor_wealth_{condition}.png")


def top10_wealth_share(metrics_df: pd.DataFrame, condition: str,
                       output_dir: str = "results/plots"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics_df["step"], metrics_df["all_top10_share"],
            color=CONDITION_COLORS.get(condition, "steelblue"), lw=1.5)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Top 10% wealth share")
    ax.set_title(f"Top 10% Wealth Share — {CONDITION_LABELS.get(condition, condition)}")
    ax.set_ylim(0, 1)
    _savefig(fig, f"{output_dir}/top10_share_{condition}.png")


def cartel_network_plot(active_cartels: Dict, firms: List,
                        condition: str, step: int,
                        output_dir: str = "results/plots"):
    """Visualise cartel membership as a graph."""
    G = nx.Graph()
    for cartel_id, members in active_cartels.items():
        members = list(members)
        for i, m in enumerate(members):
            G.add_node(m, cartel=cartel_id)
            for other in members[i + 1:]:
                G.add_edge(m, other)

    if G.number_of_nodes() == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=42)
    cartels = list({d["cartel"] for _, d in G.nodes(data=True)})
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(cartels), 1)))
    color_map = {c: palette[i] for i, c in enumerate(cartels)}
    node_colors = [color_map[G.nodes[n]["cartel"]] for n in G.nodes]
    nx.draw(G, pos, node_color=node_colors, with_labels=False,
            node_size=80, ax=ax, alpha=0.8)
    ax.set_title(f"Cartel Network — {condition} (step {step})")
    _savefig(fig, f"{output_dir}/cartel_network_{condition}_step{step}.png")


def trade_network_plot(trade_graph: nx.DiGraph, condition: str, step: int,
                       output_dir: str = "results/plots"):
    """Plot trade network with edge weights."""
    if trade_graph.number_of_nodes() < 2:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    # Sample subgraph for readability
    nodes = list(trade_graph.nodes())
    if len(nodes) > 200:
        nodes = np.random.choice(nodes, 200, replace=False).tolist()
    sg = trade_graph.subgraph(nodes)
    pos = nx.spring_layout(sg, seed=42, k=0.3)
    weights = [d.get("weight", 1.0) for _, _, d in sg.edges(data=True)]
    max_w = max(weights) if weights else 1
    widths = [max(0.3, 2.5 * w / max_w) for w in weights]
    centrality = nx.degree_centrality(sg)
    node_sizes = [200 * centrality.get(n, 0.01) + 20 for n in sg.nodes()]
    nx.draw_networkx(sg, pos, node_size=node_sizes, width=widths,
                     with_labels=False, alpha=0.7,
                     node_color="steelblue", edge_color="grey", ax=ax)
    ax.set_title(f"Trade Network — {condition} (step {step})")
    _savefig(fig, f"{output_dir}/trade_network_{condition}_step{step}.png")


def agent_mobility_heatmap(worker_positions: List[Tuple[int, int]],
                            grid_width: int, grid_height: int,
                            condition: str, step: int,
                            output_dir: str = "results/plots"):
    """2D density heatmap of worker positions."""
    grid = np.zeros((grid_width, grid_height))
    for x, y in worker_positions:
        grid[x, y] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid.T, origin="lower", cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Agent count")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Agent Density Map — {condition} (step {step})")
    _savefig(fig, f"{output_dir}/mobility_heatmap_{condition}_step{step}.png")


def resource_heatmap(resource_array: np.ndarray,
                     resource: str,
                     grid_width: int, grid_height: int,
                     condition: str, step: int,
                     output_dir: str = "results/plots"):
    """Heatmap of a single resource type.  resource_array is a (W,H) ndarray."""
    grid = resource_array
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid.T, origin="lower", cmap="Greens", aspect="auto")
    plt.colorbar(im, ax=ax, label=f"{resource} level")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{resource.capitalize()} Resources — {condition} (step {step})")
    _savefig(fig, f"{output_dir}/resource_heatmap_{resource}_{condition}_step{step}.png")


def wealth_powerlaw_plot(wealth_array: np.ndarray, condition: str,
                         output_dir: str = "results/plots"):
    """Log-log rank-wealth plot with Pareto fit."""
    w = np.sort(wealth_array[wealth_array > 0])[::-1]
    if len(w) < 10:
        return
    ranks = np.arange(1, len(w) + 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(ranks, w, ".", ms=3, alpha=0.6,
              color=CONDITION_COLORS.get(condition, "steelblue"),
              label="Data")

    # Fit Pareto to upper tail
    tail_w = w[:max(10, len(w) // 5)]
    tail_r = ranks[:len(tail_w)]
    if len(tail_w) > 5:
        log_r = np.log(tail_r)
        log_w = np.log(tail_w)
        slope, intercept, *_ = stats.linregress(log_r, log_w)
        fitted = np.exp(intercept) * tail_r ** slope
        ax.loglog(tail_r, fitted, "r--", lw=2,
                  label=f"Power-law fit (α={-slope:.2f})")

    ax.set_xlabel("Wealth rank")
    ax.set_ylabel("Wealth")
    ax.set_title(f"Wealth Power-Law — {CONDITION_LABELS.get(condition, condition)}")
    ax.legend()
    _savefig(fig, f"{output_dir}/powerlaw_{condition}.png")


# ---------------------------------------------------------------------------
# Comparative plots
# ---------------------------------------------------------------------------

def compare_floor_wealth(trajectories: Dict[str, pd.DataFrame],
                         output_dir: str = "results/plots"):
    """Compare minimum worker wealth across conditions."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond, df in trajectories.items():
        if "worker_min" not in df.columns:
            continue
        ax.plot(df["step"], df["worker_min"],
                color=CONDITION_COLORS.get(cond, "grey"),
                lw=2, label=CONDITION_LABELS.get(cond, cond))
    ax.axhline(y=1.0, color="black", ls="--", lw=1, label="Survival threshold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Minimum worker wealth")
    ax.set_title("Floor Wealth Comparison")
    ax.legend()
    _savefig(fig, f"{output_dir}/compare_floor_wealth.png")


def compare_inequality(trajectories: Dict[str, pd.DataFrame],
                       output_dir: str = "results/plots"):
    """Compare Gini coefficient over time."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond, df in trajectories.items():
        if "all_gini" not in df.columns:
            continue
        ax.plot(df["step"], df["all_gini"],
                color=CONDITION_COLORS.get(cond, "grey"),
                lw=2, label=CONDITION_LABELS.get(cond, cond))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Inequality Comparison (Gini)")
    ax.set_ylim(0, 1)
    ax.legend()
    _savefig(fig, f"{output_dir}/compare_inequality.png")


def compare_wealth_distributions(final_wealth: Dict[str, np.ndarray],
                                  output_dir: str = "results/plots"):
    """Overlaid KDE of final wealth distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, w in final_wealth.items():
        w = w[w > 0]
        if len(w) < 5:
            continue
        log_w = np.log10(w + 1)
        kde_x = np.linspace(log_w.min(), log_w.max(), 300)
        kde = stats.gaussian_kde(log_w)
        ax.plot(kde_x, kde(kde_x),
                color=CONDITION_COLORS.get(cond, "grey"),
                lw=2, label=CONDITION_LABELS.get(cond, cond))
        ax.fill_between(kde_x, kde(kde_x), alpha=0.15,
                        color=CONDITION_COLORS.get(cond, "grey"))
    ax.set_xlabel("log₁₀(Wealth + 1)")
    ax.set_ylabel("Density")
    ax.set_title("Final Wealth Distribution Comparison")
    ax.legend()
    _savefig(fig, f"{output_dir}/compare_wealth_distributions.png")


def compare_stability(trajectories: Dict[str, pd.DataFrame],
                      output_dir: str = "results/plots"):
    """Compare unemployment and n_workers over time."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for cond, df in trajectories.items():
        color = CONDITION_COLORS.get(cond, "grey")
        label = CONDITION_LABELS.get(cond, cond)
        if "unemployment_rate" in df.columns:
            axes[0].plot(df["step"], df["unemployment_rate"],
                         color=color, lw=1.5, label=label)
        if "n_workers" in df.columns:
            axes[1].plot(df["step"], df["n_workers"],
                         color=color, lw=1.5, label=label)

    axes[0].set_title("Unemployment Rate")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Rate")
    axes[0].legend()
    axes[1].set_title("Worker Population")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("N workers")
    axes[1].legend()
    fig.suptitle("System Stability Metrics")
    _savefig(fig, f"{output_dir}/compare_stability.png")


def compare_agency_floor(trajectories: Dict[str, pd.DataFrame],
                         output_dir: str = "results/plots"):
    """Compare agency floor over time."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for cond, df in trajectories.items():
        if "agency_floor" not in df.columns:
            continue
        ax.plot(df["step"], df["agency_floor"],
                color=CONDITION_COLORS.get(cond, "grey"),
                lw=2, label=CONDITION_LABELS.get(cond, cond))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Agency floor (min agency)")
    ax.set_title("Agency Floor Comparison (JAM Objective)")
    ax.legend()
    _savefig(fig, f"{output_dir}/compare_agency_floor.png")


def summary_bar_chart(summary_df: pd.DataFrame,
                      metric: str,
                      output_dir: str = "results/plots"):
    """Bar chart comparing a single metric across conditions."""
    objectives = ["SUM", "NASH", "JAM"]
    means = [summary_df.loc[summary_df["metric"] == metric, f"{o}_mean"].values[0]
             for o in objectives if metric in summary_df["metric"].values]
    ci_lo = [summary_df.loc[summary_df["metric"] == metric, f"{o}_ci_lo"].values[0]
             for o in objectives]
    ci_hi = [summary_df.loc[summary_df["metric"] == metric, f"{o}_ci_hi"].values[0]
             for o in objectives]

    if not means:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [CONDITION_COLORS[o] for o in objectives]
    yerr_lo = [m - l for m, l in zip(means, ci_lo)]
    yerr_hi = [h - m for m, h in zip(means, ci_hi)]
    ax.bar(objectives, means, color=colors, alpha=0.85,
           yerr=[yerr_lo, yerr_hi], capsize=6)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by Objective")
    _savefig(fig, f"{output_dir}/bar_{metric}.png")


# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------

def animate_wealth_distribution(metrics_history: List[Dict],
                                  condition: str,
                                  output_dir: str = "results/animations"):
    """Animate Gini + floor wealth as two time-series panels."""
    os.makedirs(output_dir, exist_ok=True)
    steps = [m["step"] for m in metrics_history]
    ginis = [m.get("all_gini", 0) for m in metrics_history]
    floors = [m.get("worker_min", 0) for m in metrics_history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    line_gini, = axes[0].plot([], [], color=CONDITION_COLORS.get(condition, "steelblue"), lw=2)
    line_floor, = axes[1].plot([], [], color=CONDITION_COLORS.get(condition, "steelblue"), lw=2)
    axes[0].set_xlim(0, max(steps))
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Gini")
    axes[0].set_title("Inequality (Gini)")
    axes[1].set_xlim(0, max(steps))
    axes[1].set_ylim(0, max(floors) * 1.2 + 1)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Min Wealth")
    axes[1].set_title("Floor Wealth")
    fig.suptitle(f"Economic Evolution — {CONDITION_LABELS.get(condition, condition)}")

    def update(frame):
        line_gini.set_data(steps[:frame], ginis[:frame])
        line_floor.set_data(steps[:frame], floors[:frame])
        return line_gini, line_floor

    # Use every 5th frame for speed
    frames = list(range(1, len(steps), max(1, len(steps) // 100)))
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=True)
    path = f"{output_dir}/wealth_evolution_{condition}.gif"
    try:
        ani.save(path, writer="pillow", fps=15)
    except Exception:
        pass  # Skip if pillow writer unavailable
    plt.close(fig)


def animate_resource_extraction(resource_snapshots: List[np.ndarray],
                                  resource: str, condition: str,
                                  output_dir: str = "results/animations"):
    """Animate resource level over time."""
    os.makedirs(output_dir, exist_ok=True)
    if not resource_snapshots:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(resource_snapshots[0].T, origin="lower",
                   cmap="Greens", vmin=0, vmax=resource_snapshots[0].max(),
                   aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{resource.capitalize()} Extraction — {condition}")
    title = ax.set_title("")

    def update(frame):
        im.set_data(resource_snapshots[frame].T)
        title.set_text(f"{resource.capitalize()} at step {frame * 10}")
        return im, title

    frames = list(range(len(resource_snapshots)))
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    path = f"{output_dir}/resource_{resource}_{condition}.gif"
    try:
        ani.save(path, writer="pillow", fps=8)
    except Exception:
        pass
    plt.close(fig)


def animate_economic_stratification(position_history: List[Dict],
                                     condition: str, grid_size: int = 80,
                                     output_dir: str = "results/animations"):
    """
    Animate agent wealth as coloured dots on the grid.
    position_history: list of dicts {agent_id: (x, y, wealth)}
    """
    os.makedirs(output_dir, exist_ok=True)
    if not position_history:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter([], [], c=[], cmap="RdYlGn",
                         s=10, vmin=0, vmax=500, alpha=0.7)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_title(f"Economic Stratification — {condition}")
    plt.colorbar(scatter, ax=ax, label="Wealth")

    def update(frame):
        data = position_history[frame]
        if not data:
            return scatter,
        xs = [v[0] for v in data.values()]
        ys = [v[1] for v in data.values()]
        ws = [min(v[2], 500) for v in data.values()]
        scatter.set_offsets(np.column_stack([xs, ys]))
        scatter.set_array(np.array(ws))
        ax.set_title(f"Stratification — {condition} (step {frame * 10})")
        return scatter,

    frames = list(range(len(position_history)))
    ani = animation.FuncAnimation(fig, update, frames=frames, blit=False)
    path = f"{output_dir}/stratification_{condition}.gif"
    try:
        ani.save(path, writer="pillow", fps=8)
    except Exception:
        pass
    plt.close(fig)


# ---------------------------------------------------------------------------
# Master plotting function
# ---------------------------------------------------------------------------

def generate_all_plots(
    metrics_history: List[Dict],
    condition: str,
    final_wealth: np.ndarray,
    food_grid: np.ndarray = None,   # (W,H) numpy array
    grid_width: int = 80,
    grid_height: int = 80,
    active_cartels: Dict = None,
    trade_graph=None,
    worker_positions: List[Tuple] = None,
    output_dir: str = "results",
):
    """Generate all static plots for a single condition/run."""
    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    metrics_df = pd.DataFrame(metrics_history)

    # Single-condition plots
    if len(final_wealth) > 0:
        wealth_distribution_histogram(final_wealth, condition, metrics_df["step"].max(), plots_dir)
        lorenz_curve(final_wealth, condition, plots_dir)
        wealth_powerlaw_plot(final_wealth, condition, plots_dir)

    if not metrics_df.empty:
        gini_over_time(metrics_df, condition, plots_dir)
        floor_wealth_trajectory(metrics_df, condition, plots_dir)
        top10_wealth_share(metrics_df, condition, plots_dir)

    if active_cartels:
        cartel_network_plot(active_cartels, [], condition,
                            metrics_df["step"].max() if not metrics_df.empty else 0,
                            plots_dir)

    if trade_graph is not None:
        trade_network_plot(trade_graph, condition,
                           metrics_df["step"].max() if not metrics_df.empty else 0,
                           plots_dir)

    if worker_positions:
        agent_mobility_heatmap(worker_positions, grid_width, grid_height,
                               condition,
                               metrics_df["step"].max() if not metrics_df.empty else 0,
                               plots_dir)

    if food_grid is not None:
        resource_heatmap(food_grid, "food", grid_width, grid_height,
                         condition,
                         metrics_df["step"].max() if not metrics_df.empty else 0,
                         plots_dir)

# Multi-Agent Economic Simulation Platform

> **Research-grade agent-based model for studying how system-level objective functions shape inequality, exploitation, and economic power concentration.**

---

## Research Question

In decentralised economies with heterogeneous agents and realistic economic mechanisms:

> Do systems optimised for **aggregate output** naturally produce inequality, exploitation, and floor collapse?
> Does **floor optimisation** structurally prevent these outcomes?

---

## Architecture

```
ac_sugarsims/
├── agents.py          # WorkerAgent, FirmAgent, LandownerAgent
├── environment.py     # EconomicModel: 80x80 grid, Mesa scheduler
├── economy.py         # Loan market, bilateral trade, price dynamics, NetworkX trade graph
├── planner.py         # PlannerAgent: SUM / NASH / JAM objective functions
├── metrics.py         # Step-level metrics, emergence detection, episode summaries
├── analysis.py        # Statistical analysis, Mann-Whitney tests, summary tables
├── visualizations.py  # All plots and animations
├── run_experiments.py # Parallel experiment runner (multiprocessing)
└── results/
    ├── raw_data/
    ├── processed_data/
    ├── plots/
    ├── animations/
    ├── statistical_tests/
    ├── summary_tables/
    └── logs/
```

---

## Installation

```bash
pip install mesa numpy pandas scipy matplotlib seaborn plotly networkx numba
```

Python 3.11+ required.

---

## Quick Start

### Test run (10 seeds x 150 steps)

```bash
python run_experiments.py --test
```

### Full experiment (20 seeds x 1500 steps)

```bash
python run_experiments.py
```

### Custom

```bash
python run_experiments.py --objectives SUM JAM --seeds 5 --steps 300 --workers 4
```

---

## Objective Functions

| Name | Formula | Expected behaviour |
|------|---------|-------------------|
| **SUM** | `R = sum(wealth_i)` | High output, high inequality, floor collapse |
| **NASH** | `R = sum(log(wealth_i + e))` | Moderate inequality, partial redistribution |
| **JAM** | `R = log(min agency_i)` | Stable floor, structural protection of least advantaged |

The planner uses **gradient-free hill climbing** to optimise its policy instruments (tax rates, UBI, minimum wage) under the assigned objective. Policy emerges from the objective -- nothing is hard-coded.

---

## Agent Types

### WorkerAgent
- Attributes: wealth, skill, metabolism, risk tolerance, mobility, network connections
- Actions: harvest, seek employment, trade, migrate, borrow, reproduce, die

### FirmAgent
- Production: Cobb-Douglas Y = A * K^alpha * L^(1-alpha)
- Strategic wage setting; natural cartel formation

### LandownerAgent
- Controls territory; extracts rent proportional to tenant income
- Expands holdings; adjusts rent dynamically

### PlannerAgent
- Policy instruments: tax rates (worker / firm / landowner), UBI, minimum wage, harvest limit
- Optimised under SUM / NASH / JAM every 25 steps

---

## Emergent Dynamics

The following emerge from agent interactions (not hard-coded):

| Phenomenon | Mechanism |
|-----------|-----------|
| Wealth inequality / power-law | Compounding returns, skill heterogeneity |
| Rent extraction | Landowner territory control |
| Wage suppression | Cartel formation, monopsony |
| Debt traps | Compounding interest, credit inequality |
| Market monopoly | Firm profit accumulation, bankruptcy of rivals |
| Poverty traps | Metabolism costs, floor compression |

---

## Metrics Tracked (every step)

- Wealth: min, median, mean, max, std, Gini, top 1/5/10% share
- Power-law exponent (Hill estimator)
- Firm: HHI, top firm share, cartel count
- Debt: outstanding, Gini, default rate
- Labour: unemployment rate, mean wage, mobility
- Trade network: density, max centrality
- Agency floor (JAM objective proxy)
- Planner policy snapshot

---

## Emergence Detection

Automatic detection at each step:
- **Monopoly**: single firm > 40% market share
- **Cartel**: active firm coalitions for wage/price coordination
- **Power-law wealth**: Hill estimator alpha < 3 in upper tail
- **Poverty trap**: bottom quintile < 2x survival threshold

---

## Outputs

### Plots (results/plots/)
- Wealth distribution histograms
- Lorenz curves
- Gini over time
- Floor wealth trajectories
- Top 10% wealth share
- Cartel network graphs
- Trade network graphs
- Agent mobility heatmaps
- Resource extraction heatmaps
- Power-law rank-wealth plots
- Comparative: SUM vs NASH vs JAM for all key metrics

### Animations (results/animations/)
- Wealth distribution evolution (Gini + floor wealth)
- Resource extraction over time
- Economic stratification (agent wealth on grid)

### Statistical Tests (results/statistical_tests/)
- Pairwise Mann-Whitney U tests
- Effect sizes (rank-biserial correlation)
- Significance flags (p < 0.05)

### Summary Tables (results/summary_tables/)
- Per-condition means + 95% CI for all metrics
- CSV and Parquet formats

---

## Full Experiment Protocol

| Parameter | Value |
|-----------|-------|
| Conditions | SUM, NASH, JAM |
| Seeds | 20 per condition |
| Steps per episode | 1500 |
| Total episodes | 60 |
| Grid | 80 x 80 |
| Workers | 400 |
| Firms | 20 |
| Landowners | 15 |

---

## Google Colab

```python
!pip install mesa numpy pandas scipy matplotlib seaborn plotly networkx numba
!git clone <repo-url> && cd ac_sugarsims
!python run_experiments.py --test --workers 1
```

---

## Reproducibility

All experiments use fixed NumPy RNG seeds (np.random.default_rng(seed)).
Results are fully deterministic for a given seed and parameter set.

---

## References

- Epstein & Axtell (1996) Growing Artificial Societies
- Axelrod (1997) The Complexity of Cooperation
- Piketty (2014) Capital in the Twenty-First Century
- Rawls (1971) A Theory of Justice (JAM / maximin principle)
- Clauset, Shalizi & Newman (2009) Power-law distributions in empirical data

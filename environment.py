"""
environment.py
--------------
80x80 grid economic world — Mesa 3.x compatible.

Resources are stored as numpy arrays (food_grid, raw_grid, land_grid,
capital_grid) which allows vectorised / JIT-compiled regeneration via
the backend selected by hardware.py.

Agents access resources via model.food_grid[x, y] etc.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from mesa import Model
from mesa.space import MultiGrid

from agents import WorkerAgent, FirmAgent, LandownerAgent
from economy import Economy
from planner import PlannerAgent
from hardware import auto_configure, build_regen_fn, AccelConfig


# ---------------------------------------------------------------------------
# Resource caps (used throughout)
# ---------------------------------------------------------------------------

FOOD_CAP       = 30.0
RAW_CAP        = 25.0
LAND_CAP       = 20.0
CAPITAL_CAP    = 50.0


# ---------------------------------------------------------------------------
# EconomicModel
# ---------------------------------------------------------------------------

class EconomicModel(Model):
    """
    Main Mesa 3.x model.

    Parameters
    ----------
    seed : int
    grid_width, grid_height : int
    n_workers, n_firms, n_landowners : int
    objective : str  — 'SUM', 'NASH', or 'JAM'
    accel_config : AccelConfig or None — if None, auto-detected
    """

    def __init__(self,
                 seed: int = 42,
                 grid_width: int = 80,
                 grid_height: int = 80,
                 n_workers: int = 400,
                 n_firms: int = 20,
                 n_landowners: int = 15,
                 objective: str = "SUM",
                 accel_config: AccelConfig = None):

        super().__init__(seed=seed)

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.n_workers_initial = n_workers
        self.objective = objective
        self.current_step = 0

        # Acceleration config (auto-detect if not supplied)
        self._accel = accel_config or auto_configure(verbose=False)
        self._regen_fn = build_regen_fn(self._accel.backend)

        # Mesa MultiGrid
        self.grid = MultiGrid(grid_width, grid_height, torus=False)

        # Resource grids as numpy arrays (W x H, float64)
        (self.food_grid,
         self.raw_grid,
         self.land_grid,
         self.capital_grid) = self._init_resource_arrays()

        # Ownership map: cell -> landowner unique_id
        self.cell_ownership: Dict[Tuple[int, int], int] = {}

        # Agent lists (maintained for fast iteration)
        self.workers: List[WorkerAgent] = []
        self.firms: List[FirmAgent] = []
        self.landowners: List[LandownerAgent] = []

        # Cartel tracking
        self._cartel_counter: int = 0
        self.active_cartels: Dict[int, set] = {}

        # Lookup cache: unique_id -> agent
        self._id_cache: Dict[int, object] = {}

        # Economy and planner (planner needs objective before agents)
        self.economy = Economy(self)
        self.planner = PlannerAgent(model=self)

        # Place agents
        self._create_workers(n_workers)
        self._create_firms(n_firms)
        self._create_landowners(n_landowners)

        # Metrics history
        self.metrics_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Resource initialisation
    # ------------------------------------------------------------------

    def _init_resource_arrays(self):
        """
        Create four (W x H) numpy float64 arrays with Gaussian-blob clusters.
        """
        W, H = self.grid_width, self.grid_height
        food    = self._gaussian_clusters(W, H, n_clusters=12, peak=FOOD_CAP)
        raw_mat = self._gaussian_clusters(W, H, n_clusters=10, peak=RAW_CAP)
        land    = self._gaussian_clusters(W, H, n_clusters=8,  peak=LAND_CAP)
        capital = np.zeros((W, H), dtype=np.float64)
        return food, raw_mat, land, capital

    def _gaussian_clusters(self, W: int, H: int,
                            n_clusters: int, peak: float,
                            sigma: float = 10.0) -> np.ndarray:
        grid = np.zeros((W, H), dtype=np.float64)
        for _ in range(n_clusters):
            cx = int(self.rng.integers(0, W))
            cy = int(self.rng.integers(0, H))
            amplitude = float(self.rng.uniform(0.5, 1.0)) * peak
            xs = np.arange(W)[:, np.newaxis]
            ys = np.arange(H)[np.newaxis, :]
            grid += amplitude * np.exp(
                -(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))
        return np.clip(grid, 0, peak)

    # ------------------------------------------------------------------
    # Agent creation
    # ------------------------------------------------------------------

    def _random_cell(self) -> Tuple[int, int]:
        return (int(self.rng.integers(0, self.grid_width)),
                int(self.rng.integers(0, self.grid_height)))

    def _create_workers(self, n: int):
        for _ in range(n):
            agent = WorkerAgent(model=self)
            pos = self._random_cell()
            self.grid.place_agent(agent, pos)
            self.workers.append(agent)
            self._id_cache[agent.unique_id] = agent

    def _create_firms(self, n: int):
        for _ in range(n):
            agent = FirmAgent(model=self)
            pos = self._random_cell()
            self.grid.place_agent(agent, pos)
            self.firms.append(agent)
            self._id_cache[agent.unique_id] = agent

    def _create_landowners(self, n: int):
        for _ in range(n):
            agent = LandownerAgent(model=self)
            pos = self._random_cell()
            self.grid.place_agent(agent, pos)
            self.landowners.append(agent)
            self._id_cache[agent.unique_id] = agent

            territory_size = int(self.rng.integers(8, 25))
            cx, cy = pos
            for _ in range(territory_size):
                dx = int(self.rng.integers(-4, 5))
                dy = int(self.rng.integers(-4, 5))
                nx = int(np.clip(cx + dx, 0, self.grid_width - 1))
                ny = int(np.clip(cy + dy, 0, self.grid_height - 1))
                cell = (nx, ny)
                if cell not in self.cell_ownership:
                    agent.controlled_cells.append(cell)
                    self.cell_ownership[cell] = agent.unique_id

    # ------------------------------------------------------------------
    # Mesa step
    # ------------------------------------------------------------------

    def step(self):
        self.current_step += 1

        # Accelerated resource regeneration (backend-dependent)
        self._regen_fn(self.food_grid, self.raw_grid, self.land_grid)

        # Planner first
        self.planner.step()

        # Shuffle-step all agents
        agent_list = list(self.workers) + list(self.firms) + list(self.landowners)
        indices = self.rng.permutation(len(agent_list))
        for i in indices:
            a = agent_list[i]
            if hasattr(a, "defunct") and a.defunct:
                continue
            a.step()

        # Economy post-processing
        self.economy.update_prices()
        self.economy.service_loans()
        self._update_market_shares()
        self.economy.refresh_trade_network()

        # Register any new workers spawned this step
        for w in self.workers:
            if w.unique_id not in self._id_cache:
                self._id_cache[w.unique_id] = w

        # Collect metrics
        from metrics import collect_step_metrics
        self.metrics_history.append(collect_step_metrics(self))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_market_shares(self):
        total = sum(f.production_this_step for f in self.firms if not f.defunct)
        if total <= 0:
            return
        for f in self.firms:
            if not f.defunct:
                f.market_share = f.production_this_step / total

    def next_cartel_id(self) -> int:
        self._cartel_counter += 1
        return self._cartel_counter

    def get_agent_by_id(self, uid: int):
        return self._id_cache.get(uid)

    def get_landowner_at(self, pos: Tuple[int, int]) -> Optional[LandownerAgent]:
        owner_id = self.cell_ownership.get(pos)
        if owner_id is None:
            return None
        a = self.get_agent_by_id(owner_id)
        return a if isinstance(a, LandownerAgent) else None

    def get_all_agent_wealths(self) -> np.ndarray:
        return np.array(
            [w.wealth for w in self.workers]
            + [f.wealth for f in self.firms if not f.defunct]
            + [lo.wealth for lo in self.landowners],
            dtype=float,
        )

    def get_worker_wealths(self) -> np.ndarray:
        return np.array([w.wealth for w in self.workers], dtype=float)

    # ------------------------------------------------------------------
    # Resource snapshot (for visualisations)
    # ------------------------------------------------------------------

    def resource_snapshot(self, resource: str) -> np.ndarray:
        """Return a copy of a resource grid array by name."""
        mapping = {
            "food": self.food_grid,
            "raw_materials": self.raw_grid,
            "land": self.land_grid,
            "capital": self.capital_grid,
        }
        arr = mapping.get(resource, self.food_grid)
        return arr.copy()

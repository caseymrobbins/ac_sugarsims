"""
environment.py — 80x80 grid economic world, Mesa 3.x.

Grid arrays (W x H numpy float64):
  food_grid, raw_grid, land_grid, water_grid, capital_grid, pollution_grid

Planner-budget bonuses (set by planner.py each step):
  _agriculture_bonus    → multiplied into regen rates
  _infrastructure_level → multiplier on all production (TFP)
  _healthcare_bonus     → reduces agent metabolism cost
  _education_quality    → improves new-agent skill gain

Pollution dynamics (each step):
  1. regen_fn applies per-cell natural decay (POLLUTION_DECAY)
  2. NumPy-roll diffusion spreads pollution to neighbours (POLLUTION_DIFFUSE)
  3. planner cleanup_investment uniformly reduces pollution_grid
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from mesa import Model
from mesa.space import MultiGrid

from agents import WorkerAgent, FirmAgent, LandownerAgent
from economy import Economy
from planner import PlannerAgent
from hardware import (auto_configure, build_regen_fn, AccelConfig,
                      FOOD_CAP, WATER_CAP, RAW_CAP, LAND_CAP,
                      POLLUTION_CAP, POLLUTION_DIFFUSE)


class EconomicModel(Model):

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

        self.grid_width  = grid_width
        self.grid_height = grid_height
        self.n_workers_initial = n_workers
        self.objective = objective
        self.current_step = 0

        # Planner-managed bonuses (updated each step by PlannerAgent)
        self._agriculture_bonus:    float = 1.0   # regen rate multiplier
        self._infrastructure_level: float = 1.0   # TFP multiplier
        self._healthcare_bonus:     float = 0.0   # metabolism reduction
        self._education_quality:    float = 1.0   # skill-gain multiplier

        # Acceleration
        self._accel   = accel_config or auto_configure(verbose=False)
        self._regen_fn = build_regen_fn(self._accel.backend)

        # Grid
        self.grid = MultiGrid(grid_width, grid_height, torus=False)

        # Resource arrays
        (self.food_grid,
         self.raw_grid,
         self.land_grid,
         self.water_grid,
         self.pollution_grid,
         self.capital_grid) = self._init_resource_arrays()

        # Cell ownership: (x,y) -> landowner unique_id
        self.cell_ownership: Dict[Tuple[int,int], int] = {}

        # Agent lists
        self.workers:    List[WorkerAgent]    = []
        self.firms:      List[FirmAgent]      = []
        self.landowners: List[LandownerAgent] = []

        # Cartel tracking
        self._cartel_counter = 0
        self.active_cartels: Dict[int, set] = {}

        # Lookup cache
        self._id_cache: Dict[int, object] = {}

        # Economy and planner (planner needs objective before agent creation)
        self.economy = Economy(self)
        self.planner = PlannerAgent(model=self)

        # Populate grid
        self._create_workers(n_workers)
        self._create_firms(n_firms)
        self._create_landowners(n_landowners)

        # Metrics history
        self.metrics_history: List[Dict] = []

    # ── Resource initialisation ───────────────────────────────────────────────

    def _init_resource_arrays(self):
        W, H = self.grid_width, self.grid_height
        food      = self._gaussian_clusters(W, H, 12, FOOD_CAP,  sigma=10.0)
        raw_mat   = self._gaussian_clusters(W, H, 10, RAW_CAP,   sigma=9.0)
        land      = self._gaussian_clusters(W, H,  8, LAND_CAP,  sigma=12.0)
        water     = self._gaussian_clusters(W, H, 15, WATER_CAP, sigma=8.0)
        pollution = np.zeros((W, H), dtype=np.float64)   # starts clean
        capital   = np.zeros((W, H), dtype=np.float64)
        return food, raw_mat, land, water, pollution, capital

    def _gaussian_clusters(self, W, H, n_clusters, peak, sigma=10.0):
        grid = np.zeros((W, H), dtype=np.float64)
        for _ in range(n_clusters):
            cx = int(self.rng.integers(0, W))
            cy = int(self.rng.integers(0, H))
            amp = float(self.rng.uniform(0.5, 1.0)) * peak
            xs = np.arange(W)[:, None]
            ys = np.arange(H)[None, :]
            grid += amp * np.exp(-((xs-cx)**2 + (ys-cy)**2) / (2*sigma**2))
        return np.clip(grid, 0, peak)

    # ── Agent creation ────────────────────────────────────────────────────────

    def _random_cell(self):
        return (int(self.rng.integers(0, self.grid_width)),
                int(self.rng.integers(0, self.grid_height)))

    def _create_workers(self, n):
        for _ in range(n):
            a = WorkerAgent(model=self)
            pos = self._random_cell()
            self.grid.place_agent(a, pos)
            self.workers.append(a)
            self._id_cache[a.unique_id] = a

    def _create_firms(self, n):
        for _ in range(n):
            a = FirmAgent(model=self)
            pos = self._random_cell()
            self.grid.place_agent(a, pos)
            self.firms.append(a)
            self._id_cache[a.unique_id] = a

    def _create_landowners(self, n):
        for _ in range(n):
            a = LandownerAgent(model=self)
            pos = self._random_cell()
            self.grid.place_agent(a, pos)
            self.landowners.append(a)
            self._id_cache[a.unique_id] = a
            cx, cy = pos
            territory_size = int(self.rng.integers(8, 25))
            for _ in range(territory_size):
                dx = int(self.rng.integers(-4, 5))
                dy = int(self.rng.integers(-4, 5))
                nx = int(np.clip(cx+dx, 0, self.grid_width-1))
                ny = int(np.clip(cy+dy, 0, self.grid_height-1))
                cell = (nx, ny)
                if cell not in self.cell_ownership:
                    a.controlled_cells.append(cell)
                    self.cell_ownership[cell] = a.unique_id

    # ── Mesa step ─────────────────────────────────────────────────────────────

    def step(self):
        self.current_step += 1

        # Resource regeneration — agriculture_bonus scales regen passes
        agri = self._agriculture_bonus
        for _ in range(max(1, int(agri))):
            self._regen_fn(self.food_grid, self.raw_grid,
                           self.land_grid, self.water_grid,
                           self.pollution_grid)

        # Pollution spatial diffusion: spread fraction to 4 neighbours
        neighbours = (
            np.roll(self.pollution_grid,  1, axis=0) +
            np.roll(self.pollution_grid, -1, axis=0) +
            np.roll(self.pollution_grid,  1, axis=1) +
            np.roll(self.pollution_grid, -1, axis=1)
        ) / 4.0
        self.pollution_grid += POLLUTION_DIFFUSE * (neighbours - self.pollution_grid)
        np.clip(self.pollution_grid, 0, POLLUTION_CAP, out=self.pollution_grid)

        # Planner first (sets bonuses, runs elections, redistributes)
        self.planner.step()

        # Shuffle-step all agents
        agent_list = list(self.workers) + list(self.firms) + list(self.landowners)
        idx = self.rng.permutation(len(agent_list))
        for i in idx:
            a = agent_list[i]
            if getattr(a, "defunct", False):
                continue
            a.step()

        # Economy post-processing
        self.economy.update_prices()
        self.economy.service_loans()
        self.economy.update_market_sentiment()
        self._update_market_shares()
        self.economy.refresh_trade_network()

        # Register any newly spawned workers
        for w in self.workers:
            if w.unique_id not in self._id_cache:
                self._id_cache[w.unique_id] = w

        # Collect metrics
        from metrics import collect_step_metrics
        self.metrics_history.append(collect_step_metrics(self))

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def infrastructure_bonus(self) -> float:
        return self._infrastructure_level

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

    def get_landowner_at(self, pos) -> Optional[LandownerAgent]:
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
            dtype=float)

    def get_worker_wealths(self) -> np.ndarray:
        return np.array([w.wealth for w in self.workers], dtype=float)

    def resource_snapshot(self, resource: str) -> np.ndarray:
        return {
            "food":        self.food_grid,
            "water":       self.water_grid,
            "raw_materials": self.raw_grid,
            "land":        self.land_grid,
            "capital":     self.capital_grid,
            "pollution":   self.pollution_grid,
        }.get(resource, self.food_grid).copy()

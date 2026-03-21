"""
environment.py
--------------
80x80 grid economic world.

Compatible with Mesa 3.x (no mesa.time, agent-based scheduling via AgentSet).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from mesa import Model
from mesa.space import MultiGrid

from agents import WorkerAgent, FirmAgent, LandownerAgent
from economy import Economy
from planner import PlannerAgent


# ---------------------------------------------------------------------------
# Resource defaults
# ---------------------------------------------------------------------------

RESOURCE_TYPES = ["food", "land", "capital", "raw_materials"]

REGEN_RATES: Dict[str, float] = {
    "food": 0.15,
    "land": 0.01,
    "capital": 0.0,
    "raw_materials": 0.05,
}

MAX_RESOURCES: Dict[str, float] = {
    "food": 30.0,
    "land": 20.0,
    "capital": 50.0,
    "raw_materials": 25.0,
}


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
    """

    def __init__(self,
                 seed: int = 42,
                 grid_width: int = 80,
                 grid_height: int = 80,
                 n_workers: int = 400,
                 n_firms: int = 20,
                 n_landowners: int = 15,
                 objective: str = "SUM"):

        super().__init__(seed=seed)
        # Mesa 3.x provides self.rng (numpy Generator) and self.random (stdlib random)

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.n_workers_initial = n_workers
        self.objective = objective
        self.current_step = 0

        # Mesa MultiGrid
        self.grid = MultiGrid(grid_width, grid_height, torus=False)

        # Resource grid
        self.grid_resources = self._init_resources()

        # Ownership map: cell -> landowner unique_id
        self.cell_ownership: Dict[Tuple[int, int], int] = {}

        # Agent lists (maintained manually for fast iteration)
        self.workers: List[WorkerAgent] = []
        self.firms: List[FirmAgent] = []
        self.landowners: List[LandownerAgent] = []

        # Cartel tracking
        self._cartel_counter: int = 0
        self.active_cartels: Dict[int, set] = {}

        # Lookup cache: unique_id -> agent (rebuilt on demand)
        self._id_cache: Dict[int, object] = {}

        # Economy sub-system (must exist before agents are created)
        self.economy = Economy(self)

        # Planner agent
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

    def _init_resources(self) -> List[List[Dict[str, float]]]:
        W, H = self.grid_width, self.grid_height
        food_map = self._gaussian_clusters(W, H, n_clusters=12, peak=MAX_RESOURCES["food"])
        land_map = self._gaussian_clusters(W, H, n_clusters=8, peak=MAX_RESOURCES["land"])
        raw_map = self._gaussian_clusters(W, H, n_clusters=10, peak=MAX_RESOURCES["raw_materials"])
        capital_map = np.zeros((W, H))

        grid = [[{} for _ in range(H)] for _ in range(W)]
        for x in range(W):
            for y in range(H):
                grid[x][y] = {
                    "food": float(food_map[x, y]),
                    "land": float(land_map[x, y]),
                    "capital": float(capital_map[x, y]),
                    "raw_materials": float(raw_map[x, y]),
                }
        return grid

    def _gaussian_clusters(self, W: int, H: int,
                            n_clusters: int, peak: float,
                            sigma: float = 10.0) -> np.ndarray:
        grid = np.zeros((W, H))
        for _ in range(n_clusters):
            cx = int(self.rng.integers(0, W))
            cy = int(self.rng.integers(0, H))
            amplitude = float(self.rng.uniform(0.5, 1.0)) * peak
            xs = np.arange(W)[:, np.newaxis]
            ys = np.arange(H)[np.newaxis, :]
            blob = amplitude * np.exp(
                -(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))
            grid += blob
        return np.clip(grid, 0, peak)

    # ------------------------------------------------------------------
    # Agent creation
    # ------------------------------------------------------------------

    def _random_cell(self) -> Tuple[int, int]:
        x = int(self.rng.integers(0, self.grid_width))
        y = int(self.rng.integers(0, self.grid_height))
        return (x, y)

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

            # Give each landowner a starting territory
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

        # Regenerate resources
        self._regenerate_resources()

        # Step planner first
        self.planner.step()

        # Step all registered agents in shuffled order
        # (exclude planner since it already stepped)
        agent_list = list(self.workers) + list(self.firms) + list(self.landowners)
        # Shuffle using model's rng
        indices = self.rng.permutation(len(agent_list))
        for i in indices:
            a = agent_list[i]
            if hasattr(a, "defunct") and a.defunct:
                continue
            a.step()

        # Economy services
        self.economy.update_prices()
        self.economy.service_loans()
        self._update_market_shares()
        self.economy.refresh_trade_network()

        # Register new workers created this step
        for w in self.workers:
            if w.unique_id not in self._id_cache:
                self._id_cache[w.unique_id] = w

        # Collect metrics
        from metrics import collect_step_metrics
        m = collect_step_metrics(self)
        self.metrics_history.append(m)

    def _regenerate_resources(self):
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                cell = self.grid_resources[x][y]
                for rtype in ["food", "raw_materials", "land"]:
                    current = cell[rtype]
                    cap = MAX_RESOURCES[rtype]
                    rate = REGEN_RATES[rtype]
                    cell[rtype] = min(cap, current + rate * current * (1 - current / cap) + 0.01)

    def _update_market_shares(self):
        total_prod = sum(f.production_this_step for f in self.firms if not f.defunct)
        if total_prod <= 0:
            return
        for f in self.firms:
            if not f.defunct:
                f.market_share = f.production_this_step / total_prod

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def next_cartel_id(self) -> int:
        self._cartel_counter += 1
        return self._cartel_counter

    def get_agent_by_id(self, unique_id: int):
        """Lookup agent by unique_id via cache."""
        return self._id_cache.get(unique_id, None)

    def get_landowner_at(self, pos: Tuple[int, int]) -> Optional[LandownerAgent]:
        owner_id = self.cell_ownership.get(pos)
        if owner_id is None:
            return None
        agent = self.get_agent_by_id(owner_id)
        return agent if isinstance(agent, LandownerAgent) else None

    def get_all_agent_wealths(self) -> np.ndarray:
        wealths = (
            [w.wealth for w in self.workers]
            + [f.wealth for f in self.firms if not f.defunct]
            + [lo.wealth for lo in self.landowners]
        )
        return np.array(wealths, dtype=float)

    def get_worker_wealths(self) -> np.ndarray:
        return np.array([w.wealth for w in self.workers], dtype=float)

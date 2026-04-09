"""
environment.py — 80x80 grid economic world, Mesa 3.x.

Changes from original:
  - NaN guard on agriculture_bonus before int() cast
  - Immigration: periodic new worker spawning as population floor
  - Initial network connections seeded for workers (breaks trade deadlock)
  - Periodic firm spawning if firm count drops too low
  - Animation frame collection (collect_animation_frame)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from mesa import Model
from mesa.space import MultiGrid

from agents import WorkerAgent, FirmAgent, LandownerAgent, EnforcerAgent
from economy import Economy
from planner import PlannerAgent
from information import (NewsFirm, GovernmentBroadcaster,
                         propagate_peer_information, compute_information_metrics)
from banking import BankAgent, compute_banking_metrics
from hardware import (auto_configure, build_regen_fn, AccelConfig,
                      FOOD_CAP, WATER_CAP, RAW_CAP, LAND_CAP,
                      POLLUTION_CAP, POLLUTION_DIFFUSE)


# Population management constants
MIN_WORKERS = 100          # immigration only at severe population loss
IMMIGRATION_RATE = 0.01    # very slow replenishment
MIN_FIRMS = 2              # firm spawning only at near-total firm death
FIRM_SPAWN_PROB = 0.03     # probability per step when below min


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

        # Conflict and legitimacy fields
        self.conflict_grid = np.zeros((grid_width, grid_height), dtype=np.float64)
        self.legitimacy_grid = np.full((grid_width, grid_height), 0.7, dtype=np.float64)
        self._surveillance_level = 0.0  # set by planner
        self._rebellion_events = 0  # accumulator per step
        self._total_crime_events = 0
        self._total_riot_events = 0
        self.enforcers: list = []

        # Cell ownership: (x,y) -> landowner unique_id
        self.cell_ownership: Dict[Tuple[int,int], int] = {}

        # Agent lists
        self.workers:    List[WorkerAgent]    = []
        self.firms:      List[FirmAgent]      = []
        self.landowners: List[LandownerAgent] = []
        self.news_firms: List[NewsFirm]       = []
        self.banks:      List[BankAgent]      = []

        # Cartel tracking
        self._cartel_counter = 0
        self.active_cartels: Dict[int, set] = {}

        # Lookup cache
        self._id_cache: Dict[int, object] = {}

        # Economy and planner
        self.economy = Economy(self)
        self.planner = PlannerAgent(model=self)

        # Populate grid
        self._create_workers(n_workers)
        self._create_firms(n_firms)
        self._create_landowners(n_landowners)
        self._create_news_firms(3)  # start with 3 news firms
        self._create_banks(2)      # start with 2 banks
        self._create_enforcers(5)  # start with 5 enforcement agents

        # Government broadcaster: public media funded by the planner.
        # Enabled by setting use_government_broadcaster = True (done in configure_model).
        # Accuracy tracks governance type and democratic health.
        self.government_broadcaster = GovernmentBroadcaster(self)
        self.use_government_broadcaster: bool = False  # off by default; enabled per-condition
        self.eh_formula: str = 'legacy'               # 'legacy' | 'paper'

        # Seed initial social network connections (fixes trade deadlock)
        self._seed_network_connections()

        # Metrics history
        self.metrics_history: List[Dict] = []
        # Bottleneck detection/regulation state
        self.bottleneck_detected: bool = False
        self.bottleneck_firm_id: Optional[int] = None
        self.bottleneck_market_share: float = 0.0
        self.bottleneck_duration: int = 0
        self.bottleneck_hhi: float = 0.0
        self.bottleneck_top_firm_share: float = 0.0
        self.bottleneck_mode: str = "none"
        self.bottleneck_regulation_policy: str = "off"  # off | enabled | aggressive
        self.bottleneck_dynamic_capture: bool = False
        self.bottleneck_breakup_threshold: int = 50
        self.bottleneck_open_access_bonus: float = 0.05
        self.max_profit_margin_cap: float = 0.15
        self.total_bottleneck_rent_redistributed: float = 0.0  # total BICF grant capital issued
        self.bottleneck_breakups: int = 0
        self._current_bottleneck_firm_id: Optional[int] = None
        self.bicf_entrants_spawned: int = 0
        self.bicf_entrant_ids: set = set()
        self._bicf_spawn_cooldown: int = 0           # steps until next entrant may be spawned

        # Animation frame history (lightweight snapshots for visualization)
        self.animation_frames: List[Dict] = []
        self._collect_animation = True  # set False for batch/headless runs

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

    def _seed_network_connections(self):
        """Give each worker 2-4 random initial trade connections."""
        worker_ids = [w.unique_id for w in self.workers]
        if len(worker_ids) < 5:
            return
        for w in self.workers:
            n_connections = int(self.rng.integers(2, 5))
            candidates = [wid for wid in worker_ids if wid != w.unique_id]
            if len(candidates) < n_connections:
                continue
            chosen = self.rng.choice(candidates, size=n_connections, replace=False)
            for cid in chosen:
                if cid not in w.network_connections:
                    w.network_connections.append(int(cid))

    def _create_news_firms(self, n):
        """Create initial news firms with varying accuracy and capital."""
        for i in range(n):
            accuracy = float(self.rng.uniform(0.3, 0.9))
            capital = float(self.rng.lognormal(mean=4.5, sigma=1.0))
            nf = NewsFirm(model=self, capital=capital, accuracy=accuracy)
            pos = self._random_cell()
            self.grid.place_agent(nf, pos)
            self.news_firms.append(nf)
            self._id_cache[nf.unique_id] = nf
            # Seed initial audience from nearby workers
            if self.workers:
                n_seed = min(int(self.rng.integers(8, 13)), len(self.workers))
                initial_subs = self.rng.choice(self.workers, size=n_seed, replace=False)
                nf.audience = [w.unique_id for w in initial_subs]
                nf.audience_size = len(nf.audience)

    def _create_banks(self, n):
        """Create initial banks."""
        for _ in range(n):
            capital = float(self.rng.lognormal(mean=5.0, sigma=1.0))
            bank = BankAgent(model=self, capital=capital)
            pos = self._random_cell()
            self.grid.place_agent(bank, pos)
            self.banks.append(bank)
            self._id_cache[bank.unique_id] = bank

    def _create_enforcers(self, n):
        """Create enforcement agents."""
        for _ in range(n):
            e = EnforcerAgent(model=self)
            pos = self._random_cell()
            self.grid.place_agent(e, pos)
            self.enforcers.append(e)
            self._id_cache[e.unique_id] = e

    # ── Mesa step ─────────────────────────────────────────────────────────────

    def step(self):
        self.current_step += 1

        # Resource regeneration — agriculture_bonus scales regen passes
        agri = self._agriculture_bonus
        # NaN/negative guard: prevent int(NaN) crash and negative regen passes
        if not np.isfinite(agri) or agri < 0:
            agri = 1.0
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

        # Conflict field diffusion (vectorized, similar to pollution)
        c_neighbours = (
            np.roll(self.conflict_grid,  1, axis=0) +
            np.roll(self.conflict_grid, -1, axis=0) +
            np.roll(self.conflict_grid,  1, axis=1) +
            np.roll(self.conflict_grid, -1, axis=1)
        )
        self.conflict_grid = 0.6 * self.conflict_grid + 0.1 * c_neighbours
        self.conflict_grid *= 0.97  # decay
        np.clip(self.conflict_grid, 0.0, 1.0, out=self.conflict_grid)

        # Legitimacy field update (vectorized)
        # Legitimacy rises with low enforcement brutality; falls with high conflict
        enforcement_pressure = self._surveillance_level * 0.005
        self.legitimacy_grid -= enforcement_pressure
        self.legitimacy_grid -= self.conflict_grid * 0.01
        # Economic growth boost (proxy: mean production normalized)
        n_active = sum(1 for f in self.firms if not f.defunct)
        if n_active > 0:
            self.legitimacy_grid += 0.002
        self.legitimacy_grid *= 0.999  # slow mean reversion
        np.clip(self.legitimacy_grid, 0.0, 1.0, out=self.legitimacy_grid)

        # Reset per-step event counters
        self._rebellion_events = 0
        self._total_crime_events = 0
        self._total_riot_events = 0

        # Planner first (sets bonuses, runs elections, redistributes)
        self.planner.step()

        # Government broadcaster: reaches ALL workers before commercial news.
        # In healthy democracies accuracy is 0.8-0.9; in captured / authoritarian
        # conditions it degrades to 0.4-0.5 / 0.3-0.4.
        self.government_broadcaster.broadcast()

        # Shuffle-step all agents (including news firms and banks)
        agent_list = (list(self.workers) + list(self.firms)
                      + list(self.landowners) + list(self.news_firms)
                      + list(self.banks))
        idx = self.rng.permutation(len(agent_list))
        for i in idx:
            a = agent_list[i]
            if getattr(a, "defunct", False):
                continue
            a.step()

        # Aggregate per-worker crime/riot events
        for w in self.workers:
            self._total_crime_events += getattr(w, '_crime_events', 0)
            self._total_riot_events += getattr(w, '_riot_events', 0)

        # Enforcer patrol step
        for e in self.enforcers:
            if not getattr(e, 'defunct', False):
                e.step()

        # Enforcement suppresses local conflict
        for e in self.enforcers:
            if e.pos is not None:
                ex, ey = int(e.pos[0]), int(e.pos[1])
                self.conflict_grid[ex, ey] = max(0.0, self.conflict_grid[ex, ey] - e.force * 0.1)
                # But brutality adds long-term tension
                self.conflict_grid[ex, ey] = min(1.0, self.conflict_grid[ex, ey] + e.aggression * 0.05)

        # Information propagation: peer-to-peer weight sharing
        propagate_peer_information(self)

        # Economy post-processing
        self.economy.update_prices()
        self.economy.service_loans()
        self.economy.update_market_sentiment()
        self._update_market_shares()
        self._detect_bottleneck()
        self._apply_bottleneck_regulation()
        self.economy.refresh_trade_network()

        # Population management: immigration and firm spawning
        self._manage_population()

        # Register any newly spawned workers/firms
        for w in self.workers:
            if w.unique_id not in self._id_cache:
                self._id_cache[w.unique_id] = w
        for f in self.firms:
            if f.unique_id not in self._id_cache:
                self._id_cache[f.unique_id] = f

        # Technology diffusion: spatial spillovers and depreciation
        from innovation import diffuse_technology, compute_innovation_metrics
        diffuse_technology(self)

        # Update trust scores for all agents (after actions, before metrics)
        from trust import update_trust_scores, compute_trust_metrics
        update_trust_scores(self)

        # Collect metrics (including information, banking, trust, and innovation)
        from metrics import collect_step_metrics
        step_metrics = collect_step_metrics(self)
        info_metrics = compute_information_metrics(self)
        bank_metrics = compute_banking_metrics(self)
        trust_metrics = compute_trust_metrics(self)
        innovation_metrics = compute_innovation_metrics(self)
        step_metrics.update(info_metrics)
        step_metrics.update(bank_metrics)
        step_metrics.update(trust_metrics)
        step_metrics.update(innovation_metrics)
        self.metrics_history.append(step_metrics)

        # Collect animation frame
        if self._collect_animation:
            from metrics import collect_animation_frame
            self.animation_frames.append(collect_animation_frame(self))

    # ── Population management ─────────────────────────────────────────────────

    def _manage_population(self):
        """Immigration and firm spawning to prevent total economic collapse."""
        # Immigration: add workers if population drops too low
        n_workers = len(self.workers)
        if n_workers < MIN_WORKERS:
            deficit = MIN_WORKERS - n_workers
            n_immigrants = max(1, int(deficit * IMMIGRATION_RATE))
            for _ in range(n_immigrants):
                w = WorkerAgent(model=self)
                pos = self._random_cell()
                self.grid.place_agent(w, pos)
                self.workers.append(w)
                self._id_cache[w.unique_id] = w
                # Give immigrant a few connections
                if len(self.workers) > 3:
                    candidates = [x.unique_id for x in self.workers
                                  if x.unique_id != w.unique_id]
                    n_conn = min(3, len(candidates))
                    chosen = self.rng.choice(candidates, size=n_conn, replace=False)
                    w.network_connections = [int(c) for c in chosen]

        # Firm spawning: ensure minimum viable economy
        active_firms = [f for f in self.firms if not f.defunct]
        if len(active_firms) < MIN_FIRMS and self.rng.random() < FIRM_SPAWN_PROB:
            f = FirmAgent(model=self)
            pos = self._random_cell()
            self.grid.place_agent(f, pos)
            self.firms.append(f)
            self._id_cache[f.unique_id] = f
            # Choose governance based on market performance
            if active_firms:
                sevc_profits = [af.profit for af in active_firms if getattr(af, 'is_sevc', True)]
                vanilla_profits = [af.profit for af in active_firms if not getattr(af, 'is_sevc', True)]
                sevc_mean = float(np.mean(sevc_profits)) if sevc_profits else 0.0
                vanilla_mean = float(np.mean(vanilla_profits)) if vanilla_profits else 0.0
                sevc_prob = (max(sevc_mean, 0) + 1.0) / (max(sevc_mean, 0) + max(vanilla_mean, 0) + 2.0)
                f.is_sevc = bool(self.rng.random() < sevc_prob)
            if not f.is_sevc:
                for k in f.strategy_weights:
                    f.strategy_weights[k] = 0.2

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def infrastructure_bonus(self) -> float:
        return max(self._infrastructure_level, 0.1)  # Never let TFP go negative

    def _update_market_shares(self):
        total = sum(f.production_this_step for f in self.firms if not f.defunct)
        if total <= 0:
            return
        for f in self.firms:
            if not f.defunct:
                f.market_share = f.production_this_step / total

    def _detect_bottleneck(self):
        active_firms = [f for f in self.firms if not f.defunct]
        self.bottleneck_detected = False
        self.bottleneck_mode = "none"
        self._current_bottleneck_firm_id = None
        if not active_firms:
            self.bottleneck_firm_id = None
            self.bottleneck_market_share = 0.0
            self.bottleneck_hhi = 0.0
            self.bottleneck_top_firm_share = 0.0
            self.bottleneck_duration = 0
            return

        shares = np.array([max(0.0, float(f.market_share)) for f in active_firms], dtype=np.float64)
        top_idx = int(np.argmax(shares))
        top_firm = active_firms[top_idx]
        top_share = float(shares[top_idx]) if len(shares) else 0.0
        hhi = float(np.sum(shares ** 2))
        capture_trigger = False
        if self.bottleneck_dynamic_capture:
            cap_vals = [float(getattr(f, "capture_ratio", 0.0)) for f in active_firms]
            capture_trigger = (float(np.mean(cap_vals)) > 0.25) if cap_vals else False

        detected = (top_share > 0.35) or (hhi > 0.25) or (top_share > 0.4) or capture_trigger
        if detected:
            if self.bottleneck_firm_id == top_firm.unique_id:
                self.bottleneck_duration += 1
            else:
                self.bottleneck_duration = 1
            self.bottleneck_detected = True
            self.bottleneck_firm_id = top_firm.unique_id
            self._current_bottleneck_firm_id = top_firm.unique_id
            self.bottleneck_market_share = top_share
            self.bottleneck_hhi = hhi
            self.bottleneck_top_firm_share = top_share
        else:
            self.bottleneck_firm_id = None
            self.bottleneck_market_share = 0.0
            self.bottleneck_hhi = hhi
            self.bottleneck_top_firm_share = top_share
            self.bottleneck_duration = 0

    def _apply_bottleneck_regulation(self):
        self._current_bottleneck_firm_id = self.bottleneck_firm_id if self.bottleneck_detected else None
        if self._bicf_spawn_cooldown > 0:
            self._bicf_spawn_cooldown -= 1
        if not self.bottleneck_detected or self.bottleneck_regulation_policy == "off":
            return
        bottleneck = next((f for f in self.firms
                           if (not f.defunct) and f.unique_id == self.bottleneck_firm_id), None)
        if bottleneck is None:
            return

        is_aggressive = self.bottleneck_regulation_policy == "aggressive"
        # B3 aggressive = larger grant + longer vesting, not faster breakup.
        grant_fraction  = 0.80 if is_aggressive else 0.50
        vesting_steps   = 40   if is_aggressive else 20
        open_access_bonus = 0.10 if is_aggressive else self.bottleneck_open_access_bonus
        margin_cap      = 0.10 if is_aggressive else self.max_profit_margin_cap

        # Compute rent signal from incumbent margin (incumbent balance sheet unchanged).
        revenue = max(float(getattr(bottleneck, "revenue", 0.0)), 0.0)
        current_profit = float(getattr(bottleneck, "profit", 0.0))
        margin = (current_profit / max(revenue, 1e-9)) if revenue > 0 else 0.0
        rent_signal = 0.0
        if margin > margin_cap and current_profit > 0:
            capped_profit = margin_cap * revenue
            rent_signal = max(0.0, current_profit - capped_profit)

        grant = rent_signal * grant_fraction

        # Spawn a BICF entrant funded by the detected rent.
        # Only spawn if cooldown has expired AND no living BICF entrant already exists.
        living_bicf = [f for f in self.firms
                       if not f.defunct and f.unique_id in self.bicf_entrant_ids]
        if self._bicf_spawn_cooldown == 0 and not living_bicf and grant > 0:
            self.bottleneck_mode = "entrant_spawned"
            entrant = self._spawn_bicf_entrant(
                grant=grant,
                vesting_steps=vesting_steps,
                open_access_bonus=open_access_bonus,
                incumbent=bottleneck,
            )
            self.total_bottleneck_rent_redistributed += grant
            self._bicf_spawn_cooldown = vesting_steps  # don't flood with entrants
            _ = entrant  # referenced for clarity; registered inside _spawn_bicf_entrant
        else:
            # Bottleneck persists but entrant already exists or grant too small:
            # keep giving the living BICF entrant its open_access_bonus.
            self.bottleneck_mode = "open_access"
            for f in living_bicf:
                f.open_access_bonus = open_access_bonus

    def _spawn_bicf_entrant(
        self,
        grant: float,
        vesting_steps: int,
        open_access_bonus: float,
        incumbent: FirmAgent,
    ) -> FirmAgent:
        """Create a new firm with structural entry advantages funded by detected rent.

        The incumbent is not touched. This is the topology change: reduce the
        entrant's barrier, not the incumbent's capacity.
        """
        entrant = FirmAgent(model=self, capital=grant)
        # Entrant starts with better-than-baseline capital stock funded by grant.
        entrant.capital_stock = max(grant * 0.5, incumbent.capital_stock * 0.25)
        # Wage competitive with incumbent so the entrant can attract workers.
        entrant.offered_wage = incumbent.offered_wage
        # SEVC and ownership structure mirrors model-wide settings.
        entrant.is_sevc = getattr(self, "use_sevc", True)
        entrant.worker_ownership_share = getattr(incumbent, "worker_ownership_share", 0.0)
        entrant.investor_ownership_share = getattr(incumbent, "investor_ownership_share", 1.0)
        # BICF structural advantages.
        entrant.bicf_entrant = True
        entrant.tax_advantage_remaining = vesting_steps
        entrant.bicf_grant_received = grant
        entrant.open_access_bonus = open_access_bonus  # production boost this step
        # Lower pollution factor: entrant represents newer technology.
        entrant.pollution_factor = max(0.0, incumbent.pollution_factor * 0.75)

        pos = self._random_cell()
        self.grid.place_agent(entrant, pos)
        self.firms.append(entrant)
        self._id_cache[entrant.unique_id] = entrant
        self.bicf_entrant_ids.add(entrant.unique_id)
        self.bicf_entrants_spawned += 1
        return entrant

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

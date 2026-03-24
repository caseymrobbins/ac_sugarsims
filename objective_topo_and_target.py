    def _objective_topo(self) -> float:
        """
        TOPO: Topology shaping toward healthy society.
        
        Don't tell the planner what healthy looks like.
        Shape the landscape so healthy is where the gradients point.
        
        Architecture: nash_base * product(health_factors)
        
        Each health factor scores 1.0 when the metric is in the
        healthy range, and drops smoothly toward 0 outside it.
        The product structure means ALL factors must be healthy
        simultaneously. Mesa optimizers profit most when the
        landscape is healthy, because their revenue comes from
        the nash_base which is gated by the health factors.
        
        Uses gaussian scoring: score = exp(-((x - center) / width)^2)
        This creates a smooth bowl centered on the healthy range
        with gradients pointing toward the center from any direction.
        """
        workers = self.model.workers
        if not workers:
            return float(math.log(EPSILON))
        
        # ── Nash base ───────────────────────────────────────────
        wealths = np.array([w.wealth for w in workers], dtype=np.float64)
        wealths = wealths[np.isfinite(wealths)]
        if len(wealths) == 0:
            return float(math.log(EPSILON))
        
        nash_base = float(np.sum(np.log(np.maximum(wealths, EPSILON))))
        
        # ── Helper: gaussian score (1.0 at center, drops outside) ──
        def gscore(value, center, width):
            """Score 1.0 at center, drops to 0.37 at center ± width."""
            return math.exp(-((value - center) / max(width, 0.01)) ** 2)
        
        # ── Health factors ──────────────────────────────────────
        
        # 1. Equity: Gini in [0.25, 0.40], center 0.325
        all_w = self.model.get_all_agent_wealths()
        all_w = all_w[np.isfinite(all_w) & (all_w > 0)]
        if len(all_w) > 1:
            s = np.sort(all_w)
            n = len(s)
            all_gini = float(
                (2 * np.sum(np.arange(1, n + 1) * s) - (n + 1) * s.sum())
                / (n * s.sum()))
        else:
            all_gini = 0.5
        equity = gscore(all_gini, 0.325, 0.15)
        
        # 2. Employment: unemployment in [0.03, 0.15], center 0.08
        n_employed = sum(1 for w in workers if w.employed)
        unemployment = 1.0 - n_employed / max(len(workers), 1)
        employment_health = gscore(unemployment, 0.08, 0.12)
        
        # 3. Skills: mean skill > 0.5, center 0.7
        skills = np.array([w.skill for w in workers], dtype=np.float64)
        mean_skill = float(np.mean(skills)) if len(skills) > 0 else 0.3
        skill_health = gscore(mean_skill, 0.7, 0.25)
        
        # 4. Epistemic: R0 in [0.5, 2.0], center 1.2
        # Compute R0 inline (simplified from information.py)
        mean_connections = float(np.mean([
            len(w.network_connections) for w in workers
        ])) if workers else 0
        trust_vals = np.array([getattr(w, 'authority_trust', 0.7) for w in workers])
        mean_trust = float(np.mean(trust_vals))
        contact_rate = mean_connections * 0.05  # PEER_SHARE_RATE
        transmission_prob = mean_trust * 0.1     # NEWS_ABSORPTION_RATE
        recovery_rate = 0.02
        info_r0 = (contact_rate * transmission_prob) / max(recovery_rate, 0.001)
        epistemic = gscore(info_r0, 1.2, 1.0)
        
        # 5. Market competition: HHI < 0.10, center 0.05
        firms = [f for f in self.model.firms if not f.defunct]
        if firms:
            shares = np.array([f.market_share for f in firms])
            hhi = float(np.sum(shares ** 2))
        else:
            hhi = 1.0
        competition = gscore(hhi, 0.05, 0.08)
        
        # 6. Population sustainability: population should be stable/growing
        # Score based on whether pop is above initial
        pop_ratio = len(workers) / max(self.model.n_workers_initial, 1)
        sustainability = min(1.0, pop_ratio)  # 1.0 at or above initial, drops below
        
        # ── Combined ────────────────────────────────────────────
        # Product of all health factors
        system_health = (equity * employment_health * skill_health 
                        * epistemic * competition * sustainability)
        
        # Gate the nash base with system health
        # When all factors are 1.0, reward = nash_base (full credit)
        # When any factor drops, reward drops proportionally
        reward = nash_base * max(system_health, 1e-10)
        
        return float(reward)

    def _objective_target(self) -> float:
        """
        TARGET: Direct scoring toward healthy society targets.
        
        Explicitly tells the planner what healthy looks like.
        Each metric is scored on how close it is to the target range.
        In range = 1.0, out of range = decays toward 0.
        
        Total score = sum of dimension scores * nash_base_sign
        
        This is the prescriptive approach: we define the target and
        the planner optimizes toward it. Less elegant than TOPO but
        more direct, and lets us test whether the target itself is
        achievable.
        
        Uses asymmetric scoring: metrics have a healthy range [lo, hi].
        Inside the range scores 1.0. Outside, score decays with
        distance from the nearest bound.
        """
        workers = self.model.workers
        if not workers:
            return -100.0
        
        # ── Helper: range score ─────────────────────────────────
        def rscore(value, lo, hi, decay=5.0):
            """
            1.0 inside [lo, hi]. Decays outside.
            decay controls how fast it drops (higher = sharper).
            """
            if lo <= value <= hi:
                return 1.0
            if value < lo:
                return math.exp(-decay * (lo - value) / max(hi - lo, 0.01))
            else:
                return math.exp(-decay * (value - hi) / max(hi - lo, 0.01))
        
        # ── Compute all metrics ─────────────────────────────────
        
        # Gini
        all_w = self.model.get_all_agent_wealths()
        all_w = all_w[np.isfinite(all_w) & (all_w > 0)]
        if len(all_w) > 1:
            s = np.sort(all_w)
            n = len(s)
            all_gini = float(
                (2 * np.sum(np.arange(1, n + 1) * s) - (n + 1) * s.sum())
                / (n * s.sum()))
        else:
            all_gini = 0.5
        
        # Alpha
        alpha = 5.0  # default healthy
        if len(all_w) >= 20:
            threshold = np.percentile(all_w, 90)
            tail = all_w[all_w >= threshold]
            if len(tail) >= 5 and threshold > 0:
                log_sum = np.sum(np.log(tail / threshold))
                if log_sum > EPSILON:
                    alpha = len(tail) / log_sum
        
        # Employment
        n_employed = sum(1 for w in workers if w.employed)
        unemployment = 1.0 - n_employed / max(len(workers), 1)
        
        # Skills
        skills = np.array([w.skill for w in workers], dtype=np.float64)
        mean_skill = float(np.mean(skills)) if len(skills) > 0 else 0.3
        
        # R0 (inline)
        mean_connections = float(np.mean([
            len(w.network_connections) for w in workers
        ])) if workers else 0
        trust_vals = np.array([getattr(w, 'authority_trust', 0.7) for w in workers])
        mean_trust = float(np.mean(trust_vals))
        contact_rate = mean_connections * 0.05
        transmission_prob = mean_trust * 0.1
        info_r0 = (contact_rate * transmission_prob) / 0.02
        
        # Epistemic health
        # Simplified: geometric mean of trust, anti-polarization, R0 health
        weight_matrix = np.array([
            [w.decision_weights.get(a, 0.5) for a in 
             ["harvest","seek_work","trade","migrate","save","invest","found_firm"]]
            for w in workers[:200]  # sample for speed
        ])
        if len(weight_matrix) > 1:
            mean_weights = weight_matrix.mean(axis=0)
            deviations = weight_matrix - mean_weights
            polarization = float(np.mean(np.sqrt(np.sum(deviations ** 2, axis=1))))
        else:
            polarization = 0.1
        
        # Market structure
        firms = [f for f in self.model.firms if not f.defunct]
        if firms:
            shares = np.array([f.market_share for f in firms])
            hhi = float(np.sum(shares ** 2))
            n_cartels = sum(1 for cid, members in self.model.active_cartels.items() 
                          if len(members) >= 2)
            cartel_pct = n_cartels / max(len(firms), 1)
        else:
            hhi = 1.0
            cartel_pct = 0.0
        
        # Debt
        debt_frac = float(np.mean([w.debt > 0 for w in workers])) if workers else 0
        
        # Worker floor
        worker_w = np.array([w.wealth for w in workers], dtype=np.float64)
        worker_min = float(np.min(worker_w)) if len(worker_w) > 0 else 0
        
        # Population
        pop_ratio = len(workers) / max(self.model.n_workers_initial, 1)
        
        # ── Score each dimension ────────────────────────────────
        scores = {
            'gini':        rscore(all_gini, 0.25, 0.40),
            'alpha':       min(1.0, alpha / 3.0),          # 1.0 when alpha >= 3
            'top10':       rscore(all_w[all_w >= np.percentile(all_w, 90)].sum() / max(all_w.sum(), 1) 
                           if len(all_w) > 10 else 0.3, 0.15, 0.35),
            'unemployment': rscore(unemployment, 0.03, 0.15),
            'skill':       rscore(mean_skill, 0.50, 1.00),
            'info_r0':     rscore(info_r0, 0.50, 2.00),
            'trust':       rscore(mean_trust, 0.50, 0.80),
            'polarization': rscore(polarization, 0.05, 0.20),
            'hhi':         rscore(hhi, 0.0, 0.10),
            'cartels':     rscore(cartel_pct, 0.0, 0.05),
            'debt':        rscore(debt_frac, 0.0, 0.20),
            'floor':       min(1.0, worker_min / 40.0),    # 1.0 when min >= 40
            'population':  min(1.0, pop_ratio),             # 1.0 when pop >= initial
        }
        
        # ── Weighted sum ────────────────────────────────────────
        # All dimensions matter, but some matter more
        weights = {
            'gini': 2.0,          # high priority: distribution
            'alpha': 1.0,
            'top10': 1.5,
            'unemployment': 2.0,  # high priority: people need work
            'skill': 1.0,
            'info_r0': 1.5,       # epistemic health matters a lot
            'trust': 1.0,
            'polarization': 0.5,
            'hhi': 1.0,
            'cartels': 0.5,
            'debt': 0.5,
            'floor': 2.0,         # high priority: nobody starving
            'population': 2.0,    # high priority: don't collapse
        }
        
        total_score = sum(scores[k] * weights[k] for k in scores)
        max_score = sum(weights.values())
        normalized_score = total_score / max_score  # 0 to 1
        
        # Scale by population (bigger healthy societies are better than
        # smaller healthy societies, prevents "shrink to win")
        pop_bonus = math.log(max(len(workers), 1))
        
        reward = normalized_score * pop_bonus * 100  # scale for RL
        
        return float(reward)

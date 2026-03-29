"""
information.py
--------------
Information production and propagation system for the economic simulation.

Two-channel model:
  1. Firms (including NewsFirms) produce information signals
  2. Signals propagate through the social network with trust decay

NewsFirm is a specialized firm that produces weight-modification signals
instead of goods. Its output is shaped by:
  - Ground truth (actual economic state, filtered by journalist skill)
  - Owner bias (if captured by a cartel, signal shifts toward cartel interests)
  - Audience capture (signal drifts toward what audience already believes)

Information propagation:
  - Each news firm broadcasts to agents within network reach
  - Signal trust decays with network distance (hops from source)
  - Agents blend incoming signals into their decision weights
  - Peer-to-peer: agents share weights with trade partners (grassroots)

Measurable outputs:
  - Weight vector divergence (polarization = bimodality of weight distribution)
  - Authority trust distribution
  - Epistemic health = distance between agent weights and locally-optimal weights
  - Information Gini = inequality in access to accurate information
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from mesa import Agent

if TYPE_CHECKING:
    from environment import EconomicModel
    from agents import WorkerAgent, FirmAgent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Action names (must match WorkerAgent.decision_weights keys)
ACTIONS = ["harvest", "seek_work", "trade", "migrate", "save", "invest", "found_firm"]
N_ACTIONS = len(ACTIONS)

# Trust decay per network hop
TRUST_DECAY_PER_HOP = 0.4       # signal retains 60% per hop
MAX_BROADCAST_HOPS = 4           # news doesn't propagate beyond this
PEER_SHARE_RATE = 0.05           # how much weight blending happens per peer interaction
NEWS_ABSORPTION_RATE = 0.1       # how much a news signal shifts weights

# News firm parameters
NEWS_FIRM_ACCURACY_COST = 3.0    # wealth per step to maintain high accuracy
AUDIENCE_CAPTURE_RATE = 0.005    # per-step drift toward audience beliefs (slow enough for editorial signal to persist)
CARTEL_BIAS_STRENGTH = 0.3       # how much cartel capture distorts the signal


# ---------------------------------------------------------------------------
# Decision weight system (used by WorkerAgent)
# ---------------------------------------------------------------------------

def init_decision_weights(rng: np.random.Generator) -> Dict[str, float]:
    """
    Generate initial decision weights for a new agent.
    Drawn from distributions that create natural personality variation.
    """
    return {
        "harvest":    float(rng.beta(3, 2)),     # most agents lean toward harvesting
        "seek_work":  float(rng.beta(2, 2)),     # moderate work-seeking
        "trade":      float(rng.beta(2, 3)),     # slightly less inclined to trade
        "migrate":    float(rng.beta(1.5, 4)),   # low migration tendency
        "save":       float(rng.beta(2, 3)),     # moderate saving
        "invest":     float(rng.beta(1.5, 4)),   # low investment (requires knowledge)
        "found_firm": float(rng.beta(1, 6)),     # very few are entrepreneurs
    }


def choose_action_from_weights(weights: Dict[str, float],
                                context: Dict[str, float],
                                rng: np.random.Generator) -> str:
    """
    Score each action using weights × context, pick the best with noise.

    context contains situational multipliers for each action:
      - harvest: food_nearby / (wealth + 1)
      - seek_work: has_firms * (1 + unemployment_penalty)
      - trade: n_connections * (surplus + 0.1)
      - migrate: resource_gradient
      - save: wealth / (metabolism * 50)
      - invest: wealth / 500 * skill
      - found_firm: (wealth / 300) * (1 - employed)
    """
    scores = {}
    for action in ACTIONS:
        w = weights.get(action, 0.5)
        c = context.get(action, 0.5)
        # Score = weight * context + bounded noise for exploration
        noise = float(rng.normal(0, 0.05))
        scores[action] = w * c + noise

    return max(scores, key=scores.get)


def compute_action_context(worker, model) -> Dict[str, float]:
    """Compute situational context multipliers for a worker's action scoring."""
    pos = worker._tpos()
    wealth = max(worker.wealth, 0.1)

    # Food nearby
    if pos is not None:
        food = float(model.food_grid[pos[0], pos[1]])
        raw = float(model.raw_grid[pos[0], pos[1]])
        food_nearby = (food + raw * 0.5) / (wealth + 1)
    else:
        food_nearby = 0.5

    # Work availability
    has_firms = 1.0 if any(not f.defunct for f in model.firms) else 0.0
    unemp_penalty = 1.0 + worker.consecutive_unemployed_steps * 0.1
    if worker.employed:
        unemp_penalty = 0.3  # employed agents rarely seek new work

    # Trade opportunity
    n_conn = len(worker.network_connections)
    surplus = max(0, wealth - 30) / 100.0

    # Migration gradient (are nearby cells better?)
    resource_gradient = 0.3  # default moderate
    if pos is not None:
        try:
            neighbours = model.grid.get_neighborhood(pos, moore=True, include_center=False, radius=2)
            if neighbours:
                local_val = food_nearby * wealth
                best_nearby = max(
                    float(model.food_grid[c[0], c[1]] + model.raw_grid[c[0], c[1]])
                    for c in neighbours[:8]  # sample for speed
                )
                resource_gradient = max(0.1, (best_nearby - food) / (food + 1))
        except Exception:
            pass

    return {
        "harvest":    food_nearby,
        "seek_work":  has_firms * unemp_penalty,
        "trade":      (n_conn + 0.5) * (surplus + 0.1),
        "migrate":    resource_gradient * worker.mobility,
        "save":       wealth / (worker.metabolism * 50 + 1),
        "invest":     wealth / 500.0 * worker.skill,
        "found_firm": (wealth / 300.0) * (1.0 - float(worker.employed)),
    }


def update_weights_from_experience(weights: Dict[str, float],
                                    action: str,
                                    outcome: float,
                                    learning_rate: float = 0.02):
    """
    Simple reinforcement: shift the chosen action's weight toward
    the outcome quality. outcome > 0 = good, < 0 = bad.
    """
    if action in weights:
        # Sigmoid-bounded update to keep weights in (0, 1)
        current = weights[action]
        adjustment = learning_rate * np.tanh(outcome)
        weights[action] = float(np.clip(current + adjustment, 0.01, 0.99))


# ---------------------------------------------------------------------------
# Information signal
# ---------------------------------------------------------------------------

class InfoSignal:
    """
    A packet of information produced by a news firm.
    Contains weight modifications, trust level, and optional scapegoating.
    """
    __slots__ = ['weight_deltas', 'trust', 'source_id', 'hops',
                 'scapegoat_identity', 'blame_target']

    def __init__(self, weight_deltas: Dict[str, float], trust: float,
                 source_id: int, hops: int = 0,
                 scapegoat_identity=None, blame_target: Optional[str] = None):
        self.weight_deltas = weight_deltas   # {action: delta} to apply to weights
        self.trust = trust                    # 0-1, decays with distance
        self.source_id = source_id
        self.hops = hops
        self.scapegoat_identity = scapegoat_identity  # identity tuple to blame
        self.blame_target = blame_target  # "workers", "firms", "government", etc.

    def decay(self) -> 'InfoSignal':
        """Return a copy with decayed trust for next hop."""
        return InfoSignal(
            weight_deltas=dict(self.weight_deltas),
            trust=self.trust * (1 - TRUST_DECAY_PER_HOP),
            source_id=self.source_id,
            hops=self.hops + 1,
            scapegoat_identity=self.scapegoat_identity,
            blame_target=self.blame_target,
        )


# ---------------------------------------------------------------------------
# NewsFirm
# ---------------------------------------------------------------------------

class NewsFirm(Agent):
    """
    A firm that produces information instead of goods.

    Revenue comes from audience size (subscriptions/ads).
    Output is an InfoSignal that modifies agent decision weights.

    Can be captured by a cartel, which biases the signal.
    Subject to audience capture (drifts toward what audience believes).
    """

    agent_type_name = "NEWS_FIRM"

    def __init__(self, model: "EconomicModel",
                 capital: float = None,
                 accuracy: float = None):
        super().__init__(model)
        rng = model.rng

        self.wealth: float = capital if capital is not None else float(rng.lognormal(mean=4.5, sigma=1.0))
        self.capital_stock: float = self.wealth * 0.3

        # Editorial quality: higher = closer to ground truth
        self.accuracy: float = accuracy if accuracy is not None else float(rng.uniform(0.4, 0.9))

        # Audience
        self.audience: List[int] = []          # worker unique_ids who subscribe
        self.audience_size: int = 0
        self.reach_radius: int = 10             # network hops for broadcast

        # Capture state
        self.captured_by_cartel: Optional[int] = None  # cartel_id if captured
        self.bias_direction: Dict[str, float] = {}     # weight deltas serving captor

        # Audience capture drift
        self._audience_mean_weights: Dict[str, float] = {a: 0.5 for a in ACTIONS}

        # Revenue model
        self.revenue: float = 0.0
        self.profit: float = 0.0
        self.defunct: bool = False
        self.age: int = 0

        # Last broadcast signal (for metrics)
        self.last_signal: Optional[InfoSignal] = None
        # Latent source credibility used by trust metrics and workers' source filtering
        self.trust_score: float = float(np.clip(self.accuracy, 0.05, 0.95))
        # Public funding flag (set by planner, cleared each step)
        self._received_public_funding: bool = False

    def step(self):
        if self.defunct:
            return
        self.age += 1

        # Pay for accuracy (journalism costs money)
        accuracy_cost = self.accuracy * NEWS_FIRM_ACCURACY_COST
        self.wealth -= accuracy_cost

        # Public funding maintains minimum editorial standards
        if self._received_public_funding:
            self.accuracy = max(0.3, self.accuracy)
            self._received_public_funding = False

        # Revenue from audience
        self.revenue = self.audience_size * 0.3  # per-subscriber fee
        self.profit = self.revenue - accuracy_cost
        self.wealth += self.revenue

        # Audience capture: drift toward what audience already believes
        self._update_audience_capture()

        # Check for cartel capture attempts
        self._check_cartel_capture()

        # Produce and broadcast signal
        signal = self._produce_signal()
        self.last_signal = signal
        self._broadcast(signal)

        # Update reputation: accurate, uncaptured outlets become more trusted
        capture_penalty = 0.0
        if self.captured_by_cartel is not None:
            capture_penalty += 0.18
        audience_pressure = min(0.15, 0.01 * self.audience_size)
        credibility_target = float(np.clip(self.accuracy - capture_penalty - audience_pressure, 0.05, 0.95))
        self.trust_score = float(np.clip(0.92 * self.trust_score + 0.08 * credibility_target, 0.0, 1.0))

        # Bankruptcy
        if self.wealth < -50:
            self._go_defunct()

    def _produce_signal(self) -> InfoSignal:
        """
        Produce an information signal based on the actual economic state,
        filtered through accuracy, bias, and audience capture.
        """
        rng = self.model.rng

        # Ground truth signal: what weight adjustments would actually help agents
        truth = self._observe_ground_truth()

        # Apply accuracy filter: lower accuracy = more noise
        noise_scale = max(0.01, 1.0 - self.accuracy)
        signal_deltas = {}
        for action in ACTIONS:
            true_val = truth.get(action, 0.0)
            noise = float(rng.normal(0, noise_scale * 0.1))
            signal_deltas[action] = true_val + noise

        # Apply cartel bias: if captured, shift signal toward cartel interests
        if self.captured_by_cartel is not None and self.bias_direction:
            for action in ACTIONS:
                bias = self.bias_direction.get(action, 0.0)
                signal_deltas[action] = (
                    (1 - CARTEL_BIAS_STRENGTH) * signal_deltas[action]
                    + CARTEL_BIAS_STRENGTH * bias
                )

        # Apply audience capture: drift toward what audience already believes
        for action in ACTIONS:
            audience_pref = self._audience_mean_weights.get(action, 0.5) - 0.5
            signal_deltas[action] += AUDIENCE_CAPTURE_RATE * audience_pref

        # Trust level based on capital investment and reputation
        base_trust = min(0.9, 0.3 + self.capital_stock * 0.001 + self.accuracy * 0.3)

        # Scapegoating: captured media or high propaganda triggers blame narratives
        scapegoat_identity = None; blame_target = None
        prop_budget = getattr(self.model, '_propaganda_budget', 0.0) if hasattr(self.model, '_propaganda_budget') else 0.0
        if self.captured_by_cartel is not None or prop_budget > 1.0:
            # Economic crisis indicator: high unemployment
            n_workers = max(len(self.model.workers), 1)
            unemp = 1 - sum(1 for w in self.model.workers if w.employed) / n_workers
            if unemp > 0.15 or self.captured_by_cartel is not None:
                from agents import IDENTITY_TYPES
                scapegoat_identity = IDENTITY_TYPES[rng.integers(len(IDENTITY_TYPES))]
                blame_target = rng.choice(["workers", "immigrants", "government"])

        return InfoSignal(
            weight_deltas=signal_deltas,
            trust=base_trust,
            source_id=self.unique_id,
            scapegoat_identity=scapegoat_identity,
            blame_target=blame_target,
        )

    def _observe_ground_truth(self) -> Dict[str, float]:
        """
        What weight adjustments would be optimal given the true economic state?
        This is the 'journalism' function: observing reality and reporting it.
        """
        model = self.model

        # Compute economy-level signals
        n_workers = len(model.workers)
        n_firms = len([f for f in model.firms if not f.defunct])
        mean_wealth = np.mean([w.wealth for w in model.workers]) if model.workers else 100
        unemployment = 1 - sum(1 for w in model.workers if w.employed) / max(n_workers, 1)
        mean_food = float(np.mean(model.food_grid))

        # Optimal weight adjustments given current conditions
        truth = {
            "harvest":    0.02 if mean_food > 20 else -0.02,       # harvest if food available
            "seek_work":  0.03 if n_firms > 5 and unemployment > 0.1 else -0.01,
            "trade":      0.01,                                      # trading is generally good
            "migrate":    0.02 if mean_food < 15 else -0.01,        # migrate if resources scarce
            "save":       0.01 if mean_wealth < 100 else -0.01,
            "invest":     0.02 if n_firms < 15 else 0.0,
            "found_firm": 0.01 if n_firms < 10 else -0.01,
        }
        return truth

    def _broadcast(self, signal: InfoSignal):
        """Broadcast signal to audience and nearby agents."""
        if not self.model.workers:
            return

        # Retain existing subscribers (drop those who left the model)
        self.audience = [wid for wid in self.audience
                         if self.model.get_agent_by_id(wid) is not None]

        # Churn: some subscribers leave each step
        if self.audience:
            churn_rate = max(0.02, 0.10 - self.trust_score * 0.08)
            self.audience = [wid for wid in self.audience
                             if self.model.rng.random() > churn_rate]

        # Recruit new subscribers proportional to capital
        existing = set(self.audience)
        n_sample = min(50, max(10, int(self.capital_stock * 0.01)), len(self.model.workers))
        if self.model.workers:
            sampled = self.model.rng.choice(self.model.workers, size=n_sample, replace=False)
            for w in sampled:
                if w.unique_id not in existing and self.model.rng.random() < min(0.5, self.capital_stock * 0.001 + self.trust_score * 0.05):
                    self.audience.append(w.unique_id)
        self.audience_size = len(self.audience)

        # Deliver signal to each audience member
        for wid in self.audience:
            worker = self.model.get_agent_by_id(wid)
            if worker is not None and hasattr(worker, 'receive_information'):
                worker.receive_information(signal)

    def _update_audience_capture(self):
        """Track what the audience currently believes (for drift)."""
        if not self.audience:
            return
        # Sample audience weights
        weight_sums = {a: 0.0 for a in ACTIONS}
        count = 0
        for wid in self.audience[:50]:  # sample for speed
            worker = self.model.get_agent_by_id(wid)
            if worker and hasattr(worker, 'decision_weights'):
                for a in ACTIONS:
                    weight_sums[a] += worker.decision_weights.get(a, 0.5)
                count += 1
        if count > 0:
            self._audience_mean_weights = {a: weight_sums[a] / count for a in ACTIONS}

    def _check_cartel_capture(self):
        """Cartels attempt to capture news firms by funding them."""
        if self.captured_by_cartel is not None:
            # Check if captor cartel still exists
            if self.captured_by_cartel not in self.model.active_cartels:
                self.captured_by_cartel = None
                self.bias_direction = {}
            return

        # Vulnerable to capture when low on wealth
        if self.wealth > 100:
            return

        # Check if any cartel wants to capture this news firm
        for cartel_id, members in self.model.active_cartels.items():
            if len(members) < 2:
                continue
            if self.model.rng.random() < 0.02:  # 2% chance per step per cartel
                # Cartel funds the news firm in exchange for bias
                self.captured_by_cartel = cartel_id
                self.wealth += 50  # cartel investment
                # Bias: suppress seek_work (keeps workers desperate),
                # suppress found_firm (reduces competition),
                # boost save (keeps workers passive)
                self.bias_direction = {
                    "harvest":    0.0,
                    "seek_work":  -0.05,
                    "trade":      0.0,
                    "migrate":    -0.03,
                    "save":       0.03,
                    "invest":     -0.03,
                    "found_firm": -0.05,
                }
                break

    def _go_defunct(self):
        """News firm shuts down."""
        self.defunct = True
        if self.pos is not None:
            self.model.grid.remove_agent(self)
        self.remove()

    def remove(self):
        if self in self.model.news_firms:
            self.model.news_firms.remove(self)
        super().remove()


# ---------------------------------------------------------------------------
# Network propagation (called from environment.step)
# ---------------------------------------------------------------------------

def propagate_peer_information(model: "EconomicModel"):
    """
    Peer-to-peer information sharing: agents share decision weights
    with their trade partners. This is the grassroots channel.

    Trust is based on relationship (trade history) and proximity.
    """
    rng = model.rng

    # Sample a fraction of workers for peer sharing (for performance)
    workers = model.workers
    if not workers:
        return

    n_share = max(1, len(workers) // 10)  # 10% per step
    sharers = rng.choice(workers, size=min(n_share, len(workers)), replace=False)

    for worker in sharers:
        if not hasattr(worker, 'decision_weights') or not worker.network_connections:
            continue

        # Pick one connection to share with
        partner_id = rng.choice(worker.network_connections)
        partner = model.get_agent_by_id(partner_id)
        if partner is None or not hasattr(partner, 'decision_weights'):
            continue

        # Blend weights: both agents move slightly toward each other
        for action in ACTIONS:
            w_val = worker.decision_weights.get(action, 0.5)
            p_val = partner.decision_weights.get(action, 0.5)
            diff = p_val - w_val
            worker.decision_weights[action] = float(
                np.clip(w_val + PEER_SHARE_RATE * diff, 0.01, 0.99))
            partner.decision_weights[action] = float(
                np.clip(p_val - PEER_SHARE_RATE * diff * 0.5, 0.01, 0.99))


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_information_metrics(model: "EconomicModel") -> Dict[str, float]:
    """
    Compute epistemic health metrics for the current step.

    Includes information R0: the effective reproduction number for
    information spread. R0 = contact_rate * transmission_prob * duration.
      - contact_rate: mean connections per agent * peer_share fraction
      - transmission_prob: mean authority_trust (willingness to accept)
      - duration: ~1/learning_rate (how long before experience overwrites)

    R0 > 1 means information (including misinformation) goes viral.
    R0 < 1 means signals die out through experience correction.
    """
    workers = model.workers
    if not workers or not hasattr(workers[0], 'decision_weights'):
        return {
            "mean_authority_trust": 0.0,
            "min_authority_trust": 0.0,
            "weight_polarization": 0.0,
            "info_r0": 0.0,
            "n_news_firms": 0,
            "n_captured_news": 0,
            "n_accurate_news": 0,
            "n_captured_accurate": 0,
            "epistemic_health": 0.0,
            "trust_gini": 0.0,
            "pct_low_trust": 0.0,
        }

    # Authority trust distribution
    trusts = np.array([getattr(w, 'authority_trust', 0.5) for w in workers])
    mean_trust = float(np.mean(trusts))

    # Trust Gini: inequality in information access
    trust_sorted = np.sort(trusts)
    n = len(trust_sorted)
    if n > 1 and trust_sorted.sum() > 0:
        trust_gini = float(
            (2 * np.sum((np.arange(1, n+1)) * trust_sorted) - (n+1) * trust_sorted.sum())
            / (n * trust_sorted.sum()))
    else:
        trust_gini = 0.0

    # Fraction with critically low trust (susceptible to capture)
    pct_low_trust = float(np.mean(trusts < 0.3))

    # Weight vector polarization
    weight_matrix = np.array([
        [w.decision_weights.get(a, 0.5) for a in ACTIONS]
        for w in workers if hasattr(w, 'decision_weights')
    ])

    if len(weight_matrix) > 1:
        mean_weights = weight_matrix.mean(axis=0)
        deviations = weight_matrix - mean_weights
        polarization = float(np.mean(np.sqrt(np.sum(deviations ** 2, axis=1))))
    else:
        polarization = 0.0

    # Information R0 calculation (SIR analogy)
    # contact_rate: fraction of population reached per step via peer sharing
    mean_connections = float(np.mean([
        len(w.network_connections) for w in workers
    ])) if workers else 0
    contact_rate = mean_connections * PEER_SHARE_RATE  # effective contacts per step

    # transmission_prob: mean trust * absorption rate
    transmission_prob = mean_trust * NEWS_ABSORPTION_RATE

    # recovery_rate: how fast experience overwrites received information
    # ~learning_rate from weight updates (0.02)
    recovery_rate = 0.02

    # R0 = (contact_rate * transmission_prob) / recovery_rate
    if recovery_rate > 0:
        info_r0 = (contact_rate * transmission_prob) / recovery_rate
    else:
        info_r0 = 0.0

    # News firm stats
    news_firms = getattr(model, 'news_firms', [])
    active_nf = [nf for nf in news_firms if not nf.defunct]
    n_captured = sum(1 for nf in active_nf if nf.captured_by_cartel is not None)
    n_accurate_news = sum(1 for nf in active_nf if nf.accuracy >= 0.4)
    n_captured_accurate = sum(1 for nf in active_nf
                              if nf.accuracy >= 0.4 and nf.captured_by_cartel is not None)

    # Boost R0 for captured news firms (superspreader effect)
    # Each captured firm adds to effective contact rate
    for nf in news_firms:
        if nf.captured_by_cartel is not None and not nf.defunct:
            # Captured firm's audience adds to biased signal R0
            info_r0 += nf.audience_size * transmission_prob / max(len(workers), 1)

    # Epistemic health: composite metric
    # High trust + low polarization + R0 near 1 = healthy information ecology
    # Low trust + high polarization + R0 >> 1 = information crisis
    eh_trust = mean_trust
    eh_polarization = max(0, 1.0 - polarization * 2)
    eh_r0 = max(0, 1.0 - abs(info_r0 - 1.0) * 0.5)  # penalty for R0 far from 1
    epistemic_health = (eh_trust * eh_polarization * eh_r0) ** (1/3)  # geometric mean

    return {
        "mean_authority_trust": mean_trust,
        "min_authority_trust": float(np.min(trusts)),
        "weight_polarization": polarization,
        "info_r0": float(info_r0),
        "n_news_firms": len(active_nf),
        "n_captured_news": n_captured,
        "n_accurate_news": n_accurate_news,
        "n_captured_accurate": n_captured_accurate,
        "epistemic_health": float(epistemic_health),
        "trust_gini": trust_gini,
        "pct_low_trust": pct_low_trust,
    }

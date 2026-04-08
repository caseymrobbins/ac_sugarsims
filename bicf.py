"""Bottleneck Innovation & Competition Framework (BICF) helpers.

This module provides a small, testable policy-evaluation surface for the
framework proposal in the project discussion.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, Mapping, Optional, Set


@dataclass(frozen=True)
class IndustryProfile:
    sector_type: str
    high_capital_barriers: bool
    concentrated_market_share: bool
    chokepoint_infrastructure: bool
    persistent_rent_extraction: bool
    low_innovation_rates: bool


@dataclass(frozen=True)
class EntrantProfile:
    incumbent_ownership_share: float
    sells_to_end_customers: bool
    price_vs_incumbent_median: float
    incumbent_linked_revenue_share: float
    market_share_after_5y: float
    board_control_by_incumbent: bool = False  # mechanism 4: no board control


@dataclass(frozen=True)
class AcquisitionEvent:
    """Represents an incumbent acquisition of a qualifying entrant (mechanism 5)."""
    technology_remains_available: bool
    technology_suppressed: bool


SECTOR_THRESHOLDS = {
    "capital_intensive": 0.15,
    "regulated": 0.20,
    "service_technology": 0.25,
}


def is_bottleneck_industry(industry: IndustryProfile) -> bool:
    """Classify bottleneck industries from the five core criteria."""
    signals = [
        industry.high_capital_barriers,
        industry.concentrated_market_share,
        industry.chokepoint_infrastructure,
        industry.persistent_rent_extraction,
        industry.low_innovation_rates,
    ]
    return sum(signals) >= 4


def required_improvement_threshold(sector_type: str) -> float:
    try:
        return SECTOR_THRESHOLDS[sector_type]
    except KeyError as exc:
        raise ValueError(f"Unknown sector_type: {sector_type}") from exc


def aggregate_improvement(improvements: Mapping[str, float]) -> float:
    """Aggregate multi-metric improvements using median for robustness."""
    if not improvements:
        return 0.0
    vals = [max(0.0, float(v)) for v in improvements.values()]
    return median(vals)


def innovation_tier(aggregate_gain: float) -> int:
    if aggregate_gain >= 0.25:
        return 2
    if aggregate_gain >= 0.10:
        return 1
    return 0


def passes_market_participation(entrant: EntrantProfile) -> bool:
    """Mechanisms 3 & 4: market participation and independence requirements."""
    return all([
        entrant.incumbent_ownership_share <= 0.20,      # mechanism 4: ≤20% ownership
        not entrant.board_control_by_incumbent,          # mechanism 4: no board control
        entrant.sells_to_end_customers,                  # mechanism 3: direct sales
        entrant.price_vs_incumbent_median <= 1.0,        # mechanism 3: at or below median price
        entrant.incumbent_linked_revenue_share <= 0.30,  # mechanism 3: ≤30% from incumbents
        entrant.market_share_after_5y >= 0.02,           # mechanism 3: ≥2% share within 5y
    ])


# ---------------------------------------------------------------------------
# Mechanism 2: Incentive structure
# ---------------------------------------------------------------------------

def incentive_package(tier: int, qualified: bool) -> Dict[str, bool]:
    """Map innovation tier to applicable incentives.

    Tier 1 (≥10% gain): tax reduction + regulatory fast-track.
    Tier 2 (≥25% gain): adds innovation grant + government legal defense.
    """
    if not qualified or tier == 0:
        return {
            "tax_reduction": False,
            "innovation_grant": False,
            "regulatory_fast_track": False,
            "legal_defense": False,
        }
    return {
        "tax_reduction": True,
        "innovation_grant": tier >= 2,    # mechanism 2 & 8: large improvement → grant + defense
        "regulatory_fast_track": True,
        "legal_defense": tier >= 2,
    }


# ---------------------------------------------------------------------------
# Mechanism 5: Acquisition safeguards
# ---------------------------------------------------------------------------

def evaluate_acquisition_safeguard(event: AcquisitionEvent) -> bool:
    """Returns True if the acquisition complies with BICF safeguards.

    After an incumbent acquires a qualifying entrant the innovation must
    remain available to the market and cannot be suppressed.
    """
    return event.technology_remains_available and not event.technology_suppressed


# ---------------------------------------------------------------------------
# Mechanism 6: Supplier retaliation protection
# ---------------------------------------------------------------------------

def supplier_retaliation_levy(
    contract_value: float,
    legitimate_exemption: bool,
) -> Dict[str, float]:
    """Compute incumbent tax and entrant transfer for retaliatory contract cancellations.

    If a supplier contract is cancelled because the supplier works with the
    entrant, the incumbent is taxed equal to the contract value; that amount
    is transferred to the entrant.  Legitimate termination exemptions bypass
    the levy entirely.
    """
    if legitimate_exemption or contract_value <= 0.0:
        return {"incumbent_tax": 0.0, "entrant_transfer": 0.0}
    return {"incumbent_tax": contract_value, "entrant_transfer": contract_value}


# ---------------------------------------------------------------------------
# Mechanism 7: Supply chain neutrality
# ---------------------------------------------------------------------------

def check_supply_chain_neutrality(
    is_critical_supplier: bool,
    access_denied_to_entrant: bool,
) -> bool:
    """Returns True if supply chain neutrality is satisfied.

    Critical suppliers and infrastructure providers must offer
    non-discriminatory access to qualifying entrants.
    """
    if is_critical_supplier and access_denied_to_entrant:
        return False
    return True


# ---------------------------------------------------------------------------
# Mechanism 10: Dynamic benchmarking
# ---------------------------------------------------------------------------

def compute_relative_improvements(
    entrant_metrics: Mapping[str, float],
    industry_baseline: Mapping[str, float],
    lower_is_better: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Compute per-metric improvement relative to the *current* industry baseline.

    This implements dynamic benchmarking: by measuring against today's
    industry median (or top-performer values), the qualification bar rises
    automatically as the industry improves.

    lower_is_better: metric names where a lower value is an improvement
                     (e.g. cost, emissions, waste). Defaults to all metrics.
    Returns a dict of fractional improvements; negative values mean the
    entrant is worse than baseline on that metric.
    """
    if lower_is_better is None:
        lower_is_better = set(industry_baseline.keys())

    result: Dict[str, float] = {}
    for metric, baseline_val in industry_baseline.items():
        if baseline_val == 0:
            continue
        entrant_val = entrant_metrics.get(metric, baseline_val)
        if metric in lower_is_better:
            improvement = (baseline_val - entrant_val) / abs(baseline_val)
        else:
            improvement = (entrant_val - baseline_val) / abs(baseline_val)
        result[metric] = improvement
    return result


# ---------------------------------------------------------------------------
# Combined qualification entry point
# ---------------------------------------------------------------------------

def evaluate_bicf_qualification(
    industry: IndustryProfile,
    entrant: EntrantProfile,
    metric_improvements: Mapping[str, float],
) -> Dict[str, object]:
    """Return a normalized qualification verdict for policy simulation."""
    bottleneck = is_bottleneck_industry(industry)
    aggregate_gain = aggregate_improvement(metric_improvements)
    threshold = required_improvement_threshold(industry.sector_type)
    innovation_pass = aggregate_gain >= threshold
    market_pass = passes_market_participation(entrant)
    qualified = bottleneck and innovation_pass and market_pass
    tier = innovation_tier(aggregate_gain)

    return {
        "qualified": qualified,
        "bottleneck_industry": bottleneck,
        "aggregate_improvement": aggregate_gain,
        "required_threshold": threshold,
        "innovation_pass": innovation_pass,
        "market_participation_pass": market_pass,
        "tier": tier,
        "incentives": incentive_package(tier, qualified),
    }

"""Bottleneck Innovation & Competition Framework (BICF) helpers.

This module provides a small, testable policy-evaluation surface for the
framework proposal in the project discussion.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, Mapping


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
    return all(
        [
            entrant.incumbent_ownership_share <= 0.20,
            entrant.sells_to_end_customers,
            entrant.price_vs_incumbent_median <= 1.0,
            entrant.incumbent_linked_revenue_share <= 0.30,
            entrant.market_share_after_5y >= 0.02,
        ]
    )


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

    return {
        "qualified": qualified,
        "bottleneck_industry": bottleneck,
        "aggregate_improvement": aggregate_gain,
        "required_threshold": threshold,
        "innovation_pass": innovation_pass,
        "market_participation_pass": market_pass,
        "tier": innovation_tier(aggregate_gain),
    }

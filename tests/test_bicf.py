import unittest

from bicf import (
    AcquisitionEvent,
    EntrantProfile,
    IndustryProfile,
    aggregate_improvement,
    check_supply_chain_neutrality,
    compute_relative_improvements,
    evaluate_acquisition_safeguard,
    evaluate_bicf_qualification,
    incentive_package,
    innovation_tier,
    is_bottleneck_industry,
    passes_market_participation,
    required_improvement_threshold,
    supplier_retaliation_levy,
)


class TestBottleneckClassification(unittest.TestCase):
    def _industry(self, **overrides):
        defaults = dict(
            sector_type="capital_intensive",
            high_capital_barriers=True,
            concentrated_market_share=True,
            chokepoint_infrastructure=True,
            persistent_rent_extraction=True,
            low_innovation_rates=False,
        )
        defaults.update(overrides)
        return IndustryProfile(**defaults)

    def test_four_signals_qualifies(self):
        self.assertTrue(is_bottleneck_industry(self._industry()))

    def test_five_signals_qualifies(self):
        self.assertTrue(is_bottleneck_industry(self._industry(low_innovation_rates=True)))

    def test_three_signals_does_not_qualify(self):
        ind = self._industry(concentrated_market_share=False, chokepoint_infrastructure=False)
        self.assertFalse(is_bottleneck_industry(ind))


class TestInnovationQualification(unittest.TestCase):
    """Mechanism 1: thresholds, aggregation, tiers."""

    def test_threshold_map(self):
        self.assertEqual(required_improvement_threshold("capital_intensive"), 0.15)
        self.assertEqual(required_improvement_threshold("regulated"), 0.20)
        self.assertEqual(required_improvement_threshold("service_technology"), 0.25)

    def test_unknown_sector_raises(self):
        with self.assertRaises(ValueError):
            required_improvement_threshold("unknown")

    def test_aggregate_improvement_uses_median(self):
        improvements = {
            "production_cost": 0.20,
            "energy_efficiency": 0.15,
            "emissions": 0.30,
            "throughput": 0.10,
            "waste": 0.05,
        }
        self.assertEqual(aggregate_improvement(improvements), 0.15)

    def test_aggregate_improvement_empty(self):
        self.assertEqual(aggregate_improvement({}), 0.0)

    def test_aggregate_improvement_clamps_negatives(self):
        # Negative improvements are treated as zero
        result = aggregate_improvement({"cost": -0.10, "energy": 0.20})
        self.assertEqual(result, 0.10)

    def test_tier_assignment(self):
        self.assertEqual(innovation_tier(0.09), 0)
        self.assertEqual(innovation_tier(0.10), 1)
        self.assertEqual(innovation_tier(0.249), 1)
        self.assertEqual(innovation_tier(0.25), 2)
        self.assertEqual(innovation_tier(0.50), 2)


class TestMarketParticipationAndIndependence(unittest.TestCase):
    """Mechanisms 3 & 4."""

    def _good_entrant(self, **overrides):
        defaults = dict(
            incumbent_ownership_share=0.20,
            sells_to_end_customers=True,
            price_vs_incumbent_median=0.98,
            incumbent_linked_revenue_share=0.30,
            market_share_after_5y=0.03,
            board_control_by_incumbent=False,
        )
        defaults.update(overrides)
        return EntrantProfile(**defaults)

    def test_passes_when_all_criteria_met(self):
        self.assertTrue(passes_market_participation(self._good_entrant()))

    def test_fails_excessive_incumbent_ownership(self):
        self.assertFalse(passes_market_participation(
            self._good_entrant(incumbent_ownership_share=0.21)
        ))

    def test_fails_board_control(self):
        self.assertFalse(passes_market_participation(
            self._good_entrant(board_control_by_incumbent=True)
        ))

    def test_fails_no_direct_sales(self):
        self.assertFalse(passes_market_participation(
            self._good_entrant(sells_to_end_customers=False)
        ))

    def test_fails_price_above_median(self):
        self.assertFalse(passes_market_participation(
            self._good_entrant(price_vs_incumbent_median=1.01)
        ))

    def test_fails_excessive_incumbent_revenue(self):
        self.assertFalse(passes_market_participation(
            self._good_entrant(incumbent_linked_revenue_share=0.31)
        ))

    def test_fails_insufficient_market_share(self):
        self.assertFalse(passes_market_participation(
            self._good_entrant(market_share_after_5y=0.019)
        ))

    def test_boundary_market_share_2pct(self):
        self.assertTrue(passes_market_participation(
            self._good_entrant(market_share_after_5y=0.02)
        ))


class TestIncentiveStructure(unittest.TestCase):
    """Mechanism 2 & 8: incentive packages by tier."""

    def test_not_qualified_gets_no_incentives(self):
        pkg = incentive_package(tier=2, qualified=False)
        self.assertFalse(any(pkg.values()))

    def test_tier_zero_gets_no_incentives(self):
        pkg = incentive_package(tier=0, qualified=True)
        self.assertFalse(any(pkg.values()))

    def test_tier_one_incentives(self):
        pkg = incentive_package(tier=1, qualified=True)
        self.assertTrue(pkg["tax_reduction"])
        self.assertTrue(pkg["regulatory_fast_track"])
        self.assertFalse(pkg["innovation_grant"])
        self.assertFalse(pkg["legal_defense"])

    def test_tier_two_incentives(self):
        pkg = incentive_package(tier=2, qualified=True)
        self.assertTrue(pkg["tax_reduction"])
        self.assertTrue(pkg["regulatory_fast_track"])
        self.assertTrue(pkg["innovation_grant"])
        self.assertTrue(pkg["legal_defense"])


class TestAcquisitionSafeguards(unittest.TestCase):
    """Mechanism 5."""

    def test_compliant_acquisition(self):
        event = AcquisitionEvent(technology_remains_available=True, technology_suppressed=False)
        self.assertTrue(evaluate_acquisition_safeguard(event))

    def test_suppressed_technology_fails(self):
        event = AcquisitionEvent(technology_remains_available=True, technology_suppressed=True)
        self.assertFalse(evaluate_acquisition_safeguard(event))

    def test_unavailable_technology_fails(self):
        event = AcquisitionEvent(technology_remains_available=False, technology_suppressed=False)
        self.assertFalse(evaluate_acquisition_safeguard(event))

    def test_both_violations_fails(self):
        event = AcquisitionEvent(technology_remains_available=False, technology_suppressed=True)
        self.assertFalse(evaluate_acquisition_safeguard(event))


class TestSupplierRetaliationProtection(unittest.TestCase):
    """Mechanism 6."""

    def test_retaliatory_cancellation_levied(self):
        result = supplier_retaliation_levy(contract_value=500_000.0, legitimate_exemption=False)
        self.assertEqual(result["incumbent_tax"], 500_000.0)
        self.assertEqual(result["entrant_transfer"], 500_000.0)

    def test_legitimate_exemption_no_levy(self):
        result = supplier_retaliation_levy(contract_value=500_000.0, legitimate_exemption=True)
        self.assertEqual(result["incumbent_tax"], 0.0)
        self.assertEqual(result["entrant_transfer"], 0.0)

    def test_zero_contract_value_no_levy(self):
        result = supplier_retaliation_levy(contract_value=0.0, legitimate_exemption=False)
        self.assertEqual(result["incumbent_tax"], 0.0)
        self.assertEqual(result["entrant_transfer"], 0.0)

    def test_negative_contract_value_no_levy(self):
        result = supplier_retaliation_levy(contract_value=-100.0, legitimate_exemption=False)
        self.assertEqual(result["incumbent_tax"], 0.0)


class TestSupplyChainNeutrality(unittest.TestCase):
    """Mechanism 7."""

    def test_critical_supplier_denying_access_violates(self):
        self.assertFalse(check_supply_chain_neutrality(
            is_critical_supplier=True, access_denied_to_entrant=True
        ))

    def test_critical_supplier_granting_access_ok(self):
        self.assertTrue(check_supply_chain_neutrality(
            is_critical_supplier=True, access_denied_to_entrant=False
        ))

    def test_non_critical_supplier_may_deny(self):
        self.assertTrue(check_supply_chain_neutrality(
            is_critical_supplier=False, access_denied_to_entrant=True
        ))


class TestDynamicBenchmarking(unittest.TestCase):
    """Mechanism 10: improvements measured against current industry baseline."""

    def test_lower_is_better_metrics(self):
        baseline = {"cost": 100.0, "emissions": 200.0}
        entrant = {"cost": 80.0, "emissions": 150.0}
        result = compute_relative_improvements(entrant, baseline, lower_is_better={"cost", "emissions"})
        self.assertAlmostEqual(result["cost"], 0.20)
        self.assertAlmostEqual(result["emissions"], 0.25)

    def test_higher_is_better_metrics(self):
        baseline = {"throughput": 100.0}
        entrant = {"throughput": 120.0}
        result = compute_relative_improvements(entrant, baseline, lower_is_better=set())
        self.assertAlmostEqual(result["throughput"], 0.20)

    def test_mixed_metric_directions(self):
        baseline = {"cost": 100.0, "throughput": 100.0}
        entrant = {"cost": 85.0, "throughput": 115.0}
        result = compute_relative_improvements(
            entrant, baseline, lower_is_better={"cost"}
        )
        self.assertAlmostEqual(result["cost"], 0.15)
        self.assertAlmostEqual(result["throughput"], 0.15)

    def test_defaults_to_all_lower_is_better(self):
        baseline = {"cost": 100.0}
        entrant = {"cost": 80.0}
        result = compute_relative_improvements(entrant, baseline)
        self.assertAlmostEqual(result["cost"], 0.20)

    def test_missing_entrant_metric_defaults_to_baseline(self):
        baseline = {"cost": 100.0, "emissions": 100.0}
        entrant = {}  # entrant has no data → same as baseline → 0% improvement
        result = compute_relative_improvements(entrant, baseline)
        self.assertAlmostEqual(result["cost"], 0.0)
        self.assertAlmostEqual(result["emissions"], 0.0)

    def test_zero_baseline_skipped(self):
        baseline = {"cost": 0.0}
        entrant = {"cost": 10.0}
        result = compute_relative_improvements(entrant, baseline)
        self.assertNotIn("cost", result)

    def test_dynamic_benchmarking_pipeline(self):
        """Full pipeline: compute improvements from live baseline, then qualify."""
        industry = IndustryProfile(
            sector_type="capital_intensive",
            high_capital_barriers=True,
            concentrated_market_share=True,
            chokepoint_infrastructure=True,
            persistent_rent_extraction=True,
            low_innovation_rates=False,
        )
        entrant_profile = EntrantProfile(
            incumbent_ownership_share=0.10,
            sells_to_end_customers=True,
            price_vs_incumbent_median=0.95,
            incumbent_linked_revenue_share=0.10,
            market_share_after_5y=0.03,
        )
        industry_baseline = {"cost_per_unit": 100.0, "energy_kwh": 500.0, "co2_kg": 200.0}
        entrant_actuals = {"cost_per_unit": 82.0, "energy_kwh": 410.0, "co2_kg": 158.0}
        improvements = compute_relative_improvements(entrant_actuals, industry_baseline)
        result = evaluate_bicf_qualification(industry, entrant_profile, improvements)
        self.assertTrue(result["qualified"])


class TestEndToEndQualification(unittest.TestCase):
    def setUp(self):
        self.industry = IndustryProfile(
            sector_type="capital_intensive",
            high_capital_barriers=True,
            concentrated_market_share=True,
            chokepoint_infrastructure=True,
            persistent_rent_extraction=True,
            low_innovation_rates=False,
        )

    def test_qualified_entrant(self):
        entrant = EntrantProfile(
            incumbent_ownership_share=0.10,
            sells_to_end_customers=True,
            price_vs_incumbent_median=1.0,
            incumbent_linked_revenue_share=0.20,
            market_share_after_5y=0.04,
        )
        improvements = {
            "production_cost": 0.18,
            "energy_efficiency": 0.22,
            "emissions": 0.19,
            "throughput": 0.15,
            "waste": 0.16,
        }
        result = evaluate_bicf_qualification(self.industry, entrant, improvements)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["tier"], 1)
        self.assertIn("incentives", result)
        self.assertTrue(result["incentives"]["tax_reduction"])
        self.assertFalse(result["incentives"]["innovation_grant"])

    def test_high_gain_entrant_gets_full_incentives(self):
        entrant = EntrantProfile(
            incumbent_ownership_share=0.05,
            sells_to_end_customers=True,
            price_vs_incumbent_median=0.90,
            incumbent_linked_revenue_share=0.05,
            market_share_after_5y=0.05,
        )
        improvements = {
            "cost": 0.30,
            "energy": 0.28,
            "emissions": 0.35,
            "throughput": 0.26,
        }
        result = evaluate_bicf_qualification(self.industry, entrant, improvements)
        self.assertTrue(result["qualified"])
        self.assertEqual(result["tier"], 2)
        self.assertTrue(result["incentives"]["innovation_grant"])
        self.assertTrue(result["incentives"]["legal_defense"])

    def test_non_bottleneck_industry_does_not_qualify(self):
        weak_industry = IndustryProfile(
            sector_type="capital_intensive",
            high_capital_barriers=True,
            concentrated_market_share=False,
            chokepoint_infrastructure=False,
            persistent_rent_extraction=False,
            low_innovation_rates=False,
        )
        entrant = EntrantProfile(
            incumbent_ownership_share=0.10,
            sells_to_end_customers=True,
            price_vs_incumbent_median=0.90,
            incumbent_linked_revenue_share=0.10,
            market_share_after_5y=0.05,
        )
        result = evaluate_bicf_qualification(weak_industry, entrant, {"cost": 0.30})
        self.assertFalse(result["qualified"])
        self.assertFalse(result["bottleneck_industry"])


if __name__ == "__main__":
    unittest.main()

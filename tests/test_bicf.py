import unittest

from bicf import (
    EntrantProfile,
    IndustryProfile,
    aggregate_improvement,
    evaluate_bicf_qualification,
    innovation_tier,
    passes_market_participation,
    required_improvement_threshold,
)


class TestBICFQualification(unittest.TestCase):
    def setUp(self):
        self.industry = IndustryProfile(
            sector_type="capital_intensive",
            high_capital_barriers=True,
            concentrated_market_share=True,
            chokepoint_infrastructure=True,
            persistent_rent_extraction=True,
            low_innovation_rates=False,
        )

    def test_threshold_map(self):
        self.assertEqual(required_improvement_threshold("capital_intensive"), 0.15)
        self.assertEqual(required_improvement_threshold("regulated"), 0.20)
        self.assertEqual(required_improvement_threshold("service_technology"), 0.25)

    def test_aggregate_improvement_uses_median(self):
        improvements = {
            "production_cost": 0.20,
            "energy_efficiency": 0.15,
            "emissions": 0.30,
            "throughput": 0.10,
            "waste": 0.05,
        }
        self.assertEqual(aggregate_improvement(improvements), 0.15)

    def test_tier_assignment(self):
        self.assertEqual(innovation_tier(0.09), 0)
        self.assertEqual(innovation_tier(0.10), 1)
        self.assertEqual(innovation_tier(0.249), 1)
        self.assertEqual(innovation_tier(0.25), 2)

    def test_market_participation_rules(self):
        good = EntrantProfile(
            incumbent_ownership_share=0.20,
            sells_to_end_customers=True,
            price_vs_incumbent_median=0.98,
            incumbent_linked_revenue_share=0.30,
            market_share_after_5y=0.03,
        )
        bad = EntrantProfile(
            incumbent_ownership_share=0.45,
            sells_to_end_customers=True,
            price_vs_incumbent_median=0.90,
            incumbent_linked_revenue_share=0.10,
            market_share_after_5y=0.03,
        )
        self.assertTrue(passes_market_participation(good))
        self.assertFalse(passes_market_participation(bad))

    def test_end_to_end_qualification(self):
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


if __name__ == "__main__":
    unittest.main()

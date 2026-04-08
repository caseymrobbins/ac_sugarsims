import sys
import types
import unittest
from types import SimpleNamespace


# Provide a tiny numpy stub so run_parallel can be imported in minimal CI envs.
if "numpy" not in sys.modules:
    fake_numpy = types.ModuleType("numpy")

    class _FakeRng:
        def shuffle(self, seq):
            # deterministic no-op shuffle for unit tests
            return None

    class _FakeRandom:
        @staticmethod
        def default_rng(_seed=None):
            return _FakeRng()

    fake_numpy.random = _FakeRandom()
    sys.modules["numpy"] = fake_numpy

from run_parallel import B2, B3, Condition, PRESETS, configure_model


class DummyFirm:
    def __init__(self):
        self.worker_ownership_share = 0.0
        self.investor_ownership_share = 1.0
        self.is_sevc = True
        self.strategy_weights = {"expand": 0.6, "innovate": 0.4}
        self.tech_level = 1.2


class DummyModel:
    def __init__(self, n_firms=3):
        self.firms = [DummyFirm() for _ in range(n_firms)]
        self.schedule = SimpleNamespace(agents=[])
        self.rng = sys.modules["numpy"].random.default_rng(123)


class TestRunParallelConfig(unittest.TestCase):
    def test_bicf_test_preset_registered(self):
        self.assertIn("bicf_test", PRESETS)
        preset = PRESETS["bicf_test"]
        self.assertEqual(len(preset["conditions"]), 1)
        self.assertEqual(preset["conditions"][0].name, "BICF_test_condition")

    def test_bottleneck_preset_registered(self):
        self.assertIn("bottleneck", PRESETS)
        preset = PRESETS["bottleneck"]
        condition_names = [c.name for c in preset["conditions"]]
        self.assertEqual(
            condition_names,
            ["B1_baseline_no_reg", "B2_bottleneck_reg", "B3_bottleneck_aggressive"],
        )

    def test_aggressive_bottleneck_policy_sets_caps(self):
        model = DummyModel()
        configured = configure_model(model, B3)

        self.assertEqual(configured.bottleneck_regulation_policy, "aggressive")
        self.assertEqual(configured.max_profit_margin_cap, 0.10)
        self.assertEqual(configured.bottleneck_open_access_bonus, 0.10)
        self.assertEqual(configured.bottleneck_breakup_threshold, 25)

    def test_enabled_bottleneck_policy_sets_moderate_caps(self):
        model = DummyModel()
        configured = configure_model(model, B2)

        self.assertEqual(configured.bottleneck_regulation_policy, "enabled")
        self.assertEqual(configured.max_profit_margin_cap, 0.15)
        self.assertEqual(configured.bottleneck_open_access_bonus, 0.05)
        self.assertEqual(configured.bottleneck_breakup_threshold, 50)

    def test_worker_ownership_applies_to_existing_firms(self):
        model = DummyModel(n_firms=2)
        condition = Condition(
            "test_worker_owned",
            "worker owned",
            "PLANNER_SEVC",
            True,
            True,
            0.1,
            True,
            True,
            "democratic",
            worker_ownership=True,
            worker_ownership_share=0.75,
        )

        configured = configure_model(model, condition)
        for firm in configured.firms:
            self.assertEqual(firm.worker_ownership_share, 0.75)
            self.assertEqual(firm.investor_ownership_share, 0.25)


if __name__ == "__main__":
    unittest.main()

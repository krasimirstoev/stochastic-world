import unittest
from unittest.mock import patch

from stochastic_world.agent_coarse import AgentCoarsePool, _safe_action_result


class AgentMultiprocessingTest(unittest.TestCase):
    def _snapshot(self, *, energy=40, health=80, shelter=70, money=10.0,
                  food=5, medicine=1, working_age=True, kind="residential"):
        return (
            42, 0, food, medicine, energy, health, shelter, money,
            True, kind, 0, 0, 0.0, 0.0,
            3, 0.10, working_age,
        )

    def test_worker_action_result_is_deterministic(self):
        snapshot = self._snapshot()
        first = _safe_action_result(snapshot, 12345, 7, 0)
        second = _safe_action_result(snapshot, 12345, 7, 0)
        self.assertEqual(first, second)

    def test_safe_result_preserves_identity_and_state_shape(self):
        result = _safe_action_result(self._snapshot(), 999, 3, 1)
        self.assertEqual(result[0], 42)
        self.assertEqual(len(result), 10)
        self.assertIsInstance(result[2], bool)

    def test_requested_workers_are_not_capped_by_district_count(self):
        with patch("stochastic_world.agent_coarse.os.cpu_count", return_value=12):
            pool = AgentCoarsePool(123, workers=10, min_active=1)
        self.assertEqual(pool.worker_count, 10)
        self.assertTrue(pool.enabled)

    def test_worker_count_caps_at_available_cpus(self):
        with patch("stochastic_world.agent_coarse.os.cpu_count", return_value=12):
            pool = AgentCoarsePool(123, workers=100, min_active=1)
        self.assertEqual(pool.worker_count, 12)


if __name__ == "__main__":
    unittest.main()

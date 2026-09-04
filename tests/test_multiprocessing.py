import unittest

from stochastic_world.multiprocessing_engine import (
    PersistentDistrictPool,
    _end_of_day_delta,
    _weighted_action,
)


class MultiprocessingPlannerTest(unittest.TestCase):
    def test_action_planning_is_deterministic(self):
        snapshot = (
            42, 3, 2, 0, 55, 64, 32, 3.0, False, "industrial",
            2, 1, 18.0, -4.0,
        )
        first = _weighted_action(snapshot, 12345678901234567890, 17, 0)
        second = _weighted_action(snapshot, 12345678901234567890, 17, 0)
        self.assertEqual(first, second)

    def test_action_seed_changes_by_round(self):
        snapshot = (
            42, 3, 8, 2, 80, 100, 70, 20.0, True, "residential",
            0, 0, 0.0, 0.0,
        )
        results = {
            _weighted_action(snapshot, 12345, 17, round_index)
            for round_index in range(20)
        }
        self.assertGreater(len(results), 1)

    def test_end_of_day_is_deterministic(self):
        snapshot = (
            9, 0, 3, 18, 50, 2.0, 45, False, True, False, True, 0, 0.15,
        )
        first = _end_of_day_delta(snapshot, 987654321, 60)
        second = _end_of_day_delta(snapshot, 987654321, 60)
        self.assertEqual(first, second)

    def test_disabled_pool_does_not_spawn(self):
        pool = PersistentDistrictPool(123, location_count=5, workers=0)
        try:
            self.assertFalse(pool.enabled)
            self.assertFalse(pool.should_parallelize(100000))
            self.assertEqual(pool.summary()["workers"], 0)
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()

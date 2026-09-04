import sys
import unittest
from unittest.mock import patch

from stochastic_world.agent_world import AgentWorkerPool
from stochastic_world.cli import _resolved_worker_arg, parse_args, resolve_engine


class AgentMultiprocessingTest(unittest.TestCase):
    def test_agent_pool_is_not_capped_by_district_count(self):
        pool = AgentWorkerPool(12345, location_count=5, workers=10, min_active=100)
        try:
            self.assertTrue(pool.enabled)
            self.assertEqual(pool.worker_count, min(10, __import__("os").cpu_count() or 10))
            self.assertFalse(pool.started)
        finally:
            pool.close()

    def test_workers_alias_enables_agent_workers(self):
        with patch.object(sys, "argv", ["world.py", "--engine", "agent", "--workers", "10"]):
            args = parse_args()
        self.assertEqual(resolve_engine(args), "agent")
        self.assertEqual(_resolved_worker_arg(args, "agent"), 10)

    def test_agent_workers_default_to_serial(self):
        with patch.object(sys, "argv", ["world.py", "--engine", "agent"]):
            args = parse_args()
        self.assertEqual(_resolved_worker_arg(args, "agent"), 0)

    def test_hybrid_workers_keep_auto_default(self):
        with patch.object(sys, "argv", ["world.py", "--engine", "hybrid"]):
            args = parse_args()
        self.assertEqual(_resolved_worker_arg(args, "hybrid"), -1)

    def test_person_sharding_uses_all_available_workers(self):
        pool = AgentWorkerPool(1, location_count=5, workers=4, min_active=1)
        rows = [(pid, 0) for pid in range(12)]
        shards = pool._person_shards(rows)
        self.assertEqual(set(shards), {0, 1, 2, 3})
        self.assertEqual(sum(len(v) for v in shards.values()), 12)


if __name__ == "__main__":
    unittest.main()

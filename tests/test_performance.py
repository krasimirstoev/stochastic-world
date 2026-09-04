import sqlite3
import unittest

from stochastic_world.performance import PhaseProfiler


class DummyWorld:
    alive_count = 123
    last_hybrid_stats = {"explicit_agents": 7}


class DummyStore:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.simulation_id = 42


class PhaseProfilerTest(unittest.TestCase):
    def test_disabled_profiler_records_nothing(self):
        profiler = PhaseProfiler(DummyWorld(), enabled=False)
        profiler.record(1, "election", 1.0)
        self.assertEqual(profiler.rows, [])
        self.assertEqual(profiler.summary(), [])

    def test_summary_and_flush(self):
        profiler = PhaseProfiler(DummyWorld(), enabled=True)
        profiler.record(1, "election", 2.0)
        profiler.record(2, "election", 4.0)
        profiler.record(2, "welfare", 1.5)

        summary = {row["phase"]: row for row in profiler.summary()}
        self.assertEqual(summary["election"]["calls"], 2)
        self.assertAlmostEqual(summary["election"]["total"], 6.0)
        self.assertAlmostEqual(summary["election"]["avg"], 3.0)
        self.assertAlmostEqual(summary["election"]["max"], 4.0)

        store = DummyStore()
        profiler.flush(store)
        rows = store.conn.execute(
            "SELECT day,phase,duration_seconds,population_alive,explicit_agents "
            "FROM performance_timings ORDER BY day,phase"
        ).fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0][3:], (123, 7))


if __name__ == "__main__":
    unittest.main()

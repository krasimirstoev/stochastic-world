import random
import unittest
from types import SimpleNamespace

from stochastic_world.hybrid import PRIORITY_HIGH, PRIORITY_MANDATORY, HybridEngine


class FakeIndex:
    def __init__(self, people, locations):
        self.by_location = {loc.id: [] for loc in locations}
        for p in people:
            self.by_location[p.location_id].append(p)

    def population(self, location_id):
        return len([p for p in self.by_location[location_id] if p.alive])

    def sample_people(self, location_id, rng, limit, exclude=()):
        pool = [p for p in self.by_location[location_id] if p.alive and p.id not in exclude]
        return pool if len(pool) <= limit else rng.sample(pool, limit)


class HybridBudgetTests(unittest.TestCase):
    def make_world(self, population=10000, districts=5):
        locations = [SimpleNamespace(id=i) for i in range(districts)]
        people = [SimpleNamespace(id=i, alive=True, location_id=i % districts) for i in range(population)]
        return SimpleNamespace(people=people, locations=locations, population_index=FakeIndex(people, locations),
                               alive_count=population, rng=random.Random(12345), current_day=1)

    def test_default_target_is_three_percent(self):
        engine = HybridEngine(self.make_world(), sample_per_district=256)
        active = engine.select_active(1)
        self.assertEqual(len(active), 300)
        self.assertEqual(engine.last_stats["budget_target"], 300)
        self.assertEqual(engine.last_stats["budget_ceiling"], 500)

    def test_high_priority_is_capped_by_soft_ceiling(self):
        world = self.make_world(); engine = HybridEngine(world)
        for pid in range(2000):
            engine.mark_interesting(pid, day=1, days=10, reason="offender", priority=PRIORITY_HIGH)
        active = engine.select_active(1)
        self.assertEqual(len(active), 500)
        self.assertGreater(engine.last_stats["pending_interesting"], 0)

    def test_mandatory_agents_can_exceed_ceiling(self):
        world = self.make_world(); engine = HybridEngine(world)
        for pid in range(700):
            engine.mark_interesting(pid, day=1, days=3, reason="detained", priority=PRIORITY_MANDATORY)
        active = engine.select_active(1)
        self.assertEqual(len(active), 700)
        self.assertEqual(engine.last_stats["mandatory_agents"], 700)


if __name__ == "__main__":
    unittest.main()

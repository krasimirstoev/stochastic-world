import random
import unittest

from stochastic_world.agent_world import ParallelAgentWorld
from stochastic_world.person import Person


class _Store:
    event_mode = "compact"


class _Index:
    def __init__(self, witnesses=()):
        self.witnesses = list(witnesses)

    def sample_people(self, location_id, rng, limit, exclude=()):
        return self.witnesses[:limit]


class CompactRelationshipPersistenceTest(unittest.TestCase):
    def _world(self):
        world = ParallelAgentWorld.__new__(ParallelAgentWorld)
        world.store = _Store()
        world.current_day = 10
        world.max_witnesses = 0
        world.visibility = 1.0
        world.rng = random.Random(1)
        world.population_index = _Index()
        world.total_observations = 0
        return world

    def test_compact_interaction_updates_memory_without_snapshots(self):
        world = self._world()
        actor = Person(1, "A")
        target = Person(2, "B")

        world.remember_interaction(actor, target, "steal", 2.0)

        self.assertIn(target.id, actor.memories)
        self.assertIn(actor.id, target.memories)
        self.assertEqual(actor.memories[target.id].thefts_committed, 1)
        self.assertEqual(target.memories[actor.id].thefts_suffered, 1)

    def test_compact_reputation_updates_witness_memory_without_persistence_snapshots(self):
        world = self._world()
        actor = Person(1, "A", location_id=0)
        target = Person(2, "B", location_id=0)
        witness = Person(3, "C", location_id=0)
        world.max_witnesses = 1
        world.population_index = _Index([witness])

        world.spread_reputation(actor, target, "attack", 1.5)

        self.assertEqual(world.total_observations, 1)
        self.assertEqual(witness.memories[actor.id].observed_attack, 1)


if __name__ == "__main__":
    unittest.main()

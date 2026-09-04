import random
import unittest

from stochastic_world.memory import InteractionMemory
from stochastic_world.person import Person
from stochastic_world.politics import PoliticalSystem


class SocialMemoryPerformanceTest(unittest.TestCase):
    def test_aggregate_memory_updates_incrementally(self):
        person = Person(1, "A")
        other = Person(2, "B")
        aggregate = person.aggregate_memory()
        self.assertEqual(aggregate["known_people"], 0)

        person.remember(other, 1, "help", "actor", 1.0)
        self.assertIs(aggregate, person.aggregate_memory())
        self.assertEqual(aggregate["known_people"], 1)
        self.assertGreater(aggregate["mean_affinity"], 0)

        person.remember(other, 2, "attack", "target", 1.0)
        self.assertIs(aggregate, person.aggregate_memory())
        self.assertGreater(aggregate["max_conflict"], 0)
        self.assertEqual(aggregate["hostile_ties"], 1)

    def test_memory_cap_evicts_oldest_relationships(self):
        person = Person(1, "A", memory_cap=2)
        others = [Person(i, str(i)) for i in range(2, 5)]
        person.remember(others[0], 1, "help", "actor")
        person.remember(others[1], 2, "help", "actor")
        person.remember(others[2], 3, "help", "actor")
        self.assertEqual(len(person.memories), 2)
        self.assertNotIn(others[0].id, person.memories)
        self.assertEqual(person.aggregate_memory()["known_people"], 2)

    def test_zero_memory_cap_means_unlimited(self):
        person = Person(1, "A", memory_cap=0)
        for i in range(2, 102):
            person.remember(Person(i, str(i)), i, "help", "actor")
        self.assertEqual(len(person.memories), 100)

    def test_lazy_decay_matches_repeated_daily_decay(self):
        lazy = InteractionMemory(2, trust=-40.0, grievance=60.0)
        eager = InteractionMemory(2, trust=-40.0, grievance=60.0)
        for _ in range(12):
            eager.decay()
        lazy.decay_through(12)
        self.assertAlmostEqual(lazy.trust, eager.trust)
        self.assertAlmostEqual(lazy.grievance, eager.grievance)

    def test_relationship_read_materializes_decay_without_creating_memory(self):
        person = Person(1, "A")
        other = Person(2, "B")
        person.remember(other, 1, "attack", "target", 1.0)
        before = person.aggregate_memory()["max_conflict"]
        memory = person.memory_by_id(other.id, 10)
        self.assertIsNotNone(memory)
        self.assertLess(person.aggregate_memory()["max_conflict"], before)
        self.assertIsNone(person.memory_by_id(999, 10))

    def test_daily_decay_only_reconciles_full_memory_weekly(self):
        person = Person(1, "A")
        other = Person(2, "B")
        person.remember(other, 1, "attack", "target", 1.0)
        memory = person.memories[other.id]
        through = memory.decayed_through_day
        person.decay_memories(2)
        self.assertEqual(memory.decayed_through_day, through)
        person.decay_memories(7)
        self.assertEqual(memory.decayed_through_day, 7)


class InitialGovernmentTest(unittest.TestCase):
    def test_forced_government_wins_day_one(self):
        politics = PoliticalSystem(12345)
        politics.force_initial_government("left")
        people = [Person(1, "A", ideology=1.0)]
        _, _, winner = politics.hold_election(1, people, {0: 0.0}, random.Random(1))
        self.assertEqual(winner.id, "left")

    def test_later_elections_are_not_forced(self):
        politics = PoliticalSystem(12345)
        politics.force_initial_government("left")
        people = [Person(1, "A", ideology=1.0)]
        _, _, winner = politics.hold_election(1461, people, {0: 0.0}, random.Random(1))
        self.assertEqual(winner.id, "right")


if __name__ == "__main__":
    unittest.main()

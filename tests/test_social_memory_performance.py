import random
import unittest

from stochastic_world.person import Person
from stochastic_world.politics import PoliticalSystem


class SocialMemoryPerformanceTest(unittest.TestCase):
    def test_aggregate_memory_is_cached_until_mutation(self):
        person = Person(1, "A")
        other = Person(2, "B")
        person.remember(other, 1, "help", "actor", 1.0)
        first = person.aggregate_memory()
        second = person.aggregate_memory()
        self.assertIs(first, second)
        person.remember(other, 2, "attack", "target", 1.0)
        self.assertIsNot(first, person.aggregate_memory())

    def test_memory_cap_evicts_oldest_relationships(self):
        person = Person(1, "A", memory_cap=2)
        others = [Person(i, str(i)) for i in range(2, 5)]
        person.remember(others[0], 1, "help", "actor")
        person.remember(others[1], 2, "help", "actor")
        person.remember(others[2], 3, "help", "actor")
        self.assertEqual(len(person.memories), 2)
        self.assertNotIn(others[0].id, person.memories)

    def test_zero_memory_cap_means_unlimited(self):
        person = Person(1, "A", memory_cap=0)
        for i in range(2, 102):
            person.remember(Person(i, str(i)), i, "help", "actor")
        self.assertEqual(len(person.memories), 100)


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

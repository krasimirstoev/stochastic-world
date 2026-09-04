import random
import unittest

from stochastic_world.geography import build_locations
from stochastic_world.person import Person
from stochastic_world.politics import ELECTION_INTERVAL_DAYS, PoliticalSystem


class GeographyPoliticsTests(unittest.TestCase):
    def test_base_geography_is_connected_without_teleport_edges(self):
        locations = build_locations(5)
        self.assertEqual(locations[0].neighbors, (1,))
        self.assertIn(4, locations[2].neighbors)
        self.assertNotIn(4, locations[0].neighbors)

    def test_election_interval_is_four_365_day_years(self):
        self.assertEqual(ELECTION_INTERVAL_DAYS, 1460)

    def test_every_living_person_gets_a_ballot(self):
        politics = PoliticalSystem(1234)
        people = [Person(i, f"Person {i}", ideology=(-0.5 if i < 3 else 0.5)) for i in range(6)]
        people[-1].alive = False
        votes, ballots, _ = politics.hold_election(1, people, {0: 0.0}, random.Random(10))
        self.assertEqual(len(ballots), 5)
        self.assertEqual(votes["left"] + votes["right"], 5)

    def test_welfare_is_larger_under_left_baseline(self):
        left = PoliticalSystem(2); right = PoliticalSystem(3)
        left.government = left.party_by_id("left"); right.government = right.party_by_id("right")
        a = Person(1, "A", money=0, food=1); b = Person(2, "B", money=0, food=1)
        left.treasury = right.treasury = 100
        left.distribute_welfare([a], random.Random(1)); right.distribute_welfare([b], random.Random(1))
        self.assertGreater(a.money + a.food, b.money + b.food)


if __name__ == "__main__":
    unittest.main()

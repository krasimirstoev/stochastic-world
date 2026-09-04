import random
import unittest

from stochastic_world.person import ADULT_AGE_DAYS, RETIREMENT_AGE_DAYS, Person
from stochastic_world.population import build_population
from stochastic_world.population_index import PopulationIndex
from stochastic_world.politics import PoliticalSystem


class DemographicBasicsTest(unittest.TestCase):
    def test_population_has_age_structure(self):
        people = build_population(500, 12345)
        self.assertTrue(any(p.is_dependent for p in people))
        self.assertTrue(any(p.is_working_age for p in people))
        self.assertTrue(any(p.retired for p in people))
        self.assertTrue(all(p.profession == "dependent" for p in people if p.age_days < ADULT_AGE_DAYS))
        self.assertTrue(all(p.profession == "retired" for p in people if p.age_days >= RETIREMENT_AGE_DAYS))

    def test_population_index_can_append_birth(self):
        people = [Person(0, "A", location_id=0)]
        index = PopulationIndex(people, 1)
        baby = Person(1, "B", location_id=0, age_days=0, profession="dependent")
        people.append(baby)
        index.add(1, 0)
        self.assertEqual(index.population(0), 2)
        self.assertEqual(index.positions[1], 1)

    def test_minors_do_not_vote(self):
        politics = PoliticalSystem(2)
        minor = Person(0, "Minor", age_days=10 * 365)
        adult = Person(1, "Adult", age_days=30 * 365)
        votes, ballots, _ = politics.hold_election(1, [minor, adult], {0: 0.0}, random.Random(7))
        self.assertEqual(sum(votes.values()), 1)
        self.assertEqual(len(ballots), 1)
        self.assertEqual(ballots[0][0].id, adult.id)


if __name__ == "__main__":
    unittest.main()

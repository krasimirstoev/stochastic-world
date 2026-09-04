import random
import unittest

from stochastic_world.geography import build_locations
from stochastic_world.labor_market import LaborMarket, build_employers
from stochastic_world.person import Person


class LaborMarketTests(unittest.TestCase):
    def test_hiring_respects_capacity(self):
        rng = random.Random(42)
        locations = build_locations(5)
        employers = build_employers(locations, 20, rng)
        market = LaborMarket(employers, rng)
        person = Person(1, "Test Person", profession="laborer", location_id=2)
        employer = market.hire(person)
        self.assertIsNotNone(employer)
        self.assertEqual(person.employer_id, employer.id)
        self.assertIn(person.id, employer.employee_ids)
        self.assertLessEqual(len(employer.employee_ids), employer.capacity)

    def test_no_local_vacancy_means_no_job(self):
        rng = random.Random(1)
        locations = build_locations(5)
        employers = build_employers(locations, 10, rng)
        market = LaborMarket(employers, rng)
        for employer in market.local_employers(0):
            employer.employee_ids = set(range(employer.capacity))
        person = Person(999, "Unemployed", profession="service_worker", location_id=0)
        self.assertIsNone(market.hire(person))
        self.assertIsNone(person.employer_id)

    def test_work_shift_changes_firm_cash(self):
        rng = random.Random(7)
        locations = build_locations(5)
        employers = build_employers(locations, 20, rng)
        market = LaborMarket(employers, rng)
        person = Person(2, "Worker", profession="laborer", location_id=2)
        employer = market.hire(person)
        self.assertIsNotNone(employer)
        before = employer.cash
        result = market.work_shift(person, locations[2])
        self.assertFalse(result["insolvent"])
        self.assertGreater(result["gross"], 0)
        self.assertNotEqual(employer.cash, before)

    def test_termination_releases_slot(self):
        rng = random.Random(9)
        locations = build_locations(5)
        employers = build_employers(locations, 20, rng)
        market = LaborMarket(employers, rng)
        person = Person(3, "Mover", profession="laborer", location_id=2)
        employer = market.hire(person)
        before = employer.vacancies
        market.terminate(person, "relocation")
        self.assertIsNone(person.employer_id)
        self.assertEqual(employer.vacancies, before + 1)


if __name__ == "__main__":
    unittest.main()

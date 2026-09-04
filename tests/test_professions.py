import unittest

from stochastic_world.geography import build_locations
from stochastic_world.person import Person
from stochastic_world.professions import (
    CLASS_ORDER,
    MOBILITY_INTERVAL_DAYS,
    mobility_decision,
    profession_for,
    workplace_fit,
)


class ProfessionAndMobilityTests(unittest.TestCase):
    def test_mobility_interval_is_180_days(self):
        self.assertEqual(MOBILITY_INTERVAL_DAYS, 180)

    def test_upward_mobility_is_only_one_class(self):
        person = Person(
            id=1,
            name="Test Person",
            social_class="working",
            profession="laborer",
            money=300,
            food=30,
            shelter=100,
            health=100,
            work_experience=300,
            career_progress=120,
        )
        new_class, _, direction = mobility_decision(person)
        self.assertEqual(new_class, "lower_middle")
        self.assertEqual(direction, "up")
        self.assertEqual(
            CLASS_ORDER.index(new_class) - CLASS_ORDER.index(person.social_class),
            1,
        )

    def test_downward_mobility_is_only_one_class(self):
        person = Person(
            id=2,
            name="Test Person 2",
            social_class="upper_middle",
            profession="engineer",
            money=0,
            food=0,
            shelter=10,
            health=20,
        )
        new_class, _, direction = mobility_decision(person)
        self.assertEqual(new_class, "middle")
        self.assertEqual(direction, "down")

    def test_profession_has_preferred_workplace_bonus(self):
        person = Person(
            id=3,
            name="Engineer",
            social_class="upper_middle",
            profession="engineer",
        )
        locations = build_locations(5)
        industrial = next(l for l in locations if l.kind == "industrial")
        market = next(l for l in locations if l.kind == "market")
        self.assertGreater(
            workplace_fit(person, industrial),
            workplace_fit(person, market),
        )
        self.assertEqual(profession_for(person).id, "engineer")


if __name__ == "__main__":
    unittest.main()

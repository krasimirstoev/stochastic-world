import random
import tempfile
import unittest
from pathlib import Path

from stochastic_world.fast_storage import BufferedEventMixin
from stochastic_world.person import Person
from stochastic_world.population_index import PopulationIndex
from stochastic_world.storage import EventStore


class BufferedStore(BufferedEventMixin, EventStore):
    pass


class FastStorageTest(unittest.TestCase):
    def test_events_are_batched_until_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "seed": 1,
                "population": 1,
                "actions_per_day": 1,
                "period": 1,
                "faker_locale": "en_US",
                "visibility": 0.0,
                "max_witnesses": 0,
                "locations_count": 5,
                "event_mode": "compact",
            }
            store = BufferedStore(Path(tmp) / "run.sqlite", Path(tmp) / "run.log", config)
            actor = Person(0, "A")
            store.event(1, 1, "death", actor=actor, cause="test")
            before = store.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            self.assertEqual(before, 0)
            store.commit_day()
            after = store.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            self.assertEqual(after, 1)
            store.finish()


class PopulationSamplingTest(unittest.TestCase):
    def test_sample_is_unique_and_respects_exclusions(self):
        people = [Person(i, str(i), location_id=0) for i in range(100)]
        index = PopulationIndex(people, 1)
        result = index.sample_people(0, random.Random(123), 16, exclude=(0, 1, 2))
        ids = [person.id for person in result]
        self.assertEqual(len(ids), 16)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse({0, 1, 2}.intersection(ids))


if __name__ == "__main__":
    unittest.main()

import random
import unittest

from stochastic_world.geography import build_locations, recommended_location_count
from stochastic_world.market import GoodsMarket
from stochastic_world.population_index import PopulationIndex, permutation_ids
from stochastic_world.transport import TransportSystem


class DummyPerson:
    def __init__(self, pid, location_id):
        self.id = pid
        self.location_id = location_id
        self.alive = True


class DummyEmployer:
    def __init__(self, kind="logistics", employees=4):
        self.kind = kind
        self.alive = True
        self.employee_ids = set(range(employees))
        self.output_good = None


class DummyLabor:
    def local_employers(self, location_id):
        return [DummyEmployer()]


class ScaleTests(unittest.TestCase):
    def test_two_million_auto_locations(self):
        self.assertEqual(recommended_location_count(2_000_000, 20_000), 100)

    def test_large_map_is_connected_locally(self):
        locations = build_locations(100, population_size=2_000_000)
        self.assertEqual(len(locations), 100)
        self.assertTrue(all(location.neighbors for location in locations))
        self.assertTrue(all(location.capacity_hint == 20_000 for location in locations))

    def test_population_index_moves_without_global_scan(self):
        people = [DummyPerson(i, i % 5) for i in range(100)]
        index = PopulationIndex(people, 5)
        index.move(7, 2, 3)
        people[7].location_id = 3
        self.assertNotIn(7, index.ids(2))
        self.assertIn(7, index.ids(3))
        sample = index.sample_people(3, random.Random(1), 8, exclude=(7,))
        self.assertLessEqual(len(sample), 8)
        self.assertTrue(all(p.location_id == 3 and p.id != 7 for p in sample))

    def test_permutation_has_every_id_once(self):
        order = list(permutation_ids(997, random.Random(42)))
        self.assertEqual(len(order), 997)
        self.assertEqual(len(set(order)), 997)
        self.assertEqual(set(order), set(range(997)))

    def test_transport_moves_surplus_to_neighbor_shortage(self):
        locations = build_locations(5, population_size=100)
        market = GoodsMarket(locations, {i: 20 for i in range(5)})
        for state in market.states.values():
            state.supplier_stock["food"].clear()
        market.add_supply(1, 10, "food", 100)
        transport = TransportSystem(locations, market, DummyLabor(), random.Random(2), capacity_per_1000=20)
        before = market.total_stock(0, "food")
        shipments = transport.rebalance(1)
        self.assertTrue(shipments)
        self.assertGreater(market.total_stock(0, "food"), before)


if __name__ == "__main__":
    unittest.main()

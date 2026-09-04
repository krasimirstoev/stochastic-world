import unittest
from types import SimpleNamespace

from stochastic_world.cli import resolve_engine
from stochastic_world.market import GoodsMarket


class DummyLocation:
    def __init__(self, location_id=0):
        self.id = location_id
        self.capacity_hint = 100


class HybridModeTests(unittest.TestCase):
    def test_auto_engine_switches_at_large_population(self):
        self.assertEqual(resolve_engine(SimpleNamespace(engine="auto", population=99_999)), "agent")
        self.assertEqual(resolve_engine(SimpleNamespace(engine="auto", population=100_000)), "hybrid")
        self.assertEqual(resolve_engine(SimpleNamespace(engine="agent", population=2_000_000)), "agent")

    def test_bulk_demand_reduces_stock(self):
        market = GoodsMarket([DummyLocation()], {0: 100})
        before = market.total_stock(0, "food")
        result = market.buy_bulk(0, "food", 25)
        self.assertEqual(result["quantity"], 25)
        self.assertAlmostEqual(market.total_stock(0, "food"), before - 25)

    def test_bulk_shortage_is_recorded_as_demand(self):
        market = GoodsMarket([DummyLocation()], {0: 1})
        market.state(0).supplier_stock["food"].clear()
        market.add_supply(0, -1, "food", 2)
        result = market.buy_bulk(0, "food", 10)
        self.assertTrue(result["shortage"])
        self.assertEqual(result["quantity"], 2)
        self.assertEqual(market.state(0).demand["food"], 10)


if __name__ == "__main__":
    unittest.main()

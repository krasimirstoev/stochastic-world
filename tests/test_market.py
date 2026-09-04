import random
import unittest

from stochastic_world.geography import build_locations
from stochastic_world.labor_market import Employer, LaborMarket
from stochastic_world.market import GoodsMarket
from stochastic_world.person import Person


class MarketTests(unittest.TestCase):
    def setUp(self):
        self.locations = build_locations(5)
        self.market = GoodsMarket(self.locations)

    def test_sale_reduces_stock_and_attributes_revenue(self):
        self.market.state(0).supplier_stock["food"].clear()
        self.market.add_supply(0, 7, "food", 10)
        before = self.market.total_stock(0, "food")
        result = self.market.buy(0, "food", 3, 100)
        self.assertEqual(result["quantity"], 3)
        self.assertEqual(self.market.total_stock(0, "food"), before - 3)
        self.assertGreater(result["seller_revenue"][7], 0)

    def test_shortage_pushes_price_up(self):
        state = self.market.state(0)
        state.supplier_stock["food"].clear()
        old = self.market.quote(0, "food")
        self.market.buy(0, "food", 8, 100)
        self.market.reprice()
        self.assertGreater(self.market.quote(0, "food"), old)

    def test_production_does_not_credit_cash_before_sale(self):
        employer = Employer(
            id=1, name="Food Works", location_id=2, kind="industrial",
            capacity=2, base_wage=5, cash=100, productivity=1.0,
            preferred_professions=("laborer",), output_good="food", output_per_shift=3.0,
        )
        market = LaborMarket([employer], random.Random(1))
        person = Person(1, "Test Worker", profession="laborer", location_id=2, employer_id=1)
        employer.employee_ids.add(person.id)
        before = employer.cash
        result = market.work_shift(person, self.locations[2])
        self.assertGreater(result["produced"], 0)
        self.assertLess(employer.cash, before)
        self.assertEqual(employer.revenue_since_review, 0)

    def test_sale_credit_reaches_employer(self):
        employer = Employer(
            id=3, name="Producer", location_id=0, kind="market",
            capacity=1, base_wage=4, cash=10, productivity=1.0,
            preferred_professions=("trader",), output_good="food", output_per_shift=2.0,
        )
        labor = LaborMarket([employer], random.Random(2))
        labor.credit_sale(3, 12.5)
        self.assertEqual(employer.cash, 22.5)
        self.assertEqual(employer.revenue_since_review, 12.5)


if __name__ == "__main__":
    unittest.main()

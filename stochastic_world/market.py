from collections import defaultdict
from dataclasses import dataclass, field


GOODS = ("food", "medicine")
BASE_PRICES = {"food": 2.0, "medicine": 5.0}


@dataclass
class MarketState:
    location_id: int
    prices: dict
    population: int = 1
    supplier_stock: dict = field(default_factory=lambda: {"food": defaultdict(float), "medicine": defaultdict(float)})
    demand: dict = field(default_factory=lambda: defaultdict(float))
    sold: dict = field(default_factory=lambda: defaultdict(float))

    def stock(self, good):
        return sum(self.supplier_stock[good].values())


class GoodsMarket:
    """Local essential-goods markets with supplier ownership of inventory."""

    def __init__(self, locations, population_by_location=None):
        population_by_location = population_by_location or {}
        self.states = {}
        for loc in locations:
            population = max(1, population_by_location.get(loc.id, loc.capacity_hint))
            state = MarketState(loc.id, dict(BASE_PRICES), population=population)
            state.supplier_stock["food"][-1] = population * 2.2
            state.supplier_stock["medicine"][-1] = max(8.0, population * 0.12)
            self.states[loc.id] = state

    def state(self, location_id):
        return self.states[location_id]

    def set_population(self, location_id, population):
        self.state(location_id).population = max(1, population)

    def add_supply(self, location_id, employer_id, good, quantity):
        if good in GOODS and quantity > 0:
            self.state(location_id).supplier_stock[good][employer_id] += float(quantity)

    def quote(self, location_id, good):
        return self.state(location_id).prices[good]

    def total_stock(self, location_id, good):
        return self.state(location_id).stock(good)

    def _consume(self, state, good, quantity):
        remaining = float(quantity); seller_revenue = defaultdict(float)
        price = max(0.25, state.prices[good])
        for supplier_id in sorted(state.supplier_stock[good]):
            if remaining <= 0: break
            stock = state.supplier_stock[good][supplier_id]; take = min(stock, remaining)
            if take <= 0: continue
            state.supplier_stock[good][supplier_id] -= take; remaining -= take
            if supplier_id >= 0: seller_revenue[supplier_id] += take * price
        sold = float(quantity) - remaining; state.sold[good] += sold
        return sold, dict(seller_revenue)

    def buy(self, location_id, good, requested, budget):
        state = self.state(location_id); price = max(0.25, state.prices[good])
        requested = max(0, int(requested)); affordable = int(max(0.0, budget) // price)
        quantity = min(requested, affordable, int(state.stock(good))); state.demand[good] += requested
        shortage = quantity < requested
        if quantity <= 0:
            return {"quantity": 0, "cost": 0.0, "unit_price": price, "shortage": shortage, "seller_revenue": {}}
        sold, seller_revenue = self._consume(state, good, quantity)
        return {"quantity": sold, "cost": round(sold * price, 2), "unit_price": price,
                "shortage": sold < requested, "seller_revenue": seller_revenue}

    def buy_bulk(self, location_id, good, requested):
        """Aggregate routine household demand without per-person bookkeeping."""
        state = self.state(location_id); requested = max(0.0, float(requested))
        quantity = min(requested, max(0.0, state.stock(good))); state.demand[good] += requested
        if quantity <= 0:
            return {"quantity": 0.0, "shortage": requested > 0, "seller_revenue": {}}
        sold, seller_revenue = self._consume(state, good, quantity)
        return {"quantity": sold, "shortage": sold + 1e-9 < requested, "seller_revenue": seller_revenue}

    def transfer(self, source_location, target_location, good, requested):
        if requested <= 0 or good not in GOODS: return 0.0
        source = self.state(source_location); target = self.state(target_location)
        remaining = float(requested); moved = 0.0
        for supplier_id in sorted(source.supplier_stock[good]):
            if remaining <= 0: break
            stock = source.supplier_stock[good][supplier_id]; take = min(stock, remaining)
            if take <= 0: continue
            source.supplier_stock[good][supplier_id] -= take; target.supplier_stock[good][supplier_id] += take
            moved += take; remaining -= take
        return moved

    def reprice(self):
        for state in self.states.values():
            for good in GOODS:
                stock = max(0.0, state.stock(good)); demand = state.demand[good]; sold = state.sold[good]
                target = state.population * (1.8 if good == "food" else 0.10)
                scarcity = max(-0.30, min(1.50, (target - stock) / max(1.0, target)))
                unmet = max(0.0, demand - sold)
                pressure = max(0.0, min(0.65, unmet / max(4.0, demand + 2.0)))
                desired = BASE_PRICES[good] * (1.0 + scarcity * 0.75 + pressure * 0.55)
                old = state.prices[good]
                state.prices[good] = round(max(BASE_PRICES[good] * 0.55,
                    min(BASE_PRICES[good] * 4.0, old * 0.65 + desired * 0.35)), 2)
                state.demand[good] = 0.0; state.sold[good] = 0.0

    def inflation_index(self):
        base = sum(BASE_PRICES.values())
        current = sum(sum(s.prices[g] for g in GOODS) for s in self.states.values()) / max(1, len(self.states))
        return current / base if base else 1.0

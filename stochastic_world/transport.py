from dataclasses import dataclass


TRANSPORT_GOODS = ("food", "medicine")


@dataclass
class Shipment:
    day: int
    source_location: int
    target_location: int
    good: str
    quantity: float
    transport_cost: float


class TransportSystem:
    """Moves surplus inventory one graph edge toward local shortages."""

    def __init__(self, locations, market, labor_market, rng, capacity_per_1000: float = 4.0):
        self.locations = locations
        self.market = market
        self.labor_market = labor_market
        self.rng = rng
        self.capacity_per_1000 = capacity_per_1000
        self.shipments = 0
        self.volume = 0.0
        self.cost = 0.0

    def _edge_capacity(self, source_id):
        firms = [
            e for e in self.labor_market.local_employers(source_id)
            if e.alive and (e.kind in ("market", "industrial", "logistics") or e.output_good is None)
        ]
        worker_capacity = sum(max(1, len(e.employee_ids)) for e in firms)
        return max(2.0, worker_capacity * self.capacity_per_1000)

    def rebalance(self, day):
        shipments = []
        for source in self.locations:
            capacity_left = self._edge_capacity(source.id)
            if capacity_left <= 0:
                continue
            for good in TRANSPORT_GOODS:
                source_state = self.market.state(source.id)
                target_stock = source_state.population * (1.8 if good == "food" else 0.10)
                surplus = max(0.0, source_state.stock(good) - target_stock * 1.35)
                if surplus <= 0:
                    continue

                neighbors = []
                for neighbor_id in source.neighbors:
                    ns = self.market.state(neighbor_id)
                    target = ns.population * (1.8 if good == "food" else 0.10)
                    shortage = max(0.0, target * 0.75 - ns.stock(good))
                    if shortage > 0:
                        neighbors.append((neighbor_id, shortage))
                neighbors.sort(key=lambda item: item[1], reverse=True)

                for target_id, shortage in neighbors:
                    if capacity_left <= 0 or surplus <= 0:
                        break
                    quantity = min(surplus, shortage, capacity_left)
                    moved = self.market.transfer(source.id, target_id, good, quantity)
                    if moved <= 0:
                        continue
                    transport_cost = round(moved * (0.08 if good == "food" else 0.18), 2)
                    shipments.append(Shipment(day, source.id, target_id, good, moved, transport_cost))
                    self.shipments += 1
                    self.volume += moved
                    self.cost += transport_cost
                    surplus -= moved
                    capacity_left -= moved
        return shipments

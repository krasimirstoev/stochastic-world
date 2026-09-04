from collections import deque
from dataclasses import dataclass
from math import ceil


@dataclass
class PoliceDistrict:
    location_id: int
    officers: int
    incidents_today: int = 0
    responses_today: int = 0
    arrests_today: int = 0


class PoliceSystem:
    """Aggregate local police capacity; no per-officer agents are instantiated."""

    def __init__(self, locations, population_index, rng, officers_per_1000: float = 2.2):
        self.rng = rng
        self.index = population_index
        self.officers_per_1000 = max(0.0, officers_per_1000)
        self.districts = {
            loc.id: PoliceDistrict(loc.id, self._desired_officers(loc.id))
            for loc in locations
        }
        self.history = {loc.id: deque(maxlen=30) for loc in locations}

    def _desired_officers(self, location_id):
        population = self.index.population(location_id)
        return max(1, ceil(population * self.officers_per_1000 / 1000.0)) if population else 0

    def rebalance(self):
        for location_id, district in self.districts.items():
            district.officers = self._desired_officers(location_id)

    def coverage(self, location_id):
        district = self.districts[location_id]
        population = max(1, self.index.population(location_id))
        officers_per_1000 = district.officers * 1000.0 / population
        load = district.incidents_today / max(1, district.officers)
        return max(0.02, min(0.92, 0.22 + officers_per_1000 * 0.12 - load * 0.035))

    def respond(self, day, location_id, crime_type, offender, victim, magnitude=1.0):
        district = self.districts[location_id]
        district.incidents_today += 1
        severity = 1.15 if crime_type == "attack" else 0.90
        probability = min(0.97, self.coverage(location_id) * severity)
        responded = self.rng.random() < probability
        arrested = False
        detention_days = 0
        fine = 0

        if responded:
            district.responses_today += 1
            arrest_probability = min(0.90, 0.42 + 0.12 * magnitude + (0.12 if crime_type == "attack" else 0.0))
            arrested = self.rng.random() < arrest_probability
            if arrested:
                district.arrests_today += 1
                detention_days = self.rng.randint(2, 8 if crime_type == "attack" else 5)
                offender.detained_until_day = max(offender.detained_until_day, day + detention_days)
                fine = min(offender.money, self.rng.randint(1, 5) + (2 if crime_type == "attack" else 0))
                offender.money -= fine
                offender.arrests += 1

        return {
            "responded": responded,
            "arrested": arrested,
            "detention_days": detention_days,
            "fine": fine,
            "coverage": round(self.coverage(location_id), 4),
        }

    def end_day(self):
        snapshots = {}
        for location_id, district in self.districts.items():
            snapshots[location_id] = {
                "officers": district.officers,
                "incidents": district.incidents_today,
                "responses": district.responses_today,
                "arrests": district.arrests_today,
                "coverage": self.coverage(location_id),
            }
            self.history[location_id].append(district.incidents_today)
            district.incidents_today = 0
            district.responses_today = 0
            district.arrests_today = 0
        return snapshots

from math import ceil

from .hybrid import HybridEngine


class LifeHybridEngine(HybridEngine):
    """Hybrid engine extensions for dynamically growing, age-structured populations."""

    def ensure_person_capacity(self, size):
        missing = size - len(self.last_touched_day)
        if missing > 0:
            self.last_touched_day.extend([0] * missing)

    def _sample_to_budget(self, selected, budget):
        if budget <= len(selected):
            return set()
        remaining = budget - len(selected)
        alive = max(1, self.world.alive_count)
        sampled = set()
        locations = list(self.world.locations)
        self.world.rng.shuffle(locations)
        for location in locations:
            if remaining <= 0:
                break
            pop = self.world.population_index.population(location.id)
            if pop <= 0:
                continue
            quota = min(self.sample_per_district, max(1, ceil(budget * pop / alive)), remaining)
            # Oversample locally and keep adults; this stays bounded and avoids an O(population) age scan.
            candidate_limit = min(max(quota * 3, quota + 8), pop)
            candidates = self.world.population_index.sample_people(
                location.id, self.world.rng, candidate_limit, exclude=tuple(selected | sampled))
            for person in candidates:
                if person.alive and person.is_adult:
                    sampled.add(person.id)
                    if len(sampled) >= budget - len(selected):
                        break
            remaining = budget - len(selected) - len(sampled)
        return sampled

    def catch_up(self, person, day):
        self.ensure_person_capacity(len(self.world.people))
        if person.is_dependent:
            last = self.last_touched_day[person.id]
            skipped = max(0, day - int(last) - 1) if last else max(0, day - max(1, person.birth_day))
            if skipped > 0:
                self.world.demographics.support_dependent(person)
                person.food = max(0, person.food - skipped // 10)
                person.energy = max(45, person.energy - skipped // 12)
                location = self.world.location_of(person)
                person.shelter = max(0, person.shelter - int(skipped * (0.03 + 0.02 * location.shelter_decay_bonus)))
                if person.food == 0 and skipped >= 21:
                    person.health -= min(20, skipped // 10)
            person.unemployment_days = 0
            self.last_touched_day[person.id] = day
            return
        if person.retired:
            last = self.last_touched_day[person.id]
            skipped = max(0, day - int(last) - 1) if last else max(0, day - 1)
            if skipped > 0:
                food_cost = skipped * 0.75 * self.world.goods_market.quote(person.location_id, "food")
                spent = min(max(0.0, person.money), food_cost)
                person.money -= spent; person.market_spending += spent
                person.food = max(0, person.food - skipped // 12)
                person.energy = max(35, person.energy - skipped // 10)
            person.unemployment_days = 0
            self.last_touched_day[person.id] = day
            return
        return super().catch_up(person, day)

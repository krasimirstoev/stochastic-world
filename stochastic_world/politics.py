from dataclasses import dataclass

from faker import Faker


ELECTION_INTERVAL_DAYS = 1460


@dataclass(frozen=True)
class Party:
    id: str
    name: str
    ideology: float
    tax_rate: float
    welfare_cash: int
    welfare_food: int
    welfare_medicine_chance: float
    welfare_money_threshold: int


LEFT = Party("left", "Civic Left", -0.65, 0.18, 4, 2, 0.18, 14)
RIGHT = Party("right", "Civic Right", 0.65, 0.10, 1, 0, 0.04, 7)
PARTIES = (LEFT, RIGHT)


class PoliticalSystem:
    """Two-party left/right baseline. Coefficients are experiment assumptions, not voter facts."""

    def __init__(self, seed: int, locale: str = "en_US"):
        fake = Faker(locale); fake.seed_instance(seed ^ 0xA11CE)
        self.representatives = {party.id: fake.name() for party in PARTIES}
        self.government = LEFT if seed % 2 == 0 else RIGHT
        self.election_number = 0
        self.treasury = 0.0
        self.last_election_day = 0

    def party_by_id(self, party_id):
        return LEFT if party_id == "left" else RIGHT

    def vote(self, person, local_crime_rate: float, rng):
        left_score = -abs(person.ideology - LEFT.ideology)
        right_score = -abs(person.ideology - RIGHT.ideology)
        if person.money < 8 or person.food <= 3: left_score += 0.22
        if person.welfare_received > person.taxes_paid: left_score += 0.06
        if person.taxes_paid > person.welfare_received + 15: right_score += 0.06
        right_score += min(0.28, local_crime_rate * 0.9)
        right_score += min(0.18, person.crime_suffered * 0.025)
        left_score += rng.uniform(-0.08, 0.08); right_score += rng.uniform(-0.08, 0.08)
        return LEFT if left_score >= right_score else RIGHT

    def update_attitudes(self, person, local_crime_rate: float):
        shift = 0.0
        if person.money < 6 or person.food <= 2: shift -= 0.0015
        if local_crime_rate > 0.10: shift += min(0.0030, local_crime_rate * 0.012)
        person.shift_ideology(shift)

    def collect_tax(self, person, gross_income: int):
        if gross_income <= 0: return 0
        rate = self.government.tax_rate
        if self.government.id == "left" and person.social_class in ("upper_middle", "affluent"): rate += 0.05
        tax = min(person.money, max(0, int(round(gross_income * rate))))
        person.money -= tax; person.taxes_paid += tax; self.treasury += tax
        person.shift_ideology(min(0.0025, tax * 0.00035))
        return tax

    def distribute_welfare(self, people, rng):
        party = self.government; transfers = []
        for person in people:
            if not person.alive or person.money > party.welfare_money_threshold: continue
            cost = party.welfare_cash + party.welfare_food
            if self.treasury < cost: continue
            person.money += party.welfare_cash; person.food += party.welfare_food
            medicine = 0
            if rng.random() < party.welfare_medicine_chance and self.treasury >= cost + 2:
                person.medicine += 1; medicine = 1; cost += 2
            self.treasury -= cost; person.welfare_received += cost
            person.shift_ideology(-min(0.0035, cost * 0.00045))
            transfers.append((person, party.welfare_cash, party.welfare_food, medicine, cost))
        return transfers

    def hold_election(self, day, people, crime_rate_by_location, rng):
        votes = {party.id: 0 for party in PARTIES}; ballots = []
        for person in people:
            if not person.alive or not getattr(person, "is_adult", True): continue
            party = self.vote(person, crime_rate_by_location.get(person.location_id, 0.0), rng)
            votes[party.id] += 1; ballots.append((person, party))
        winner_id = "left" if votes["left"] >= votes["right"] else "right"
        self.government = self.party_by_id(winner_id)
        self.election_number += 1; self.last_election_day = day
        return votes, ballots, self.government

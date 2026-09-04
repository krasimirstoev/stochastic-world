from dataclasses import dataclass, field
from math import ceil

from .professions import profession_for, workplace_fit


BUSINESS_INTERVAL_DAYS = 30
TARGET_EMPLOYEES_PER_FIRM = 180


@dataclass
class Employer:
    id: int
    name: str
    location_id: int
    kind: str
    capacity: int
    base_wage: float
    cash: float
    productivity: float
    preferred_professions: tuple[str, ...]
    output_good: str | None = None
    output_per_shift: float = 0.0
    employee_ids: set[int] = field(default_factory=set)
    revenue_since_review: float = 0.0
    payroll_since_review: float = 0.0
    units_produced_since_review: float = 0.0
    alive: bool = True

    @property
    def vacancies(self):
        return max(0, self.capacity - len(self.employee_ids)) if self.alive else 0


FIRM_TEMPLATES = {
    "residential": ("Community Services", 5.0, 1.00, ("service_worker", "teacher", "clerk"), None, 0.0),
    "market": ("Food Cooperative", 5.4, 1.08, ("trader", "clerk", "service_worker", "manager"), "food", 2.1),
    "industrial": ("Food Works", 6.0, 1.18, ("laborer", "technician", "engineer", "manager"), "food", 2.8),
    "clinic": ("Medical Cooperative", 5.8, 1.10, ("nurse", "technician", "clerk"), "medicine", 0.75),
    "outskirts": ("Resource Farms", 4.6, 0.92, ("laborer", "service_worker"), "food", 3.2),
}


def build_employers(locations, population_size: int, rng):
    employers, eid = [], 0
    people_per_location = max(1, ceil(population_size / len(locations)))
    firms_per_location = max(2, ceil(people_per_location * 0.72 / TARGET_EMPLOYEES_PER_FIRM))
    base_capacity = max(12, ceil(people_per_location * 0.72 / firms_per_location))

    for location in locations:
        label, wage, productivity, preferred, output_good, output_per_shift = FIRM_TEMPLATES.get(
            location.kind, FIRM_TEMPLATES["residential"]
        )
        for number in range(firms_per_location):
            spread = max(2, base_capacity // 8)
            capacity = max(8, base_capacity + rng.randint(-spread, spread))
            employers.append(Employer(
                id=eid,
                name=f"{label} {location.id + 1}-{number + 1}",
                location_id=location.id,
                kind=location.kind,
                capacity=capacity,
                base_wage=max(2.5, wage * rng.uniform(0.90, 1.10)),
                cash=capacity * wage * rng.uniform(12.0, 22.0),
                productivity=productivity * rng.uniform(0.92, 1.08),
                preferred_professions=preferred,
                output_good=output_good,
                output_per_shift=output_per_shift * rng.uniform(0.90, 1.10),
            ))
            eid += 1

        if location.kind in ("market", "industrial"):
            cap = max(8, ceil(people_per_location / 900))
            employers.append(Employer(
                id=eid,
                name=f"Logistics Hub {location.id + 1}",
                location_id=location.id,
                kind="logistics",
                capacity=cap,
                base_wage=5.6 * rng.uniform(0.92, 1.08),
                cash=cap * 100.0,
                productivity=1.0,
                preferred_professions=("laborer", "technician", "trader", "manager"),
            ))
            eid += 1
    return employers


class LaborMarket:
    def __init__(self, employers, rng):
        self.employers = employers
        self.rng = rng
        self.by_id = {e.id: e for e in employers}
        self.by_location = {}
        for employer in employers:
            self.by_location.setdefault(employer.location_id, []).append(employer)
        self.next_employer_id = max(self.by_id, default=-1) + 1

    def _add_employer(self, employer):
        self.employers.append(employer)
        self.by_id[employer.id] = employer
        self.by_location.setdefault(employer.location_id, []).append(employer)

    def employer(self, employer_id):
        employer = self.by_id.get(employer_id)
        return employer if employer and employer.alive else None

    def employer_any(self, employer_id):
        return self.by_id.get(employer_id)

    def local_employers(self, location_id):
        return [e for e in self.by_location.get(location_id, ()) if e.alive]

    def vacancies(self, location_id=None):
        firms = self.employers if location_id is None else self.by_location.get(location_id, ())
        return sum(e.vacancies for e in firms if e.alive)

    def unemployment_rate(self, people):
        living = 0
        unemployed = 0
        for person in people:
            if not person.alive:
                continue
            living += 1
            unemployed += person.employer_id is None
        return unemployed / living if living else 0.0

    def suitable_employers(self, person, sample_limit=24):
        profession = profession_for(person)
        candidates = [e for e in self.by_location.get(person.location_id, ()) if e.alive and e.vacancies > 0]
        if len(candidates) > sample_limit:
            candidates = self.rng.sample(candidates, sample_limit)
        ranked = []
        for employer in candidates:
            preference = 1.35 if person.profession in employer.preferred_professions else 0.85
            wage = employer.base_wage * profession.income_multiplier
            ranked.append((employer, max(0.1, preference * wage)))
        return ranked

    def hire(self, person):
        candidates = self.suitable_employers(person)
        if not candidates:
            return None
        employer = self.rng.choices([x[0] for x in candidates], weights=[x[1] for x in candidates], k=1)[0]
        employer.employee_ids.add(person.id)
        person.employer_id = employer.id
        person.unemployment_days = 0
        person.jobs_held += 1
        return employer

    def fast_hire(self, person, attempts=12):
        firms = self.by_location.get(person.location_id, ())
        if not firms:
            return None
        for _ in range(min(attempts, max(1, len(firms)))):
            employer = firms[self.rng.randrange(len(firms))]
            if employer.alive and employer.vacancies > 0:
                employer.employee_ids.add(person.id)
                person.employer_id = employer.id
                person.unemployment_days = 0
                person.jobs_held += 1
                return employer
        return self.hire(person)

    def terminate(self, person, reason="separation"):
        employer = self.employer_any(person.employer_id)
        if employer:
            employer.employee_ids.discard(person.id)
        old_id = person.employer_id
        person.employer_id = None
        return old_id, reason

    def wage_for(self, person, employer, location):
        profession = profession_for(person)
        fit = workplace_fit(person, location)
        scarcity = 1.10 if employer.vacancies > max(1, employer.capacity // 3) else 1.0
        preferred = 1.08 if person.profession in employer.preferred_professions else 0.92
        return max(1, round(employer.base_wage * profession.income_multiplier * fit * scarcity * preferred))

    def work_shift(self, person, location):
        employer = self.employer(person.employer_id)
        if employer is None or employer.location_id != person.location_id:
            return None
        gross = self.wage_for(person, employer, location)
        if employer.cash < gross:
            return {"employer": employer, "gross": 0, "produced_good": None, "produced": 0.0, "insolvent": True}
        employer.cash -= gross
        employer.payroll_since_review += gross

        produced = 0.0
        if employer.output_good:
            produced = max(
                0.0,
                employer.output_per_shift
                * employer.productivity
                * workplace_fit(person, location)
                * self.rng.uniform(0.85, 1.15),
            )
            employer.units_produced_since_review += produced
        elif employer.kind != "logistics":
            service_revenue = gross * employer.productivity * self.rng.uniform(1.12, 1.45)
            employer.cash += service_revenue
            employer.revenue_since_review += service_revenue

        return {
            "employer": employer,
            "gross": gross,
            "produced_good": employer.output_good,
            "produced": produced,
            "insolvent": False,
        }

    def credit_sale(self, employer_id, revenue):
        employer = self.employer(employer_id)
        if employer and revenue > 0:
            employer.cash += revenue
            employer.revenue_since_review += revenue

    def business_review(self, people):
        changes = []
        for employer in self.employers:
            if not employer.alive:
                continue
            margin = employer.revenue_since_review - employer.payroll_since_review
            if employer.cash < 4 or (margin < -22 and employer.cash < max(35, employer.capacity * 2)):
                employer.alive = False
                laid_off = list(employer.employee_ids)
                for pid in laid_off:
                    person = people[pid]
                    if person.alive:
                        person.employer_id = None
                employer.employee_ids.clear()
                changes.append(("closed", employer, laid_off))
            elif margin > employer.capacity * 2.5 and employer.cash > employer.capacity * 15 and employer.capacity < 1000:
                old = employer.capacity
                employer.capacity += max(1, min(20, employer.capacity // 20))
                changes.append(("expanded", employer, (old, employer.capacity)))
            elif margin < -8 and employer.capacity > 8 and len(employer.employee_ids) < employer.capacity * 0.7:
                old = employer.capacity
                employer.capacity = max(8, int(employer.capacity * 0.95))
                changes.append(("contracted", employer, (old, employer.capacity)))
            employer.revenue_since_review = 0.0
            employer.payroll_since_review = 0.0
            employer.units_produced_since_review = 0.0

        unemployment = self.unemployment_rate(people)
        if unemployment > 0.18 and self.rng.random() < min(0.65, unemployment):
            living_locations = list(self.by_location)
            if living_locations:
                location_id = self.rng.choice(living_locations)
                good = "food" if self.rng.random() < 0.78 else "medicine"
                capacity = self.rng.randint(20, 80)
                employer = Employer(
                    id=self.next_employer_id,
                    name=f"New Venture {self.next_employer_id}",
                    location_id=location_id,
                    kind="new_venture",
                    capacity=capacity,
                    base_wage=self.rng.uniform(4.5, 6.5),
                    cash=capacity * self.rng.uniform(50, 90),
                    productivity=self.rng.uniform(0.95, 1.15),
                    preferred_professions=("laborer", "service_worker", "technician", "trader"),
                    output_good=good,
                    output_per_shift=self.rng.uniform(1.5, 3.0) if good == "food" else self.rng.uniform(0.45, 0.9),
                )
                self.next_employer_id += 1
                self._add_employer(employer)
                changes.append(("created", employer, None))
        return changes

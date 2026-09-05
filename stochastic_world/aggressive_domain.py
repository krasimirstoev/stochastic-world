"""Location-owned BSP action engine for very large aggressive simulations.

The design follows domain decomposition / owner-computes: each worker owns one or
more locations, applies all local agent actions directly to shared hot state, and
returns only compact economy / police aggregates at the barrier. Main no longer
replays one intent per action.
"""

import math
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from multiprocessing import shared_memory
import struct
from time import perf_counter

from . import agent_shards as planner
from .aggressive_economy import (
    LOCATION_KINDS,
    PROFESSION_NAMES,
    PROFESSION_TO_CODE,
    SOCIAL_CLASSES,
    SOCIAL_CLASS_TO_CODE,
)
from .agent_shards_shared_ram import (
    _refresh_worker_social_aggregate,
    _remember_worker_social,
)
from .multiprocessing_engine import _DeterministicStream, _seed_for, _weighted_action
from .professions import PROFESSIONS


_PHASE_SAFE = 0x5AFE
_PHASE_SOCIAL = 0x50C1A1
_PHASE_MOVE = 0x4D4F5645
_PHASE_WORK = 0x574F524B
_PHASE_BUY = 0x425559
_PHASE_POLICE = 0x504F4C49

# location, food, medicine, energy, health, shelter, money, employer_id,
# profession_code, social_class_code, ideology, detained_until, alive,
# working_age, dependent, adult, taxes_paid, welfare_received, work_experience,
# career_progress, lifetime_gross_income, market_spending, shortage_experiences,
# crime_suffered, arrests, unemployment_days, lifetime_unemployment_days,
# days_in_class, jobs_held
_AGENT = struct.Struct("<iiiiiidiiidiBBBBddidddiiiiiii")


class SharedDomainAgentState:
    """Authoritative hot / medium agent state for the domain-owned action phase."""

    def __init__(self, capacity, *, descriptor=None):
        if descriptor is None:
            self.capacity = max(1, int(capacity))
            self._state = shared_memory.SharedMemory(
                create=True, size=self.capacity * _AGENT.size
            )
            self._owner = True
        else:
            self.capacity = int(descriptor["capacity"])
            self._state = shared_memory.SharedMemory(name=descriptor["name"])
            self._owner = False

    @classmethod
    def attach(cls, descriptor):
        return cls(1, descriptor=descriptor)

    @property
    def descriptor(self):
        return {"capacity": self.capacity, "name": self._state.name}

    @property
    def allocated_bytes(self):
        return self._state.size

    def write_person(self, person):
        pid = int(person.id)
        if pid < 0 or pid >= self.capacity:
            return
        _AGENT.pack_into(
            self._state.buf,
            pid * _AGENT.size,
            int(person.location_id),
            int(person.food),
            int(person.medicine),
            int(person.energy),
            int(person.health),
            int(person.shelter),
            float(person.money),
            -1 if person.employer_id is None else int(person.employer_id),
            int(PROFESSION_TO_CODE.get(person.profession, 0)),
            int(SOCIAL_CLASS_TO_CODE.get(person.social_class, 0)),
            float(person.ideology),
            int(person.detained_until_day),
            int(bool(person.alive)),
            int(bool(person.is_working_age)),
            int(bool(person.is_dependent)),
            int(bool(person.is_adult)),
            float(person.taxes_paid),
            float(person.welfare_received),
            int(person.work_experience),
            float(person.career_progress),
            float(person.lifetime_gross_income),
            float(person.market_spending),
            int(person.shortage_experiences),
            int(person.crime_suffered),
            int(person.arrests),
            int(person.unemployment_days),
            int(person.lifetime_unemployment_days),
            int(person.days_in_class),
            int(person.jobs_held),
        )

    def read(self, pid):
        return list(_AGENT.unpack_from(self._state.buf, int(pid) * _AGENT.size))

    def write(self, pid, values):
        _AGENT.pack_into(self._state.buf, int(pid) * _AGENT.size, *values)

    def sync_world(self, world):
        for person in world.people:
            self.write_person(person)

    def close(self, *, unlink=False):
        self._state.close()
        if unlink and self._owner:
            try:
                self._state.unlink()
            except FileNotFoundError:
                pass


LID = 0
FOOD = 1
MED = 2
ENERGY = 3
HEALTH = 4
SHELTER = 5
MONEY = 6
EMPLOYER = 7
PROFESSION = 8
SOCIAL_CLASS = 9
IDEOLOGY = 10
DETAINED = 11
ALIVE = 12
WORKING_AGE = 13
DEPENDENT = 14
ADULT = 15
TAXES = 16
WELFARE = 17
WORK_EXP = 18
CAREER = 19
LIFETIME_GROSS = 20
MARKET_SPENDING = 21
SHORTAGES = 22
CRIME_SUFFERED = 23
ARRESTS = 24
UNEMPLOYMENT = 25
LIFETIME_UNEMPLOYMENT = 26
DAYS_IN_CLASS = 27
JOBS_HELD = 28


def _coprime_step(n, seed):
    if n <= 1:
        return 1
    step = 1 + (int(seed) % (n - 1))
    while math.gcd(step, n) != 1:
        step += 1
        if step >= n:
            step = 1
    return step


def _iter_permuted(size, seed):
    """O(1)-memory deterministic affine permutation of range(size)."""
    size = int(size)
    if size <= 0:
        return
    start = int(seed) % size
    step = _coprime_step(size, int(seed) >> 17)
    for k in range(size):
        yield (start + k * step) % size


def _weighted_choice(rows, weights, stream):
    if not rows:
        return None
    total = sum(weights)
    if total <= 0:
        return rows[stream.randint(0, len(rows) - 1)]
    needle = stream.random() * total
    upto = 0.0
    for row, weight in zip(rows, weights):
        upto += weight
        if needle <= upto:
            return row
    return rows[-1]


def _location_packet(world):
    packet = []
    for location in world.locations:
        market = world.goods_market.state(location.id)
        packet.append(
            (
                int(location.id),
                location.kind,
                tuple(int(x) for x in location.neighbors),
                int(location.scavenge_food_max),
                float(location.medicine_chance),
                float(market.prices["food"]),
                float(market.prices["medicine"]),
                dict(market.supplier_stock["food"]),
                dict(market.supplier_stock["medicine"]),
                int(world.population_index.population(location.id)),
            )
        )
    return tuple(packet)


def _employer_packet(world):
    rows = []
    for e in world.labor_market.employers:
        rows.append(
            (
                int(e.id),
                int(e.location_id),
                e.kind,
                int(e.capacity),
                int(len(e.employee_ids)),
                float(e.base_wage),
                float(e.cash),
                float(e.productivity),
                e.output_good,
                float(e.output_per_shift),
                tuple(e.preferred_professions),
                float(e.payroll_since_review),
                float(e.revenue_since_review),
                float(e.units_produced_since_review),
                int(bool(e.alive)),
            )
        )
    return tuple(rows)


def build_domain_packet(world):
    government = world.politics.government
    police = {
        int(lid): int(district.officers)
        for lid, district in world.police.districts.items()
    }
    return {
        "locations": _location_packet(world),
        "employers": _employer_packet(world),
        "government": (government.id, float(government.tax_rate)),
        "police": police,
    }


def _market_from_location_row(row):
    (
        lid, kind, neighbors, scavenge_food_max, medicine_chance,
        food_price, medicine_price, food_suppliers, medicine_suppliers, population,
    ) = row
    return {
        "id": int(lid),
        "kind": kind,
        "neighbors": neighbors,
        "scavenge_food_max": int(scavenge_food_max),
        "medicine_chance": float(medicine_chance),
        "prices": {"food": float(food_price), "medicine": float(medicine_price)},
        "suppliers": {
            "food": {int(k): float(v) for k, v in food_suppliers.items()},
            "medicine": {int(k): float(v) for k, v in medicine_suppliers.items()},
        },
        "demand": {"food": 0.0, "medicine": 0.0},
        "sold": {"food": 0.0, "medicine": 0.0},
        "population": int(population),
    }


def _stock(market, good):
    return sum(market["suppliers"][good].values())


def _consume_market(market, good, quantity, employers):
    remaining = float(quantity)
    price = max(0.25, float(market["prices"][good]))
    sold = 0.0
    suppliers = market["suppliers"][good]
    for supplier_id in sorted(suppliers):
        if remaining <= 0:
            break
        available = max(0.0, float(suppliers[supplier_id]))
        take = min(available, remaining)
        if take <= 0:
            continue
        suppliers[supplier_id] = available - take
        remaining -= take
        sold += take
        if supplier_id >= 0:
            employer = employers.get(supplier_id)
            if employer is not None:
                revenue = take * price
                employer["cash"] += revenue
                employer["revenue"] += revenue
    market["sold"][good] += sold
    return sold


def _prepare_employers(rows):
    result = {}
    by_location = defaultdict(list)
    for row in rows:
        (
            eid, lid, kind, capacity, employees, base_wage, cash, productivity,
            output_good, output_per_shift, preferred, payroll, revenue, units, alive,
        ) = row
        item = {
            "id": int(eid),
            "location": int(lid),
            "kind": kind,
            "capacity": int(capacity),
            "employees": int(employees),
            "base_wage": float(base_wage),
            "cash": float(cash),
            "productivity": float(productivity),
            "output_good": output_good,
            "output_per_shift": float(output_per_shift),
            "preferred": set(preferred),
            "payroll": float(payroll),
            "revenue": float(revenue),
            "units": float(units),
            "alive": bool(alive),
        }
        result[item["id"]] = item
        by_location[item["location"]].append(item)
    return result, by_location


def _hire(state, local_employers, stream):
    candidates = [
        e for e in local_employers
        if e["alive"] and e["employees"] < e["capacity"]
    ]
    if not candidates:
        return None
    profession_name = PROFESSION_NAMES[int(state[PROFESSION])]
    profession = PROFESSIONS[profession_name]
    weights = []
    for e in candidates:
        preference = 1.35 if profession_name in e["preferred"] else 0.85
        weights.append(max(0.1, preference * e["base_wage"] * profession.income_multiplier))
    employer = _weighted_choice(candidates, weights, stream)
    if employer is None:
        return None
    employer["employees"] += 1
    state[EMPLOYER] = employer["id"]
    state[UNEMPLOYMENT] = 0
    state[JOBS_HELD] += 1
    return employer


def _work(state, location, employers, local_employers, market, master_seed, day, pid, round_index):
    stream = _DeterministicStream(_seed_for(master_seed, day, pid, _PHASE_WORK, round_index))
    if state[ENERGY] < 8:
        state[ENERGY] = min(100, state[ENERGY] + stream.randint(12, 24))
        state[HEALTH] = min(100, state[HEALTH] + stream.randint(0, 2))
        return

    employer = employers.get(int(state[EMPLOYER]))
    if employer is None or not employer["alive"]:
        state[EMPLOYER] = -1
        employer = _hire(state, local_employers, stream)
        if employer is None:
            return

    if employer["location"] != int(state[LID]):
        return

    profession_name = PROFESSION_NAMES[int(state[PROFESSION])]
    profession = PROFESSIONS[profession_name]
    fit = 1.15 if location["kind"] in profession.workplace_kinds else 0.82
    vacancies = max(0, employer["capacity"] - employer["employees"])
    scarcity = 1.10 if vacancies > max(1, employer["capacity"] // 3) else 1.0
    preferred = 1.08 if profession_name in employer["preferred"] else 0.92
    gross = max(
        1,
        round(
            employer["base_wage"]
            * profession.income_multiplier
            * fit
            * scarcity
            * preferred
        ),
    )
    if employer["cash"] < gross:
        employer["employees"] = max(0, employer["employees"] - 1)
        state[EMPLOYER] = -1
        return

    employer["cash"] -= gross
    employer["payroll"] += gross

    if employer["output_good"]:
        produced = max(
            0.0,
            employer["output_per_shift"]
            * employer["productivity"]
            * fit
            * (0.85 + stream.random() * 0.30),
        )
        employer["units"] += produced
        market["suppliers"][employer["output_good"]][employer["id"]] = (
            market["suppliers"][employer["output_good"]].get(employer["id"], 0.0)
            + produced
        )
    elif employer["kind"] != "logistics":
        service_revenue = gross * employer["productivity"] * (1.12 + stream.random() * 0.33)
        employer["cash"] += service_revenue
        employer["revenue"] += service_revenue

    state[MONEY] += gross
    state[LIFETIME_GROSS] += gross
    state[WORK_EXP] += 1
    state[CAREER] += profession.advancement_rate * fit

    energy_cost = max(3, round(stream.randint(6, 12) * profession.energy_multiplier))
    government_id, tax_rate = location["government"]
    rate = float(tax_rate)
    social_name = SOCIAL_CLASSES[int(state[SOCIAL_CLASS])]
    if government_id == "left" and social_name in ("upper_middle", "affluent"):
        rate += 0.05
    tax = min(state[MONEY], max(0, int(round(gross * rate))))
    state[MONEY] -= tax
    state[TAXES] += tax
    state[IDEOLOGY] = max(
        -1.0,
        min(1.0, state[IDEOLOGY] + min(0.0025, tax * 0.00035)),
    )
    location["treasury_delta"] += tax
    state[ENERGY] = max(0, state[ENERGY] - energy_cost)


def _buy(state, market, master_seed, day, pid, round_index):
    options = []
    if state[FOOD] <= 6:
        options.append(("food", 3))
    if state[MED] <= 1:
        options.append(("medicine", 1))
    if not options:
        return
    stream = _DeterministicStream(_seed_for(master_seed, day, pid, _PHASE_BUY, round_index))
    good, requested = options[0] if len(options) == 1 else options[stream.randint(0, len(options) - 1)]
    price = max(0.25, float(market["prices"][good]))
    affordable = int(max(0.0, state[MONEY]) // price)
    available = int(max(0.0, _stock(market, good)))
    quantity = min(int(requested), affordable, available)
    market["demand"][good] += requested
    if quantity <= 0:
        state[SHORTAGES] += int(requested > 0)
        state[IDEOLOGY] = max(-1.0, state[IDEOLOGY] - 0.00025)
        return
    sold = _consume_market(market, good, quantity, market["employers"])
    cost = round(sold * price, 2)
    state[MONEY] -= cost
    state[MARKET_SPENDING] += cost
    if good == "food":
        state[FOOD] += int(sold)
    else:
        state[MED] += int(sold)
    if sold + 1e-9 < requested:
        state[SHORTAGES] += 1
        state[IDEOLOGY] = max(-1.0, state[IDEOLOGY] - 0.00025)


def _rest_or_safe(state, action, location, master_seed, day, pid, round_index):
    stream = _DeterministicStream(_seed_for(master_seed, day, pid, _PHASE_SAFE, round_index))
    if action == "rest":
        state[ENERGY] = min(100, state[ENERGY] + stream.randint(12, 24))
        state[HEALTH] = min(100, state[HEALTH] + stream.randint(0, 2))
    elif action == "heal":
        if state[MED] > 0 and state[HEALTH] < 100:
            state[MED] -= 1
            state[HEALTH] = min(100, state[HEALTH] + stream.randint(8, 18))
    elif action == "repair":
        if state[MONEY] >= 3 and state[SHELTER] < 100:
            state[MONEY] -= 3
            state[SHELTER] = min(100, state[SHELTER] + stream.randint(8, 16))
    elif action == "scavenge":
        state[ENERGY] = max(0, state[ENERGY] - stream.randint(4, 9))
        state[FOOD] += stream.randint(0, location["scavenge_food_max"])
        state[MED] += int(stream.random() < location["medicine_chance"])


def _choose_move(state, location, all_locations, employers, master_seed, day, pid, round_index):
    if state[ENERGY] < 4 or not location["neighbors"]:
        return None
    stream = _DeterministicStream(_seed_for(master_seed, day, pid, _PHASE_MOVE, round_index))
    employer = employers.get(int(state[EMPLOYER]))
    profession_name = PROFESSION_NAMES[int(state[PROFESSION])]
    profession = PROFESSIONS[profession_name]
    options = []
    weights = []
    for lid in location["neighbors"]:
        candidate = all_locations.get(int(lid))
        if candidate is None:
            continue
        weight = 1.0
        if employer and employer["location"] == int(lid):
            weight *= 5.0
        if candidate["kind"] in profession.workplace_kinds:
            weight *= 2.2
        if employer is None and candidate["vacancies"] > 0:
            weight *= 1.8
        if state[FOOD] <= 3 and candidate["food_stock"] > 0:
            weight *= 2.5
        if state[MED] == 0 and candidate["kind"] == "clinic":
            weight *= 3.0
        options.append(int(lid))
        weights.append(weight)
    destination = _weighted_choice(options, weights, stream)
    if destination is None or destination == int(state[LID]):
        return None
    state[ENERGY] = max(0, state[ENERGY] - stream.randint(3, 7))
    if employer and employer["location"] != destination:
        employer["employees"] = max(0, employer["employees"] - 1)
        state[EMPLOYER] = -1
    return destination


def _police_response(state, action, magnitude, police_state, master_seed, day, pid, round_index):
    police_state["incidents"] += 1
    officers = police_state["officers"]
    population = max(1, police_state["population"])
    officers_per_1000 = officers * 1000.0 / population
    load = police_state["incidents"] / max(1, officers)
    coverage = max(0.02, min(0.92, 0.22 + officers_per_1000 * 0.12 - load * 0.035))
    severity = 1.15 if action == "attack" else 0.90
    probability = min(0.97, coverage * severity)
    stream = _DeterministicStream(_seed_for(master_seed, day, pid, _PHASE_POLICE, round_index))
    if stream.random() >= probability:
        return False
    police_state["responses"] += 1
    arrest_probability = min(0.90, 0.42 + 0.12 * magnitude + (0.12 if action == "attack" else 0.0))
    if stream.random() >= arrest_probability:
        return False
    police_state["arrests"] += 1
    detention = stream.randint(2, 8 if action == "attack" else 5)
    state[DETAINED] = max(int(state[DETAINED]), int(day) + detention)
    fine = min(state[MONEY], stream.randint(1, 5) + (2 if action == "attack" else 0))
    state[MONEY] -= fine
    state[ARRESTS] += 1
    return True


def _social_action(state, action, location, social, domain_state, memories, social_state,
                   master_seed, day, pid, round_index, encounter_sample, max_witnesses,
                   visibility, counters, police_state):
    stream = _DeterministicStream(_seed_for(master_seed, day, pid, _PHASE_SOCIAL, round_index))
    candidates = social.sample(location["id"], (pid,), encounter_sample, stream)
    target_row = planner._weighted_pick(candidates, memories, action, stream, day)
    if target_row is None:
        return
    target_id = int(target_row[0])
    target = domain_state.read(target_id)
    if not target[ALIVE] or int(target[LID]) != location["id"]:
        return

    witnesses = ()
    if max_witnesses and stream.random() <= visibility:
        rows = social.sample(location["id"], (pid, target_id), max_witnesses, stream)
        witnesses = tuple(int(row[0]) for row in rows)

    payload = None
    magnitude = 0.0
    if action == "help":
        amount = 0
        resource = None
        if target[HEALTH] < 70 and state[MED] > 0:
            state[MED] -= 1
            target[MED] += 1
            resource = "medicine"
            amount = 1
        elif state[FOOD] > 2:
            amount = stream.randint(1, min(2, state[FOOD] - 1))
            state[FOOD] -= amount
            target[FOOD] += amount
            resource = "food"
        payload = (resource, amount)
        if amount:
            counters["helps"] += 1
            magnitude = float(amount)
    elif action == "steal":
        resource = None
        amount = 0
        if stream.random() < 0.45:
            options = []
            if target[FOOD] > 0:
                options.extend(("food",) * 4)
            if target[MONEY] > 0:
                options.extend(("money",) * 2)
            if target[MED] > 0:
                options.append("medicine")
            if options:
                resource = options[stream.randint(0, len(options) - 1)]
                if resource == "food":
                    amount = min(target[FOOD], stream.randint(1, 3))
                    target[FOOD] -= amount
                    state[FOOD] += amount
                elif resource == "money":
                    amount = min(target[MONEY], float(stream.randint(1, 5)))
                    target[MONEY] -= amount
                    state[MONEY] += amount
                else:
                    amount = 1
                    target[MED] -= 1
                    state[MED] += 1
        payload = (resource, amount)
        if amount:
            counters["thefts"] += 1
            target[CRIME_SUFFERED] += 1
            counters["crimes"] += 1
            magnitude = max(1.0, float(amount) / 2.0)
            if _police_response(
                state, action, magnitude, police_state, master_seed, day, pid, round_index
            ):
                counters["arrests"] += 1
    else:
        damage = stream.randint(5, 20)
        energy_cost = stream.randint(4, 9)
        state[ENERGY] = max(0, state[ENERGY] - energy_cost)
        target[HEALTH] -= damage
        target[CRIME_SUFFERED] += 1
        counters["attacks"] += 1
        counters["crimes"] += 1
        payload = (damage, energy_cost)
        magnitude = float(damage) / 10.0
        if _police_response(
            state, action, magnitude, police_state, master_seed, day, pid, round_index
        ):
            counters["arrests"] += 1
        if target[HEALTH] <= 0:
            target[ALIVE] = 0

    valid_witnesses = 0
    for witness_id in witnesses:
        witness = domain_state.read(witness_id)
        if witness[ALIVE] and int(witness[LID]) == location["id"]:
            valid_witnesses += 1
    counters["observations"] += valid_witnesses
    domain_state.write(target_id, target)

    if magnitude > 0:
        social_plan = (action, target_id, payload, witnesses)
        _remember_worker_social(social_state, memories, social_plan, day)


def _process_location(
    location,
    all_locations,
    employer_rows,
    domain_state,
    social,
    social_cache,
    master_seed,
    day,
    actions_per_day,
    encounter_sample,
    max_witnesses,
    visibility,
    government,
    officers,
):
    employers, employers_by_location = _prepare_employers(employer_rows)
    local_employers = employers_by_location.get(location["id"], [])
    market = location["market"]
    market["employers"] = employers
    location["government"] = government
    location["treasury_delta"] = 0.0

    for other in all_locations.values():
        other["vacancies"] = sum(
            max(0, e["capacity"] - e["employees"])
            for e in employers_by_location.get(other["id"], ())
            if e["alive"]
        )
        other["food_stock"] = _stock(other["market"], "food")
        other["medicine_stock"] = _stock(other["market"], "medicine")

    police_state = {
        "officers": int(officers),
        "population": int(location["market"]["population"]),
        "incidents": 0,
        "responses": 0,
        "arrests": 0,
    }
    counters = Counter()

    size = social.count(location["id"])
    perm_seed = _seed_for(master_seed, day, location["id"], 0x444F4D41)
    for index in _iter_permuted(size, perm_seed):
        pid = int(social.row(location["id"], index)[0])
        if not social.is_eligible(pid):
            continue
        state = domain_state.read(pid)
        if not state[ALIVE] or int(state[LID]) != location["id"] or day < state[DETAINED]:
            continue

        memories = social_cache.setdefault(pid, {})
        social_state = {
            "pid": pid,
            "_known_people": 0,
            "_affinity_sum": 0.0,
            "_max_conflict_target": None,
            "positive_ties": 0,
            "hostile_ties": 0,
            "max_conflict": 0.0,
            "mean_affinity": 0.0,
        }
        if memories:
            _refresh_worker_social_aggregate(social_state, memories, day)

        pending_move = None
        for round_index in range(int(actions_per_day)):
            if not state[ALIVE] or day < state[DETAINED]:
                break

            snapshot = (
                pid,
                int(state[LID]),
                int(state[FOOD]),
                int(state[MED]),
                int(state[ENERGY]),
                int(state[HEALTH]),
                int(state[SHELTER]),
                float(state[MONEY]),
                int(state[EMPLOYER]) >= 0,
                location["kind"],
                int(social_state.get("positive_ties", 0)),
                int(social_state.get("hostile_ties", 0)),
                float(social_state.get("max_conflict", 0.0)),
                float(social_state.get("mean_affinity", 0.0)),
            )
            _, action = _weighted_action(snapshot, master_seed, day, round_index)
            counters[f"action_{action}"] += 1
            if not state[WORKING_AGE] and action not in {
                "scavenge", "buy_supplies", "rest", "heal", "repair", "help"
            }:
                action = "rest"

            if action in ("rest", "heal", "repair", "scavenge"):
                _rest_or_safe(state, action, location, master_seed, day, pid, round_index)
            elif action == "work":
                _work(
                    state, location, employers, local_employers, market,
                    master_seed, day, pid, round_index,
                )
            elif action == "buy_supplies":
                _buy(state, market, master_seed, day, pid, round_index)
            elif action == "move":
                destination = _choose_move(
                    state, location, all_locations, employers,
                    master_seed, day, pid, round_index,
                )
                if destination is not None:
                    pending_move = int(destination)
                    counters["moves"] += 1
            elif action in ("help", "steal", "attack"):
                _social_action(
                    state, action, location, social, domain_state, memories,
                    social_state, master_seed, day, pid, round_index,
                    encounter_sample, max_witnesses, visibility, counters,
                    police_state,
                )

        if pending_move is not None:
            state[LID] = pending_move
        domain_state.write(pid, state)

    market.pop("employers", None)
    employer_result = []
    for e in employers.values():
        if e["location"] != location["id"]:
            continue
        employer_result.append(
            (
                e["id"], e["location"], e["capacity"], e["employees"],
                e["cash"], e["productivity"], e["payroll"], e["revenue"],
                e["units"], int(e["alive"]),
            )
        )
    market_result = (
        location["id"],
        dict(market["prices"]),
        dict(market["suppliers"]["food"]),
        dict(market["suppliers"]["medicine"]),
        dict(market["demand"]),
        dict(market["sold"]),
    )
    police_result = (
        location["id"],
        police_state["officers"],
        police_state["incidents"],
        police_state["responses"],
        police_state["arrests"],
    )
    return {
        "location": location["id"],
        "counters": dict(counters),
        "treasury_delta": float(location["treasury_delta"]),
        "employers": employer_result,
        "market": market_result,
        "police": police_result,
    }


def _domain_worker(
    worker_id,
    owned_locations,
    input_queue,
    result_queue,
    master_seed,
    domain_descriptor,
    social_descriptor,
):
    from .aggressive_social import SharedSocialState

    domain_state = SharedDomainAgentState.attach(domain_descriptor)
    social = SharedSocialState.attach(social_descriptor)
    social_cache = {}
    try:
        while True:
            task = input_queue.get()
            if task is None:
                return
            (
                day,
                actions_per_day,
                encounter_sample,
                max_witnesses,
                visibility,
                packet,
            ) = task
            started = perf_counter()

            location_rows = {
                int(row[0]): _market_from_location_row(row)
                for row in packet["locations"]
            }
            all_locations = {
                lid: {
                    "id": lid,
                    "kind": row["kind"],
                    "neighbors": row["neighbors"],
                    "scavenge_food_max": row["scavenge_food_max"],
                    "medicine_chance": row["medicine_chance"],
                    "market": row,
                    "vacancies": 0,
                    "food_stock": _stock(row, "food"),
                    "medicine_stock": _stock(row, "medicine"),
                }
                for lid, row in location_rows.items()
            }

            results = []
            for lid in owned_locations:
                location = all_locations.get(int(lid))
                if location is None:
                    continue
                results.append(
                    _process_location(
                        location,
                        all_locations,
                        packet["employers"],
                        domain_state,
                        social,
                        social_cache,
                        master_seed,
                        int(day),
                        int(actions_per_day),
                        int(encounter_sample),
                        int(max_witnesses),
                        float(visibility),
                        packet["government"],
                        packet["police"].get(int(lid), 0),
                    )
                )

            result_queue.put(
                (worker_id, perf_counter() - started, results)
            )
    finally:
        social.close()
        domain_state.close()


class DomainOwnerPool:
    """Persistent location-owner workers with BSP barriers."""

    def __init__(self, master_seed, domain_state, social_state, location_count,
                 workers=0, min_active=100_000):
        cpu_count = os.cpu_count() or 1
        requested = max(0, int(workers))
        self.worker_count = min(requested, cpu_count) if requested else 0
        self.master_seed = int(master_seed)
        self.domain_state = domain_state
        self.social_state = social_state
        self.domain_descriptor = domain_state.descriptor
        self.social_descriptor = social_state.descriptor
        self.location_count = int(location_count)
        self.min_active = max(1, int(min_active))
        self.enabled = self.worker_count >= 2
        self.started = False
        self._ctx = None
        self._queues = []
        self._result_queue = None
        self._processes = []
        self._active_workers = []
        self.stats = Counter()
        self.worker_seconds = 0.0
        self.dispatch_seconds = 0.0

    def should_parallelize(self, active_count):
        return self.enabled and int(active_count) >= self.min_active

    def _ensure_started(self):
        if self.started or not self.enabled:
            return
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        self._ctx = mp.get_context(method)
        self._result_queue = self._ctx.Queue()
        assignments = [[] for _ in range(self.worker_count)]
        for lid in range(self.location_count):
            assignments[lid % self.worker_count].append(lid)
        for worker_id, owned in enumerate(assignments):
            if not owned:
                self._queues.append(None)
                self._processes.append(None)
                continue
            queue = self._ctx.Queue(maxsize=2)
            process = self._ctx.Process(
                target=_domain_worker,
                args=(
                    worker_id,
                    tuple(owned),
                    queue,
                    self._result_queue,
                    self.master_seed,
                    self.domain_descriptor,
                    self.social_descriptor,
                ),
                name=f"stochastic-domain-{worker_id}",
                daemon=True,
            )
            process.start()
            self._queues.append(queue)
            self._processes.append(process)
            self._active_workers.append(worker_id)
        self.started = True

    def run_day(self, day, actions_per_day, encounter_sample, max_witnesses,
                visibility, packet):
        self._ensure_started()
        task = (
            int(day),
            int(actions_per_day),
            int(encounter_sample),
            int(max_witnesses),
            float(visibility),
            packet,
        )
        started = perf_counter()
        for worker_id in self._active_workers:
            self._queues[worker_id].put(task)
            self.stats["tasks"] += 1
        results = []
        for _ in self._active_workers:
            _worker_id, seconds, payload = self._result_queue.get()
            self.worker_seconds += seconds
            results.extend(payload)
        self.dispatch_seconds += perf_counter() - started
        self.stats["days"] += 1
        return results

    def summary(self):
        return {
            "days": int(self.stats["days"]),
            "tasks": int(self.stats["tasks"]),
            "workers": len(self._active_workers),
            "worker_seconds": float(self.worker_seconds),
            "dispatch_seconds": float(self.dispatch_seconds),
            "shared_bytes": int(self.domain_state.allocated_bytes),
        }

    def close(self):
        if not self.started:
            return
        for worker_id in self._active_workers:
            self._queues[worker_id].put(None)
        for worker_id in self._active_workers:
            process = self._processes[worker_id]
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            self._queues[worker_id].close()
        if self._result_queue is not None:
            self._result_queue.close()
        self._active_workers.clear()
        self.started = False

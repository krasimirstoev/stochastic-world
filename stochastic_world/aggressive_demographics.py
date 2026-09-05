"""Vectorized demographic sidecar for the 100k+ SoA engine.

Demographic / household state is cold data: action workers do not need it on
normal days. Keeping it in a main-process NumPy sidecar avoids materializing
every Person on monthly lifecycle boundaries while preserving births,
partnerships, aging, retirement and natural mortality.
"""

from __future__ import annotations

import numpy as np

from .aggressive_economy import (
    PROFESSION_NAMES,
    PROFESSION_TO_CODE,
    SOCIAL_CLASSES,
)
from .aggressive_soa import (
    B_ADULT,
    B_ALIVE,
    B_DEPENDENT,
    B_WORKING_AGE,
    F_IDEOLOGY,
    F_MONEY,
    I_EMPLOYER,
    I_FOOD,
    I_HEALTH,
    I_LID,
    I_MED,
    I_PROFESSION,
    I_SHELTER,
    I_SOCIAL_CLASS,
    I_UNEMPLOYMENT,
)
from .demographics import DEMOGRAPHIC_INTERVAL_DAYS, PREGNANCY_DAYS
from .person import ADULT_AGE_DAYS, RETIREMENT_AGE_DAYS, Person
from .professions import CLASS_PROFESSIONS

_DEMOGRAPHIC_PHASE = 0xD3A06A9
_PARTNERSHIP_PHASE = 0x50415254
_PREGNANCY_PHASE = 0x50524547
_BIRTH_PHASE = 0x42495254

_FEMALE = 0
_MALE = 1

_CLASS_PROFESSION_CODES = tuple(
    np.asarray(
        [PROFESSION_TO_CODE[name] for name in CLASS_PROFESSIONS[class_name]],
        dtype=np.int32,
    )
    for class_name in SOCIAL_CLASSES
)


def _rng(seed: int, day: int, phase: int):
    mixed = (
        int(seed) ^ (int(day) * 0x9E3779B97F4A7C15) ^ int(phase)
    ) & ((1 << 63) - 1)
    return np.random.default_rng(mixed)


class SoADemographicState:
    """Cold demographic state parallel to SharedSoAAgentState."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = max(1, int(capacity))
        self.seed = int(seed)
        self.age_days = np.zeros(self.capacity, dtype=np.int32)
        self.birth_day = np.zeros(self.capacity, dtype=np.int32)
        self.mother_id = np.full(self.capacity, -1, dtype=np.int32)
        self.father_id = np.full(self.capacity, -1, dtype=np.int32)
        self.partner_id = np.full(self.capacity, -1, dtype=np.int32)
        self.household_id = np.full(self.capacity, -1, dtype=np.int32)
        self.generation = np.zeros(self.capacity, dtype=np.int16)
        self.pregnant_until = np.zeros(self.capacity, dtype=np.int32)
        self.pregnancy_partner = np.full(self.capacity, -1, dtype=np.int32)
        self.sex = np.zeros(self.capacity, dtype=np.uint8)
        self.retired = np.zeros(self.capacity, dtype=np.uint8)

    @property
    def allocated_bytes(self):
        return sum(
            arr.nbytes
            for arr in (
                self.age_days,
                self.birth_day,
                self.mother_id,
                self.father_id,
                self.partner_id,
                self.household_id,
                self.generation,
                self.pregnant_until,
                self.pregnancy_partner,
                self.sex,
                self.retired,
            )
        )

    def sync_world(self, world):
        if len(world.people) > self.capacity:
            raise RuntimeError("SoA demographic capacity exhausted")
        for person in world.people:
            self.write_person(person)

    def write_person(self, person):
        pid = int(person.id)
        if pid < 0 or pid >= self.capacity:
            raise RuntimeError("SoA demographic capacity exhausted")
        self.age_days[pid] = int(person.age_days)
        self.birth_day[pid] = int(person.birth_day)
        self.mother_id[pid] = -1 if person.mother_id is None else int(person.mother_id)
        self.father_id[pid] = -1 if person.father_id is None else int(person.father_id)
        self.partner_id[pid] = -1 if person.partner_id is None else int(person.partner_id)
        self.household_id[pid] = (
            -1 if person.household_id is None else int(person.household_id)
        )
        self.generation[pid] = int(person.generation)
        self.pregnant_until[pid] = int(person.pregnant_until_day)
        self.pregnancy_partner[pid] = (
            -1
            if person.pregnancy_partner_id is None
            else int(person.pregnancy_partner_id)
        )
        self.sex[pid] = _MALE if person.sex == "male" else _FEMALE
        self.retired[pid] = int(bool(person.retired))

    def apply_to_person(self, person):
        pid = int(person.id)
        person.age_days = int(self.age_days[pid])
        person.birth_day = int(self.birth_day[pid])
        mother = int(self.mother_id[pid])
        father = int(self.father_id[pid])
        partner = int(self.partner_id[pid])
        household = int(self.household_id[pid])
        pregnancy_partner = int(self.pregnancy_partner[pid])
        person.mother_id = None if mother < 0 else mother
        person.father_id = None if father < 0 else father
        person.partner_id = None if partner < 0 else partner
        person.household_id = None if household < 0 else household
        person.generation = int(self.generation[pid])
        person.pregnant_until_day = int(self.pregnant_until[pid])
        person.pregnancy_partner_id = (
            None if pregnancy_partner < 0 else pregnancy_partner
        )
        person.sex = "male" if self.sex[pid] == _MALE else "female"
        person.retired = bool(self.retired[pid])
        if person.age_days < ADULT_AGE_DAYS:
            person.profession = "dependent"
        elif person.retired:
            person.profession = "retired"

    def _merge_households(self, world, keep_id: int, merge_id: int):
        if keep_id < 0:
            return merge_id
        if merge_id < 0 or keep_id == merge_id:
            return keep_id
        demographics = world.demographics
        moved = tuple(demographics.members.get(merge_id, ()))
        if moved:
            demographics.members[keep_id].update(moved)
            self.household_id[np.asarray(moved, dtype=np.int64)] = int(keep_id)
        demographics.members.pop(merge_id, None)
        demographics.households.pop(merge_id, None)
        return keep_id

    def _form_partnerships(self, world, day: int, n: int):
        flags = world.soa_state.flags
        ints = world.soa_state.ints
        alive = flags[B_ALIVE, :n] != 0
        age = self.age_days[:n]
        free = self.partner_id[:n] < 0
        eligible_age = (age >= 20 * 365) & (age <= 45 * 365)
        rng = _rng(self.seed, day, _PARTNERSHIP_PHASE)
        formed = 0

        for lid in range(len(world.locations)):
            local = ints[I_LID, :n] == lid
            females = np.flatnonzero(
                alive & local & free & eligible_age & (self.sex[:n] == _FEMALE)
            )
            males = np.flatnonzero(
                alive & local & free & eligible_age & (self.sex[:n] == _MALE)
            )
            if females.size == 0 or males.size == 0:
                continue
            females = rng.permutation(females)
            males = rng.permutation(males)
            size = min(females.size, males.size)
            females = females[:size]
            males = males[:size]
            compatible = np.abs(age[females] - age[males]) <= 15 * 365
            chosen = compatible & (rng.random(size) <= 0.025)
            for female, male in zip(females[chosen], males[chosen]):
                female = int(female)
                male = int(male)
                if self.partner_id[female] >= 0 or self.partner_id[male] >= 0:
                    continue
                self.partner_id[female] = male
                self.partner_id[male] = female
                keep = int(self.household_id[female])
                merge = int(self.household_id[male])
                hid = self._merge_households(world, keep, merge)
                if hid >= 0:
                    self.household_id[female] = hid
                    self.household_id[male] = hid
                    world.demographics.members[hid].add(female)
                    world.demographics.members[hid].add(male)
                formed += 1
                world.demographics._life_event(
                    day,
                    "partnership",
                    world.people[female],
                    world.people[male],
                    hid,
                )
        world.demographics.total_partnerships += formed
        return formed

    def _start_pregnancies(self, world, day: int, n: int):
        flags = world.soa_state.flags
        ints = world.soa_state.ints
        floats = world.soa_state.floats
        alive = flags[B_ALIVE, :n] != 0
        female = self.sex[:n] == _FEMALE
        age = self.age_days[:n]
        partner = self.partner_id[:n]
        household = self.household_id[:n]
        candidate = (
            alive
            & female
            & (self.pregnant_until[:n] == 0)
            & (partner >= 0)
            & (age >= 18 * 365)
            & (age <= 42 * 365)
            & (ints[I_HEALTH, :n] >= 50)
        )
        ids = np.flatnonzero(candidate)
        if ids.size == 0:
            return 0

        partner_ids = partner[ids]
        valid_partner = (
            (partner_ids >= 0)
            & (partner_ids < n)
            & alive[partner_ids]
            & (household[partner_ids] == household[ids])
        )
        ids = ids[valid_partner]
        if ids.size == 0:
            return 0

        annual = np.zeros(ids.size, dtype=np.float64)
        years = age[ids] / 365.0
        annual[(years >= 18) & (years < 25)] = 0.075
        annual[(years >= 25) & (years < 35)] = 0.115
        annual[(years >= 35) & (years < 40)] = 0.065
        annual[(years >= 40) & (years <= 42)] = 0.018

        mothers = self.mother_id[:n]
        valid_children = alive & (mothers >= 0) & (mothers < n)
        child_counts = (
            np.bincount(mothers[valid_children], minlength=n)
            if np.any(valid_children)
            else np.zeros(n, dtype=np.int64)
        )
        parity = np.maximum(0.25, 1.0 - child_counts[ids] * 0.18)

        hids = household
        valid_h = alive & (hids >= 0) & (flags[B_ADULT, :n] != 0)
        max_hid = int(np.max(hids[valid_h])) + 1 if np.any(valid_h) else 1
        h_count = np.bincount(hids[valid_h], minlength=max_hid).astype(np.float64)
        h_money = np.bincount(
            hids[valid_h],
            weights=floats[F_MONEY, :n][valid_h],
            minlength=max_hid,
        )
        h_food = np.bincount(
            hids[valid_h],
            weights=ints[I_FOOD, :n][valid_h],
            minlength=max_hid,
        )
        h_shelter = np.bincount(
            hids[valid_h],
            weights=ints[I_SHELTER, :n][valid_h],
            minlength=max_hid,
        )
        mother_h = hids[ids]
        condition = np.full(ids.size, 0.4, dtype=np.float64)
        in_range = (mother_h >= 0) & (mother_h < max_hid)
        good_h = np.zeros(ids.size, dtype=bool)
        if np.any(in_range):
            good_h[in_range] = h_count[mother_h[in_range]] > 0
        if np.any(good_h):
            hh = mother_h[good_h]
            count = h_count[hh]
            condition[good_h] = np.clip(
                0.55
                + (h_money[hh] / count) / 120.0
                + (h_food[hh] / count) / 45.0
                + (h_shelter[hh] / count) / 300.0,
                0.35,
                1.2,
            )

        probability = (
            annual * (DEMOGRAPHIC_INTERVAL_DAYS / 365.0) * condition * parity
        )
        rng = _rng(self.seed, day, _PREGNANCY_PHASE)
        selected = ids[rng.random(ids.size) < probability]
        if selected.size == 0:
            return 0
        self.pregnant_until[selected] = int(day + PREGNANCY_DAYS)
        self.pregnancy_partner[selected] = self.partner_id[selected]
        for pid in selected:
            pid = int(pid)
            partner_id = int(self.pregnancy_partner[pid])
            world.demographics._life_event(
                day,
                "pregnancy",
                world.people[pid],
                world.people[partner_id]
                if 0 <= partner_id < len(world.people)
                else None,
                int(self.household_id[pid]),
                due_day=int(self.pregnant_until[pid]),
            )
        return int(selected.size)

    def _append_birth(self, world, mother_id: int, day: int, rng):
        pid = len(world.people)
        if pid >= self.capacity or pid >= world.soa_state.capacity:
            raise RuntimeError(
                "SoA capacity exhausted by demographic growth; "
                "increase large-world headroom"
            )

        father_id = int(self.pregnancy_partner[mother_id])
        if (
            father_id < 0
            or father_id >= pid
            or world.soa_state.flags[B_ALIVE, father_id] == 0
        ):
            father_id = -1
        mother_generation = int(self.generation[mother_id])
        father_generation = (
            int(self.generation[father_id]) if father_id >= 0 else mother_generation
        )
        generation = max(mother_generation, father_generation) + 1
        ideology = float(world.soa_state.floats[F_IDEOLOGY, mother_id])
        if father_id >= 0:
            ideology = (
                ideology + float(world.soa_state.floats[F_IDEOLOGY, father_id])
            ) / 2.0
        ideology = float(np.clip(ideology + rng.normal(0.0, 0.16), -1.0, 1.0))
        sex = _FEMALE if rng.random() < 0.5 else _MALE
        household_id = int(self.household_id[mother_id])
        location_id = int(world.soa_state.ints[I_LID, mother_id])
        social_code = int(world.soa_state.ints[I_SOCIAL_CLASS, mother_id])
        social_class = (
            SOCIAL_CLASSES[social_code]
            if 0 <= social_code < len(SOCIAL_CLASSES)
            else SOCIAL_CLASSES[0]
        )

        child = Person(
            id=pid,
            name=f"A{pid}",
            social_class=social_class,
            profession="dependent",
            ideology=ideology,
            location_id=location_id,
            food=8,
            money=0.0,
            medicine=1,
            energy=90,
            shelter=max(35, int(world.soa_state.ints[I_SHELTER, mother_id])),
            health=int(rng.integers(88, 101)),
            age_days=0,
            sex="male" if sex == _MALE else "female",
            birth_day=day,
            mother_id=mother_id,
            father_id=None if father_id < 0 else father_id,
            household_id=None if household_id < 0 else household_id,
            generation=generation,
        )
        child.memory_cap = getattr(world, "memory_cap", child.memory_cap)
        world.people.append(child)
        world.store.register_person(child)

        world.soa_state.write_person(child)
        self.write_person(child)
        if household_id >= 0:
            world.demographics.members[household_id].add(pid)
        world.demographics._persist_person(child)

        world.alive_count += 1
        world.invalidate_living_cache()
        world._domain_location_population[location_id] = (
            int(world._domain_location_population.get(location_id, 0)) + 1
        )
        world.goods_market.set_population(
            location_id,
            world._domain_location_population[location_id],
        )
        world.demographics.total_births += 1
        world.demographics._life_event(
            day,
            "birth",
            child,
            world.people[mother_id],
            household_id,
            father_id=None if father_id < 0 else father_id,
            generation=generation,
        )
        self.pregnant_until[mother_id] = 0
        self.pregnancy_partner[mother_id] = -1
        return pid

    def _complete_pregnancies(self, world, day: int, n: int):
        flags = world.soa_state.flags
        due = np.flatnonzero(
            (flags[B_ALIVE, :n] != 0)
            & (self.pregnant_until[:n] > 0)
            & (self.pregnant_until[:n] <= day)
        )
        if due.size == 0:
            return 0
        rng = _rng(self.seed, day, _BIRTH_PHASE)
        for mother_id in due:
            self._append_birth(world, int(mother_id), day, rng)
        return int(due.size)

    def _support_dependents(self, world, n: int):
        flags = world.soa_state.flags
        ints = world.soa_state.ints
        floats = world.soa_state.floats
        dependent_ids = np.flatnonzero(
            (flags[B_ALIVE, :n] != 0) & (flags[B_DEPENDENT, :n] != 0)
        )
        supported = 0
        for pid_raw in dependent_ids:
            pid = int(pid_raw)
            hid = int(self.household_id[pid])
            if hid < 0:
                continue
            members = world.demographics.members.get(hid, ())
            adults = [
                int(aid)
                for aid in members
                if 0 <= int(aid) < n
                and flags[B_ALIVE, int(aid)]
                and flags[B_ADULT, int(aid)]
            ]
            if not adults:
                continue
            donor = max(
                adults,
                key=lambda aid: (
                    int(ints[I_FOOD, aid]),
                    float(floats[F_MONEY, aid]),
                ),
            )
            if ints[I_FOOD, pid] <= 3 and ints[I_FOOD, donor] > 5:
                amount = min(2, int(ints[I_FOOD, donor]) - 4)
                ints[I_FOOD, donor] -= amount
                ints[I_FOOD, pid] += amount
            if ints[I_HEALTH, pid] < 70 and ints[I_MED, pid] == 0:
                medic = max(adults, key=lambda aid: int(ints[I_MED, aid]))
                if ints[I_MED, medic] > 1:
                    ints[I_MED, medic] -= 1
                    ints[I_MED, pid] += 1
            max_shelter = max(int(ints[I_SHELTER, aid]) for aid in adults)
            ints[I_SHELTER, pid] = max(
                int(ints[I_SHELTER, pid]),
                max_shelter - 8,
            )
            supported += 1
        return supported

    def _age_retire_mortality(self, world, day: int, n: int):
        flags = world.soa_state.flags
        ints = world.soa_state.ints
        old_age = self.age_days[:n].copy()
        alive = flags[B_ALIVE, :n] != 0
        self.age_days[:n] = np.maximum(
            self.age_days[:n],
            int(day) - self.birth_day[:n],
        )

        became_adult = (
            alive
            & (old_age < ADULT_AGE_DAYS)
            & (self.age_days[:n] >= ADULT_AGE_DAYS)
        )
        adult_ids = np.flatnonzero(became_adult)
        if adult_ids.size:
            flags[B_ADULT, adult_ids] = 1
            flags[B_DEPENDENT, adult_ids] = 0
            flags[B_WORKING_AGE, adult_ids] = 1
            ints[I_UNEMPLOYMENT, adult_ids] = 0
            rng = _rng(self.seed, day, _DEMOGRAPHIC_PHASE ^ 0xA0)
            class_codes = ints[I_SOCIAL_CLASS, adult_ids]
            for class_code, choices in enumerate(_CLASS_PROFESSION_CODES):
                ids = adult_ids[class_codes == class_code]
                if ids.size:
                    ints[I_PROFESSION, ids] = choices[
                        rng.integers(0, len(choices), size=ids.size)
                    ]
            world.demographics.total_adulthoods += int(adult_ids.size)
            for pid_raw in adult_ids:
                pid = int(pid_raw)
                pcode = int(ints[I_PROFESSION, pid])
                world.demographics._life_event(
                    day,
                    "coming_of_age",
                    world.people[pid],
                    household_id=int(self.household_id[pid]),
                    profession=(
                        PROFESSION_NAMES[pcode]
                        if 0 <= pcode < len(PROFESSION_NAMES)
                        else "laborer"
                    ),
                )

        retiring = (
            alive
            & (old_age < RETIREMENT_AGE_DAYS)
            & (self.age_days[:n] >= RETIREMENT_AGE_DAYS)
            & (self.retired[:n] == 0)
        )
        retire_ids = np.flatnonzero(retiring)
        if retire_ids.size:
            self.retired[retire_ids] = 1
            flags[B_WORKING_AGE, retire_ids] = 0
            ints[I_EMPLOYER, retire_ids] = -1
            ints[I_UNEMPLOYMENT, retire_ids] = 0
            world.demographics.total_retirements += int(retire_ids.size)
            for pid_raw in retire_ids:
                pid = int(pid_raw)
                world.demographics._life_event(
                    day,
                    "retirement",
                    world.people[pid],
                    household_id=int(self.household_id[pid]),
                )

        age_years = self.age_days[:n].astype(np.float64) / 365.0
        base = np.empty(n, dtype=np.float64)
        base[:] = 0.0006
        base[age_years < 1] = 0.004
        base[(age_years >= 1) & (age_years < 15)] = 0.0002
        base[(age_years >= 45) & (age_years < 65)] = 0.004
        base[(age_years >= 65) & (age_years < 75)] = 0.018
        base[(age_years >= 75) & (age_years < 85)] = 0.055
        base[(age_years >= 85) & (age_years < 95)] = 0.16
        old = age_years >= 95
        base[old] = np.minimum(
            0.65,
            0.22 + (age_years[old] - 95.0) * 0.035,
        )
        health_factor = (
            1.0 + np.maximum(0.0, 70.0 - ints[I_HEALTH, :n]) / 45.0
        )
        annual = np.minimum(0.95, base * health_factor)
        monthly = 1.0 - np.power(
            1.0 - annual,
            DEMOGRAPHIC_INTERVAL_DAYS / 365.0,
        )
        rng = _rng(self.seed, day, _DEMOGRAPHIC_PHASE ^ 0xD1E)
        dead = alive & (rng.random(n) < monthly)
        dead_ids = np.flatnonzero(dead)
        if dead_ids.size:
            partners = self.partner_id[dead_ids].copy()
            valid = (partners >= 0) & (partners < n)
            if np.any(valid):
                dead_valid = dead_ids[valid]
                partner_valid = partners[valid]
                linked = self.partner_id[partner_valid] == dead_valid
                self.partner_id[partner_valid[linked]] = -1
            flags[B_ALIVE, dead_ids] = 0
            flags[B_WORKING_AGE, dead_ids] = 0
            ints[I_EMPLOYER, dead_ids] = -1
            self.pregnant_until[dead_ids] = 0
            self.pregnancy_partner[dead_ids] = -1
            for pid in dead_ids:
                world.store.event(
                    day,
                    world.next_sequence(),
                    "death",
                    actor=world.people[int(pid)],
                    location_id=int(ints[I_LID, int(pid)]),
                    cause="natural_age",
                )
            count = int(dead_ids.size)
            world.total_deaths += count
            world.alive_count = max(0, int(world.alive_count) - count)
            world.demographics.total_natural_deaths += count

        working = (
            (flags[B_ALIVE, :n] != 0)
            & (flags[B_ADULT, :n] != 0)
            & (self.retired[:n] == 0)
            & (self.age_days[:n] < RETIREMENT_AGE_DAYS)
        )
        flags[B_WORKING_AGE, :n] = working.astype(np.uint8)
        flags[B_DEPENDENT, :n] = (
            (flags[B_ALIVE, :n] != 0)
            & (self.age_days[:n] < ADULT_AGE_DAYS)
        ).astype(np.uint8)
        flags[B_ADULT, :n] = (
            (flags[B_ALIVE, :n] != 0)
            & (self.age_days[:n] >= ADULT_AGE_DAYS)
        ).astype(np.uint8)
        world.demographics.working_age_count = int(np.count_nonzero(working))

        employed_mask = (
            (flags[B_ALIVE, :n] != 0) & (ints[I_EMPLOYER, :n] >= 0)
        )
        employer_ids = ints[I_EMPLOYER, :n][employed_mask]
        max_eid = max(
            (int(e.id) for e in world.labor_market.employers),
            default=-1,
        )
        counts = (
            np.bincount(employer_ids, minlength=max_eid + 1)
            if employer_ids.size and max_eid >= 0
            else np.zeros(max(0, max_eid + 1), dtype=np.int64)
        )
        for employer in world.labor_market.employers:
            eid = int(employer.id)
            count = int(counts[eid]) if 0 <= eid < counts.size else 0
            world._domain_employee_counts[eid] = count
            employer._domain_employee_count = count
        return int(adult_ids.size), int(retire_ids.size), int(dead_ids.size)

    def cycle(self, world, day: int):
        n = len(world.people)
        self._age_retire_mortality(world, day, n)
        births = self._complete_pregnancies(world, day, n)
        n = len(world.people)
        partnerships = self._form_partnerships(world, day, n)
        pregnancies = self._start_pregnancies(world, day, n)
        supported = self._support_dependents(world, n)

        alive = world.soa_state.flags[B_ALIVE, :n] != 0
        locs = world.soa_state.ints[I_LID, :n]
        for lid in range(len(world.locations)):
            count = int(np.count_nonzero(alive & (locs == lid)))
            world._domain_location_population[lid] = count
            world.goods_market.set_population(lid, count)
        world.invalidate_living_cache()
        return {
            "births": births,
            "partnerships": partnerships,
            "pregnancies": pregnancies,
            "supported_dependents": supported,
        }

    def write_stats(self, world, day: int):
        n = len(world.people)
        alive = world.soa_state.flags[B_ALIVE, :n] != 0
        ages = self.age_days[:n][alive]
        population = int(ages.size)
        years = ages.astype(np.float64) / 365.0
        bands = (
            int(np.count_nonzero(years < 15)),
            int(np.count_nonzero((years >= 15) & (years < 25))),
            int(np.count_nonzero((years >= 25) & (years < 45))),
            int(np.count_nonzero((years >= 45) & (years < 65))),
            int(np.count_nonzero(years >= 65)),
        )
        hids = self.household_id[:n][alive]
        active_households = int(np.unique(hids[hids >= 0]).size)
        pregnant = int(np.count_nonzero(self.pregnant_until[:n][alive] > 0))
        max_generation = (
            int(np.max(self.generation[:n][alive])) if population else 0
        )
        median_age = float(np.median(years)) if population else 0.0
        world.store.conn.execute(
            "INSERT OR REPLACE INTO demographic_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                world.store.simulation_id,
                day,
                population,
                world.demographics.total_births,
                world.demographics.total_natural_deaths,
                world.total_deaths,
                median_age,
                *bands,
                active_households,
                pregnant,
                max_generation,
            ),
        )

    def persist_people(self, world):
        """Bulk-persist demographic state for reporting / final inspection."""
        n = len(world.people)
        rows = []
        household_member_rows = []
        sid = world.store.simulation_id
        for pid in range(n):
            mother = int(self.mother_id[pid])
            father = int(self.father_id[pid])
            partner = int(self.partner_id[pid])
            household = int(self.household_id[pid])
            rows.append(
                (
                    int(self.age_days[pid]),
                    "male" if self.sex[pid] == _MALE else "female",
                    int(self.birth_day[pid]),
                    None if mother < 0 else mother,
                    None if father < 0 else father,
                    None if partner < 0 else partner,
                    None if household < 0 else household,
                    int(self.generation[pid]),
                    int(self.retired[pid]),
                    sid,
                    pid,
                )
            )
            household_member_rows.append(
                (sid, pid, household, max(0, int(self.birth_day[pid])))
            )
        world.store.conn.executemany(
            """UPDATE persons SET age_days=?,sex=?,birth_day=?,mother_id=?,father_id=?,partner_id=?,
               household_id=?,generation=?,retired=? WHERE simulation_id=? AND person_id=?""",
            rows,
        )
        world.store.conn.executemany(
            """INSERT INTO household_members(simulation_id,person_id,household_id,joined_day)
               VALUES(?,?,?,?) ON CONFLICT(simulation_id,person_id)
               DO UPDATE SET household_id=excluded.household_id""",
            household_member_rows,
        )
        world.store.conn.execute(
            "DELETE FROM households WHERE simulation_id=?",
            (sid,),
        )
        household_rows = [
            (sid, h.id, h.created_day, h.location_id)
            for h in world.demographics.households.values()
        ]
        if household_rows:
            world.store.conn.executemany(
                "INSERT OR REPLACE INTO households VALUES(?,?,?,?)",
                household_rows,
            )

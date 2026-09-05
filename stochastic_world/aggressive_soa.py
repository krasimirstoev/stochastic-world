"""Compiled structure-of-arrays BSP engine for very large aggressive worlds.

For 100k+ agents, hot state is kept in shared NumPy arrays. Each location owner
enters one Numba kernel per simulated day; the kernel executes action selection,
local economy, social interactions, end-of-day decay and statistics without
per-agent Python dispatch. A CSR location index and fixed 64-slot relationship
table keep targeting and social memory bounded and cache-friendly.
"""

import math
import multiprocessing as mp
import os
from collections import Counter
from multiprocessing import shared_memory
from time import perf_counter

import numpy as np
from numba import njit

from .aggressive_economy import LOCATION_KINDS, PROFESSION_NAMES, SOCIAL_CLASSES
from .professions import PROFESSIONS
from .aggressive_jit import (
    _MASK64,
    _INV_U64,
    _PHASE_ACTION,
    _PHASE_END_OF_DAY,
    _weighted_code_nb,
    _eod_nb,
    _seed_nb,
    _splitmix64_nb,
)


MEMORY_CAP = 64
MEMORY_SCALE = 80.0
MEMORY_CLAMP = 8000
MAX_ENCOUNTER_SAMPLE = 32

I_LID = 0
I_FOOD = 1
I_MED = 2
I_ENERGY = 3
I_HEALTH = 4
I_SHELTER = 5
I_EMPLOYER = 6
I_PROFESSION = 7
I_SOCIAL_CLASS = 8
I_DETAINED = 9
I_WORK_EXP = 10
I_SHORTAGES = 11
I_CRIME_SUFFERED = 12
I_ARRESTS = 13
I_UNEMPLOYMENT = 14
I_LIFETIME_UNEMPLOYMENT = 15
I_DAYS_IN_CLASS = 16
I_JOBS_HELD = 17
INT_FIELDS = 18

F_MONEY = 0
F_IDEOLOGY = 1
F_TAXES = 2
F_WELFARE = 3
F_CAREER = 4
F_LIFETIME_GROSS = 5
F_MARKET_SPENDING = 6
FLOAT_FIELDS = 7

B_ALIVE = 0
B_WORKING_AGE = 1
B_DEPENDENT = 2
B_ADULT = 3
FLAG_FIELDS = 4

EI_LOCATION = 0
EI_KIND = 1
EI_CAPACITY = 2
EI_EMPLOYEES = 3
EI_OUTPUT = 4
EI_ALIVE = 5
EMP_INT_FIELDS = 6

EF_BASE_WAGE = 0
EF_CASH = 1
EF_PRODUCTIVITY = 2
EF_OUTPUT_PER_SHIFT = 3
EF_PAYROLL = 4
EF_REVENUE = 5
EF_UNITS = 6
EMP_FLOAT_FIELDS = 7

LI_KIND = 0
LI_VACANCIES = 1
LI_POPULATION = 2
LI_SCAVENGE_MAX = 3
LI_SHELTER_DECAY = 4
LOC_INT_FIELDS = 5

LF_FOOD_STOCK = 0
LF_MED_STOCK = 1
LF_FOOD_PRICE = 2
LF_MED_PRICE = 3
LF_MED_CHANCE = 4
LF_HISTORY_SUM = 5
LF_HISTORY_LEN = 6
LOC_FLOAT_FIELDS = 7

A_MOVE = 0
A_WORK = 1
A_SCAVENGE = 2
A_BUY = 3
A_REST = 4
A_HEAL = 5
A_REPAIR = 6
A_HELP = 7
A_STEAL = 8
A_ATTACK = 9
A_IDLE = 10
ACTION_COUNT = 11

C_HELPS = 0
C_THEFTS = 1
C_ATTACKS = 2
C_OBSERVATIONS = 3
C_ARRESTS = 4
C_MOVES = 5
C_CRIMES = 6
COUNTER_COUNT = 7

_PHASE_SAFE = 0x5AFE
_PHASE_SOCIAL = 0x50C1A1
_PHASE_MOVE = 0x4D4F5645
_PHASE_WORK = 0x574F524B
_PHASE_BUY = 0x425559
_PHASE_POLICE = 0x504F4C49
_PHASE_ORDER = 0x444F4D41

LOCATION_TO_CODE = {name: i for i, name in enumerate(LOCATION_KINDS)}
INDUSTRIAL_CODE = LOCATION_TO_CODE.get("industrial", 2)
OUTSKIRTS_CODE = LOCATION_TO_CODE.get("outskirts", 4)
CLINIC_CODE = LOCATION_TO_CODE.get("clinic", 3)
LOGISTICS_CODE = LOCATION_TO_CODE.get("logistics", 5)

_PROF_INCOME = np.array(
    [PROFESSIONS.get(name, PROFESSIONS["laborer"]).income_multiplier for name in PROFESSION_NAMES],
    dtype=np.float64,
)
_PROF_ENERGY = np.array(
    [PROFESSIONS.get(name, PROFESSIONS["laborer"]).energy_multiplier for name in PROFESSION_NAMES],
    dtype=np.float64,
)
_PROF_ADVANCE = np.array(
    [PROFESSIONS.get(name, PROFESSIONS["laborer"]).advancement_rate for name in PROFESSION_NAMES],
    dtype=np.float64,
)
_prof_masks = []
for _name in PROFESSION_NAMES:
    _prof = PROFESSIONS.get(_name, PROFESSIONS["laborer"])
    _mask = 0
    for _kind in _prof.workplace_kinds:
        _code = LOCATION_TO_CODE.get(_kind)
        if _code is not None:
            _mask |= 1 << _code
    _prof_masks.append(_mask)
_PROF_WORK_MASK = np.array(_prof_masks, dtype=np.uint64)


def _create_segment(size):
    return shared_memory.SharedMemory(create=True, size=max(1, int(size)))


class SharedSoAAgentState:
    def __init__(self, capacity, *, descriptor=None):
        if descriptor is None:
            self.capacity = max(1, int(capacity))
            self._ints_shm = _create_segment(INT_FIELDS * self.capacity * np.dtype(np.int32).itemsize)
            self._floats_shm = _create_segment(FLOAT_FIELDS * self.capacity * np.dtype(np.float64).itemsize)
            self._flags_shm = _create_segment(FLAG_FIELDS * self.capacity * np.dtype(np.uint8).itemsize)
            self._owner = True
        else:
            self.capacity = int(descriptor["capacity"])
            self._ints_shm = shared_memory.SharedMemory(name=descriptor["ints"])
            self._floats_shm = shared_memory.SharedMemory(name=descriptor["floats"])
            self._flags_shm = shared_memory.SharedMemory(name=descriptor["flags"])
            self._owner = False
        self.ints = np.ndarray((INT_FIELDS, self.capacity), dtype=np.int32, buffer=self._ints_shm.buf)
        self.floats = np.ndarray((FLOAT_FIELDS, self.capacity), dtype=np.float64, buffer=self._floats_shm.buf)
        self.flags = np.ndarray((FLAG_FIELDS, self.capacity), dtype=np.uint8, buffer=self._flags_shm.buf)
        if self._owner:
            self.ints.fill(0)
            self.floats.fill(0.0)
            self.flags.fill(0)

    @classmethod
    def attach(cls, descriptor):
        return cls(1, descriptor=descriptor)

    @property
    def descriptor(self):
        return {"capacity": self.capacity, "ints": self._ints_shm.name, "floats": self._floats_shm.name, "flags": self._flags_shm.name}

    @property
    def allocated_bytes(self):
        return self._ints_shm.size + self._floats_shm.size + self._flags_shm.size

    def write_person(self, person):
        pid = int(person.id)
        if pid < 0 or pid >= self.capacity:
            return
        self.ints[I_LID, pid] = int(person.location_id)
        self.ints[I_FOOD, pid] = int(person.food)
        self.ints[I_MED, pid] = int(person.medicine)
        self.ints[I_ENERGY, pid] = int(person.energy)
        self.ints[I_HEALTH, pid] = int(person.health)
        self.ints[I_SHELTER, pid] = int(person.shelter)
        self.ints[I_EMPLOYER, pid] = -1 if person.employer_id is None else int(person.employer_id)
        self.ints[I_PROFESSION, pid] = int(PROFESSION_NAMES.index(person.profession)) if person.profession in PROFESSION_NAMES else 0
        self.ints[I_SOCIAL_CLASS, pid] = int(SOCIAL_CLASSES.index(person.social_class)) if person.social_class in SOCIAL_CLASSES else 0
        self.ints[I_DETAINED, pid] = int(person.detained_until_day)
        self.ints[I_WORK_EXP, pid] = int(person.work_experience)
        self.ints[I_SHORTAGES, pid] = int(person.shortage_experiences)
        self.ints[I_CRIME_SUFFERED, pid] = int(person.crime_suffered)
        self.ints[I_ARRESTS, pid] = int(person.arrests)
        self.ints[I_UNEMPLOYMENT, pid] = int(person.unemployment_days)
        self.ints[I_LIFETIME_UNEMPLOYMENT, pid] = int(person.lifetime_unemployment_days)
        self.ints[I_DAYS_IN_CLASS, pid] = int(person.days_in_class)
        self.ints[I_JOBS_HELD, pid] = int(person.jobs_held)
        self.floats[F_MONEY, pid] = float(person.money)
        self.floats[F_IDEOLOGY, pid] = float(person.ideology)
        self.floats[F_TAXES, pid] = float(person.taxes_paid)
        self.floats[F_WELFARE, pid] = float(person.welfare_received)
        self.floats[F_CAREER, pid] = float(person.career_progress)
        self.floats[F_LIFETIME_GROSS, pid] = float(person.lifetime_gross_income)
        self.floats[F_MARKET_SPENDING, pid] = float(person.market_spending)
        self.flags[B_ALIVE, pid] = int(bool(person.alive))
        self.flags[B_WORKING_AGE, pid] = int(bool(person.is_working_age))
        self.flags[B_DEPENDENT, pid] = int(bool(person.is_dependent))
        self.flags[B_ADULT, pid] = int(bool(person.is_adult))

    def sync_world(self, world):
        for person in world.people:
            self.write_person(person)

    def close(self, *, unlink=False):
        segments = (self._ints_shm, self._floats_shm, self._flags_shm)
        for segment in segments:
            segment.close()
        if unlink and self._owner:
            for segment in segments:
                try: segment.unlink()
                except FileNotFoundError: pass


class SharedRelationMemory:
    def __init__(self, capacity, *, descriptor=None):
        if descriptor is None:
            self.capacity = max(1, int(capacity))
            slots = self.capacity * MEMORY_CAP
            self._targets_shm = _create_segment(slots * np.dtype(np.int32).itemsize)
            self._trust_shm = _create_segment(slots * np.dtype(np.int16).itemsize)
            self._grievance_shm = _create_segment(slots * np.dtype(np.int16).itemsize)
            self._familiarity_shm = _create_segment(slots * np.dtype(np.uint16).itemsize)
            self._day_shm = _create_segment(slots * np.dtype(np.uint16).itemsize)
            self._owner = True
        else:
            self.capacity = int(descriptor["capacity"])
            self._targets_shm = shared_memory.SharedMemory(name=descriptor["targets"])
            self._trust_shm = shared_memory.SharedMemory(name=descriptor["trust"])
            self._grievance_shm = shared_memory.SharedMemory(name=descriptor["grievance"])
            self._familiarity_shm = shared_memory.SharedMemory(name=descriptor["familiarity"])
            self._day_shm = shared_memory.SharedMemory(name=descriptor["day"])
            self._owner = False
        shape = (self.capacity, MEMORY_CAP)
        self.targets = np.ndarray(shape, dtype=np.int32, buffer=self._targets_shm.buf)
        self.trust = np.ndarray(shape, dtype=np.int16, buffer=self._trust_shm.buf)
        self.grievance = np.ndarray(shape, dtype=np.int16, buffer=self._grievance_shm.buf)
        self.familiarity = np.ndarray(shape, dtype=np.uint16, buffer=self._familiarity_shm.buf)
        self.days = np.ndarray(shape, dtype=np.uint16, buffer=self._day_shm.buf)
        if self._owner:
            self.targets.fill(-1); self.trust.fill(0); self.grievance.fill(0); self.familiarity.fill(0); self.days.fill(0)

    @classmethod
    def attach(cls, descriptor): return cls(1, descriptor=descriptor)

    @property
    def descriptor(self):
        return {"capacity": self.capacity, "targets": self._targets_shm.name, "trust": self._trust_shm.name, "grievance": self._grievance_shm.name, "familiarity": self._familiarity_shm.name, "day": self._day_shm.name}

    @property
    def allocated_bytes(self):
        return sum(s.size for s in (self._targets_shm, self._trust_shm, self._grievance_shm, self._familiarity_shm, self._day_shm))

    def close(self, *, unlink=False):
        segments = (self._targets_shm, self._trust_shm, self._grievance_shm, self._familiarity_shm, self._day_shm)
        for segment in segments: segment.close()
        if unlink and self._owner:
            for segment in segments:
                try: segment.unlink()
                except FileNotFoundError: pass


class SharedCSRLocationIndex:
    def __init__(self, capacity, location_count, *, descriptor=None):
        if descriptor is None:
            self.capacity = max(1, int(capacity)); self.location_count = max(1, int(location_count))
            self._pids_shm = _create_segment(self.capacity * np.dtype(np.int32).itemsize)
            self._offsets_shm = _create_segment((self.location_count + 1) * np.dtype(np.int32).itemsize)
            self._owner = True
        else:
            self.capacity = int(descriptor["capacity"]); self.location_count = int(descriptor["location_count"])
            self._pids_shm = shared_memory.SharedMemory(name=descriptor["pids"]); self._offsets_shm = shared_memory.SharedMemory(name=descriptor["offsets"]); self._owner = False
        self.pids = np.ndarray((self.capacity,), dtype=np.int32, buffer=self._pids_shm.buf)
        self.offsets = np.ndarray((self.location_count + 1,), dtype=np.int32, buffer=self._offsets_shm.buf)
        if self._owner: self.pids.fill(-1); self.offsets.fill(0)

    @classmethod
    def attach(cls, descriptor): return cls(1, 1, descriptor=descriptor)

    @property
    def descriptor(self): return {"capacity": self.capacity, "location_count": self.location_count, "pids": self._pids_shm.name, "offsets": self._offsets_shm.name}

    @property
    def allocated_bytes(self): return self._pids_shm.size + self._offsets_shm.size

    def close(self, *, unlink=False):
        segments = (self._pids_shm, self._offsets_shm)
        for segment in segments: segment.close()
        if unlink and self._owner:
            for segment in segments:
                try: segment.unlink()
                except FileNotFoundError: pass


@njit(cache=True, nogil=True)
def _build_csr_index_nb(ints, flags, population_size, location_count, out_pids, offsets):
    counts = np.zeros(location_count, dtype=np.int32); alive = 0
    for pid in range(population_size):
        if flags[B_ALIVE, pid] == 0: continue
        lid = ints[I_LID, pid]
        if 0 <= lid < location_count: counts[lid] += 1; alive += 1
    offsets[0] = 0
    for lid in range(location_count): offsets[lid + 1] = offsets[lid] + counts[lid]
    cursor = offsets[:-1].copy()
    for pid in range(population_size):
        if flags[B_ALIVE, pid] == 0: continue
        lid = ints[I_LID, pid]
        if 0 <= lid < location_count:
            pos = cursor[lid]; out_pids[pos] = pid; cursor[lid] = pos + 1
    return alive


@njit(cache=True, nogil=True, inline="always")
def _gcd_nb(a, b):
    while b: a, b = b, a % b
    return a

@njit(cache=True, nogil=True, inline="always")
def _coprime_step_nb(size, raw):
    if size <= 1: return 1
    step = 1 + int(raw % np.uint64(size - 1))
    while _gcd_nb(step, size) != 1:
        step += 1
        if step >= size: step = 1
    return step

@njit(cache=True, nogil=True, inline="always")
def _next_u64(seed): seed = _splitmix64_nb(seed); return seed, seed

@njit(cache=True, nogil=True, inline="always")
def _randint_nb(seed, low, high):
    seed, raw = _next_u64(seed); return seed, low + int(raw % np.uint64(high - low + 1))

@njit(cache=True, nogil=True, inline="always")
def _random_nb(seed): seed, raw = _next_u64(seed); return seed, float(raw) * _INV_U64


@njit(cache=True, nogil=True, inline="always")
def _memory_find_slot(targets, familiarity, days, actor, target):
    start = (int(target) * 2654435761) & (MEMORY_CAP - 1); empty = -1
    for k in range(MEMORY_CAP):
        slot = (start + k) & (MEMORY_CAP - 1); value = targets[actor, slot]
        if value == target: return slot
        if value == -1: empty = slot; break
    if empty >= 0: return -(empty + 2)
    best = 0; best_fam = int(familiarity[actor, 0]); best_day = int(days[actor, 0])
    for slot in range(1, MEMORY_CAP):
        fam = int(familiarity[actor, slot]); last = int(days[actor, slot])
        if fam < best_fam or (fam == best_fam and last < best_day): best = slot; best_fam = fam; best_day = last
    return -(best + 2)

@njit(cache=True, nogil=True, inline="always")
def _materialize_slot(targets, trust, grievance, familiarity, days, actor, target, day):
    marker = _memory_find_slot(targets, familiarity, days, actor, target)
    if marker < 0: return -1, 0.0, 0.0, 0
    slot = marker; last = int(days[actor, slot]); t = int(trust[actor, slot]); g = int(grievance[actor, slot])
    if day > last:
        elapsed = day - last; g = max(0, g - 28 * elapsed); decay = 7 * elapsed
        if t > 0: t = max(0, t - decay)
        elif t < 0: t = min(0, t + decay)
        trust[actor, slot] = t; grievance[actor, slot] = g; days[actor, slot] = np.uint16(day & 0xFFFF)
    return slot, t / MEMORY_SCALE, g / MEMORY_SCALE, int(familiarity[actor, slot])

@njit(cache=True, nogil=True)
def _memory_aggregate(targets, trust, grievance, familiarity, days, actor, day):
    pos = 0; hostile = 0; max_conflict = 0.0; affinity_sum = 0.0; known = 0; reconcile = (day % 7) == 0
    for slot in range(MEMORY_CAP):
        target = int(targets[actor, slot])
        if target < 0: continue
        t = int(trust[actor, slot]); g = int(grievance[actor, slot])
        if reconcile:
            last = int(days[actor, slot])
            if day > last:
                elapsed = day - last; g = max(0, g - 28 * elapsed); decay = 7 * elapsed
                if t > 0: t = max(0, t - decay)
                elif t < 0: t = min(0, t + decay)
                trust[actor, slot] = t; grievance[actor, slot] = g; days[actor, slot] = np.uint16(day & 0xFFFF)
        tf = t / MEMORY_SCALE; gf = g / MEMORY_SCALE; affinity = max(-100.0, min(100.0, tf - gf)); conflict = max(0.0, gf - min(0.0, tf))
        affinity_sum += affinity; known += 1
        if affinity >= 15.0: pos += 1
        if conflict >= 20.0: hostile += 1
        if conflict > max_conflict: max_conflict = conflict
    return pos, hostile, max_conflict, affinity_sum / known if known else 0.0

@njit(cache=True, nogil=True)
def _remember_actor_nb(targets, trust, grievance, familiarity, days, actor, target, action_code, magnitude, day):
    marker = _memory_find_slot(targets, familiarity, days, actor, target)
    if marker >= 0:
        slot = marker; _slot, tf, gf, fam = _materialize_slot(targets, trust, grievance, familiarity, days, actor, target, day); t = int(round(tf * MEMORY_SCALE)); g = int(round(gf * MEMORY_SCALE))
    else:
        slot = -marker - 2; targets[actor, slot] = target; t = 0; g = 0; fam = 0
    fam = min(65535, fam + 1)
    if action_code == A_HELP: t += int(round(4.0 * magnitude * MEMORY_SCALE))
    elif action_code == A_STEAL: t -= int(round(2.0 * magnitude * MEMORY_SCALE)); g += int(round(1.0 * magnitude * MEMORY_SCALE))
    else: t -= int(round(3.0 * magnitude * MEMORY_SCALE)); g += int(round(2.0 * magnitude * MEMORY_SCALE))
    trust[actor, slot] = np.int16(max(-MEMORY_CLAMP, min(MEMORY_CLAMP, t))); grievance[actor, slot] = np.int16(max(0, min(MEMORY_CLAMP, g))); familiarity[actor, slot] = np.uint16(fam); days[actor, slot] = np.uint16(day & 0xFFFF)

@njit(cache=True, nogil=True, inline="always")
def _candidate_weight(actor, target, mode, day, ints, floats, flags, mem_targets, mem_trust, mem_grievance, mem_familiarity, mem_days):
    slot, tf, gf, fam = _materialize_slot(mem_targets, mem_trust, mem_grievance, mem_familiarity, mem_days, actor, target, day); has_memory = slot >= 0
    affinity = max(-100.0, min(100.0, tf - gf)); conflict = max(0.0, gf - min(0.0, tf))
    if mode == A_HELP: return 1.0 if not has_memory else max(0.1, 1.0 + max(0.0, affinity) / 10.0 + fam / 20.0)
    if mode == A_ATTACK: return 1.0 if not has_memory else max(0.1, 1.0 + conflict / 5.0)
    wealth = ints[I_FOOD, target] + ints[I_MED, target] * 2 + max(0.0, floats[F_MONEY, target]) / 4.0
    value = 1.0 + wealth / 25.0 if not has_memory else 1.0 + conflict / 18.0 + max(0.0, -affinity) / 30.0 + wealth / 25.0
    return max(0.1, value)

@njit(cache=True, nogil=True)
def _pick_target_nb(pids, actor, lid, mode, encounter_sample, seed, day, ints, floats, flags, mem_targets, mem_trust, mem_grievance, mem_familiarity, mem_days):
    size = pids.shape[0]
    if size <= 1: return seed, -1
    want = min(MAX_ENCOUNTER_SAMPLE, max(1, encounter_sample), size - 1)
    seed, raw = _next_u64(seed); start = np.int64(raw % np.uint64(size)); seed, raw = _next_u64(seed); step = np.int64(_coprime_step_nb(size, raw))
    total = 0.0; seen = 0; index = start
    for _ in range(size):
        target = int(pids[index]); index = (index + step) % size
        if target == actor or flags[B_ALIVE, target] == 0 or ints[I_LID, target] != lid: continue
        total += _candidate_weight(actor, target, mode, day, ints, floats, flags, mem_targets, mem_trust, mem_grievance, mem_familiarity, mem_days); seen += 1
        if seen >= want: break
    if total <= 0.0 or seen == 0: return seed, -1
    seed, rnd = _random_nb(seed); needle = rnd * total; upto = 0.0; seen2 = 0; index = start; fallback = -1
    for _ in range(size):
        target = int(pids[index]); index = (index + step) % size
        if target == actor or flags[B_ALIVE, target] == 0 or ints[I_LID, target] != lid: continue
        fallback = target; upto += _candidate_weight(actor, target, mode, day, ints, floats, flags, mem_targets, mem_trust, mem_grievance, mem_familiarity, mem_days)
        if needle <= upto: return seed, target
        seen2 += 1
        if seen2 >= seen: break
    return seed, fallback

@njit(cache=True, nogil=True)
def _count_witnesses_nb(pids, actor, target, lid, max_witnesses, seed, ints, flags):
    if max_witnesses <= 0 or pids.shape[0] <= 2: return seed, 0
    want = min(max_witnesses, pids.shape[0] - 2); seed, raw = _next_u64(seed); start = np.int64(raw % np.uint64(pids.shape[0])); seed, raw = _next_u64(seed); step = np.int64(_coprime_step_nb(pids.shape[0], raw)); count = 0; index = start
    for _ in range(pids.shape[0]):
        pid = int(pids[index]); index = (index + step) % pids.shape[0]
        if pid == actor or pid == target: continue
        if flags[B_ALIVE, pid] and ints[I_LID, pid] == lid:
            count += 1
            if count >= want: break
    return seed, count

@njit(cache=True, nogil=True)
def _hire_nb(pid, pcode, local_eids, emp_i, emp_f, emp_pref, seed, ints):
    total = 0.0
    for j in range(local_eids.shape[0]):
        eid = int(local_eids[j])
        if emp_i[EI_ALIVE, eid] == 0 or emp_i[EI_EMPLOYEES, eid] >= emp_i[EI_CAPACITY, eid]: continue
        preferred = 1.35 if (int(emp_pref[eid]) & (1 << pcode)) else 0.85; total += max(0.1, preferred * emp_f[EF_BASE_WAGE, eid])
    if total <= 0.0: return seed, -1
    seed, rnd = _random_nb(seed); needle = rnd * total; upto = 0.0; chosen = -1
    for j in range(local_eids.shape[0]):
        eid = int(local_eids[j])
        if emp_i[EI_ALIVE, eid] == 0 or emp_i[EI_EMPLOYEES, eid] >= emp_i[EI_CAPACITY, eid]: continue
        preferred = 1.35 if (int(emp_pref[eid]) & (1 << pcode)) else 0.85; upto += max(0.1, preferred * emp_f[EF_BASE_WAGE, eid]); chosen = eid
        if needle <= upto: break
    if chosen >= 0: emp_i[EI_EMPLOYEES, chosen] += 1; ints[I_EMPLOYER, pid] = chosen; ints[I_UNEMPLOYMENT, pid] = 0; ints[I_JOBS_HELD, pid] += 1
    return seed, chosen

@njit(cache=True, nogil=True)
def _consume_nb(good, quantity, price, supplier_eids, stock, reserve, current_lid, emp_i, emp_f):
    remaining = float(quantity); sold = 0.0; take = min(max(0.0, reserve), remaining); reserve -= take; remaining -= take; sold += take
    for j in range(supplier_eids.shape[0]):
        if remaining <= 0.0: break
        eid = int(supplier_eids[j])
        if eid < 0 or eid >= stock.shape[0]: continue
        available = max(0.0, stock[eid]); take = min(available, remaining)
        if take <= 0.0: continue
        stock[eid] = available - take; remaining -= take; sold += take
        if emp_i[EI_LOCATION, eid] == current_lid:
            revenue = take * price; emp_f[EF_CASH, eid] += revenue; emp_f[EF_REVENUE, eid] += revenue
    return sold, reserve


@njit(cache=True, nogil=True)
def _run_location_day_nb(pids, day, actions_per_day, master_seed, lid, ints, floats, flags, mem_targets, mem_trust, mem_grievance, mem_familiarity, mem_days, loc_i, loc_f, neighbors, neighbor_counts, emp_i, emp_f, emp_pref, local_eids, food_supplier_eids, med_supplier_eids, stock_food, stock_med, reserve_food, reserve_med, tax_rate, government_left, encounter_sample, max_witnesses, visibility, officers, fuse_eod):
    action_counts = np.zeros(ACTION_COUNT, dtype=np.int64); counters = np.zeros(COUNTER_COUNT, dtype=np.int64); treasury_delta = 0.0; demand_food = 0.0; demand_med = 0.0; sold_food = 0.0; sold_med = 0.0; police_incidents = 0; police_responses = 0; police_arrests = 0
    size = pids.shape[0]; perm_seed = _seed_nb(master_seed, day, lid, _PHASE_ORDER, 0); start = np.int64(perm_seed % np.uint64(max(1, size))) if size else np.int64(0); step = np.int64(_coprime_step_nb(size, _splitmix64_nb(perm_seed))) if size > 1 else np.int64(1)
    local_kind = int(loc_i[LI_KIND, lid]); action_loc_code = 1 if local_kind == INDUSTRIAL_CODE else (2 if local_kind == OUTSKIRTS_CODE else 0)
    for order_pos in range(size):
        pid = int(pids[(start + order_pos * step) % size])
        if flags[B_ALIVE, pid] == 0 or flags[B_ADULT, pid] == 0 or ints[I_LID, pid] != lid or day < ints[I_DETAINED, pid]: continue
        positive, hostile, max_conflict, mean_affinity = _memory_aggregate(mem_targets, mem_trust, mem_grievance, mem_familiarity, mem_days, pid, day); pending_move = -1
        for round_index in range(actions_per_day):
            if flags[B_ALIVE, pid] == 0 or day < ints[I_DETAINED, pid]: break
            code = int(_weighted_code_nb(pid, ints[I_FOOD, pid], ints[I_MED, pid], ints[I_ENERGY, pid], ints[I_HEALTH, pid], ints[I_SHELTER, pid], floats[F_MONEY, pid], ints[I_EMPLOYER, pid] >= 0, action_loc_code, positive, hostile, max_conflict, mean_affinity, np.uint64(master_seed), day, round_index))
            if 0 <= code < ACTION_COUNT: action_counts[code] += 1
            if flags[B_WORKING_AGE, pid] == 0 and code not in (A_SCAVENGE, A_BUY, A_REST, A_HEAL, A_REPAIR, A_HELP): code = A_REST
            if code == A_IDLE: continue
            if code in (A_REST, A_HEAL, A_REPAIR, A_SCAVENGE):
                seed = _seed_nb(master_seed, day, pid, _PHASE_SAFE, round_index)
                if code == A_REST:
                    seed, gain = _randint_nb(seed, 12, 24); ints[I_ENERGY, pid] = min(100, ints[I_ENERGY, pid] + gain); seed, heal = _randint_nb(seed, 0, 2); ints[I_HEALTH, pid] = min(100, ints[I_HEALTH, pid] + heal)
                elif code == A_HEAL:
                    if ints[I_MED, pid] > 0 and ints[I_HEALTH, pid] < 100:
                        ints[I_MED, pid] -= 1; seed, heal = _randint_nb(seed, 8, 18); ints[I_HEALTH, pid] = min(100, ints[I_HEALTH, pid] + heal)
                elif code == A_REPAIR:
                    if floats[F_MONEY, pid] >= 3.0 and ints[I_SHELTER, pid] < 100:
                        floats[F_MONEY, pid] -= 3.0; seed, gain = _randint_nb(seed, 8, 16); ints[I_SHELTER, pid] = min(100, ints[I_SHELTER, pid] + gain)
                else:
                    seed, cost = _randint_nb(seed, 4, 9); ints[I_ENERGY, pid] = max(0, ints[I_ENERGY, pid] - cost); seed, found = _randint_nb(seed, 0, max(0, int(loc_i[LI_SCAVENGE_MAX, lid]))); ints[I_FOOD, pid] += found; seed, rnd = _random_nb(seed); ints[I_MED, pid] += int(rnd < loc_f[LF_MED_CHANCE, lid])
                continue
            if code == A_WORK:
                seed = _seed_nb(master_seed, day, pid, _PHASE_WORK, round_index)
                if ints[I_ENERGY, pid] < 8:
                    seed, gain = _randint_nb(seed, 12, 24); ints[I_ENERGY, pid] = min(100, ints[I_ENERGY, pid] + gain); seed, heal = _randint_nb(seed, 0, 2); ints[I_HEALTH, pid] = min(100, ints[I_HEALTH, pid] + heal); continue
                pcode = int(ints[I_PROFESSION, pid]); pcode = pcode if 0 <= pcode < _PROF_INCOME.shape[0] else 0; eid = int(ints[I_EMPLOYER, pid])
                if eid < 0 or eid >= emp_i.shape[1] or emp_i[EI_ALIVE, eid] == 0:
                    ints[I_EMPLOYER, pid] = -1; seed, eid = _hire_nb(pid, pcode, local_eids, emp_i, emp_f, emp_pref, seed, ints)
                    if eid < 0: continue
                if emp_i[EI_LOCATION, eid] != lid: continue
                fit = 1.15 if (int(_PROF_WORK_MASK[pcode]) & (1 << local_kind)) else 0.82; vacancies = max(0, int(emp_i[EI_CAPACITY, eid] - emp_i[EI_EMPLOYEES, eid])); scarcity = 1.10 if vacancies > max(1, int(emp_i[EI_CAPACITY, eid] // 3)) else 1.0; preferred = 1.08 if (int(emp_pref[eid]) & (1 << pcode)) else 0.92; gross = max(1, int(round(emp_f[EF_BASE_WAGE, eid] * _PROF_INCOME[pcode] * fit * scarcity * preferred)))
                if emp_f[EF_CASH, eid] < gross: emp_i[EI_EMPLOYEES, eid] = max(0, emp_i[EI_EMPLOYEES, eid] - 1); ints[I_EMPLOYER, pid] = -1; continue
                emp_f[EF_CASH, eid] -= gross; emp_f[EF_PAYROLL, eid] += gross; output_code = int(emp_i[EI_OUTPUT, eid])
                if output_code != 0:
                    seed, rnd = _random_nb(seed); produced = max(0.0, emp_f[EF_OUTPUT_PER_SHIFT, eid] * emp_f[EF_PRODUCTIVITY, eid] * fit * (0.85 + rnd * 0.30)); emp_f[EF_UNITS, eid] += produced
                    if output_code == 1: stock_food[eid] += produced
                    else: stock_med[eid] += produced
                elif emp_i[EI_KIND, eid] != LOGISTICS_CODE:
                    seed, rnd = _random_nb(seed); service_revenue = gross * emp_f[EF_PRODUCTIVITY, eid] * (1.12 + rnd * 0.33); emp_f[EF_CASH, eid] += service_revenue; emp_f[EF_REVENUE, eid] += service_revenue
                floats[F_MONEY, pid] += gross; floats[F_LIFETIME_GROSS, pid] += gross; ints[I_WORK_EXP, pid] += 1; floats[F_CAREER, pid] += _PROF_ADVANCE[pcode] * fit; seed, raw_cost = _randint_nb(seed, 6, 12); energy_cost = max(3, int(round(raw_cost * _PROF_ENERGY[pcode]))); rate = tax_rate; class_code = int(ints[I_SOCIAL_CLASS, pid]); rate += 0.05 if government_left and class_code >= 3 else 0.0; tax = min(floats[F_MONEY, pid], max(0, int(round(gross * rate)))); floats[F_MONEY, pid] -= tax; floats[F_TAXES, pid] += tax; floats[F_IDEOLOGY, pid] = max(-1.0, min(1.0, floats[F_IDEOLOGY, pid] + min(0.0025, tax * 0.00035))); treasury_delta += tax; ints[I_ENERGY, pid] = max(0, ints[I_ENERGY, pid] - energy_cost); continue
            if code == A_BUY:
                want_food = ints[I_FOOD, pid] <= 6; want_med = ints[I_MED, pid] <= 1
                if not want_food and not want_med: continue
                seed = _seed_nb(master_seed, day, pid, _PHASE_BUY, round_index); good = 1 if want_food else 2; requested = 3 if want_food else 1
                if want_food and want_med:
                    seed, pick = _randint_nb(seed, 0, 1)
                    if pick == 1: good = 2; requested = 1
                if good == 1:
                    price = max(0.25, loc_f[LF_FOOD_PRICE, lid]); total_stock = reserve_food
                    for sj in range(food_supplier_eids.shape[0]):
                        se = int(food_supplier_eids[sj]); total_stock += max(0.0, stock_food[se]) if 0 <= se < stock_food.shape[0] else 0.0
                    affordable = int(max(0.0, floats[F_MONEY, pid]) // price); quantity = min(requested, affordable, int(max(0.0, total_stock))); demand_food += requested
                    if quantity <= 0: ints[I_SHORTAGES, pid] += 1; floats[F_IDEOLOGY, pid] = max(-1.0, floats[F_IDEOLOGY, pid] - 0.00025); continue
                    sold, reserve_food = _consume_nb(1, quantity, price, food_supplier_eids, stock_food, reserve_food, lid, emp_i, emp_f); sold_food += sold; cost = round(sold * price, 2); floats[F_MONEY, pid] -= cost; floats[F_MARKET_SPENDING, pid] += cost; ints[I_FOOD, pid] += int(sold)
                else:
                    price = max(0.25, loc_f[LF_MED_PRICE, lid]); total_stock = reserve_med
                    for sj in range(med_supplier_eids.shape[0]):
                        se = int(med_supplier_eids[sj]); total_stock += max(0.0, stock_med[se]) if 0 <= se < stock_med.shape[0] else 0.0
                    affordable = int(max(0.0, floats[F_MONEY, pid]) // price); quantity = min(requested, affordable, int(max(0.0, total_stock))); demand_med += requested
                    if quantity <= 0: ints[I_SHORTAGES, pid] += 1; floats[F_IDEOLOGY, pid] = max(-1.0, floats[F_IDEOLOGY, pid] - 0.00025); continue
                    sold, reserve_med = _consume_nb(2, quantity, price, med_supplier_eids, stock_med, reserve_med, lid, emp_i, emp_f); sold_med += sold; cost = round(sold * price, 2); floats[F_MONEY, pid] -= cost; floats[F_MARKET_SPENDING, pid] += cost; ints[I_MED, pid] += int(sold)
                if sold + 1e-9 < requested: ints[I_SHORTAGES, pid] += 1; floats[F_IDEOLOGY, pid] = max(-1.0, floats[F_IDEOLOGY, pid] - 0.00025)
                continue
            if code == A_MOVE:
                if ints[I_ENERGY, pid] < 4: continue
                ncount = int(neighbor_counts[lid])
                if ncount <= 0: continue
                seed = _seed_nb(master_seed, day, pid, _PHASE_MOVE, round_index); pcode = int(ints[I_PROFESSION, pid]); pcode = pcode if 0 <= pcode < _PROF_WORK_MASK.shape[0] else 0; eid = int(ints[I_EMPLOYER, pid]); total_weight = 0.0
                for nj in range(ncount):
                    dest = int(neighbors[lid, nj]); weight = 1.0
                    if 0 <= eid < emp_i.shape[1] and emp_i[EI_ALIVE, eid] and emp_i[EI_LOCATION, eid] == dest: weight *= 5.0
                    if int(_PROF_WORK_MASK[pcode]) & (1 << int(loc_i[LI_KIND, dest])): weight *= 2.2
                    if eid < 0 and loc_i[LI_VACANCIES, dest] > 0: weight *= 1.8
                    if ints[I_FOOD, pid] <= 3 and loc_f[LF_FOOD_STOCK, dest] > 0: weight *= 2.5
                    if ints[I_MED, pid] == 0 and loc_i[LI_KIND, dest] == CLINIC_CODE: weight *= 3.0
                    total_weight += weight
                if total_weight <= 0: continue
                seed, rnd = _random_nb(seed); needle = rnd * total_weight; upto = 0.0; destination = -1
                for nj in range(ncount):
                    dest = int(neighbors[lid, nj]); weight = 1.0
                    if 0 <= eid < emp_i.shape[1] and emp_i[EI_ALIVE, eid] and emp_i[EI_LOCATION, eid] == dest: weight *= 5.0
                    if int(_PROF_WORK_MASK[pcode]) & (1 << int(loc_i[LI_KIND, dest])): weight *= 2.2
                    if eid < 0 and loc_i[LI_VACANCIES, dest] > 0: weight *= 1.8
                    if ints[I_FOOD, pid] <= 3 and loc_f[LF_FOOD_STOCK, dest] > 0: weight *= 2.5
                    if ints[I_MED, pid] == 0 and loc_i[LI_KIND, dest] == CLINIC_CODE: weight *= 3.0
                    upto += weight; destination = dest
                    if needle <= upto: break
                if destination >= 0 and destination != lid:
                    seed, cost = _randint_nb(seed, 3, 7); ints[I_ENERGY, pid] = max(0, ints[I_ENERGY, pid] - cost)
                    if 0 <= eid < emp_i.shape[1] and emp_i[EI_ALIVE, eid] and emp_i[EI_LOCATION, eid] != destination: emp_i[EI_EMPLOYEES, eid] = max(0, emp_i[EI_EMPLOYEES, eid] - 1); ints[I_EMPLOYER, pid] = -1
                    pending_move = destination; counters[C_MOVES] += 1
                continue
            seed = _seed_nb(master_seed, day, pid, _PHASE_SOCIAL, round_index); seed, target = _pick_target_nb(pids, pid, lid, code, encounter_sample, seed, day, ints, floats, flags, mem_targets, mem_trust, mem_grievance, mem_familiarity, mem_days)
            if target < 0: continue
            seed, vis = _random_nb(seed); witnesses = 0
            if max_witnesses > 0 and vis <= visibility: seed, witnesses = _count_witnesses_nb(pids, pid, target, lid, max_witnesses, seed, ints, flags)
            counters[C_OBSERVATIONS] += witnesses; magnitude = 0.0
            if code == A_HELP:
                amount = 0
                if ints[I_HEALTH, target] < 70 and ints[I_MED, pid] > 0: ints[I_MED, pid] -= 1; ints[I_MED, target] += 1; amount = 1
                elif ints[I_FOOD, pid] > 2: seed, amount = _randint_nb(seed, 1, min(2, ints[I_FOOD, pid] - 1)); ints[I_FOOD, pid] -= amount; ints[I_FOOD, target] += amount
                if amount > 0: counters[C_HELPS] += 1; magnitude = float(amount)
            elif code == A_STEAL:
                seed, rnd = _random_nb(seed); amount = 0.0
                if rnd < 0.45:
                    option_count = (4 if ints[I_FOOD, target] > 0 else 0) + (2 if floats[F_MONEY, target] > 0 else 0) + (1 if ints[I_MED, target] > 0 else 0)
                    if option_count > 0:
                        seed, pick = _randint_nb(seed, 0, option_count - 1); cursor = 0; resource = 0
                        if ints[I_FOOD, target] > 0:
                            if pick < cursor + 4: resource = 1
                            cursor += 4
                        if resource == 0 and floats[F_MONEY, target] > 0:
                            if pick < cursor + 2: resource = 2
                            cursor += 2
                        if resource == 0 and ints[I_MED, target] > 0: resource = 3
                        if resource == 1: seed, take = _randint_nb(seed, 1, 3); take = min(take, ints[I_FOOD, target]); ints[I_FOOD, target] -= take; ints[I_FOOD, pid] += take; amount = float(take)
                        elif resource == 2: seed, take = _randint_nb(seed, 1, 5); cash_take = min(float(take), floats[F_MONEY, target]); floats[F_MONEY, target] -= cash_take; floats[F_MONEY, pid] += cash_take; amount = cash_take
                        else: ints[I_MED, target] -= 1; ints[I_MED, pid] += 1; amount = 1.0
                if amount > 0: counters[C_THEFTS] += 1; counters[C_CRIMES] += 1; ints[I_CRIME_SUFFERED, target] += 1; magnitude = max(1.0, amount / 2.0)
            else:
                seed, damage = _randint_nb(seed, 5, 20); seed, energy_cost = _randint_nb(seed, 4, 9); ints[I_ENERGY, pid] = max(0, ints[I_ENERGY, pid] - energy_cost); ints[I_HEALTH, target] -= damage; ints[I_CRIME_SUFFERED, target] += 1; counters[C_ATTACKS] += 1; counters[C_CRIMES] += 1; magnitude = damage / 10.0
                if ints[I_HEALTH, target] <= 0: flags[B_ALIVE, target] = 0
            if magnitude > 0.0:
                if code in (A_STEAL, A_ATTACK):
                    police_incidents += 1; population = max(1, int(loc_i[LI_POPULATION, lid])); officers_per_1000 = officers * 1000.0 / population; load = police_incidents / max(1, officers); coverage = max(0.02, min(0.92, 0.22 + officers_per_1000 * 0.12 - load * 0.035)); severity = 1.15 if code == A_ATTACK else 0.90; probability = min(0.97, coverage * severity); pseed = _seed_nb(master_seed, day, pid, _PHASE_POLICE, round_index); pseed, prnd = _random_nb(pseed)
                    if prnd < probability:
                        police_responses += 1; arrest_probability = min(0.90, 0.42 + 0.12 * magnitude + (0.12 if code == A_ATTACK else 0.0)); pseed, arnd = _random_nb(pseed)
                        if arnd < arrest_probability:
                            police_arrests += 1; counters[C_ARRESTS] += 1; max_detention = 8 if code == A_ATTACK else 5; pseed, detention = _randint_nb(pseed, 2, max_detention); ints[I_DETAINED, pid] = max(ints[I_DETAINED, pid], day + detention); pseed, fine_base = _randint_nb(pseed, 1, 5); fine = min(floats[F_MONEY, pid], float(fine_base + (2 if code == A_ATTACK else 0))); floats[F_MONEY, pid] -= fine; ints[I_ARRESTS, pid] += 1
                _remember_actor_nb(mem_targets, mem_trust, mem_grievance, mem_familiarity, mem_days, pid, target, code, magnitude, day); positive, hostile, max_conflict, mean_affinity = _memory_aggregate(mem_targets, mem_trust, mem_grievance, mem_familiarity, mem_days, pid, day)
        if pending_move >= 0: ints[I_LID, pid] = pending_move
    alive = 0; food_sum = 0.0; money_sum = 0.0; med_sum = 0.0; energy_sum = 0.0; shelter_sum = 0.0; health_sum = 0.0; ideology_sum = 0.0; taxes_sum = 0.0; welfare_sum = 0.0; left_count = 0; right_count = 0; workforce = 0; employed = 0; deaths = 0
    loc_stats = np.zeros((loc_i.shape[1], 4), dtype=np.float64); social_stats = np.zeros((len(SOCIAL_CLASSES), 7), dtype=np.float64); employer_counts = np.zeros(emp_i.shape[1], dtype=np.int64)
    if fuse_eod:
        for idx in range(size):
            pid = int(pids[idx])
            if flags[B_ALIVE, pid] == 0: continue
            current_lid = int(ints[I_LID, pid]); current_lid = current_lid if 0 <= current_lid < loc_i.shape[1] else lid; history_sum = loc_f[LF_HISTORY_SUM, current_lid]; history_len = int(loc_f[LF_HISTORY_LEN, current_lid]); todays = counters[C_CRIMES] if current_lid == lid else 0; denom_days = min(30, history_len + 1); crime_rate = (history_sum + todays) / (max(1, int(loc_i[LI_POPULATION, current_lid])) * max(1, denom_days)); values = _eod_nb(pid, ints[I_FOOD, pid], ints[I_ENERGY, pid], ints[I_SHELTER, pid], ints[I_HEALTH, pid], floats[F_MONEY, pid], ints[I_UNEMPLOYMENT, pid], ints[I_EMPLOYER, pid] >= 0, flags[B_WORKING_AGE, pid] != 0, flags[B_DEPENDENT, pid] != 0, flags[B_ADULT, pid] != 0, int(loc_i[LI_SHELTER_DECAY, current_lid]), crime_rate, np.uint64(master_seed), day); ints[I_FOOD, pid] = int(values[0]); ints[I_ENERGY, pid] = int(values[1]); ints[I_SHELTER, pid] = int(values[2]); ints[I_HEALTH, pid] = int(values[3]); ints[I_UNEMPLOYMENT, pid] = int(values[4]); ints[I_LIFETIME_UNEMPLOYMENT, pid] += int(values[5]); floats[F_IDEOLOGY, pid] = max(-1.0, min(1.0, floats[F_IDEOLOGY, pid] + float(values[6]))); ints[I_DAYS_IN_CLASS, pid] += 1
            if ints[I_HEALTH, pid] <= 0: flags[B_ALIVE, pid] = 0; deaths += 1; continue
            alive += 1; food = float(ints[I_FOOD, pid]); money = floats[F_MONEY, pid]; med = float(ints[I_MED, pid]); energy = float(ints[I_ENERGY, pid]); shelter = float(ints[I_SHELTER, pid]); health = float(ints[I_HEALTH, pid]); ideology = floats[F_IDEOLOGY, pid]; food_sum += food; money_sum += money; med_sum += med; energy_sum += energy; shelter_sum += shelter; health_sum += health; ideology_sum += ideology; taxes_sum += floats[F_TAXES, pid]; welfare_sum += floats[F_WELFARE, pid]; left_count += int(ideology < 0); right_count += int(ideology >= 0); loc_stats[current_lid, 0] += 1; loc_stats[current_lid, 1] += food; loc_stats[current_lid, 2] += money; loc_stats[current_lid, 3] += health; class_code = int(ints[I_SOCIAL_CLASS, pid])
            if 0 <= class_code < social_stats.shape[0]: social_stats[class_code, 0] += 1; social_stats[class_code, 1] += money; social_stats[class_code, 2] += food; social_stats[class_code, 3] += shelter; social_stats[class_code, 4] += health; social_stats[class_code, 5] += ideology; social_stats[class_code, 6] += ints[I_WORK_EXP, pid]
            if flags[B_WORKING_AGE, pid]: workforce += 1; employed += int(ints[I_EMPLOYER, pid] >= 0)
            eid = int(ints[I_EMPLOYER, pid]); employer_counts[eid] += 1 if 0 <= eid < employer_counts.shape[0] else 0
    scalar_stats = np.array([alive, food_sum, money_sum, med_sum, energy_sum, shelter_sum, health_sum, ideology_sum, taxes_sum, welfare_sum, left_count, right_count, workforce, employed, deaths], dtype=np.float64); police = np.array([police_incidents, police_responses, police_arrests], dtype=np.int64); market = np.array([reserve_food, reserve_med, demand_food, demand_med, sold_food, sold_med], dtype=np.float64)
    return action_counts, counters, treasury_delta, police, market, scalar_stats, loc_stats, social_stats, employer_counts


def _location_packet_arrays(packet):
    rows = packet["locations"]; location_count = len(rows); max_neighbors = max((len(row[2]) for row in rows), default=0); loc_i = np.zeros((LOC_INT_FIELDS, location_count), dtype=np.int32); loc_f = np.zeros((LOC_FLOAT_FIELDS, location_count), dtype=np.float64); neighbors = np.full((location_count, max(1, max_neighbors)), -1, dtype=np.int32); neighbor_counts = np.zeros(location_count, dtype=np.int32); histories = packet.get("crime_history", {}); shelter_decay = packet.get("shelter_decay", {})
    for row in rows:
        lid, kind, nbrs, scavenge, med_chance, food_price, med_price, food_suppliers, med_suppliers, population = row; lid = int(lid); loc_i[LI_KIND, lid] = LOCATION_TO_CODE.get(kind, 0); loc_i[LI_POPULATION, lid] = int(population); loc_i[LI_SCAVENGE_MAX, lid] = int(scavenge); loc_i[LI_SHELTER_DECAY, lid] = int(shelter_decay.get(lid, 0)); loc_f[LF_FOOD_STOCK, lid] = float(sum(food_suppliers.values())); loc_f[LF_MED_STOCK, lid] = float(sum(med_suppliers.values())); loc_f[LF_FOOD_PRICE, lid] = float(food_price); loc_f[LF_MED_PRICE, lid] = float(med_price); loc_f[LF_MED_CHANCE, lid] = float(med_chance); hist = tuple(histories.get(lid, ())); loc_f[LF_HISTORY_SUM, lid] = float(sum(hist)); loc_f[LF_HISTORY_LEN, lid] = float(len(hist)); neighbor_counts[lid] = len(nbrs)
        for j, neighbor in enumerate(nbrs): neighbors[lid, j] = int(neighbor)
    return loc_i, loc_f, neighbors, neighbor_counts


def _employer_packet_arrays(packet, loc_i):
    rows = packet["employers"]; max_eid = max((int(row[0]) for row in rows), default=-1); count = max(1, max_eid + 1); emp_i = np.zeros((EMP_INT_FIELDS, count), dtype=np.int32); emp_f = np.zeros((EMP_FLOAT_FIELDS, count), dtype=np.float64); emp_pref = np.zeros(count, dtype=np.uint64); by_location = [[] for _ in range(loc_i.shape[1])]
    for row in rows:
        eid, lid, kind, capacity, employees, base_wage, cash, productivity, output_good, output_per_shift, preferred, payroll, revenue, units, alive = row; eid = int(eid); lid = int(lid); emp_i[EI_LOCATION, eid] = lid; emp_i[EI_KIND, eid] = LOCATION_TO_CODE.get(kind, 0); emp_i[EI_CAPACITY, eid] = int(capacity); emp_i[EI_EMPLOYEES, eid] = int(employees); emp_i[EI_OUTPUT, eid] = 1 if output_good == "food" else (2 if output_good == "medicine" else 0); emp_i[EI_ALIVE, eid] = int(bool(alive)); emp_f[EF_BASE_WAGE, eid] = float(base_wage); emp_f[EF_CASH, eid] = float(cash); emp_f[EF_PRODUCTIVITY, eid] = float(productivity); emp_f[EF_OUTPUT_PER_SHIFT, eid] = float(output_per_shift); emp_f[EF_PAYROLL, eid] = float(payroll); emp_f[EF_REVENUE, eid] = float(revenue); emp_f[EF_UNITS, eid] = float(units); mask = 0
        for name in preferred:
            if name in PROFESSION_NAMES: mask |= 1 << PROFESSION_NAMES.index(name)
        emp_pref[eid] = np.uint64(mask); by_location[lid].append(eid)
    for lid in range(loc_i.shape[1]): loc_i[LI_VACANCIES, lid] = sum(max(0, int(emp_i[EI_CAPACITY, eid] - emp_i[EI_EMPLOYEES, eid])) for eid in by_location[lid] if emp_i[EI_ALIVE, eid])
    return emp_i, emp_f, emp_pref, by_location


def _market_arrays(location_row, employer_count, local_eids, emp_i):
    food_dict = {int(k): float(v) for k, v in location_row[7].items()}; med_dict = {int(k): float(v) for k, v in location_row[8].items()}; stock_food = np.zeros(employer_count, dtype=np.float64); stock_med = np.zeros(employer_count, dtype=np.float64); reserve_food = float(food_dict.pop(-1, 0.0)); reserve_med = float(med_dict.pop(-1, 0.0)); food_keys = {k for k in food_dict if 0 <= k < employer_count}; med_keys = {k for k in med_dict if 0 <= k < employer_count}
    for raw_eid in local_eids:
        eid = int(raw_eid)
        if emp_i[EI_OUTPUT, eid] == 1: food_keys.add(eid)
        elif emp_i[EI_OUTPUT, eid] == 2: med_keys.add(eid)
    food_eids = np.array(sorted(food_keys), dtype=np.int32); med_eids = np.array(sorted(med_keys), dtype=np.int32)
    for eid, amount in food_dict.items():
        if 0 <= eid < employer_count: stock_food[eid] = amount
    for eid, amount in med_dict.items():
        if 0 <= eid < employer_count: stock_med[eid] = amount
    return stock_food, stock_med, reserve_food, reserve_med, food_eids, med_eids


def _stats_from_kernel(scalar, loc_stats, social_stats, employer_counts):
    return {"alive": int(scalar[0]), "food": float(scalar[1]), "money": float(scalar[2]), "medicine": float(scalar[3]), "energy": float(scalar[4]), "shelter": float(scalar[5]), "health": float(scalar[6]), "ideology": float(scalar[7]), "taxes": float(scalar[8]), "welfare": float(scalar[9]), "left": int(scalar[10]), "right": int(scalar[11]), "workforce": int(scalar[12]), "employed": int(scalar[13]), "deaths": int(scalar[14]), "locations": {lid: [int(loc_stats[lid, 0]), float(loc_stats[lid, 1]), float(loc_stats[lid, 2]), float(loc_stats[lid, 3])] for lid in range(loc_stats.shape[0]) if loc_stats[lid, 0] > 0}, "social": [list(map(float, social_stats[i])) for i in range(social_stats.shape[0])], "employer_counts": {eid: int(employer_counts[eid]) for eid in np.nonzero(employer_counts)[0]}}


def _merge_stats(target, source):
    if target is None: return source
    for key in ("alive", "left", "right", "workforce", "employed", "deaths"): target[key] += int(source.get(key, 0))
    for key in ("food", "money", "medicine", "energy", "shelter", "health", "ideology", "taxes", "welfare"): target[key] += float(source.get(key, 0.0))
    for lid, row in source.get("locations", {}).items():
        dst = target["locations"].setdefault(int(lid), [0, 0.0, 0.0, 0.0])
        for i in range(4): dst[i] += row[i]
    for i, row in enumerate(source.get("social", ())):
        dst = target["social"][i]
        for j in range(7): dst[j] += row[j]
    for eid, count in source.get("employer_counts", {}).items(): target["employer_counts"][int(eid)] = target["employer_counts"].get(int(eid), 0) + int(count)
    return target


def _worker_main(worker_id, owned_locations, input_queue, result_queue, master_seed, state_desc, memory_desc, index_desc):
    state = SharedSoAAgentState.attach(state_desc); memory = SharedRelationMemory.attach(memory_desc); index = SharedCSRLocationIndex.attach(index_desc)
    try:
        while True:
            task = input_queue.get()
            if task is None: return
            day, actions_per_day, encounter_sample, max_witnesses, visibility, packet, fuse_eod = task; started = perf_counter(); loc_i, loc_f, neighbors, neighbor_counts = _location_packet_arrays(packet); emp_i, emp_f, emp_pref, by_location = _employer_packet_arrays(packet, loc_i); government_id, tax_rate = packet["government"]; government_left = government_id == "left"; results = []; worker_stats = None; location_rows = {int(row[0]): row for row in packet["locations"]}
            for lid in owned_locations:
                start = int(index.offsets[lid]); end = int(index.offsets[lid + 1]); pids = index.pids[start:end]; location_row = location_rows[int(lid)]; local_eids = np.array(by_location[int(lid)], dtype=np.int32); stock_food, stock_med, reserve_food, reserve_med, food_eids, med_eids = _market_arrays(location_row, emp_i.shape[1], local_eids, emp_i)
                action_counts, counters, treasury_delta, police, market, scalar, loc_stats, social_stats, employer_counts = _run_location_day_nb(pids, int(day), int(actions_per_day), np.uint64(master_seed), int(lid), state.ints, state.floats, state.flags, memory.targets, memory.trust, memory.grievance, memory.familiarity, memory.days, loc_i, loc_f, neighbors, neighbor_counts, emp_i, emp_f, emp_pref, local_eids, food_eids, med_eids, stock_food, stock_med, float(reserve_food), float(reserve_med), float(tax_rate), bool(government_left), int(encounter_sample), int(max_witnesses), float(visibility), int(packet["police"].get(int(lid), 0)), bool(fuse_eod))
                food_suppliers = {-1: float(market[0])}; med_suppliers = {-1: float(market[1])}
                for eid in food_eids:
                    amount = float(stock_food[int(eid)])
                    if amount: food_suppliers[int(eid)] = amount
                for eid in med_eids:
                    amount = float(stock_med[int(eid)])
                    if amount: med_suppliers[int(eid)] = amount
                employer_rows = [(int(eid), int(emp_i[EI_LOCATION, int(eid)]), int(emp_i[EI_CAPACITY, int(eid)]), int(emp_i[EI_EMPLOYEES, int(eid)]), float(emp_f[EF_CASH, int(eid)]), float(emp_f[EF_PRODUCTIVITY, int(eid)]), float(emp_f[EF_PAYROLL, int(eid)]), float(emp_f[EF_REVENUE, int(eid)]), float(emp_f[EF_UNITS, int(eid)]), int(emp_i[EI_ALIVE, int(eid)])) for eid in local_eids]
                counter_dict = {"helps": int(counters[C_HELPS]), "thefts": int(counters[C_THEFTS]), "attacks": int(counters[C_ATTACKS]), "observations": int(counters[C_OBSERVATIONS]), "arrests": int(counters[C_ARRESTS]), "moves": int(counters[C_MOVES]), "crimes": int(counters[C_CRIMES])}; action_names = ("move", "work", "scavenge", "buy_supplies", "rest", "heal", "repair", "help", "steal", "attack", "idle")
                for code, name in enumerate(action_names):
                    if action_counts[code]: counter_dict[f"action_{name}"] = int(action_counts[code])
                results.append({"location": int(lid), "counters": counter_dict, "treasury_delta": float(treasury_delta), "employers": employer_rows, "market": (int(lid), {"food": float(loc_f[LF_FOOD_PRICE, lid]), "medicine": float(loc_f[LF_MED_PRICE, lid])}, food_suppliers, med_suppliers, {"food": float(market[2]), "medicine": float(market[3])}, {"food": float(market[4]), "medicine": float(market[5])}), "police": (int(lid), int(packet["police"].get(int(lid), 0)), int(police[0]), int(police[1]), int(police[2]))})
                if fuse_eod: worker_stats = _merge_stats(worker_stats, _stats_from_kernel(scalar, loc_stats, social_stats, employer_counts))
            if worker_stats is None: worker_stats = {"alive": 0, "food": 0.0, "money": 0.0, "medicine": 0.0, "energy": 0.0, "shelter": 0.0, "health": 0.0, "ideology": 0.0, "taxes": 0.0, "welfare": 0.0, "left": 0, "right": 0, "workforce": 0, "employed": 0, "locations": {}, "social": [[0.0] * 7 for _ in SOCIAL_CLASSES], "employer_counts": {}, "deaths": 0}
            result_queue.put((worker_id, perf_counter() - started, results, worker_stats))
    finally:
        index.close(); memory.close(); state.close()


class SoADomainPool:
    def __init__(self, master_seed, state, memory, index, location_count, workers=0):
        cpu_count = os.cpu_count() or 1; requested = max(0, int(workers)); self.worker_count = min(requested, cpu_count, max(1, int(location_count))) if requested else 0; self.master_seed = int(master_seed) & _MASK64; self.state = state; self.memory = memory; self.index = index; self.location_count = int(location_count); self.enabled = self.worker_count >= 2; self.started = False; self._ctx = None; self._queues = []; self._result_queue = None; self._processes = []; self._active_workers = []; self.stats = Counter(); self.worker_seconds = 0.0; self.dispatch_seconds = 0.0
    def _ensure_started(self):
        if self.started or not self.enabled: return
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"; self._ctx = mp.get_context(method); self._result_queue = self._ctx.Queue(); assignments = [[] for _ in range(self.worker_count)]
        for lid in range(self.location_count): assignments[lid % self.worker_count].append(lid)
        for worker_id, owned in enumerate(assignments):
            queue = self._ctx.Queue(maxsize=2); process = self._ctx.Process(target=_worker_main, args=(worker_id, tuple(owned), queue, self._result_queue, self.master_seed, self.state.descriptor, self.memory.descriptor, self.index.descriptor), name=f"stochastic-soa-domain-{worker_id}", daemon=True); process.start(); self._queues.append(queue); self._processes.append(process); self._active_workers.append(worker_id)
        self.started = True
    def rebuild_index(self, population_size): return int(_build_csr_index_nb(self.state.ints, self.state.flags, min(int(population_size), self.state.capacity), self.location_count, self.index.pids, self.index.offsets))
    def run_day(self, day, actions_per_day, encounter_sample, max_witnesses, visibility, packet, fuse_eod=True):
        self._ensure_started(); task = (int(day), int(actions_per_day), int(encounter_sample), int(max_witnesses), float(visibility), packet, bool(fuse_eod)); started = perf_counter()
        for worker_id in self._active_workers: self._queues[worker_id].put(task); self.stats["tasks"] += 1
        results = []; stats = None
        for _ in self._active_workers:
            _wid, seconds, payload, worker_stats = self._result_queue.get(); self.worker_seconds += seconds; results.extend(payload); stats = _merge_stats(stats, worker_stats)
        self.dispatch_seconds += perf_counter() - started; self.stats["days"] += 1; return results, stats
    def summary(self): return {"days": int(self.stats["days"]), "tasks": int(self.stats["tasks"]), "workers": len(self._active_workers), "worker_seconds": float(self.worker_seconds), "dispatch_seconds": float(self.dispatch_seconds), "shared_bytes": int(self.state.allocated_bytes + self.memory.allocated_bytes + self.index.allocated_bytes)}
    def close(self):
        if not self.started: return
        for worker_id in self._active_workers: self._queues[worker_id].put(None)
        for worker_id in self._active_workers:
            process = self._processes[worker_id]; process.join(timeout=5)
            if process.is_alive(): process.terminate(); process.join(timeout=1)
            self._queues[worker_id].close()
        if self._result_queue is not None: self._result_queue.close()
        self._active_workers.clear(); self.started = False


def warmup():
    dummy_state = SharedSoAAgentState(4); dummy_memory = SharedRelationMemory(4); dummy_index = SharedCSRLocationIndex(4, 1)
    try:
        dummy_state.flags[B_ALIVE, 0] = 1; dummy_state.flags[B_ADULT, 0] = 1; dummy_state.flags[B_WORKING_AGE, 0] = 1; dummy_state.ints[I_FOOD, 0] = 5; dummy_state.ints[I_MED, 0] = 1; dummy_state.ints[I_ENERGY, 0] = 80; dummy_state.ints[I_HEALTH, 0] = 90; dummy_state.ints[I_SHELTER, 0] = 70; dummy_state.ints[I_EMPLOYER, 0] = -1; _build_csr_index_nb(dummy_state.ints, dummy_state.flags, 1, 1, dummy_index.pids, dummy_index.offsets); loc_i = np.zeros((LOC_INT_FIELDS, 1), dtype=np.int32); loc_f = np.zeros((LOC_FLOAT_FIELDS, 1), dtype=np.float64); loc_i[LI_POPULATION, 0] = 1; loc_i[LI_SCAVENGE_MAX, 0] = 2; loc_f[LF_FOOD_PRICE, 0] = 2.0; loc_f[LF_MED_PRICE, 0] = 5.0; neighbors = np.full((1, 1), -1, dtype=np.int32); neighbor_counts = np.zeros(1, dtype=np.int32); emp_i = np.zeros((EMP_INT_FIELDS, 1), dtype=np.int32); emp_f = np.zeros((EMP_FLOAT_FIELDS, 1), dtype=np.float64); emp_pref = np.zeros(1, dtype=np.uint64); empty = np.zeros(0, dtype=np.int32); stock = np.zeros(1, dtype=np.float64); _run_location_day_nb(dummy_index.pids[:1], 1, 1, np.uint64(1), 0, dummy_state.ints, dummy_state.floats, dummy_state.flags, dummy_memory.targets, dummy_memory.trust, dummy_memory.grievance, dummy_memory.familiarity, dummy_memory.days, loc_i, loc_f, neighbors, neighbor_counts, emp_i, emp_f, emp_pref, empty, empty, empty, stock.copy(), stock.copy(), 10.0, 1.0, 0.2, False, 16, 3, 0.65, 1, True)
    finally:
        dummy_index.close(unlink=True); dummy_memory.close(unlink=True); dummy_state.close(unlink=True)
    return True

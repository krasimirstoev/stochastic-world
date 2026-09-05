"""Temporal-blocked domain kernel for very large aggressive simulations.

The domain state remains authoritative between cold barriers.  Workers execute
both the action phase and end-of-day lifecycle locally, and return only compact
location / economy / statistics aggregates.  Main therefore avoids daily
Person-object reconciliation and the second end-of-day marshalling pipeline.
"""

import multiprocessing as mp
import os
from collections import Counter, defaultdict
from time import perf_counter

from .aggressive_domain import (
    ADULT,
    ALIVE,
    ARRESTS,
    CAREER,
    CRIME_SUFFERED,
    DAYS_IN_CLASS,
    DEPENDENT,
    DETAINED,
    EMPLOYER,
    ENERGY,
    FOOD,
    HEALTH,
    IDEOLOGY,
    JOBS_HELD,
    LID,
    LIFETIME_GROSS,
    LIFETIME_UNEMPLOYMENT,
    MARKET_SPENDING,
    MED,
    MONEY,
    PROFESSION,
    SHELTER,
    SHORTAGES,
    SOCIAL_CLASS,
    TAXES,
    UNEMPLOYMENT,
    WELFARE,
    WORKING_AGE,
    WORK_EXP,
    SharedDomainAgentState,
    _AGENT,
    _market_from_location_row,
    _process_location,
    _stock,
)
from .aggressive_economy import SOCIAL_CLASSES
from .aggressive_social import _COUNT, _ROW, SharedSocialState
from .multiprocessing_engine import _end_of_day_delta


SOCIAL_CLASS_COUNT = len(SOCIAL_CLASSES)


def sync_social_from_domain(social, domain_state, day, population_size):
    """Rebuild social arenas directly from authoritative domain memory."""
    population_size = min(int(population_size), domain_state.capacity, social.population_capacity)
    social._counts.buf[:] = b"\x00" * social._counts.size
    social._eligible.buf[:population_size] = b"\x00" * population_size
    counts = [0] * social.location_count
    rows_buf = social._rows.buf
    eligible_buf = social._eligible.buf
    state_buf = domain_state._state.buf
    capacity = social.population_capacity
    eligible_count = 0

    for pid in range(population_size):
        values = _AGENT.unpack_from(state_buf, pid * _AGENT.size)
        if not values[ALIVE]:
            continue
        lid = int(values[LID])
        if lid < 0 or lid >= social.location_count:
            continue
        index = counts[lid]
        _ROW.pack_into(
            rows_buf,
            (lid * capacity + index) * _ROW.size,
            pid,
            int(values[FOOD]),
            int(values[MED]),
            float(values[MONEY]),
            int(values[HEALTH]),
        )
        counts[lid] = index + 1
        if values[ADULT] and int(day) >= int(values[DETAINED]):
            eligible_buf[pid] = 1
            eligible_count += 1

    for lid, count in enumerate(counts):
        _COUNT.pack_into(social._counts.buf, lid * _COUNT.size, int(count))
    return eligible_count


def _crime_rate(history, todays_crimes, population):
    values = list(history)
    values.append(int(todays_crimes))
    if len(values) > 30:
        values = values[-30:]
    return sum(values) / (max(1, int(population)) * max(1, len(values)))


def _blank_stats():
    return {
        "alive": 0,
        "food": 0.0,
        "money": 0.0,
        "medicine": 0.0,
        "energy": 0.0,
        "shelter": 0.0,
        "health": 0.0,
        "ideology": 0.0,
        "taxes": 0.0,
        "welfare": 0.0,
        "left": 0,
        "right": 0,
        "workforce": 0,
        "employed": 0,
        "locations": {},
        "social": [[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _ in range(SOCIAL_CLASS_COUNT)],
        "employer_counts": {},
        "deaths": 0,
    }


def _accumulate_state(stats, state):
    if not state[ALIVE]:
        return
    stats["alive"] += 1
    food = float(state[FOOD]); money = float(state[MONEY]); medicine = float(state[MED])
    energy = float(state[ENERGY]); shelter = float(state[SHELTER]); health = float(state[HEALTH])
    ideology = float(state[IDEOLOGY])
    stats["food"] += food; stats["money"] += money; stats["medicine"] += medicine
    stats["energy"] += energy; stats["shelter"] += shelter; stats["health"] += health
    stats["ideology"] += ideology; stats["taxes"] += float(state[TAXES]); stats["welfare"] += float(state[WELFARE])
    if ideology < 0: stats["left"] += 1
    else: stats["right"] += 1

    lid = int(state[LID])
    loc = stats["locations"].setdefault(lid, [0, 0.0, 0.0, 0.0])
    loc[0] += 1; loc[1] += food; loc[2] += money; loc[3] += health

    class_code = int(state[SOCIAL_CLASS])
    if 0 <= class_code < SOCIAL_CLASS_COUNT:
        bucket = stats["social"][class_code]
        bucket[0] += 1; bucket[1] += money; bucket[2] += food; bucket[3] += shelter
        bucket[4] += health; bucket[5] += ideology; bucket[6] += float(state[WORK_EXP])

    if state[WORKING_AGE]:
        stats["workforce"] += 1
        if int(state[EMPLOYER]) >= 0:
            stats["employed"] += 1

    employer_id = int(state[EMPLOYER])
    if employer_id >= 0:
        stats["employer_counts"][employer_id] = stats["employer_counts"].get(employer_id, 0) + 1


def _run_eod_for_source_location(
    source_lid,
    social,
    domain_state,
    master_seed,
    day,
    location_meta,
    crime_history,
    today_crimes,
):
    """Apply EOD exactly once to agents that started the day in source_lid."""
    stats = _blank_stats()
    size = social.count(source_lid)
    for index in range(size):
        pid = int(social.row(source_lid, index)[0])
        state = domain_state.read(pid)
        if not state[ALIVE]:
            continue

        current_lid = int(state[LID])
        meta = location_meta.get(current_lid) or location_meta[source_lid]
        history = crime_history.get(current_lid, ())
        crimes = int(today_crimes.get(current_lid, 0))
        rate = _crime_rate(history, crimes, meta["population"])
        snapshot = (
            pid,
            int(state[FOOD]),
            int(state[ENERGY]),
            int(state[SHELTER]),
            int(state[HEALTH]),
            float(state[MONEY]),
            int(state[UNEMPLOYMENT]),
            int(state[EMPLOYER]) >= 0,
            bool(state[WORKING_AGE]),
            bool(state[DEPENDENT]),
            bool(state[ADULT]),
            int(meta["shelter_decay_bonus"]),
            float(rate),
        )
        (
            _pid, food, energy, shelter, health, unemployment_days,
            lifetime_inc, ideology_shift, _damage, _causes,
        ) = _end_of_day_delta(snapshot, master_seed, day)
        state[FOOD] = int(food)
        state[ENERGY] = int(energy)
        state[SHELTER] = int(shelter)
        state[HEALTH] = int(health)
        state[UNEMPLOYMENT] = int(unemployment_days)
        state[LIFETIME_UNEMPLOYMENT] += int(lifetime_inc)
        state[DAYS_IN_CLASS] += 1
        if ideology_shift:
            state[IDEOLOGY] = max(-1.0, min(1.0, float(state[IDEOLOGY]) + float(ideology_shift)))
        if state[HEALTH] <= 0:
            state[ALIVE] = 0
            stats["deaths"] += 1
        domain_state.write(pid, state)
        _accumulate_state(stats, state)
    return stats


def _merge_stats(target, source):
    for key in ("alive", "left", "right", "workforce", "employed", "deaths"):
        target[key] += int(source.get(key, 0))
    for key in ("food", "money", "medicine", "energy", "shelter", "health", "ideology", "taxes", "welfare"):
        target[key] += float(source.get(key, 0.0))
    for lid, row in source.get("locations", {}).items():
        dst = target["locations"].setdefault(int(lid), [0, 0.0, 0.0, 0.0])
        for i in range(4): dst[i] += row[i]
    for i, row in enumerate(source.get("social", ())):
        dst = target["social"][i]
        for j in range(7): dst[j] += row[j]
    for eid, count in source.get("employer_counts", {}).items():
        target["employer_counts"][int(eid)] = target["employer_counts"].get(int(eid), 0) + int(count)


def _temporal_worker(
    worker_id,
    owned_locations,
    input_queue,
    result_queue,
    master_seed,
    domain_descriptor,
    social_descriptor,
):
    domain_state = SharedDomainAgentState.attach(domain_descriptor)
    social = SharedSocialState.attach(social_descriptor)
    social_cache = {}
    try:
        while True:
            task = input_queue.get()
            if task is None:
                return
            (
                day, actions_per_day, encounter_sample, max_witnesses,
                visibility, packet, fuse_eod,
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
            location_meta = {
                int(row[0]): {
                    "population": int(row[-1]),
                    "shelter_decay_bonus": int(packet["shelter_decay"].get(int(row[0]), 0)),
                }
                for row in packet["locations"]
            }

            results = []
            todays_crimes = {}
            for lid in owned_locations:
                location = all_locations.get(int(lid))
                if location is None:
                    continue
                result = _process_location(
                    location, all_locations, packet["employers"], domain_state,
                    social, social_cache, master_seed, int(day), int(actions_per_day),
                    int(encounter_sample), int(max_witnesses), float(visibility),
                    packet["government"], packet["police"].get(int(lid), 0),
                )
                todays_crimes[int(lid)] = int(result.get("counters", {}).get("crimes", 0))
                results.append(result)

            worker_stats = _blank_stats()
            if fuse_eod:
                # Include crimes from locations owned by other workers in the rate
                # approximation through the packet's previous history; today's
                # remote crime total is intentionally one barrier late.
                for lid in owned_locations:
                    partial = _run_eod_for_source_location(
                        int(lid), social, domain_state, master_seed, int(day),
                        location_meta, packet.get("crime_history", {}), todays_crimes,
                    )
                    _merge_stats(worker_stats, partial)

            result_queue.put((worker_id, perf_counter() - started, results, worker_stats))
    finally:
        social.close()
        domain_state.close()


class TemporalDomainPool:
    """Persistent domain workers with action+EOD temporal blocking."""

    def __init__(self, master_seed, domain_state, social_state, location_count, workers=0):
        cpu_count = os.cpu_count() or 1
        requested = max(0, int(workers))
        self.worker_count = min(requested, cpu_count) if requested else 0
        self.master_seed = int(master_seed)
        self.domain_state = domain_state
        self.social_state = social_state
        self.location_count = int(location_count)
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
                self._queues.append(None); self._processes.append(None); continue
            queue = self._ctx.Queue(maxsize=2)
            process = self._ctx.Process(
                target=_temporal_worker,
                args=(worker_id, tuple(owned), queue, self._result_queue,
                      self.master_seed, self.domain_state.descriptor,
                      self.social_state.descriptor),
                name=f"stochastic-temporal-domain-{worker_id}", daemon=True,
            )
            process.start()
            self._queues.append(queue); self._processes.append(process)
            self._active_workers.append(worker_id)
        self.started = True

    def run_day(self, day, actions_per_day, encounter_sample, max_witnesses,
                visibility, packet, fuse_eod=True):
        self._ensure_started()
        task = (int(day), int(actions_per_day), int(encounter_sample),
                int(max_witnesses), float(visibility), packet, bool(fuse_eod))
        started = perf_counter()
        for worker_id in self._active_workers:
            self._queues[worker_id].put(task)
            self.stats["tasks"] += 1
        results = []
        stats = _blank_stats()
        for _ in self._active_workers:
            _wid, seconds, payload, worker_stats = self._result_queue.get()
            self.worker_seconds += seconds
            results.extend(payload)
            _merge_stats(stats, worker_stats)
        self.dispatch_seconds += perf_counter() - started
        self.stats["days"] += 1
        return results, stats

    def summary(self):
        return {
            "days": int(self.stats["days"]), "tasks": int(self.stats["tasks"]),
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
                process.terminate(); process.join(timeout=1)
            self._queues[worker_id].close()
        if self._result_queue is not None:
            self._result_queue.close()
        self._active_workers.clear(); self.started = False

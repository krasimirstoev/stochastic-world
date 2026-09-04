"""Shared-memory day-shard backend for --aggressive-parallel.

Pid ownership is stable, planner input/output lives in shared memory, and queue
payloads contain only pid lists plus small day metadata. Workers also prepare
high-volume economy intents from a day-start shared economy snapshot so main is
primarily a validation/commit engine.
"""

import multiprocessing as mp
import os
from collections import Counter, defaultdict
from time import perf_counter

from .aggressive_economy import (
    LOCATION_KINDS,
    OUTPUT_NAMES,
    PROFESSION_NAMES,
    SharedEconomyState,
)
from .aggressive_shared import SharedAgentBuffers
from . import agent_shards as planner
from .multiprocessing_engine import _DeterministicStream, _seed_for, _weighted_action
from .professions import PROFESSIONS

_PHASE_SOCIAL = 0x50C1A1
_PHASE_MOVE = 0x4D4F5645
_PHASE_WORK = 0x574F524B
_PHASE_BUY = 0x425559
_SAFE_ACTIONS = {"rest", "heal", "repair", "scavenge"}
_SOCIAL_ACTIONS = {"help", "steal", "attack"}


def _fast_sample_candidates(pool, actor_id, limit, stream):
    size = len(pool)
    limit = max(0, int(limit))
    if size <= 0 or limit <= 0:
        return []
    if size <= limit + 2:
        usable = [row for row in pool if row[0] != actor_id]
        if not usable:
            return []
        want = min(limit, len(usable))
        if want == len(usable):
            return usable
        indexes = list(range(len(usable)))
        result = []
        for _ in range(want):
            pick = stream.randint(0, len(indexes) - 1)
            result.append(usable[indexes.pop(pick)])
        return result
    want = min(limit, size - 1 if actor_id >= 0 else size)
    result = []
    seen = set()
    attempts = 0
    max_attempts = max(32, want * 8)
    while len(result) < want and attempts < max_attempts:
        attempts += 1
        index = stream.randint(0, size - 1)
        if index in seen:
            continue
        seen.add(index)
        row = pool[index]
        if row[0] == actor_id:
            continue
        result.append(row)
    if len(result) < want:
        for index, row in enumerate(pool):
            if index in seen or row[0] == actor_id:
                continue
            result.append(row)
            if len(result) >= want:
                break
    return result


def _sample_excluding(pool, exclude_ids, limit, stream):
    size = len(pool)
    limit = max(0, int(limit))
    if size <= 0 or limit <= 0:
        return []
    excluded = set(exclude_ids)
    if size <= limit + len(excluded) + 2:
        usable = [row for row in pool if row[0] not in excluded]
        if not usable:
            return []
        want = min(limit, len(usable))
        if want == len(usable):
            return usable
        result = []
        indexes = list(range(len(usable)))
        for _ in range(want):
            pick = stream.randint(0, len(indexes) - 1)
            result.append(usable[indexes.pop(pick)])
        return result
    want = min(limit, max(0, size - len(excluded)))
    result = []
    seen = set()
    attempts = 0
    max_attempts = max(32, want * 8)
    while len(result) < want and attempts < max_attempts:
        attempts += 1
        index = stream.randint(0, size - 1)
        if index in seen:
            continue
        seen.add(index)
        row = pool[index]
        if row[0] in excluded:
            continue
        result.append(row)
    if len(result) < want:
        for index, row in enumerate(pool):
            if index in seen or row[0] in excluded:
                continue
            result.append(row)
            if len(result) >= want:
                break
    return result


def _fast_social_plan(state, action, memories, pools, master_seed, day, round_index, encounter_sample, max_witnesses, visibility):
    stream = _DeterministicStream(_seed_for(master_seed, day, state["pid"], _PHASE_SOCIAL, round_index))
    pool = pools.get(state["location_id"], ())
    candidates = _sample_excluding(pool, (state["pid"],), encounter_sample, stream)
    target = planner._weighted_pick(candidates, memories, action, stream, day)
    if target is None:
        return (action, None, None, ())
    target_id, target_food, target_medicine, target_money, target_health = target
    if action == "help":
        if target_health < 70 and state["medicine"] > 0:
            payload = ("medicine", 1)
        elif state["food"] > 2:
            payload = ("food", stream.randint(1, min(2, state["food"] - 1)))
        else:
            payload = (None, 0)
    elif action == "steal":
        resource = None
        amount = 0
        if stream.random() < 0.45:
            options = []
            if target_food > 0:
                options.extend(("food",) * 4)
            if target_money > 0:
                options.extend(("money",) * 2)
            if target_medicine > 0:
                options.append("medicine")
            if options:
                resource = options[stream.randint(0, len(options) - 1)]
                if resource == "food":
                    amount = min(target_food, stream.randint(1, 3))
                elif resource == "money":
                    amount = min(target_money, stream.randint(1, 5))
                else:
                    amount = 1
        payload = (resource, amount)
    else:
        payload = (stream.randint(5, 20), stream.randint(4, 9))
    witness_ids = ()
    if max_witnesses and stream.random() <= visibility:
        witnesses = _sample_excluding(pool, (state["pid"], target_id), max_witnesses, stream)
        witness_ids = tuple(row[0] for row in witnesses)
    return (action, target_id, payload, witness_ids)


def _weighted_choice(options, weights, stream):
    total = sum(weights)
    if not options or total <= 0:
        return None
    needle = stream.random() * total
    upto = 0.0
    for option, weight in zip(options, weights):
        upto += weight
        if needle <= upto:
            return option
    return options[-1]


def _prepare_move(state, employer_id, profession_code, economy, master_seed, day, round_index):
    if state["energy"] < 4:
        return None
    current_id = int(state["location_id"])
    if current_id < 0 or current_id >= len(economy.neighbors):
        return ("shared", "move")
    options = economy.neighbors[current_id]
    if not options:
        return None
    profession_name = PROFESSION_NAMES[profession_code] if 0 <= profession_code < len(PROFESSION_NAMES) else PROFESSION_NAMES[0]
    profession = PROFESSIONS[profession_name]
    employer = economy.read_employer(employer_id) if employer_id >= 0 else None
    weights = []
    valid = []
    for location_id in options:
        location = economy.read_location(location_id)
        if location is None:
            continue
        kind_code, food_stock, _medicine_stock, vacancies, _population = location
        kind = LOCATION_KINDS[kind_code] if kind_code < len(LOCATION_KINDS) else "residential"
        weight = 1.0
        if employer and int(employer[0]) == int(location_id):
            weight *= 5.0
        if kind in profession.workplace_kinds:
            weight *= 2.2
        if employer is None and vacancies > 0:
            weight *= 1.8
        if state["food"] <= 3 and food_stock > 0:
            weight *= 2.5
        if state["medicine"] == 0 and kind == "clinic":
            weight *= 3.0
        valid.append(int(location_id))
        weights.append(weight)
    if not valid:
        return None
    stream = _DeterministicStream(_seed_for(master_seed, day, state["pid"], _PHASE_MOVE, round_index))
    destination = _weighted_choice(valid, weights, stream)
    if destination is None or destination == current_id:
        return None
    return ("move_prepared", int(destination), stream.randint(3, 7))


def _prepare_work(state, employer_id, profession_code, economy, master_seed, day, round_index):
    if state["energy"] < 8 or employer_id < 0:
        return ("shared", "work")
    employer = economy.read_employer(employer_id)
    if employer is None:
        return ("shared", "work")
    (
        employer_location, capacity, employees, base_wage, cash, productivity,
        output_code, output_per_shift, preferred_mask, _alive,
    ) = employer
    if int(employer_location) != int(state["location_id"]):
        return None
    profession_name = PROFESSION_NAMES[profession_code] if 0 <= profession_code < len(PROFESSION_NAMES) else PROFESSION_NAMES[0]
    profession = PROFESSIONS[profession_name]
    fit = 1.15 if state["location_kind"] in profession.workplace_kinds else 0.82
    vacancies = max(0, int(capacity) - int(employees))
    scarcity = 1.10 if vacancies > max(1, int(capacity) // 3) else 1.0
    preferred = 1.08 if profession_code < 64 and (int(preferred_mask) & (1 << profession_code)) else 0.92
    gross = max(1, round(float(base_wage) * profession.income_multiplier * fit * scarcity * preferred))
    if float(cash) < gross:
        return ("shared", "work")
    stream = _DeterministicStream(_seed_for(master_seed, day, state["pid"], _PHASE_WORK, round_index))
    energy = max(3, round(stream.randint(6, 12) * profession.energy_multiplier))
    produced = 0.0
    service_revenue = 0.0
    output_good = OUTPUT_NAMES[int(output_code)] if 0 <= int(output_code) < len(OUTPUT_NAMES) else None
    if output_good:
        produced = max(0.0, float(output_per_shift) * float(productivity) * fit * (0.85 + stream.random() * 0.30))
    elif output_code == 0:
        # Logistics firms have no service revenue; their kind is not encoded here.
        # A zero output_per_shift identifies them in the current model.
        if float(output_per_shift) != 0.0:
            service_revenue = gross * float(productivity) * (1.12 + stream.random() * 0.33)
    career_delta = profession.advancement_rate * fit
    return (
        "work_prepared", int(employer_id), int(gross), int(energy), output_good,
        float(produced), float(service_revenue), float(career_delta),
    )


def _prepare_buy(state, master_seed, day, round_index):
    options = []
    if state["food"] <= 6:
        options.append(("food", 3))
    if state["medicine"] <= 1:
        options.append(("medicine", 1))
    if not options:
        return None
    if len(options) == 1:
        good, requested = options[0]
    else:
        stream = _DeterministicStream(_seed_for(master_seed, day, state["pid"], _PHASE_BUY, round_index))
        good, requested = options[stream.randint(0, len(options) - 1)]
    return ("buy_prepared", good, requested)


def _plan_shared_row(snapshot, memories, pools, economy, master_seed, day, actions_per_day, encounter_sample, max_witnesses, visibility):
    (
        pid, location_id, food, medicine, energy, health, shelter, money,
        has_employer, location_kind, positive_ties, hostile_ties,
        max_conflict, mean_affinity, scavenge_food_max, medicine_chance,
        is_working_age,
    ) = snapshot
    employer_id, profession_code, _social_class_code = economy.read_person(pid)
    state = {
        "pid": pid,
        "location_id": location_id,
        "food": food,
        "medicine": medicine,
        "energy": energy,
        "health": health,
        "shelter": shelter,
        "money": money,
        "has_employer": has_employer,
        "location_kind": location_kind,
        "positive_ties": positive_ties,
        "hostile_ties": hostile_ties,
        "max_conflict": max_conflict,
        "mean_affinity": mean_affinity,
        "scavenge_food_max": scavenge_food_max,
        "medicine_chance": medicine_chance,
    }
    intents = []
    for round_index in range(int(actions_per_day)):
        _, action = _weighted_action(planner._action_snapshot(state), master_seed, day, round_index)
        if not is_working_age and action not in {"scavenge", "buy_supplies", "rest", "heal", "repair", "help"}:
            action = "rest"
        if action in _SAFE_ACTIONS:
            planner._safe_apply(state, action, master_seed, day, round_index)
            intents.append(None)
        elif action in _SOCIAL_ACTIONS:
            social = _fast_social_plan(
                state, action, memories, pools, master_seed, day, round_index,
                encounter_sample, max_witnesses, visibility,
            )
            planner._remember_planned_social(memories, social, day)
            intents.append(("social",) + social)
        elif action == "idle":
            intents.append(None)
        elif action == "move":
            intents.append(_prepare_move(state, employer_id, profession_code, economy, master_seed, day, round_index))
        elif action == "work":
            intents.append(_prepare_work(state, employer_id, profession_code, economy, master_seed, day, round_index))
        elif action == "buy_supplies":
            intents.append(_prepare_buy(state, master_seed, day, round_index))
        else:
            intents.append(("shared", action))
    final_state = (
        state["food"], state["medicine"], state["energy"],
        state["health"], state["shelter"], state["money"],
    )
    return pid, final_state, tuple(intents)


def _worker_main(worker_id, input_queue, result_queue, master_seed, shared_descriptor, economy_descriptor):
    shared = SharedAgentBuffers.attach(shared_descriptor)
    economy = SharedEconomyState.attach(economy_descriptor)
    social_cache = {}
    try:
        while True:
            task = input_queue.get()
            if task is None:
                return
            day, pids, pools, actions_per_day, encounter_sample, max_witnesses, visibility = task
            started = perf_counter()
            for pid in pids:
                snapshot = shared.read_snapshot(pid)
                memories = social_cache.setdefault(pid, {})
                _pid, final_state, intents = _plan_shared_row(
                    snapshot,
                    memories,
                    pools,
                    economy,
                    master_seed,
                    day,
                    actions_per_day,
                    encounter_sample,
                    max_witnesses,
                    visibility,
                )
                shared.write_result(pid, final_state, intents)
            result_queue.put((worker_id, perf_counter() - started, len(pids)))
    finally:
        economy.close()
        shared.close()


class SharedPersistentDayShardPool:
    """Persistent fixed-owner workers with zero-copy planner result transport."""

    def __init__(self, master_seed, shared_buffers, economy_state, workers=0, min_active=64):
        requested = max(0, int(workers))
        cpu_count = os.cpu_count() or 1
        self.worker_count = min(requested, cpu_count) if requested else 0
        self.master_seed = int(master_seed)
        self.shared_buffers = shared_buffers
        self.shared_descriptor = shared_buffers.descriptor
        self.economy_state = economy_state
        self.economy_descriptor = economy_state.descriptor
        self.min_active = max(1, int(min_active))
        self.enabled = self.worker_count >= 2
        self.started = False
        self._ctx = None
        self._queues = []
        self._result_queue = None
        self._processes = []
        self.stats = Counter()
        self.phase_seconds = defaultdict(float)

    def should_parallelize(self, active_count):
        return self.enabled and int(active_count) >= self.min_active

    def _ensure_started(self):
        if not self.enabled or self.started:
            return
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        self._ctx = mp.get_context(method)
        self._result_queue = self._ctx.Queue()
        for worker_id in range(self.worker_count):
            queue = self._ctx.Queue(maxsize=2)
            process = self._ctx.Process(
                target=_worker_main,
                args=(
                    worker_id,
                    queue,
                    self._result_queue,
                    self.master_seed,
                    self.shared_descriptor,
                    self.economy_descriptor,
                ),
                name=f"stochastic-shm-shard-{worker_id}",
                daemon=True,
            )
            process.start()
            self._queues.append(queue)
            self._processes.append(process)
        self.started = True

    def plan_day(self, day, pids, pools, actions_per_day, encounter_sample, max_witnesses, visibility):
        if not pids:
            return
        if not self.should_parallelize(len(pids)):
            raise RuntimeError("shared shard pool called below parallel threshold")
        self._ensure_started()
        shards = [[] for _ in range(self.worker_count)]
        for pid in pids:
            shards[int(pid) % self.worker_count].append(int(pid))
        active = [(worker_id, shard) for worker_id, shard in enumerate(shards) if shard]
        started = perf_counter()
        for worker_id, shard in active:
            self._queues[worker_id].put(
                (
                    int(day), shard, pools, int(actions_per_day), int(encounter_sample),
                    int(max_witnesses), float(visibility),
                )
            )
            self.stats["tasks"] += 1
            self.stats["items_sent"] += len(shard)
        for _ in active:
            _worker_id, worker_seconds, item_count = self._result_queue.get()
            self.phase_seconds["worker_cpu"] += worker_seconds
            self.stats["items_returned"] += item_count
        self.phase_seconds["dispatch_wall"] += perf_counter() - started
        self.stats["days"] += 1

    def summary(self):
        return {
            "enabled": self.enabled,
            "started": self.started,
            "workers": self.worker_count,
            "min_active": self.min_active,
            "days": int(self.stats["days"]),
            "tasks": int(self.stats["tasks"]),
            "items_sent": int(self.stats["items_sent"]),
            "items_returned": int(self.stats["items_returned"]),
            "worker_seconds": float(self.phase_seconds["worker_cpu"]),
            "dispatch_seconds": float(self.phase_seconds["dispatch_wall"]),
            "shared_bytes": int(self.shared_buffers.allocated_bytes + self.economy_state.allocated_bytes),
        }

    def close(self):
        if not self.started:
            return
        for queue in self._queues:
            queue.put(None)
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        for queue in self._queues:
            queue.close()
        if self._result_queue is not None:
            self._result_queue.close()
        self._queues.clear()
        self._processes.clear()
        self.started = False

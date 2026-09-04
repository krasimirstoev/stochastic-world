"""RAM-first shard backend for --aggressive-parallel.

Workers receive only control metadata. Fixed pid ownership, eligibility and
social candidates are all discovered from shared memory, eliminating the daily
candidate-pool pickle and pid-shard queue payloads.
"""

import multiprocessing as mp
import os
from collections import Counter, defaultdict
from time import perf_counter

from . import agent_shards as planner
from .aggressive_economy import SharedEconomyState
from .aggressive_shared import SharedAgentBuffers
from .aggressive_social import SharedSocialState
from .agent_shards_shared import (
    _SAFE_ACTIONS,
    _SOCIAL_ACTIONS,
    _prepare_buy,
    _prepare_move,
    _prepare_work,
)
from .multiprocessing_engine import _DeterministicStream, _seed_for, _weighted_action


_PHASE_SOCIAL = 0x50C1A1
_MEMORY_RECONCILE_INTERVAL_DAYS = 7


def _memory_scores(memory):
    if memory is None:
        return 0.0, 0.0
    trust, grievance, _familiarity, _decayed_day = memory
    affinity = max(-100.0, min(100.0, trust - grievance))
    conflict = max(0.0, grievance - min(0.0, trust))
    return affinity, conflict


def _refresh_worker_social_aggregate(state, memories, day):
    """Build the action-weight social aggregate from worker-owned memory.

    Main-side Person.memories is intentionally not maintained in aggressive mode,
    so using snapshot aggregate values there would freeze social action weights.
    Reconcile lazy decay weekly, matching Person.decay_memories(), then keep the
    aggregate incremental for the rest of the day's action rounds.
    """
    if not memories:
        state["_known_people"] = 0
        state["_affinity_sum"] = 0.0
        state["_max_conflict_target"] = None
        state["positive_ties"] = 0
        state["hostile_ties"] = 0
        state["max_conflict"] = 0.0
        state["mean_affinity"] = 0.0
        return

    reconcile = int(day) % _MEMORY_RECONCILE_INTERVAL_DAYS == 0
    affinity_sum = 0.0
    positive_ties = 0
    hostile_ties = 0
    max_conflict = 0.0
    max_target = None
    for target_id in tuple(memories):
        memory = (
            planner._materialize_memory(memories, target_id, day)
            if reconcile
            else memories.get(target_id)
        )
        affinity, conflict = _memory_scores(memory)
        affinity_sum += affinity
        positive_ties += int(affinity >= 15.0)
        hostile_ties += int(conflict >= 20.0)
        if conflict > max_conflict:
            max_conflict = conflict
            max_target = target_id

    known = len(memories)
    state["_known_people"] = known
    state["_affinity_sum"] = affinity_sum
    state["_max_conflict_target"] = max_target
    state["positive_ties"] = positive_ties
    state["hostile_ties"] = hostile_ties
    state["max_conflict"] = max_conflict
    state["mean_affinity"] = affinity_sum / known if known else 0.0


def _recompute_worker_max_conflict(state, memories):
    max_conflict = 0.0
    max_target = None
    for target_id, memory in memories.items():
        _affinity, conflict = _memory_scores(memory)
        if conflict > max_conflict:
            max_conflict = conflict
            max_target = target_id
    state["max_conflict"] = max_conflict
    state["_max_conflict_target"] = max_target


def _remember_worker_social(state, memories, social_plan, day):
    """Apply actor memory and update action-weight aggregates incrementally."""
    action, target_id, payload, _witness_ids = social_plan
    if target_id is None or payload is None:
        return

    before_exists = target_id in memories
    before = planner._materialize_memory(memories, target_id, day)
    old_affinity, old_conflict = _memory_scores(before)

    planner._remember_planned_social(memories, social_plan, day)
    after = memories.get(target_id)
    if after is None:
        return
    new_affinity, new_conflict = _memory_scores(after)

    if not before_exists:
        state["_known_people"] += 1
    state["_affinity_sum"] += new_affinity - old_affinity
    state["positive_ties"] += int(new_affinity >= 15.0) - int(old_affinity >= 15.0)
    state["hostile_ties"] += int(new_conflict >= 20.0) - int(old_conflict >= 20.0)

    current_max_target = state.get("_max_conflict_target")
    if new_conflict >= state["max_conflict"]:
        state["max_conflict"] = new_conflict
        state["_max_conflict_target"] = target_id
    elif current_max_target == target_id and new_conflict < old_conflict:
        _recompute_worker_max_conflict(state, memories)

    known = state["_known_people"]
    state["mean_affinity"] = state["_affinity_sum"] / known if known else 0.0


def _social_plan_ram(state, action, memories, social, master_seed, day, round_index,
                     encounter_sample, max_witnesses, visibility):
    stream = _DeterministicStream(
        _seed_for(master_seed, day, state["pid"], _PHASE_SOCIAL, round_index)
    )
    candidates = social.sample(
        state["location_id"], (state["pid"],), encounter_sample, stream
    )
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
        witnesses = social.sample(
            state["location_id"],
            (state["pid"], target_id),
            max_witnesses,
            stream,
        )
        witness_ids = tuple(row[0] for row in witnesses)
    return (action, target_id, payload, witness_ids)


def _plan_row_ram(snapshot, memories, economy, social, master_seed, day,
                  actions_per_day, encounter_sample, max_witnesses, visibility):
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

    # Once an actor has worker-owned memories, those are authoritative for
    # social action weights in aggressive mode. This avoids the stale main-side
    # aggregate that caused conflict to collapse after relationship bookkeeping
    # moved out of the main process.
    if memories:
        _refresh_worker_social_aggregate(state, memories, day)
    else:
        state["_known_people"] = 0
        state["_affinity_sum"] = 0.0
        state["_max_conflict_target"] = None

    intents = []
    for round_index in range(int(actions_per_day)):
        _, action = _weighted_action(
            planner._action_snapshot(state), master_seed, day, round_index
        )
        if not is_working_age and action not in {
            "scavenge", "buy_supplies", "rest", "heal", "repair", "help"
        }:
            action = "rest"
        if action in _SAFE_ACTIONS:
            planner._safe_apply(state, action, master_seed, day, round_index)
            intents.append(None)
        elif action in _SOCIAL_ACTIONS:
            social_plan = _social_plan_ram(
                state, action, memories, social, master_seed, day, round_index,
                encounter_sample, max_witnesses, visibility,
            )
            _remember_worker_social(state, memories, social_plan, day)
            intents.append(("social",) + social_plan)
        elif action == "idle":
            intents.append(None)
        elif action == "move":
            intents.append(
                _prepare_move(
                    state, employer_id, profession_code, economy,
                    master_seed, day, round_index,
                )
            )
        elif action == "work":
            intents.append(
                _prepare_work(
                    state, employer_id, profession_code, economy,
                    master_seed, day, round_index,
                )
            )
        elif action == "buy_supplies":
            intents.append(_prepare_buy(state, master_seed, day, round_index))
        else:
            intents.append(("shared", action))

    final_state = (
        state["food"], state["medicine"], state["energy"],
        state["health"], state["shelter"], state["money"],
    )
    return pid, final_state, tuple(intents)


def _worker_main_ram(worker_id, worker_count, input_queue, result_queue,
                     master_seed, shared_descriptor, economy_descriptor,
                     social_descriptor):
    shared = SharedAgentBuffers.attach(shared_descriptor)
    economy = SharedEconomyState.attach(economy_descriptor)
    social = SharedSocialState.attach(social_descriptor)
    social_cache = {}
    try:
        while True:
            task = input_queue.get()
            if task is None:
                return
            (
                day, population_size, actions_per_day, encounter_sample,
                max_witnesses, visibility,
            ) = task
            started = perf_counter()
            processed = 0
            for pid in range(worker_id, int(population_size), worker_count):
                if not social.is_eligible(pid):
                    continue
                snapshot = shared.read_snapshot(pid)
                memories = social_cache.setdefault(pid, {})
                _pid, final_state, intents = _plan_row_ram(
                    snapshot,
                    memories,
                    economy,
                    social,
                    master_seed,
                    day,
                    actions_per_day,
                    encounter_sample,
                    max_witnesses,
                    visibility,
                )
                shared.write_result(pid, final_state, intents)
                processed += 1
            result_queue.put((worker_id, perf_counter() - started, processed))
    finally:
        social.close()
        economy.close()
        shared.close()


class RamPersistentDayShardPool:
    """Persistent workers driven almost entirely by shared-memory state."""

    def __init__(self, master_seed, shared_buffers, economy_state, social_state,
                 workers=0, min_active=64):
        requested = max(0, int(workers))
        cpu_count = os.cpu_count() or 1
        self.worker_count = min(requested, cpu_count) if requested else 0
        self.master_seed = int(master_seed)
        self.shared_buffers = shared_buffers
        self.economy_state = economy_state
        self.social_state = social_state
        self.shared_descriptor = shared_buffers.descriptor
        self.economy_descriptor = economy_state.descriptor
        self.social_descriptor = social_state.descriptor
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
                target=_worker_main_ram,
                args=(
                    worker_id,
                    self.worker_count,
                    queue,
                    self._result_queue,
                    self.master_seed,
                    self.shared_descriptor,
                    self.economy_descriptor,
                    self.social_descriptor,
                ),
                name=f"stochastic-ram-shard-{worker_id}",
                daemon=True,
            )
            process.start()
            self._queues.append(queue)
            self._processes.append(process)
        self.started = True

    def plan_day(self, day, population_size, eligible_count, actions_per_day,
                 encounter_sample, max_witnesses, visibility):
        if eligible_count <= 0:
            return
        if not self.should_parallelize(eligible_count):
            raise RuntimeError("RAM shard pool called below parallel threshold")
        self._ensure_started()
        task = (
            int(day), int(population_size), int(actions_per_day),
            int(encounter_sample), int(max_witnesses), float(visibility),
        )
        started = perf_counter()
        for queue in self._queues:
            queue.put(task)
            self.stats["tasks"] += 1
        processed = 0
        for _ in self._queues:
            _worker_id, worker_seconds, item_count = self._result_queue.get()
            self.phase_seconds["worker_cpu"] += worker_seconds
            processed += item_count
        self.stats["items_sent"] += int(eligible_count)
        self.stats["items_returned"] += processed
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
            "shared_bytes": int(
                self.shared_buffers.allocated_bytes
                + self.economy_state.allocated_bytes
                + self.social_state.allocated_bytes
            ),
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

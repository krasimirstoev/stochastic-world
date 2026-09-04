"""Shared-memory day-shard backend for --aggressive-parallel.

This keeps the conservative agent_shards backend untouched. Pid ownership is
stable, planner input/output lives in shared memory, and queue payloads contain
only pid lists plus small day metadata.
"""

import multiprocessing as mp
import os
from collections import Counter, defaultdict
from time import perf_counter

from .aggressive_shared import SharedAgentBuffers
from . import agent_shards as planner
from .multiprocessing_engine import _DeterministicStream, _seed_for

_PHASE_SOCIAL = 0x50C1A1


def _fast_sample_candidates(pool, actor_id, limit, stream):
    """Sample O(limit) rows instead of copying/filtering the whole location."""
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


def _worker_main(worker_id, input_queue, result_queue, master_seed, shared_descriptor):
    shared = SharedAgentBuffers.attach(shared_descriptor)
    social_cache = {}
    planner._sample_candidates = _fast_sample_candidates
    planner._social_plan = _fast_social_plan
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
                _pid, final_state, intents = planner._plan_shard_row(
                    snapshot,
                    memories,
                    pools,
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
        shared.close()


class SharedPersistentDayShardPool:
    """Persistent fixed-owner workers with zero-copy planner result transport."""

    def __init__(self, master_seed, shared_buffers, workers=0, min_active=64):
        requested = max(0, int(workers))
        cpu_count = os.cpu_count() or 1
        self.worker_count = min(requested, cpu_count) if requested else 0
        self.master_seed = int(master_seed)
        self.shared_buffers = shared_buffers
        self.shared_descriptor = shared_buffers.descriptor
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
                    int(day),
                    shard,
                    pools,
                    int(actions_per_day),
                    int(encounter_sample),
                    int(max_witnesses),
                    float(visibility),
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
            "shared_bytes": int(self.shared_buffers.allocated_bytes),
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

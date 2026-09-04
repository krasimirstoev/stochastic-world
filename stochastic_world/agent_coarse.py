import multiprocessing as mp
import os
from collections import Counter, defaultdict
from time import perf_counter

from .multiprocessing_engine import (
    _DeterministicStream,
    _end_of_day_delta,
    _seed_for,
    _weighted_action,
)


_PHASE_SAFE_ACTION = 0x5AFE


def _safe_action_result(snapshot, master_seed, day, round_index):
    """Plan an action and fully execute agent-local actions inside the worker.

    The first 14 fields intentionally match multiprocessing_engine._weighted_action.
    Extra fields describe the location and lifecycle state needed for safe actions.
    """
    pid = snapshot[0]
    food = snapshot[2]
    medicine = snapshot[3]
    energy = snapshot[4]
    health = snapshot[5]
    shelter = snapshot[6]
    money = snapshot[7]
    scavenge_food_max = snapshot[14]
    medicine_chance = snapshot[15]
    is_working_age = snapshot[16]

    _, action = _weighted_action(snapshot[:14], master_seed, day, round_index)

    if not is_working_age and action not in {
        "scavenge", "buy_supplies", "rest", "heal", "repair", "help"
    }:
        action = "rest"

    stream = _DeterministicStream(
        _seed_for(master_seed, day, pid, _PHASE_SAFE_ACTION, round_index)
    )

    if action == "rest":
        energy_gain = stream.randint(12, 24)
        health_gain = stream.randint(0, 2)
        return (
            pid, action, True,
            food, medicine, min(100, energy + energy_gain),
            min(100, health + health_gain), shelter, money,
            (energy_gain, health_gain),
        )

    if action == "heal":
        if medicine <= 0 or health >= 100:
            return (pid, action, True, food, medicine, energy, health, shelter, money, None)
        gain = stream.randint(8, 18)
        return (
            pid, action, True,
            food, medicine - 1, energy, min(100, health + gain), shelter, money,
            (gain,),
        )

    if action == "repair":
        if money < 3 or shelter >= 100:
            return (pid, action, True, food, medicine, energy, health, shelter, money, None)
        gain = stream.randint(8, 16)
        return (
            pid, action, True,
            food, medicine, energy, health, min(100, shelter + gain), money - 3,
            (gain,),
        )

    if action == "scavenge":
        cost = stream.randint(4, 9)
        food_found = stream.randint(0, scavenge_food_max)
        medicine_found = int(stream.random() < medicine_chance)
        return (
            pid, action, True,
            food + food_found, medicine + medicine_found,
            max(0, energy - cost), health, shelter, money,
            (food_found, medicine_found, cost),
        )

    return (pid, action, False, food, medicine, energy, health, shelter, money, None)


def _worker_main(worker_id, input_queue, result_queue, master_seed):
    while True:
        task = input_queue.get()
        if task is None:
            return
        kind, day, round_index, payload = task
        started = perf_counter()
        if kind == "round":
            results = [
                _safe_action_result(row, master_seed, day, round_index)
                for row in payload
            ]
        elif kind == "end":
            results = [
                _end_of_day_delta(row, master_seed, day)
                for row in payload
            ]
        else:
            raise RuntimeError(f"unknown agent coarse task: {kind}")
        result_queue.put(
            (worker_id, kind, perf_counter() - started, len(payload), results)
        )


class AgentCoarsePool:
    """Persistent coarse-grained agent workers.

    Each worker receives one large contiguous batch. Workers perform decision
    calculation plus full agent-local actions, and also calculate end-of-day
    survival deltas. Shared world mutations remain intents for deterministic main
    process application.
    """

    def __init__(self, master_seed, workers=0, min_active=1024):
        self.master_seed = int(master_seed)
        requested = max(0, int(workers))
        cpu_count = os.cpu_count() or 1
        self.worker_count = min(requested, cpu_count) if requested else 0
        self.min_active = max(1, int(min_active))
        self.enabled = self.worker_count >= 2
        self._ctx = None
        self._queues = []
        self._result_queue = None
        self._processes = []
        self.started = False
        self.stats = Counter()
        self.phase_seconds = defaultdict(float)

    def should_parallelize(self, active_count):
        return self.enabled and int(active_count) >= self.min_active

    def _ensure_started(self):
        if not self.enabled or self.started:
            return
        methods = mp.get_all_start_methods()
        method = "fork" if "fork" in methods else "spawn"
        self._ctx = mp.get_context(method)
        self._result_queue = self._ctx.Queue()
        for worker_id in range(self.worker_count):
            queue = self._ctx.Queue(maxsize=2)
            process = self._ctx.Process(
                target=_worker_main,
                args=(worker_id, queue, self._result_queue, self.master_seed),
                name=f"stochastic-agent-{worker_id}",
                daemon=True,
            )
            process.start()
            self._queues.append(queue)
            self._processes.append(process)
        self.started = True

    def _chunks(self, rows):
        n = len(rows)
        if n == 0:
            return []
        chunk_size = max(1, (n + self.worker_count - 1) // self.worker_count)
        return [rows[start:start + chunk_size] for start in range(0, n, chunk_size)]

    def _dispatch(self, kind, day, round_index, rows):
        self._ensure_started()
        started = perf_counter()
        chunks = self._chunks(rows)
        for worker_id, payload in enumerate(chunks):
            self._queues[worker_id].put((kind, int(day), int(round_index), payload))
            self.stats["tasks"] += 1
            self.stats["items_sent"] += len(payload)

        results = []
        for _ in chunks:
            _worker_id, returned_kind, worker_seconds, item_count, result = self._result_queue.get()
            if returned_kind != kind:
                raise RuntimeError(f"agent worker phase mismatch: expected {kind}, got {returned_kind}")
            self.phase_seconds[f"{kind}_worker_cpu"] += worker_seconds
            self.stats["items_returned"] += item_count
            results.extend(result)

        self.phase_seconds[f"{kind}_dispatch_wall"] += perf_counter() - started
        self.stats[f"{kind}_calls"] += 1
        return results

    def plan_round(self, day, round_index, snapshots):
        results = self._dispatch("round", day, round_index, snapshots)
        safe = sum(1 for row in results if row[2])
        self.stats["safe_actions"] += safe
        self.stats["shared_intents"] += len(results) - safe
        return results

    def plan_end_of_day(self, day, snapshots):
        return self._dispatch("end", day, 0, snapshots)

    def summary(self):
        return {
            "enabled": self.enabled,
            "started": self.started,
            "workers": self.worker_count,
            "min_active": self.min_active,
            "tasks": int(self.stats["tasks"]),
            "items_sent": int(self.stats["items_sent"]),
            "items_returned": int(self.stats["items_returned"]),
            "action_calls": int(self.stats["round_calls"]),
            "end_of_day_calls": int(self.stats["end_calls"]),
            "safe_actions": int(self.stats["safe_actions"]),
            "shared_intents": int(self.stats["shared_intents"]),
            "action_worker_seconds": float(self.phase_seconds["round_worker_cpu"]),
            "action_dispatch_seconds": float(self.phase_seconds["round_dispatch_wall"]),
            "end_of_day_worker_seconds": float(self.phase_seconds["end_worker_cpu"]),
            "end_of_day_dispatch_seconds": float(self.phase_seconds["end_dispatch_wall"]),
        }

    def close(self):
        if not self.enabled or not self.started:
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
        self._processes.clear()
        self._queues.clear()
        self.started = False

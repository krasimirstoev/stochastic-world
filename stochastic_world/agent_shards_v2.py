import multiprocessing as mp
import os
from collections import Counter, defaultdict
from time import perf_counter

from .agent_shards import _plan_shard_row


def _worker_main_v2(worker_id, input_queue, result_queue, master_seed):
    memory_cache = {}
    while True:
        task = input_queue.get()
        if task is None:
            return
        day, rows, pools, actions_per_day, encounter_sample, max_witnesses, visibility = task
        started = perf_counter()
        results = []
        memory_updates = 0
        for snapshot, memories_update in rows:
            pid = snapshot[0]
            if memories_update is not None:
                memory_cache[pid] = memories_update
                memory_updates += 1
            results.append(
                _plan_shard_row(
                    (snapshot, memory_cache.get(pid, ())),
                    pools,
                    master_seed,
                    day,
                    actions_per_day,
                    encounter_sample,
                    max_witnesses,
                    visibility,
                )
            )
        result_queue.put(
            (worker_id, perf_counter() - started, len(rows), memory_updates, results)
        )


class PersistentDayShardPoolV2:
    """Fixed-owner day shards with persistent per-agent social-memory caches.

    Scalar agent snapshots are still refreshed daily because food, money, health,
    location and employment are authoritative in the main process. Relationship
    tuples are sent only for agents whose memories changed since their previous
    shard dispatch. Fixed pid ownership guarantees that each worker can retain
    the correct cache across days without cross-worker synchronization.
    """

    def __init__(self, master_seed, workers=0, min_active=64):
        requested = max(0, int(workers))
        cpu_count = os.cpu_count() or 1
        self.worker_count = min(requested, cpu_count) if requested else 0
        self.master_seed = int(master_seed)
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
                target=_worker_main_v2,
                args=(worker_id, queue, self._result_queue, self.master_seed),
                name=f"stochastic-shard-v2-{worker_id}",
                daemon=True,
            )
            process.start()
            self._queues.append(queue)
            self._processes.append(process)
        self.started = True

    def plan_day(self, day, rows, pools, actions_per_day, encounter_sample, max_witnesses, visibility):
        if not rows:
            return []
        if not self.should_parallelize(len(rows)):
            return [
                _plan_shard_row(
                    (snapshot, memories_update or ()),
                    pools,
                    self.master_seed,
                    day,
                    actions_per_day,
                    encounter_sample,
                    max_witnesses,
                    visibility,
                )
                for snapshot, memories_update in rows
            ]

        self._ensure_started()
        shards = [[] for _ in range(self.worker_count)]
        for row in rows:
            shards[row[0][0] % self.worker_count].append(row)
        active = [(worker_id, shard) for worker_id, shard in enumerate(shards) if shard]

        started = perf_counter()
        for worker_id, shard in active:
            self._queues[worker_id].put(
                (day, shard, pools, actions_per_day, encounter_sample, max_witnesses, visibility)
            )
            self.stats["tasks"] += 1
            self.stats["items_sent"] += len(shard)

        results = []
        for _ in active:
            _worker_id, worker_seconds, item_count, memory_updates, planned = self._result_queue.get()
            self.phase_seconds["worker_cpu"] += worker_seconds
            self.stats["items_returned"] += item_count
            self.stats["memory_updates"] += memory_updates
            results.extend(planned)

        self.phase_seconds["dispatch_wall"] += perf_counter() - started
        self.stats["days"] += 1
        return results

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
            "memory_updates": int(self.stats["memory_updates"]),
            "worker_seconds": float(self.phase_seconds["worker_cpu"]),
            "dispatch_seconds": float(self.phase_seconds["dispatch_wall"]),
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

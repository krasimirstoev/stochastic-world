import multiprocessing as mp
import os
from collections import Counter, defaultdict
from time import perf_counter


_MASK64 = (1 << 64) - 1
_PHASE_ACTION = 0xA11CE
_PHASE_END_OF_DAY = 0xE0D


def _fold_seed(value):
    value = int(value)
    folded = 0
    while value:
        folded ^= value & _MASK64
        value >>= 64
    return folded & _MASK64


def _splitmix64(value):
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _seed_for(master_seed, day, person_id, phase, round_index=0):
    value = _fold_seed(master_seed)
    value ^= (int(day) * 0xD6E8FEB86659FD93) & _MASK64
    value ^= (int(person_id) * 0xA5A3564E27F8862B) & _MASK64
    value ^= (int(phase) * 0x9E3779B185EBCA87) & _MASK64
    value ^= (int(round_index) * 0xC2B2AE3D27D4EB4F) & _MASK64
    return _splitmix64(value)


class _DeterministicStream:
    __slots__ = ("state",)

    def __init__(self, seed):
        self.state = int(seed) & _MASK64

    def u64(self):
        self.state = _splitmix64(self.state)
        return self.state

    def random(self):
        return self.u64() / float(1 << 64)

    def randint(self, low, high):
        span = int(high) - int(low) + 1
        return int(low) + (self.u64() % span)


def _weighted_action(snapshot, master_seed, day, round_index):
    (
        pid, _district_id, food, medicine, energy, health, shelter, money,
        has_employer, location_kind, positive_ties, hostile_ties,
        max_conflict, mean_affinity,
    ) = snapshot

    weights = {
        "move": 6.0, "work": 24.0, "scavenge": 11.0, "buy_supplies": 9.0,
        "rest": 13.0, "heal": 4.0, "repair": 4.0, "help": 8.0,
        "steal": 4.0, "attack": 1.0, "idle": 16.0,
    }
    if food <= 3:
        weights["scavenge"] *= 2.6
        weights["buy_supplies"] *= 3.0
        weights["steal"] *= 2.0
        weights["move"] *= 1.5
    if medicine == 0 and health < 75:
        weights["buy_supplies"] *= 2.8
        weights["move"] *= 1.6
    if energy <= 30:
        weights["rest"] *= 4.0
        weights["work"] *= 0.35
        weights["move"] *= 0.4
        weights["attack"] *= 0.5
    if health < 70:
        weights["heal"] *= 4.0 if medicine else 0.1
        weights["attack"] *= 0.5
    if shelter < 45:
        weights["repair"] *= 4.0
    if money < 4:
        weights["work"] *= 1.8
        weights["buy_supplies"] *= 0.35
        weights["move"] *= 1.3
    if not has_employer:
        weights["work"] *= 1.65
    if location_kind == "industrial":
        weights["work"] *= 1.25
    if location_kind == "outskirts":
        weights["scavenge"] *= 1.5
    if positive_ties:
        weights["help"] *= 1 + min(2.0, positive_ties / 4)
    if hostile_ties:
        weights["attack"] *= 1 + max_conflict / 12
        weights["steal"] *= 1 + max_conflict / 60
    if mean_affinity > 10:
        weights["help"] *= 1.4
        weights["attack"] *= 0.65
    elif mean_affinity < -10:
        weights["help"] *= 0.6
        weights["attack"] *= 1.5

    total = sum(weights.values())
    stream = _DeterministicStream(_seed_for(master_seed, day, pid, _PHASE_ACTION, round_index))
    pick = stream.random() * total
    cursor = 0.0
    for action, weight in weights.items():
        cursor += weight
        if pick < cursor:
            return pid, action
    return pid, "idle"


def _end_of_day_delta(snapshot, master_seed, day):
    (
        pid, food, energy, shelter, health, money, unemployment_days,
        employed, is_working_age, is_dependent, is_adult,
        shelter_decay_bonus, local_crime_rate,
    ) = snapshot
    stream = _DeterministicStream(_seed_for(master_seed, day, pid, _PHASE_END_OF_DAY))

    new_unemployment = unemployment_days
    lifetime_unemployment_increment = 0
    ideology_shift = 0.0
    if is_working_age:
        if employed:
            new_unemployment = 0
        else:
            new_unemployment += 1
            lifetime_unemployment_increment = 1
        if new_unemployment > 30:
            ideology_shift -= 0.0004
    else:
        new_unemployment = 0

    food -= 1
    energy = max(0, energy - (2 if is_dependent else 3))
    shelter = max(0, shelter - stream.randint(0, 2) - shelter_decay_bonus)

    if is_adult:
        if money < 6 or food <= 2:
            ideology_shift -= 0.0015
        if local_crime_rate > 0.10:
            ideology_shift += min(0.0030, local_crime_rate * 0.012)

    damage = 0
    causes = []
    if food < 0:
        food = 0
        damage += stream.randint(4, 10)
        causes.append("starvation")
    if energy == 0:
        damage += stream.randint(2, 6)
        causes.append("exhaustion")
    if shelter <= 20 and stream.random() < 0.35:
        damage += stream.randint(2, 7)
        causes.append("exposure")
    health -= damage

    return (
        pid, food, energy, shelter, health, new_unemployment,
        lifetime_unemployment_increment, ideology_shift, damage, tuple(causes),
    )


def _worker_main(worker_id, input_queue, result_queue, master_seed):
    while True:
        task = input_queue.get()
        if task is None:
            return
        kind, day, round_index, payload = task
        started = perf_counter()
        if kind == "actions":
            items = [_weighted_action(row, master_seed, day, round_index) for row in payload]
        elif kind == "end_of_day":
            items = [_end_of_day_delta(row, master_seed, day) for row in payload]
        else:
            raise RuntimeError(f"unknown district worker task: {kind}")
        result_queue.put((worker_id, kind, perf_counter() - started, len(payload), items))


class PersistentDistrictPool:
    """Persistent spawn-based workers with stable district-to-worker sharding."""

    def __init__(self, master_seed, location_count, workers=-1, min_active=1024):
        self.master_seed = int(master_seed)
        cpu_count = os.cpu_count() or 1
        requested = int(workers)
        if requested < 0:
            requested = max(1, cpu_count - 1)
        self.worker_count = min(max(0, requested), max(1, int(location_count)))
        self.min_active = max(1, int(min_active))
        self.enabled = self.worker_count >= 2
        self._ctx = None
        self._queues = []
        self._result_queue = None
        self._processes = []
        self.stats = Counter()
        self.phase_seconds = defaultdict(float)
        self.started = False

    def _ensure_started(self):
        if not self.enabled or self.started:
            return
        self._ctx = mp.get_context("spawn")
        self._result_queue = self._ctx.Queue()
        for worker_id in range(self.worker_count):
            queue = self._ctx.Queue(maxsize=2)
            process = self._ctx.Process(
                target=_worker_main,
                args=(worker_id, queue, self._result_queue, self.master_seed),
                name=f"stochastic-district-{worker_id}",
                daemon=True,
            )
            process.start()
            self._queues.append(queue)
            self._processes.append(process)
        self.started = True

    def shard_for_district(self, district_id):
        if not self.enabled:
            return 0
        return int(district_id) % self.worker_count

    def should_parallelize(self, active_count):
        return self.enabled and int(active_count) >= self.min_active

    def _dispatch(self, kind, day, round_index, rows_by_worker):
        self._ensure_started()
        started = perf_counter()
        expected = 0
        for worker_id in range(self.worker_count):
            payload = rows_by_worker.get(worker_id, ())
            if not payload:
                continue
            self._queues[worker_id].put((kind, int(day), int(round_index), payload))
            expected += 1
            self.stats["tasks"] += 1
            self.stats["items_sent"] += len(payload)
        items = []
        for _ in range(expected):
            worker_id, returned_kind, worker_seconds, item_count, result = self._result_queue.get()
            if returned_kind != kind:
                raise RuntimeError(f"district worker phase mismatch: expected {kind}, got {returned_kind}")
            self.phase_seconds[f"{kind}_worker_cpu"] += worker_seconds
            self.stats["items_returned"] += item_count
            items.extend(result)
        elapsed = perf_counter() - started
        self.phase_seconds[f"{kind}_dispatch_wall"] += elapsed
        self.stats[f"{kind}_calls"] += 1
        return items

    def plan_actions(self, day, round_index, snapshots):
        rows_by_worker = defaultdict(list)
        for row in snapshots:
            rows_by_worker[self.shard_for_district(row[1])].append(row)
        return self._dispatch("actions", day, round_index, rows_by_worker)

    def plan_end_of_day(self, day, snapshots):
        rows_by_worker = defaultdict(list)
        for district_id, row in snapshots:
            rows_by_worker[self.shard_for_district(district_id)].append(row)
        return self._dispatch("end_of_day", day, 0, rows_by_worker)

    def summary(self):
        return {
            "enabled": self.enabled,
            "started": self.started,
            "workers": self.worker_count,
            "min_active": self.min_active,
            "tasks": int(self.stats["tasks"]),
            "items_sent": int(self.stats["items_sent"]),
            "items_returned": int(self.stats["items_returned"]),
            "action_calls": int(self.stats["actions_calls"]),
            "end_of_day_calls": int(self.stats["end_of_day_calls"]),
            "action_worker_seconds": float(self.phase_seconds["actions_worker_cpu"]),
            "action_dispatch_seconds": float(self.phase_seconds["actions_dispatch_wall"]),
            "end_of_day_worker_seconds": float(self.phase_seconds["end_of_day_worker_cpu"]),
            "end_of_day_dispatch_seconds": float(self.phase_seconds["end_of_day_dispatch_wall"]),
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

import multiprocessing as mp
import os
from collections import Counter, defaultdict
from time import perf_counter

from stochastic_world.multiprocessing_engine import _DeterministicStream, _seed_for, _weighted_action


_PHASE_SAFE = 0x5AFE
_PHASE_SOCIAL = 0x50C1A1
_SAFE_ACTIONS = {"rest", "heal", "repair", "scavenge"}
_SOCIAL_ACTIONS = {"help", "steal", "attack"}


def _sample_candidates(pool, actor_id, limit, stream):
    usable = [row for row in pool if row[0] != actor_id]
    if not usable or limit <= 0:
        return []
    want = min(int(limit), len(usable))
    if want == len(usable):
        return usable
    result = []
    seen = set()
    while len(result) < want:
        index = stream.randint(0, len(usable) - 1)
        if index in seen:
            continue
        seen.add(index)
        result.append(usable[index])
    return result


def _weighted_pick(candidates, memories, mode, stream):
    if not candidates:
        return None
    weighted = []
    total = 0.0
    for target in candidates:
        target_id, food, medicine, money, _health = target
        memory = memories.get(target_id)
        if memory is None:
            affinity = 0.0
            conflict = 0.0
            familiarity = 0
        else:
            trust, grievance, familiarity = memory
            affinity = max(-100.0, min(100.0, trust - grievance))
            conflict = max(0.0, grievance - min(0.0, trust))
        if mode == "help":
            value = 1.0 if memory is None else 1.0 + max(0.0, affinity) / 10.0 + familiarity / 20.0
        elif mode == "attack":
            value = 1.0 if memory is None else 1.0 + conflict / 5.0
        else:
            wealth = food + medicine * 2 + max(0.0, money) / 4.0
            value = (
                1.0 + wealth / 25.0
                if memory is None
                else 1.0 + conflict / 18.0 + max(0.0, -affinity) / 30.0 + wealth / 25.0
            )
        weight = max(0.1, value)
        weighted.append((target, weight))
        total += weight
    needle = stream.random() * total
    upto = 0.0
    for target, weight in weighted:
        upto += weight
        if needle <= upto:
            return target
    return weighted[-1][0]


def _action_snapshot(state):
    return (
        state["pid"], state["location_id"], state["food"], state["medicine"],
        state["energy"], state["health"], state["shelter"], state["money"],
        state["has_employer"], state["location_kind"], state["positive_ties"],
        state["hostile_ties"], state["max_conflict"], state["mean_affinity"],
    )


def _safe_plan(state, action, master_seed, day, round_index):
    stream = _DeterministicStream(_seed_for(master_seed, day, state["pid"], _PHASE_SAFE, round_index))
    if action == "rest":
        energy_gain = stream.randint(12, 24)
        health_gain = stream.randint(0, 2)
        state["energy"] = min(100, state["energy"] + energy_gain)
        state["health"] = min(100, state["health"] + health_gain)
        return (energy_gain, health_gain)
    if action == "heal":
        if state["medicine"] <= 0 or state["health"] >= 100:
            return None
        gain = stream.randint(8, 18)
        state["medicine"] -= 1
        state["health"] = min(100, state["health"] + gain)
        return (gain,)
    if action == "repair":
        if state["money"] < 3 or state["shelter"] >= 100:
            return None
        gain = stream.randint(8, 16)
        state["money"] -= 3
        state["shelter"] = min(100, state["shelter"] + gain)
        return (gain,)
    cost = stream.randint(4, 9)
    food_found = stream.randint(0, state["scavenge_food_max"])
    medicine_found = int(stream.random() < state["medicine_chance"])
    state["energy"] = max(0, state["energy"] - cost)
    state["food"] += food_found
    state["medicine"] += medicine_found
    return (food_found, medicine_found, cost)


def _social_plan(state, action, memories, pools, master_seed, day, round_index, encounter_sample, max_witnesses, visibility):
    stream = _DeterministicStream(_seed_for(master_seed, day, state["pid"], _PHASE_SOCIAL, round_index))
    pool = pools.get(state["location_id"], ())
    candidates = _sample_candidates(pool, state["pid"], encounter_sample, stream)
    target = _weighted_pick(candidates, memories, action, stream)
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
        witness_pool = [row for row in pool if row[0] not in (state["pid"], target_id)]
        witnesses = _sample_candidates(witness_pool, -1, max_witnesses, stream)
        witness_ids = tuple(row[0] for row in witnesses)
    return (action, target_id, payload, witness_ids)


def _plan_shard_row(row, pools, master_seed, day, actions_per_day, encounter_sample, max_witnesses, visibility):
    snapshot, memories_tuple = row
    (
        pid, location_id, food, medicine, energy, health, shelter, money,
        has_employer, location_kind, positive_ties, hostile_ties,
        max_conflict, mean_affinity, scavenge_food_max, medicine_chance,
        is_working_age,
    ) = snapshot
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
    memories = dict(memories_tuple)
    plans = []
    for round_index in range(int(actions_per_day)):
        _, action = _weighted_action(_action_snapshot(state), master_seed, day, round_index)
        if not is_working_age and action not in {"scavenge", "buy_supplies", "rest", "heal", "repair", "help"}:
            action = "rest"
        if action in _SAFE_ACTIONS:
            event_data = _safe_plan(state, action, master_seed, day, round_index)
            plans.append(("safe", action, event_data))
        elif action in _SOCIAL_ACTIONS:
            social = _social_plan(
                state, action, memories, pools, master_seed, day, round_index,
                encounter_sample, max_witnesses, visibility,
            )
            plans.append(("social",) + social)
        else:
            plans.append(("shared", action))
    return (pid, tuple(plans))


def _worker_main(worker_id, input_queue, result_queue, master_seed):
    while True:
        task = input_queue.get()
        if task is None:
            return
        day, rows, pools, actions_per_day, encounter_sample, max_witnesses, visibility = task
        started = perf_counter()
        results = [
            _plan_shard_row(
                row, pools, master_seed, day, actions_per_day,
                encounter_sample, max_witnesses, visibility,
            )
            for row in rows
        ]
        result_queue.put((worker_id, perf_counter() - started, len(rows), results))


class PersistentDayShardPool:
    """Persistent fixed-owner workers that plan an agent's entire day at once."""

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
                target=_worker_main,
                args=(worker_id, queue, self._result_queue, self.master_seed),
                name=f"stochastic-shard-{worker_id}",
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
                    row, pools, self.master_seed, day, actions_per_day,
                    encounter_sample, max_witnesses, visibility,
                )
                for row in rows
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
            _worker_id, worker_seconds, item_count, planned = self._result_queue.get()
            self.phase_seconds["worker_cpu"] += worker_seconds
            self.stats["items_returned"] += item_count
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

"""Dynamically balanced owner pool for the aggressive SoA engine.

Locations remain single-owner within a BSP superstep, but ownership is
recomputed every day from the current CSR population counts.  This preserves
race-free local social mutation while avoiding permanent worker/location
pairings as districts depopulate at different rates.
"""

import multiprocessing as mp
import os
from collections import Counter
from time import perf_counter

import numpy as np

from .aggressive_economy import SOCIAL_CLASSES
from .aggressive_soa import (
    C_ARRESTS,
    C_ATTACKS,
    C_CRIMES,
    C_HELPS,
    C_MOVES,
    C_OBSERVATIONS,
    C_THEFTS,
    EF_CASH,
    EF_PAYROLL,
    EF_PRODUCTIVITY,
    EF_REVENUE,
    EF_UNITS,
    EI_ALIVE,
    EI_CAPACITY,
    EI_EMPLOYEES,
    EI_LOCATION,
    LF_FOOD_PRICE,
    LF_MED_PRICE,
    SharedCSRLocationIndex,
    SharedRelationMemory,
    SharedSoAAgentState,
    _employer_packet_arrays,
    _location_packet_arrays,
    _market_arrays,
    _merge_stats,
    _run_location_day_nb,
    _stats_from_kernel,
)


def _balanced_worker_main(
    worker_id,
    input_queue,
    result_queue,
    master_seed,
    state_desc,
    memory_desc,
    index_desc,
):
    state = SharedSoAAgentState.attach(state_desc)
    memory = SharedRelationMemory.attach(memory_desc)
    index = SharedCSRLocationIndex.attach(index_desc)
    try:
        while True:
            task = input_queue.get()
            if task is None:
                return
            (
                owned_locations,
                day,
                actions_per_day,
                encounter_sample,
                max_witnesses,
                visibility,
                packet,
                fuse_eod,
            ) = task
            started = perf_counter()
            loc_i, loc_f, neighbors, neighbor_counts = _location_packet_arrays(packet)
            emp_i, emp_f, emp_pref, by_location = _employer_packet_arrays(packet, loc_i)
            government_id, tax_rate = packet["government"]
            government_left = government_id == "left"
            results = []
            worker_stats = None
            location_rows = {int(row[0]): row for row in packet["locations"]}

            for lid in owned_locations:
                lid = int(lid)
                start = int(index.offsets[lid])
                end = int(index.offsets[lid + 1])
                pids = index.pids[start:end]
                location_row = location_rows[lid]
                local_eids = np.array(by_location[lid], dtype=np.int32)
                (
                    stock_food,
                    stock_med,
                    reserve_food,
                    reserve_med,
                    food_eids,
                    med_eids,
                ) = _market_arrays(location_row, emp_i.shape[1], local_eids, emp_i)

                (
                    action_counts,
                    counters,
                    treasury_delta,
                    police,
                    market,
                    scalar,
                    loc_stats,
                    social_stats,
                    employer_counts,
                ) = _run_location_day_nb(
                    pids,
                    int(day),
                    int(actions_per_day),
                    np.uint64(master_seed),
                    lid,
                    state.ints,
                    state.floats,
                    state.flags,
                    memory.targets,
                    memory.trust,
                    memory.grievance,
                    memory.familiarity,
                    memory.days,
                    loc_i,
                    loc_f,
                    neighbors,
                    neighbor_counts,
                    emp_i,
                    emp_f,
                    emp_pref,
                    local_eids,
                    food_eids,
                    med_eids,
                    stock_food,
                    stock_med,
                    float(reserve_food),
                    float(reserve_med),
                    float(tax_rate),
                    bool(government_left),
                    int(encounter_sample),
                    int(max_witnesses),
                    float(visibility),
                    int(packet["police"].get(lid, 0)),
                    bool(fuse_eod),
                )

                food_suppliers = {-1: float(market[0])}
                med_suppliers = {-1: float(market[1])}
                for eid in food_eids:
                    amount = float(stock_food[int(eid)])
                    if amount:
                        food_suppliers[int(eid)] = amount
                for eid in med_eids:
                    amount = float(stock_med[int(eid)])
                    if amount:
                        med_suppliers[int(eid)] = amount

                employer_rows = [
                    (
                        int(eid),
                        int(emp_i[EI_LOCATION, int(eid)]),
                        int(emp_i[EI_CAPACITY, int(eid)]),
                        int(emp_i[EI_EMPLOYEES, int(eid)]),
                        float(emp_f[EF_CASH, int(eid)]),
                        float(emp_f[EF_PRODUCTIVITY, int(eid)]),
                        float(emp_f[EF_PAYROLL, int(eid)]),
                        float(emp_f[EF_REVENUE, int(eid)]),
                        float(emp_f[EF_UNITS, int(eid)]),
                        int(emp_i[EI_ALIVE, int(eid)]),
                    )
                    for eid in local_eids
                ]
                counter_dict = {
                    "helps": int(counters[C_HELPS]),
                    "thefts": int(counters[C_THEFTS]),
                    "attacks": int(counters[C_ATTACKS]),
                    "observations": int(counters[C_OBSERVATIONS]),
                    "arrests": int(counters[C_ARRESTS]),
                    "moves": int(counters[C_MOVES]),
                    "crimes": int(counters[C_CRIMES]),
                }
                action_names = (
                    "move",
                    "work",
                    "scavenge",
                    "buy_supplies",
                    "rest",
                    "heal",
                    "repair",
                    "help",
                    "steal",
                    "attack",
                    "idle",
                )
                for code, name in enumerate(action_names):
                    if action_counts[code]:
                        counter_dict[f"action_{name}"] = int(action_counts[code])

                results.append(
                    {
                        "location": lid,
                        "counters": counter_dict,
                        "treasury_delta": float(treasury_delta),
                        "employers": employer_rows,
                        "market": (
                            lid,
                            {
                                "food": float(loc_f[LF_FOOD_PRICE, lid]),
                                "medicine": float(loc_f[LF_MED_PRICE, lid]),
                            },
                            food_suppliers,
                            med_suppliers,
                            {"food": float(market[2]), "medicine": float(market[3])},
                            {"food": float(market[4]), "medicine": float(market[5])},
                        ),
                        "police": (
                            lid,
                            int(packet["police"].get(lid, 0)),
                            int(police[0]),
                            int(police[1]),
                            int(police[2]),
                        ),
                    }
                )
                if fuse_eod:
                    worker_stats = _merge_stats(
                        worker_stats,
                        _stats_from_kernel(
                            scalar, loc_stats, social_stats, employer_counts
                        ),
                    )

            if worker_stats is None:
                worker_stats = {
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
                    "social": [[0.0] * 7 for _ in SOCIAL_CLASSES],
                    "employer_counts": {},
                    "deaths": 0,
                }
            result_queue.put(
                (worker_id, perf_counter() - started, results, worker_stats)
            )
    finally:
        index.close()
        memory.close()
        state.close()


class BalancedSoADomainPool:
    """Persistent BSP pool with daily LPT location assignment."""

    def __init__(
        self,
        master_seed,
        state,
        memory,
        index,
        location_count,
        workers=0,
    ):
        cpu_count = os.cpu_count() or 1
        requested = max(0, int(workers))
        self.worker_count = (
            min(requested, cpu_count, max(1, int(location_count)))
            if requested
            else 0
        )
        self.master_seed = int(master_seed)
        self.state = state
        self.memory = memory
        self.index = index
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
        self.balance_max_ratio = 1.0
        self.balance_ratio_sum = 0.0

    def _ensure_started(self):
        if self.started or not self.enabled:
            return
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        self._ctx = mp.get_context(method)
        self._result_queue = self._ctx.Queue()
        for worker_id in range(self.worker_count):
            queue = self._ctx.Queue(maxsize=2)
            process = self._ctx.Process(
                target=_balanced_worker_main,
                args=(
                    worker_id,
                    queue,
                    self._result_queue,
                    self.master_seed,
                    self.state.descriptor,
                    self.memory.descriptor,
                    self.index.descriptor,
                ),
                name=f"stochastic-soa-balanced-{worker_id}",
                daemon=True,
            )
            process.start()
            self._queues.append(queue)
            self._processes.append(process)
            self._active_workers.append(worker_id)
        self.started = True

    def rebuild_index(self, population_size):
        # Keep the exact compiled CSR builder owned by the original index/pool
        # contract. Import lazily to avoid another public surface in this module.
        from .aggressive_soa import _build_csr_index_nb

        return int(
            _build_csr_index_nb(
                self.state.ints,
                self.state.flags,
                min(int(population_size), self.state.capacity),
                self.location_count,
                self.index.pids,
                self.index.offsets,
            )
        )

    def _balanced_assignments(self):
        """Largest-processing-time-first bin packing by live location size."""
        counts = [
            int(self.index.offsets[lid + 1] - self.index.offsets[lid])
            for lid in range(self.location_count)
        ]
        order = sorted(range(self.location_count), key=counts.__getitem__, reverse=True)
        assignments = [[] for _ in range(self.worker_count)]
        loads = [0] * self.worker_count
        for lid in order:
            worker_id = min(range(self.worker_count), key=loads.__getitem__)
            assignments[worker_id].append(lid)
            loads[worker_id] += counts[lid]
        nonzero = [value for value in loads if value > 0]
        if nonzero:
            mean = sum(nonzero) / len(nonzero)
            ratio = max(nonzero) / mean if mean else 1.0
        else:
            ratio = 1.0
        self.balance_ratio_sum += ratio
        self.balance_max_ratio = max(self.balance_max_ratio, ratio)
        return assignments

    def run_day(
        self,
        day,
        actions_per_day,
        encounter_sample,
        max_witnesses,
        visibility,
        packet,
        fuse_eod=True,
    ):
        self._ensure_started()
        assignments = self._balanced_assignments()
        started = perf_counter()
        for worker_id in self._active_workers:
            task = (
                tuple(assignments[worker_id]),
                int(day),
                int(actions_per_day),
                int(encounter_sample),
                int(max_witnesses),
                float(visibility),
                packet,
                bool(fuse_eod),
            )
            self._queues[worker_id].put(task)
            self.stats["tasks"] += 1
        results = []
        stats = None
        for _ in self._active_workers:
            _wid, seconds, payload, worker_stats = self._result_queue.get()
            self.worker_seconds += seconds
            results.extend(payload)
            stats = _merge_stats(stats, worker_stats)
        self.dispatch_seconds += perf_counter() - started
        self.stats["days"] += 1
        return results, stats

    def summary(self):
        days = int(self.stats["days"])
        return {
            "days": days,
            "tasks": int(self.stats["tasks"]),
            "workers": len(self._active_workers),
            "worker_seconds": float(self.worker_seconds),
            "dispatch_seconds": float(self.dispatch_seconds),
            "shared_bytes": int(
                self.state.allocated_bytes
                + self.memory.allocated_bytes
                + self.index.allocated_bytes
            ),
            "balance_avg_ratio": (
                float(self.balance_ratio_sum / days) if days else 1.0
            ),
            "balance_max_ratio": float(self.balance_max_ratio),
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
                process.terminate()
                process.join(timeout=1)
            self._queues[worker_id].close()
        if self._result_queue is not None:
            self._result_queue.close()
        self._active_workers.clear()
        self.started = False

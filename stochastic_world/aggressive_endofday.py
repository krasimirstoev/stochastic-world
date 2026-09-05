"""Shared-memory end-of-day engine for very large aggressive simulations.

The conservative agent engine builds hundreds of thousands of Python snapshot
and result tuples for this phase.  This backend keeps fixed input/output records
and an alive mask in shared memory; worker queues carry control metadata only.
"""

import multiprocessing as mp
import os
import struct
from collections import Counter, defaultdict
from multiprocessing import shared_memory
from time import perf_counter

from .multiprocessing_engine import _end_of_day_delta


# food, energy, shelter, health, money, unemployment_days, employed,
# working_age, dependent, adult, shelter_decay_bonus, local_crime_rate
_INPUT = struct.Struct("<iiiidiBBBBid")
# food, energy, shelter, health, unemployment_days, lifetime_unemployment_inc,
# ideology_shift, damage, cause_mask
_OUTPUT = struct.Struct("<iiiiiidiB")

_CAUSE_STARVATION = 1
_CAUSE_EXHAUSTION = 2
_CAUSE_EXPOSURE = 4


class SharedEndOfDayBuffers:
    def __init__(self, capacity, *, descriptor=None):
        if descriptor is None:
            self.capacity = max(1, int(capacity))
            self._input = shared_memory.SharedMemory(create=True, size=self.capacity * _INPUT.size)
            self._output = shared_memory.SharedMemory(create=True, size=self.capacity * _OUTPUT.size)
            self._active = shared_memory.SharedMemory(create=True, size=self.capacity)
            self._active.buf[:] = b"\x00" * self._active.size
            self._owner = True
        else:
            self.capacity = int(descriptor["capacity"])
            self._input = shared_memory.SharedMemory(name=descriptor["input_name"])
            self._output = shared_memory.SharedMemory(name=descriptor["output_name"])
            self._active = shared_memory.SharedMemory(name=descriptor["active_name"])
            self._owner = False

    @classmethod
    def attach(cls, descriptor):
        return cls(1, descriptor=descriptor)

    @property
    def descriptor(self):
        return {
            "capacity": self.capacity,
            "input_name": self._input.name,
            "output_name": self._output.name,
            "active_name": self._active.name,
        }

    @property
    def allocated_bytes(self):
        return self._input.size + self._output.size + self._active.size

    def begin_sync(self, population_size):
        self._active.buf[: int(population_size)] = b"\x00" * int(population_size)

    def write_person(self, person, shelter_decay_bonus, crime_rate):
        pid = int(person.id)
        self._active.buf[pid] = 1
        _INPUT.pack_into(
            self._input.buf,
            pid * _INPUT.size,
            int(person.food),
            int(person.energy),
            int(person.shelter),
            int(person.health),
            float(person.money),
            int(person.unemployment_days),
            int(person.employer_id is not None),
            int(person.is_working_age),
            int(person.is_dependent),
            int(person.is_adult),
            int(shelter_decay_bonus),
            float(crime_rate),
        )

    def is_active(self, pid):
        return bool(self._active.buf[int(pid)])

    def read_input(self, pid):
        return _INPUT.unpack_from(self._input.buf, int(pid) * _INPUT.size)

    def write_delta(self, pid, delta):
        (
            _pid, food, energy, shelter, health, unemployment_days,
            lifetime_inc, ideology_shift, damage, causes,
        ) = delta
        mask = 0
        if "starvation" in causes:
            mask |= _CAUSE_STARVATION
        if "exhaustion" in causes:
            mask |= _CAUSE_EXHAUSTION
        if "exposure" in causes:
            mask |= _CAUSE_EXPOSURE
        _OUTPUT.pack_into(
            self._output.buf,
            int(pid) * _OUTPUT.size,
            int(food), int(energy), int(shelter), int(health),
            int(unemployment_days), int(lifetime_inc), float(ideology_shift),
            int(damage), int(mask),
        )

    def read_delta(self, pid):
        values = _OUTPUT.unpack_from(self._output.buf, int(pid) * _OUTPUT.size)
        food, energy, shelter, health, unemployment_days, lifetime_inc, ideology_shift, damage, mask = values
        causes = []
        if mask & _CAUSE_STARVATION:
            causes.append("starvation")
        if mask & _CAUSE_EXHAUSTION:
            causes.append("exhaustion")
        if mask & _CAUSE_EXPOSURE:
            causes.append("exposure")
        return (
            int(pid), food, energy, shelter, health, unemployment_days,
            lifetime_inc, ideology_shift, damage, tuple(causes),
        )

    def close(self, *, unlink=False):
        segments = (self._input, self._output, self._active)
        for segment in segments:
            segment.close()
        if unlink and self._owner:
            for segment in segments:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass


def _worker_main(worker_id, worker_count, input_queue, result_queue, master_seed, descriptor):
    buffers = SharedEndOfDayBuffers.attach(descriptor)
    try:
        while True:
            task = input_queue.get()
            if task is None:
                return
            day, population_size = task
            started = perf_counter()
            processed = 0
            for pid in range(worker_id, int(population_size), int(worker_count)):
                if not buffers.is_active(pid):
                    continue
                (
                    food, energy, shelter, health, money, unemployment_days,
                    employed, working_age, dependent, adult,
                    shelter_decay_bonus, crime_rate,
                ) = buffers.read_input(pid)
                snapshot = (
                    pid, food, energy, shelter, health, money, unemployment_days,
                    bool(employed), bool(working_age), bool(dependent), bool(adult),
                    shelter_decay_bonus, crime_rate,
                )
                buffers.write_delta(pid, _end_of_day_delta(snapshot, master_seed, day))
                processed += 1
            result_queue.put((worker_id, perf_counter() - started, processed))
    finally:
        buffers.close()


class SharedEndOfDayPool:
    def __init__(self, master_seed, buffers, workers=0):
        requested = max(0, int(workers))
        cpu_count = os.cpu_count() or 1
        self.worker_count = min(requested, cpu_count) if requested else 0
        self.master_seed = int(master_seed)
        self.buffers = buffers
        self.descriptor = buffers.descriptor
        self.enabled = self.worker_count >= 2
        self.started = False
        self._ctx = None
        self._queues = []
        self._result_queue = None
        self._processes = []
        self.stats = Counter()
        self.phase_seconds = defaultdict(float)

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
                args=(worker_id, self.worker_count, queue, self._result_queue,
                      self.master_seed, self.descriptor),
                name=f"stochastic-eod-shm-{worker_id}",
                daemon=True,
            )
            process.start()
            self._queues.append(queue)
            self._processes.append(process)
        self.started = True

    def run(self, day, population_size):
        if not self.enabled:
            raise RuntimeError("shared end-of-day pool requires at least two workers")
        self._ensure_started()
        started = perf_counter()
        task = (int(day), int(population_size))
        for queue in self._queues:
            queue.put(task)
            self.stats["tasks"] += 1
        processed = 0
        for _ in self._queues:
            _worker_id, seconds, count = self._result_queue.get()
            self.phase_seconds["worker_cpu"] += seconds
            processed += count
        self.phase_seconds["dispatch_wall"] += perf_counter() - started
        self.stats["days"] += 1
        self.stats["items"] += processed
        return processed

    def summary(self):
        return {
            "days": int(self.stats["days"]),
            "tasks": int(self.stats["tasks"]),
            "items": int(self.stats["items"]),
            "worker_seconds": float(self.phase_seconds["worker_cpu"]),
            "dispatch_seconds": float(self.phase_seconds["dispatch_wall"]),
            "shared_bytes": int(self.buffers.allocated_bytes),
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

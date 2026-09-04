"""Over-allocated shared social arenas for the aggressive engine.

One full population-capacity arena is reserved per location on purpose.  The
extra RAM buys a very simple hot path: workers sample social candidates directly
from shared pages and read an eligibility byte-mask instead of receiving Python
rows, pools or pid shard lists through multiprocessing queues.
"""

from multiprocessing import shared_memory
import struct


_ROW = struct.Struct("<iiidi")  # pid, food, medicine, money, health
_COUNT = struct.Struct("<i")


class SharedSocialState:
    def __init__(self, population_capacity, location_count, *, descriptor=None):
        if descriptor is None:
            self.population_capacity = max(1, int(population_capacity))
            self.location_count = max(1, int(location_count))
            self._counts = shared_memory.SharedMemory(
                create=True, size=self.location_count * _COUNT.size
            )
            self._rows = shared_memory.SharedMemory(
                create=True,
                size=self.location_count * self.population_capacity * _ROW.size,
            )
            self._eligible = shared_memory.SharedMemory(
                create=True, size=self.population_capacity
            )
            self._owner = True
            self._counts.buf[:] = b"\x00" * self._counts.size
            self._eligible.buf[:] = b"\x00" * self._eligible.size
        else:
            self.population_capacity = int(descriptor["population_capacity"])
            self.location_count = int(descriptor["location_count"])
            self._counts = shared_memory.SharedMemory(name=descriptor["counts_name"])
            self._rows = shared_memory.SharedMemory(name=descriptor["rows_name"])
            self._eligible = shared_memory.SharedMemory(name=descriptor["eligible_name"])
            self._owner = False

    @classmethod
    def attach(cls, descriptor):
        return cls(1, 1, descriptor=descriptor)

    @property
    def descriptor(self):
        return {
            "population_capacity": self.population_capacity,
            "location_count": self.location_count,
            "counts_name": self._counts.name,
            "rows_name": self._rows.name,
            "eligible_name": self._eligible.name,
        }

    @property
    def allocated_bytes(self):
        return self._counts.size + self._rows.size + self._eligible.size

    def _row_offset(self, location_id, index):
        slot = int(location_id) * self.population_capacity + int(index)
        return slot * _ROW.size

    def count(self, location_id):
        lid = int(location_id)
        if lid < 0 or lid >= self.location_count:
            return 0
        return _COUNT.unpack_from(self._counts.buf, lid * _COUNT.size)[0]

    def row(self, location_id, index):
        return _ROW.unpack_from(self._rows.buf, self._row_offset(location_id, index))

    def is_eligible(self, pid):
        pid = int(pid)
        return 0 <= pid < self.population_capacity and bool(self._eligible.buf[pid])

    def sync_world(self, world, day):
        """Rebuild candidate arenas and the worker eligibility mask in one pass."""
        self._counts.buf[:] = b"\x00" * self._counts.size
        self._eligible.buf[:] = b"\x00" * self._eligible.size
        counts = [0] * self.location_count
        eligible_count = 0
        rows_buf = self._rows.buf
        eligible_buf = self._eligible.buf
        capacity = self.population_capacity
        for person in world.people:
            pid = int(person.id)
            if pid < 0 or pid >= capacity:
                continue
            if person.alive and person.is_adult and int(day) >= person.detained_until_day:
                eligible_buf[pid] = 1
                eligible_count += 1
            if not person.alive:
                continue
            lid = int(person.location_id)
            if lid < 0 or lid >= self.location_count:
                continue
            index = counts[lid]
            if index >= capacity:
                continue
            _ROW.pack_into(
                rows_buf,
                (lid * capacity + index) * _ROW.size,
                pid,
                int(person.food),
                int(person.medicine),
                float(person.money),
                int(person.health),
            )
            counts[lid] = index + 1
        for lid, count in enumerate(counts):
            _COUNT.pack_into(self._counts.buf, lid * _COUNT.size, int(count))
        return eligible_count

    def sample(self, location_id, exclude_ids, limit, stream):
        """Return O(limit) unique rows without materializing a Python pool."""
        lid = int(location_id)
        size = self.count(lid)
        want = max(0, int(limit))
        if size <= 0 or want <= 0:
            return []
        excluded = set(int(x) for x in exclude_ids)
        want = min(want, max(0, size - len(excluded)))
        if want <= 0:
            return []
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
            row = self.row(lid, index)
            if row[0] in excluded:
                continue
            result.append(row)
        if len(result) < want:
            for index in range(size):
                if index in seen:
                    continue
                row = self.row(lid, index)
                if row[0] in excluded:
                    continue
                result.append(row)
                if len(result) >= want:
                    break
        return result

    def close(self, *, unlink=False):
        segments = (self._counts, self._rows, self._eligible)
        for segment in segments:
            segment.close()
        if unlink and self._owner:
            for segment in segments:
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass

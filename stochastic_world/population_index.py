from math import gcd


class PopulationIndex:
    """O(1) location membership updates and bounded random local sampling."""

    def __init__(self, people, location_count: int):
        self.people = people
        self.members = [[] for _ in range(location_count)]
        self.positions = [-1] * len(people)
        for person in people:
            self.add(person.id, person.location_id)

    def add(self, person_id: int, location_id: int):
        if person_id >= len(self.positions):
            self.positions.extend([-1] * (person_id + 1 - len(self.positions)))
        bucket = self.members[location_id]
        self.positions[person_id] = len(bucket)
        bucket.append(person_id)

    def remove(self, person_id: int, location_id: int):
        if person_id >= len(self.positions):
            return
        pos = self.positions[person_id]
        if pos < 0:
            return
        bucket = self.members[location_id]
        last_id = bucket[-1]
        bucket[pos] = last_id
        self.positions[last_id] = pos
        bucket.pop()
        self.positions[person_id] = -1

    def move(self, person_id: int, old_location: int, new_location: int):
        if old_location == new_location:
            return
        self.remove(person_id, old_location)
        self.add(person_id, new_location)

    def ids(self, location_id: int):
        return self.members[location_id]

    def population(self, location_id: int) -> int:
        return len(self.members[location_id])

    def sample_people(self, location_id: int, rng, limit: int, exclude=()):
        bucket = self.members[location_id]
        if not bucket or limit <= 0:
            return []
        excluded = set(exclude)
        usable = len(bucket) - sum(
            1 for pid in excluded
            if 0 <= pid < len(self.positions)
            and self.positions[pid] >= 0
            and self.people[pid].location_id == location_id
        )
        if usable <= 0:
            return []

        want = min(limit, usable)
        result, seen = [], set()
        attempts = 0
        max_attempts = max(16, want * 8)
        while len(result) < want and attempts < max_attempts:
            pid = bucket[rng.randrange(len(bucket))]
            attempts += 1
            if pid in excluded or pid in seen:
                continue
            person = self.people[pid]
            if not person.alive:
                continue
            seen.add(pid)
            result.append(person)

        if len(result) < want and len(bucket) <= max(64, want * 4):
            for pid in bucket:
                if len(result) >= want:
                    break
                if pid in excluded or pid in seen:
                    continue
                person = self.people[pid]
                if person.alive:
                    seen.add(pid)
                    result.append(person)
        return result


def permutation_ids(n: int, rng):
    """Yield a pseudo-random permutation of range(n) without allocating an O(n) order list."""
    if n <= 0:
        return
    offset = rng.randrange(n)
    if n == 1:
        yield 0
        return
    step = rng.randrange(1, n)
    while gcd(step, n) != 1:
        step = rng.randrange(1, n)
    current = offset
    for _ in range(n):
        yield current
        current = (current + step) % n

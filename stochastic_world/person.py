from dataclasses import dataclass, field

from .memory import InteractionMemory


ADULT_AGE_DAYS = 18 * 365
RETIREMENT_AGE_DAYS = 67 * 365
MEMORY_RECONCILE_INTERVAL_DAYS = 7


def _empty_aggregate():
    return {
        "known_people": 0,
        "mean_affinity": 0.0,
        "max_conflict": 0.0,
        "positive_ties": 0,
        "hostile_ties": 0,
    }


@dataclass(slots=True)
class Person:
    id: int
    name: str
    social_class: str = "working"
    profession: str = "laborer"
    ideology: float = 0.0
    location_id: int = 0
    food: int = 10
    money: float = 20.0
    medicine: int = 2
    energy: int = 80
    shelter: int = 70
    health: int = 100
    alive: bool = True
    taxes_paid: float = 0.0
    welfare_received: float = 0.0
    crime_suffered: int = 0
    work_experience: int = 0
    career_progress: float = 0.0
    lifetime_gross_income: float = 0.0
    days_in_class: int = 0
    employer_id: int | None = None
    unemployment_days: int = 0
    lifetime_unemployment_days: int = 0
    jobs_held: int = 0
    market_spending: float = 0.0
    shortage_experiences: int = 0
    detained_until_day: int = 0
    arrests: int = 0
    age_days: int = 30 * 365
    sex: str = "female"
    birth_day: int = 0
    mother_id: int | None = None
    father_id: int | None = None
    partner_id: int | None = None
    household_id: int | None = None
    generation: int = 0
    pregnant_until_day: int = 0
    pregnancy_partner_id: int | None = None
    retired: bool = False
    memories: dict[int, InteractionMemory] = field(default_factory=dict)
    memory_cap: int = 64
    _aggregate_cache: dict = field(default_factory=_empty_aggregate, init=False, repr=False)
    _affinity_sum: float = field(default=0.0, init=False, repr=False)
    _positive_ties: int = field(default=0, init=False, repr=False)
    _hostile_ties: int = field(default=0, init=False, repr=False)
    _max_conflict: float = field(default=0.0, init=False, repr=False)
    _max_conflict_id: int | None = field(default=None, init=False, repr=False)

    @property
    def age_years(self) -> float:
        return self.age_days / 365.0

    @property
    def is_adult(self) -> bool:
        return self.age_days >= ADULT_AGE_DAYS

    @property
    def is_working_age(self) -> bool:
        return self.is_adult and self.age_days < RETIREMENT_AGE_DAYS and not self.retired

    @property
    def is_dependent(self) -> bool:
        return not self.is_adult

    def _sync_aggregate_cache(self) -> None:
        count = len(self.memories)
        self._aggregate_cache["known_people"] = count
        self._aggregate_cache["mean_affinity"] = self._affinity_sum / count if count else 0.0
        self._aggregate_cache["max_conflict"] = self._max_conflict
        self._aggregate_cache["positive_ties"] = self._positive_ties
        self._aggregate_cache["hostile_ties"] = self._hostile_ties

    def _recompute_max_conflict(self) -> None:
        best_id = None
        best_value = 0.0
        for other_id, memory in self.memories.items():
            value = memory.conflict_score
            if value > best_value:
                best_id = other_id
                best_value = value
        self._max_conflict_id = best_id
        self._max_conflict = best_value

    def _update_contribution(self, memory, old_affinity, old_conflict) -> None:
        new_affinity = memory.affinity
        new_conflict = memory.conflict_score
        self._affinity_sum += new_affinity - old_affinity
        self._positive_ties += int(new_affinity >= 15) - int(old_affinity >= 15)
        self._hostile_ties += int(new_conflict >= 20) - int(old_conflict >= 20)

        if new_conflict >= self._max_conflict:
            self._max_conflict = new_conflict
            self._max_conflict_id = memory.other_id
        elif self._max_conflict_id == memory.other_id and new_conflict < old_conflict:
            self._recompute_max_conflict()
        self._sync_aggregate_cache()

    def _materialize_through(self, memory, through_day: int) -> None:
        old_affinity = memory.affinity
        old_conflict = memory.conflict_score
        if memory.decay_through(through_day):
            self._update_contribution(memory, old_affinity, old_conflict)

    def memory_by_id(self, other_id: int, day: int | None = None):
        memory = self.memories.get(other_id)
        if memory is not None and day is not None:
            self._materialize_through(memory, max(0, int(day) - 1))
        return memory

    def _evict_memory_if_needed(self) -> None:
        cap = int(self.memory_cap)
        if cap <= 0 or len(self.memories) < cap:
            return
        evict_id = min(
            self.memories,
            key=lambda other_id: (
                self.memories[other_id].last_day if self.memories[other_id].last_day is not None else -1,
                self.memories[other_id].familiarity,
            ),
        )
        memory = self.memories.pop(evict_id)
        affinity = memory.affinity
        conflict = memory.conflict_score
        self._affinity_sum -= affinity
        self._positive_ties -= int(affinity >= 15)
        self._hostile_ties -= int(conflict >= 20)
        if self._max_conflict_id == evict_id:
            self._recompute_max_conflict()
        self._sync_aggregate_cache()

    def memory_of(self, other: "Person", day: int | None = None) -> InteractionMemory:
        memory = self.memories.get(other.id)
        if memory is None:
            self._evict_memory_if_needed()
            decayed_through = max(0, int(day) - 1) if day is not None else 0
            memory = InteractionMemory(other_id=other.id, decayed_through_day=decayed_through)
            self.memories[other.id] = memory
            self._sync_aggregate_cache()
        elif day is not None:
            self._materialize_through(memory, max(0, int(day) - 1))
        return memory

    def remember(self, other: "Person", day: int, action: str, role: str, magnitude: float = 1.0):
        memory = self.memory_of(other, day)
        old_affinity = memory.affinity
        old_conflict = memory.conflict_score
        memory.remember(day, action, role, magnitude)
        self._update_contribution(memory, old_affinity, old_conflict)

    def observe(self, actor: "Person", day: int, action: str, magnitude: float = 1.0):
        memory = self.memory_of(actor, day)
        old_affinity = memory.affinity
        old_conflict = memory.conflict_score
        memory.remember(day, action, "witness", magnitude)
        self._update_contribution(memory, old_affinity, old_conflict)

    def aggregate_memory(self):
        return self._aggregate_cache

    def shift_ideology(self, amount: float):
        self.ideology = max(-1.0, min(1.0, self.ideology + amount))

    def decay_memories(self, day: int | None = None):
        if not self.memories:
            return
        if day is None:
            for memory in self.memories.values():
                old_affinity = memory.affinity
                old_conflict = memory.conflict_score
                memory.decay()
                self._update_contribution(memory, old_affinity, old_conflict)
            return
        day = int(day)
        if day % MEMORY_RECONCILE_INTERVAL_DAYS:
            return
        for memory in self.memories.values():
            self._materialize_through(memory, day)

from dataclasses import dataclass, field

from .memory import InteractionMemory


ADULT_AGE_DAYS = 18 * 365
RETIREMENT_AGE_DAYS = 67 * 365


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
    _memory_version: int = field(default=0, init=False, repr=False)
    _aggregate_cache_version: int = field(default=-1, init=False, repr=False)
    _aggregate_cache: dict = field(default_factory=dict, init=False, repr=False)

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

    def _invalidate_memory_cache(self) -> None:
        self._memory_version += 1

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
        self.memories.pop(evict_id, None)
        self._invalidate_memory_cache()

    def memory_of(self, other: "Person") -> InteractionMemory:
        memory = self.memories.get(other.id)
        if memory is None:
            self._evict_memory_if_needed()
            memory = InteractionMemory(other_id=other.id)
            self.memories[other.id] = memory
            self._invalidate_memory_cache()
        return memory

    def remember(self, other: "Person", day: int, action: str, role: str, magnitude: float = 1.0):
        self.memory_of(other).remember(day, action, role, magnitude)
        self._invalidate_memory_cache()

    def observe(self, actor: "Person", day: int, action: str, magnitude: float = 1.0):
        self.memory_of(actor).remember(day, action, "witness", magnitude)
        self._invalidate_memory_cache()

    def aggregate_memory(self):
        if self._aggregate_cache_version == self._memory_version:
            return self._aggregate_cache
        if not self.memories:
            result = {
                "known_people": 0,
                "mean_affinity": 0.0,
                "max_conflict": 0.0,
                "positive_ties": 0,
                "hostile_ties": 0,
            }
        else:
            count = len(self.memories)
            affinities = [m.affinity for m in self.memories.values()]
            conflicts = [m.conflict_score for m in self.memories.values()]
            result = {
                "known_people": count,
                "mean_affinity": sum(affinities) / count,
                "max_conflict": max(conflicts, default=0.0),
                "positive_ties": sum(1 for value in affinities if value >= 15),
                "hostile_ties": sum(1 for value in conflicts if value >= 20),
            }
        self._aggregate_cache = result
        self._aggregate_cache_version = self._memory_version
        return result

    def shift_ideology(self, amount: float):
        self.ideology = max(-1.0, min(1.0, self.ideology + amount))

    def decay_memories(self):
        if not self.memories:
            return
        for memory in self.memories.values():
            memory.decay()
        self._invalidate_memory_cache()

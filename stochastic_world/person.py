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

    def memory_of(self, other: "Person") -> InteractionMemory:
        memory = self.memories.get(other.id)
        if memory is None:
            memory = InteractionMemory(other_id=other.id)
            self.memories[other.id] = memory
        return memory

    def remember(self, other: "Person", day: int, action: str, role: str, magnitude: float = 1.0):
        self.memory_of(other).remember(day, action, role, magnitude)

    def observe(self, actor: "Person", day: int, action: str, magnitude: float = 1.0):
        self.memory_of(actor).remember(day, action, "witness", magnitude)

    def aggregate_memory(self):
        if not self.memories:
            return {"known_people": 0, "mean_affinity": 0.0, "max_conflict": 0.0, "positive_ties": 0, "hostile_ties": 0}
        count = len(self.memories)
        return {
            "known_people": count,
            "mean_affinity": sum(m.affinity for m in self.memories.values()) / count,
            "max_conflict": max((m.conflict_score for m in self.memories.values()), default=0.0),
            "positive_ties": sum(1 for m in self.memories.values() if m.affinity >= 15),
            "hostile_ties": sum(1 for m in self.memories.values() if m.conflict_score >= 20),
        }

    def shift_ideology(self, amount: float):
        self.ideology = max(-1.0, min(1.0, self.ideology + amount))

    def decay_memories(self):
        for memory in self.memories.values():
            memory.decay()

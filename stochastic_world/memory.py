from collections import deque
from dataclasses import dataclass, field


@dataclass
class InteractionMemory:
    other_id: int
    trust: float = 0.0
    grievance: float = 0.0
    familiarity: int = 0
    help_given: int = 0
    help_received: int = 0
    thefts_committed: int = 0
    thefts_suffered: int = 0
    attacks_committed: int = 0
    attacks_suffered: int = 0
    observed_help: int = 0
    observed_theft: int = 0
    observed_attack: int = 0
    last_day: int | None = None
    decayed_through_day: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=10))

    def remember(self, day: int, action: str, role: str, magnitude: float = 1.0) -> None:
        self.familiarity += 1
        self.last_day = day
        self.recent.append((day, action, role, round(magnitude, 3)))
        if role == "witness":
            if action == "help":
                self.observed_help += 1; self.trust += 2.5 * magnitude
            elif action == "steal":
                self.observed_theft += 1; self.trust -= 5.0 * magnitude; self.grievance += 2.5 * magnitude
            elif action == "attack":
                self.observed_attack += 1; self.trust -= 8.0 * magnitude; self.grievance += 4.0 * magnitude
            self._clamp(); return
        if action == "help":
            if role == "actor":
                self.help_given += 1; self.trust += 4 * magnitude
            else:
                self.help_received += 1; self.trust += 8 * magnitude; self.grievance -= 3 * magnitude
        elif action == "steal":
            if role == "actor":
                self.thefts_committed += 1; self.trust -= 2 * magnitude; self.grievance += 1 * magnitude
            else:
                self.thefts_suffered += 1; self.trust -= 12 * magnitude; self.grievance += 18 * magnitude
        elif action == "attack":
            if role == "actor":
                self.attacks_committed += 1; self.trust -= 3 * magnitude; self.grievance += 2 * magnitude
            else:
                self.attacks_suffered += 1; self.trust -= 18 * magnitude; self.grievance += 28 * magnitude
        self._clamp()

    def _clamp(self) -> None:
        self.trust = max(-100.0, min(100.0, self.trust))
        self.grievance = max(0.0, min(100.0, self.grievance))

    def decay_through(self, day: int, amount: float = 0.35) -> bool:
        day = int(day)
        if day <= self.decayed_through_day:
            return False
        days = day - self.decayed_through_day
        old_trust, old_grievance = self.trust, self.grievance
        self.grievance = max(0.0, self.grievance - amount * days)
        trust_decay = amount * 0.25 * days
        if self.trust > 0:
            self.trust = max(0.0, self.trust - trust_decay)
        elif self.trust < 0:
            self.trust = min(0.0, self.trust + trust_decay)
        self.decayed_through_day = day
        return self.trust != old_trust or self.grievance != old_grievance

    def decay(self, amount: float = 0.35) -> None:
        self.decay_through(self.decayed_through_day + 1, amount)

    @property
    def affinity(self) -> float:
        return max(-100.0, min(100.0, self.trust - self.grievance))

    @property
    def conflict_score(self) -> float:
        return max(0.0, self.grievance - min(0.0, self.trust))

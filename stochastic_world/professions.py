from dataclasses import dataclass


MOBILITY_INTERVAL_DAYS = 180
CLASS_ORDER = ("working", "lower_middle", "middle", "upper_middle", "affluent")


@dataclass(frozen=True)
class Profession:
    id: str
    name: str
    income_multiplier: float
    energy_multiplier: float
    workplace_kinds: tuple[str, ...]
    advancement_rate: float


PROFESSIONS = {
    "laborer": Profession("laborer", "Laborer", 0.90, 1.15, ("industrial", "outskirts"), 0.7),
    "service_worker": Profession("service_worker", "Service worker", 0.88, 1.00, ("market", "residential"), 0.8),
    "technician": Profession("technician", "Technician", 1.05, 1.00, ("industrial", "clinic"), 1.0),
    "clerk": Profession("clerk", "Clerk", 1.00, 0.90, ("market", "clinic"), 1.0),
    "teacher": Profession("teacher", "Teacher", 1.12, 0.90, ("residential",), 1.1),
    "nurse": Profession("nurse", "Nurse", 1.15, 1.05, ("clinic",), 1.15),
    "trader": Profession("trader", "Trader", 1.18, 0.95, ("market",), 1.15),
    "engineer": Profession("engineer", "Engineer", 1.35, 0.95, ("industrial",), 1.3),
    "manager": Profession("manager", "Manager", 1.45, 0.85, ("industrial", "market"), 1.25),
    "entrepreneur": Profession("entrepreneur", "Entrepreneur", 1.70, 1.00, ("market", "industrial"), 1.35),
    "executive": Profession("executive", "Executive", 1.85, 0.80, ("market", "industrial"), 1.25),
}

CLASS_PROFESSIONS = {
    "working": ("laborer", "service_worker"),
    "lower_middle": ("technician", "clerk", "service_worker"),
    "middle": ("teacher", "nurse", "trader", "technician"),
    "upper_middle": ("engineer", "manager", "trader"),
    "affluent": ("entrepreneur", "executive", "manager"),
}


def choose_profession(social_class: str, rng) -> str:
    return rng.choice(CLASS_PROFESSIONS[social_class])


def profession_for(person) -> Profession:
    return PROFESSIONS[person.profession]


def workplace_fit(person, location) -> float:
    profession = profession_for(person)
    return 1.15 if location.kind in profession.workplace_kinds else 0.82


def socioeconomic_score(person) -> float:
    return (
        person.money
        + person.food * 1.5
        + person.shelter * 0.55
        + person.health * 0.30
        + min(120.0, person.career_progress * 1.5)
    )


UP_THRESHOLDS = {
    "working": (105.0, 35),
    "lower_middle": (145.0, 80),
    "middle": (205.0, 150),
    "upper_middle": (285.0, 240),
}

DOWN_THRESHOLDS = {
    "lower_middle": 62.0,
    "middle": 92.0,
    "upper_middle": 125.0,
    "affluent": 175.0,
}


def mobility_decision(person):
    score = socioeconomic_score(person)
    index = CLASS_ORDER.index(person.social_class)

    if person.social_class in UP_THRESHOLDS:
        threshold, experience = UP_THRESHOLDS[person.social_class]
        if score >= threshold and person.work_experience >= experience:
            return CLASS_ORDER[index + 1], score, "up"

    if person.social_class in DOWN_THRESHOLDS:
        if score < DOWN_THRESHOLDS[person.social_class]:
            return CLASS_ORDER[index - 1], score, "down"

    return person.social_class, score, "stable"

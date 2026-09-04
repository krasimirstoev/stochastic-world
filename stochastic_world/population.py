import random

from faker import Faker

from .person import Person, RETIREMENT_AGE_DAYS
from .professions import choose_profession


CLASS_PROFILES = {
    "working":      {"weight": 38, "money": 10, "food": 8,  "shelter": 58, "ideology": -0.22},
    "lower_middle": {"weight": 27, "money": 18, "food": 10, "shelter": 68, "ideology": -0.08},
    "middle":       {"weight": 22, "money": 28, "food": 12, "shelter": 76, "ideology":  0.04},
    "upper_middle": {"weight": 10, "money": 45, "food": 14, "shelter": 84, "ideology":  0.18},
    "affluent":     {"weight": 3,  "money": 80, "food": 16, "shelter": 92, "ideology":  0.32},
}

AGE_BANDS = (
    (0, 14, 16),
    (15, 24, 12),
    (25, 44, 30),
    (45, 64, 25),
    (65, 84, 15),
    (85, 95, 2),
)


def _initial_age_days(rng):
    lo, hi, _ = rng.choices(AGE_BANDS, weights=[x[2] for x in AGE_BANDS], k=1)[0]
    return rng.randint(lo * 365, (hi + 1) * 365 - 1)


def build_population(size: int, seed: int, locale: str = "en_US") -> list[Person]:
    fake = Faker(locale)
    fake.seed_instance(seed)
    rng = random.Random(seed ^ 0x5A17C1A55)
    people = []
    classes = list(CLASS_PROFILES)
    weights = [CLASS_PROFILES[c]["weight"] for c in classes]

    for pid in range(size):
        social_class = rng.choices(classes, weights=weights, k=1)[0]
        profile = CLASS_PROFILES[social_class]
        age_days = _initial_age_days(rng)
        ideology = max(-1.0, min(1.0, profile["ideology"] + rng.gauss(0, 0.22)))
        if age_days < 18 * 365:
            profession = "dependent"
        elif age_days >= RETIREMENT_AGE_DAYS:
            profession = "retired"
        else:
            profession = choose_profession(social_class, rng)
        people.append(Person(
            id=pid,
            name=fake.name(),
            social_class=social_class,
            profession=profession,
            ideology=ideology,
            money=max(0, profile["money"] + rng.randint(-4, 4)),
            food=max(3, profile["food"] + rng.randint(-2, 2)),
            shelter=max(30, min(100, profile["shelter"] + rng.randint(-6, 6))),
            age_days=age_days,
            birth_day=-age_days,
            sex="female" if rng.random() < 0.5 else "male",
            retired=age_days >= RETIREMENT_AGE_DAYS,
        ))
    return people

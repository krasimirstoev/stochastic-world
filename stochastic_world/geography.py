from dataclasses import dataclass
from math import ceil


DEFAULT_TARGET_NEIGHBORHOOD_SIZE = 20_000
DISTRICT_TEMPLATES = (
    ("Residential", "residential", 0.90, 2, 0.05, False, 0),
    ("Market", "market", 1.00, 1, 0.10, True, 0),
    ("Industrial", "industrial", 1.35, 1, 0.05, False, 1),
    ("Clinic", "clinic", 0.90, 1, 0.35, True, 0),
    ("Outskirts", "outskirts", 0.65, 4, 0.12, False, 2),
)


@dataclass(frozen=True)
class Location:
    id: int
    name: str
    kind: str
    neighbors: tuple[int, ...]
    capacity_hint: int
    work_multiplier: float = 1.0
    scavenge_food_max: int = 2
    medicine_chance: float = 0.08
    market: bool = False
    shelter_decay_bonus: int = 0


def recommended_location_count(population_size: int, target_neighborhood_size: int = DEFAULT_TARGET_NEIGHBORHOOD_SIZE) -> int:
    if population_size < 1:
        raise ValueError("population_size must be >= 1")
    if target_neighborhood_size < 1:
        raise ValueError("target_neighborhood_size must be >= 1")
    return max(5, ceil(population_size / target_neighborhood_size))


def build_locations(
    count: int | None = None,
    *,
    population_size: int = 100,
    target_neighborhood_size: int = DEFAULT_TARGET_NEIGHBORHOOD_SIZE,
):
    if count is None or count <= 0:
        count = recommended_location_count(population_size, target_neighborhood_size)
    count = max(5, count)
    capacity_hint = ceil(population_size / count)

    raw = []
    for idx in range(count):
        zone = idx // len(DISTRICT_TEMPLATES) + 1
        local = idx % len(DISTRICT_TEMPLATES)
        base_name, kind, work, food, medicine, market, decay = DISTRICT_TEMPLATES[local]
        raw.append((f"{base_name} {zone}", kind, work, food, medicine, market, decay))

    adjacency = {i: set() for i in range(count)}
    zones = ceil(count / 5)
    for zone in range(zones):
        base = zone * 5
        edges = ((0, 1), (1, 2), (1, 3), (2, 4))
        for a, b in edges:
            ia, ib = base + a, base + b
            if ia < count and ib < count:
                adjacency[ia].add(ib)
                adjacency[ib].add(ia)

    for zone in range(zones - 1):
        a, b = zone * 5, (zone + 1) * 5
        for local in (1, 2):
            ia, ib = a + local, b + local
            if ia < count and ib < count:
                adjacency[ia].add(ib)
                adjacency[ib].add(ia)

    for idx in range(count):
        if not adjacency[idx] and count > 1:
            neighbor = idx - 1 if idx else 1
            adjacency[idx].add(neighbor)
            adjacency[neighbor].add(idx)

    return [
        Location(
            idx,
            name,
            kind,
            tuple(sorted(adjacency[idx])),
            capacity_hint,
            work,
            food,
            medicine,
            market,
            decay,
        )
        for idx, (name, kind, work, food, medicine, market, decay) in enumerate(raw)
    ]

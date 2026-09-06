"""Array-native aggregate elections for the large aggressive SoA engine."""

import numpy as np

from .aggressive_soa import (
    B_ADULT,
    B_ALIVE,
    F_IDEOLOGY,
    F_MONEY,
    F_TAXES,
    F_WELFARE,
    I_CRIME_SUFFERED,
    I_FOOD,
    I_LID,
)
from .politics import LEFT, RIGHT

_ELECTION_PHASE = 0x454C454354494F4E


def _rng(seed, day):
    mixed = (
        int(seed)
        ^ (int(day) * 0x9E3779B97F4A7C15)
        ^ _ELECTION_PHASE
    ) & ((1 << 63) - 1)
    return np.random.default_rng(mixed)


def run_soa_election(world, day):
    """Return aggregate votes and winner without materializing ``Person`` state."""
    world._initialize_soa()
    n = len(world.people)
    flags = world.soa_state.flags
    ints = world.soa_state.ints
    floats = world.soa_state.floats

    eligible = (flags[B_ALIVE, :n] != 0) & (flags[B_ADULT, :n] != 0)
    ids = np.flatnonzero(eligible)
    if ids.size == 0:
        votes = {"left": 0, "right": 0}
        return votes, world.politics.party_by_id("left")

    ideology = floats[F_IDEOLOGY, ids]
    money = floats[F_MONEY, ids]
    food = ints[I_FOOD, ids]
    taxes = floats[F_TAXES, ids]
    welfare = floats[F_WELFARE, ids]
    suffered = ints[I_CRIME_SUFFERED, ids]
    lids = ints[I_LID, ids]

    left_score = -np.abs(ideology - float(LEFT.ideology))
    right_score = -np.abs(ideology - float(RIGHT.ideology))
    left_score += ((money < 8.0) | (food <= 3)) * 0.22
    left_score += (welfare > taxes) * 0.06
    right_score += (taxes > welfare + 15.0) * 0.06

    crime_by_location = np.zeros(len(world.locations), dtype=np.float64)
    for location in world.locations:
        lid = int(location.id)
        history = world.crime_history[lid]
        population = max(1, int(world._domain_location_population.get(lid, 0)))
        if history:
            crime_by_location[lid] = sum(history) / (population * len(history))
    right_score += np.minimum(0.28, crime_by_location[lids] * 0.9)
    right_score += np.minimum(0.18, suffered.astype(np.float64) * 0.025)

    rng = _rng(world._soa_seed, day)
    left_score += rng.uniform(-0.08, 0.08, size=ids.size)
    right_score += rng.uniform(-0.08, 0.08, size=ids.size)
    left_votes = int(np.count_nonzero(left_score >= right_score))
    right_votes = int(ids.size - left_votes)
    votes = {"left": left_votes, "right": right_votes}
    winner_id = "left" if left_votes >= right_votes else "right"
    return votes, world.politics.party_by_id(winner_id)

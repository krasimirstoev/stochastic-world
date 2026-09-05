"""Vectorized social mobility for the large aggressive SoA engine."""

from __future__ import annotations

import numpy as np

from .aggressive_economy import PROFESSION_TO_CODE, SOCIAL_CLASSES
from .aggressive_soa import (
    B_ALIVE,
    B_WORKING_AGE,
    F_CAREER,
    F_IDEOLOGY,
    F_MONEY,
    I_DAYS_IN_CLASS,
    I_FOOD,
    I_HEALTH,
    I_PROFESSION,
    I_SHELTER,
    I_SOCIAL_CLASS,
    I_WORK_EXP,
)
from .professions import CLASS_PROFESSIONS

_MOBILITY_PHASE = 0x4D4F4249

_UP_SCORE = np.asarray((105.0, 145.0, 205.0, 285.0, np.inf), dtype=np.float64)
_UP_EXP = np.asarray((35, 80, 150, 240, 2**31 - 1), dtype=np.int32)
_DOWN_SCORE = np.asarray((-np.inf, 62.0, 92.0, 125.0, 175.0), dtype=np.float64)

_CLASS_PROFESSION_CODES = tuple(
    np.asarray([PROFESSION_TO_CODE[name] for name in CLASS_PROFESSIONS[class_name]], dtype=np.int32)
    for class_name in SOCIAL_CLASSES
)


def _rng(seed: int, day: int):
    mixed = (int(seed) ^ (int(day) * 0x9E3779B97F4A7C15) ^ _MOBILITY_PHASE) & ((1 << 63) - 1)
    return np.random.default_rng(mixed)


def _adjust_social_stats(world, stats, changed, old_class, new_class):
    """Move already-computed EOD aggregates between social-class buckets."""
    if not stats or changed.size == 0:
        return
    social = stats.get("social")
    if not social:
        return
    ints = world.soa_state.ints
    floats = world.soa_state.floats
    for class_code in range(len(SOCIAL_CLASSES)):
        leaving = changed[old_class == class_code]
        entering = changed[new_class == class_code]
        bucket = social[class_code]
        if leaving.size:
            bucket[0] -= int(leaving.size)
            bucket[1] -= float(floats[F_MONEY, leaving].sum())
            bucket[2] -= float(ints[I_FOOD, leaving].sum())
            bucket[3] -= float(ints[I_SHELTER, leaving].sum())
            bucket[4] -= float(ints[I_HEALTH, leaving].sum())
            bucket[5] -= float(floats[F_IDEOLOGY, leaving].sum())
            bucket[6] -= float(ints[I_WORK_EXP, leaving].sum())
        if entering.size:
            bucket[0] += int(entering.size)
            bucket[1] += float(floats[F_MONEY, entering].sum())
            bucket[2] += float(ints[I_FOOD, entering].sum())
            bucket[3] += float(ints[I_SHELTER, entering].sum())
            bucket[4] += float(ints[I_HEALTH, entering].sum())
            bucket[5] += float(floats[F_IDEOLOGY, entering].sum())
            bucket[6] += float(ints[I_WORK_EXP, entering].sum())


def run_soa_mobility(world, day: int, stats=None):
    """Apply the exact score thresholds over authoritative arrays.

    Compact aggressive mode intentionally stores one aggregate mobility event
    instead of per-agent mobility_history + social_mobility rows.
    """
    n = len(world.people)
    if n <= 0:
        return 0
    ints = world.soa_state.ints
    floats = world.soa_state.floats
    flags = world.soa_state.flags
    eligible = (flags[B_ALIVE, :n] != 0) & (flags[B_WORKING_AGE, :n] != 0)
    ids = np.flatnonzero(eligible)
    if ids.size == 0:
        return 0

    current = ints[I_SOCIAL_CLASS, ids].astype(np.int16, copy=True)
    score = (
        floats[F_MONEY, ids]
        + ints[I_FOOD, ids].astype(np.float64) * 1.5
        + ints[I_SHELTER, ids].astype(np.float64) * 0.55
        + ints[I_HEALTH, ids].astype(np.float64) * 0.30
        + np.minimum(120.0, floats[F_CAREER, ids] * 1.5)
    )
    experience = ints[I_WORK_EXP, ids]
    new_class = current.copy()
    direction = np.zeros(ids.size, dtype=np.int8)

    valid_class = (current >= 0) & (current < len(SOCIAL_CLASSES))
    up = valid_class & (current < len(SOCIAL_CLASSES) - 1)
    up &= score >= _UP_SCORE[np.clip(current, 0, len(SOCIAL_CLASSES) - 1)]
    up &= experience >= _UP_EXP[np.clip(current, 0, len(SOCIAL_CLASSES) - 1)]
    down = valid_class & (current > 0)
    down &= score < _DOWN_SCORE[np.clip(current, 0, len(SOCIAL_CLASSES) - 1)]

    new_class[up] += 1
    direction[up] = 1
    new_class[down] -= 1
    direction[down] = -1
    changed_mask = direction != 0
    changed = ids[changed_mask]
    if changed.size == 0:
        return 0

    old_changed = current[changed_mask]
    new_changed = new_class[changed_mask]
    rng = _rng(world._soa_seed, day)
    for class_code, choices in enumerate(_CLASS_PROFESSION_CODES):
        class_ids = changed[new_changed == class_code]
        if class_ids.size:
            ints[I_PROFESSION, class_ids] = choices[
                rng.integers(0, len(choices), size=class_ids.size)
            ]

    ints[I_SOCIAL_CLASS, changed] = new_changed.astype(np.int32)
    ints[I_DAYS_IN_CLASS, changed] = 0
    up_ids = changed[direction[changed_mask] > 0]
    down_ids = changed[direction[changed_mask] < 0]
    if up_ids.size:
        floats[F_CAREER, up_ids] *= 0.55
    if down_ids.size:
        floats[F_CAREER, down_ids] *= 0.70

    _adjust_social_stats(world, stats, changed, old_changed, new_changed)

    count = int(changed.size)
    world.total_mobility_changes += count
    transitions = {}
    for old_code in range(len(SOCIAL_CLASSES)):
        for new_code in range(len(SOCIAL_CLASSES)):
            if old_code == new_code:
                continue
            amount = int(np.count_nonzero((old_changed == old_code) & (new_changed == new_code)))
            if amount:
                transitions[f"{SOCIAL_CLASSES[old_code]}->{SOCIAL_CLASSES[new_code]}"] = amount
    world.store.event(
        day,
        world.next_sequence(),
        "social_mobility",
        aggregate_only=1,
        changed=count,
        up=int(up_ids.size),
        down=int(down_ids.size),
        transitions=transitions,
    )
    return count

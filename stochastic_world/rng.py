import os
import random


def entropy_seed() -> int:
    return int.from_bytes(os.urandom(32), "big")


def make_rng(seed: int | None = None) -> tuple[int, random.Random]:
    actual_seed = entropy_seed() if seed is None else seed
    return actual_seed, random.Random(actual_seed)


def derive_run_seeds(master_seed: int, runs: int) -> list[int]:
    if runs == 1:
        return [master_seed]

    master_rng = random.Random(master_seed)
    return [master_rng.getrandbits(256) for _ in range(runs)]

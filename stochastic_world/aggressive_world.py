"""Public entry point for the aggressive engine."""

from time import perf_counter

from . import population as _population
from .aggressive_jit import install as _install_jit

try:
    from .aggressive_world_demographic import AggressiveParallelAgentWorld as _BaseWorld
except ImportError:
    try:
        from .aggressive_world_soa import AggressiveParallelAgentWorld as _BaseWorld
    except ImportError:
        from .aggressive_world_temporal import AggressiveParallelAgentWorld as _BaseWorld


class AggressiveParallelAgentWorld(_BaseWorld):
    """SoA / temporal aggressive world with optional Numba acceleration."""

    def __init__(self, *args, **kwargs):
        jit_started = perf_counter()
        self._aggressive_jit_enabled = bool(_install_jit())
        jit_seconds = perf_counter() - jit_started

        self._startup_seed_employment = 0.0
        world_started = perf_counter()
        super().__init__(*args, **kwargs)
        world_seconds = perf_counter() - world_started

        population_profile = dict(getattr(_population, "LAST_BUILD_PROFILE", {}) or {})
        store_profile = dict(getattr(self.store, "startup_profile", {}) or {})
        population_seconds = float(population_profile.get("seconds", 0.0))
        sqlite_persons = float(store_profile.get("sqlite_persons", 0.0))
        seed_employment = float(getattr(self, "_startup_seed_employment", 0.0))
        self._aggressive_startup_profile = {
            "jit_warmup": jit_seconds,
            "population": population_seconds,
            "sqlite_persons": sqlite_persons,
            "seed_employment": seed_employment,
            "world_other": max(
                0.0,
                world_seconds - population_seconds - sqlite_persons - seed_employment,
            ),
            "world_total": world_seconds,
            "fast_identity": bool(population_profile.get("fast_identity", False)),
            "person_rows": int(store_profile.get("person_rows", 0)),
            "initial_employment_rows_skipped": int(
                store_profile.get("initial_employment_rows_skipped", 0)
            ),
        }
        self._aggressive_election_seconds = 0.0
        self._aggressive_election_calls = 0
        self._aggressive_soa_election_calls = 0

    def _seed_employment(self):
        started = perf_counter()
        result = super()._seed_employment()
        self._startup_seed_employment = perf_counter() - started
        return result

    def _large_compact_election(self):
        return len(self.people) >= 100_000 and self.store.event_mode == "compact"

    def run_election(self):
        if not self._large_compact_election():
            return super().run_election()

        started = perf_counter()
        forced = self.current_day == 1 and self.politics.initial_government in ("left", "right")
        votes = {"left": 0, "right": 0}

        # Day 1 still uses the original Person path when the result is not
        # forced.  Those objects are authoritative before the first action and
        # this preserves the established seed-12345 opening election.  Later
        # elections read the authoritative SoA arrays directly and avoid a full
        # 100k+/million-agent materialization barrier.
        soa_election = bool(
            self.current_day != 1
            and getattr(self, "soa_mode", False)
            and getattr(self, "_soa_initialized", False)
        )
        if forced:
            winner = self.politics.party_by_id(self.politics.initial_government)
        elif soa_election:
            from .aggressive_election import run_soa_election

            votes, winner = run_soa_election(self, self.current_day)
            self._aggressive_soa_election_calls += 1
        else:
            crime_rates = self.crime_rates()
            for person in self.people:
                if not person.alive or not getattr(person, "is_adult", True):
                    continue
                party = self.politics.vote(
                    person,
                    crime_rates.get(person.location_id, 0.0),
                    self.rng,
                )
                votes[party.id] += 1
            winner_id = "left" if votes["left"] >= votes["right"] else "right"
            winner = self.politics.party_by_id(winner_id)

        self.politics.government = winner
        self.politics.election_number += 1
        self.politics.last_election_day = self.current_day
        # Large compact runs preserve aggregate election outcomes but deliberately
        # omit per-voter ballot persistence.
        self.store.election(self.current_day, self.politics, votes, (), winner)
        self.store.event(
            self.current_day,
            self.next_sequence(),
            "election",
            left_votes=votes["left"],
            right_votes=votes["right"],
            winner=winner.id,
            representative=self.politics.representatives[winner.id],
            forced=int(forced),
            aggregate_only=1,
            soa=int(soa_election),
        )
        self._aggressive_election_seconds += perf_counter() - started
        self._aggressive_election_calls += 1

    def close_parallel(self):
        if getattr(self, "temporal_mode", False):
            mode = "numba=on" if self._aggressive_jit_enabled else "numba=off (python fallback)"
            print(f"  aggressive JIT kernels: {mode}")
            p = getattr(self, "_aggressive_startup_profile", {})
            if p:
                print("  aggressive startup profile:")
                print(f"    jit_warmup             {p['jit_warmup']:8.3f}s")
                print(f"    population             {p['population']:8.3f}s")
                print(f"    sqlite_persons         {p['sqlite_persons']:8.3f}s")
                print(f"    seed_employment        {p['seed_employment']:8.3f}s")
                print(f"    world_other            {p['world_other']:8.3f}s")
                print(f"    world_total            {p['world_total']:8.3f}s")
                print(
                    "    startup_mode           "
                    f"fast_identity={'on' if p['fast_identity'] else 'off'} "
                    f"person_rows={p['person_rows']} "
                    f"initial_employment_rows_skipped={p['initial_employment_rows_skipped']}"
                )
            if self._aggressive_election_calls:
                print(
                    "  aggressive aggregate elections: "
                    f"calls={self._aggressive_election_calls} "
                    f"soa={self._aggressive_soa_election_calls} "
                    f"wall={self._aggressive_election_seconds:.3f}s"
                )
        super().close_parallel()


__all__ = ["AggressiveParallelAgentWorld"]

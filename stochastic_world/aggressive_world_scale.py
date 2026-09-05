"""Large-population fast lane layered on the RAM-first aggressive engine.

For 100k+ agents the dominant cost moves away from worker planning and into
Python object marshalling at end-of-day plus repeated full-population statistics
scans.  This layer keeps end-of-day transport in shared memory and folds the
population statistics into one pass.
"""

from collections import defaultdict
from time import perf_counter

from .aggressive_endofday import SharedEndOfDayBuffers, SharedEndOfDayPool
from .aggressive_world_ram import AggressiveParallelAgentWorld as RamAggressiveWorld
from .demographics import DEMOGRAPHIC_INTERVAL_DAYS
from .labor_market import BUSINESS_INTERVAL_DAYS
from .politics import ELECTION_INTERVAL_DAYS
from .professions import MOBILITY_INTERVAL_DAYS


_LARGE_POPULATION = 100_000
_SOCIAL_CLASSES = ("working", "lower_middle", "middle", "upper_middle", "affluent")


class AggressiveParallelAgentWorld(RamAggressiveWorld):
    """Aggressive engine with a shared daily pipeline for very large worlds."""

    def __init__(self, *args, agent_workers=0, agent_worker_min_active=64, **kwargs):
        super().__init__(
            *args,
            agent_workers=agent_workers,
            agent_worker_min_active=agent_worker_min_active,
            **kwargs,
        )
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        self.large_scale_mode = len(self.people) >= _LARGE_POPULATION
        self.shared_end_buffers = SharedEndOfDayBuffers(self.shared_buffers.population)
        self.shared_end_pool = SharedEndOfDayPool(
            seed,
            self.shared_end_buffers,
            workers=agent_workers,
        )
        self._eod_population_stats = None

    def _new_population_accumulator(self):
        return {
            "alive": 0,
            "food": 0.0,
            "money": 0.0,
            "medicine": 0.0,
            "energy": 0.0,
            "shelter": 0.0,
            "health": 0.0,
            "ideology": 0.0,
            "taxes": 0.0,
            "welfare": 0.0,
            "left": 0,
            "right": 0,
            "workforce": 0,
            "employed": 0,
            "loc_n": [0] * len(self.locations),
            "loc_food": [0.0] * len(self.locations),
            "loc_money": [0.0] * len(self.locations),
            "loc_health": [0.0] * len(self.locations),
            "social": {
                name: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                for name in _SOCIAL_CLASSES
            },
        }

    @staticmethod
    def _accumulate_population_row(stats, person):
        stats["alive"] += 1
        food = float(person.food)
        money = float(person.money)
        medicine = float(person.medicine)
        energy = float(person.energy)
        shelter = float(person.shelter)
        health = float(person.health)
        ideology = float(person.ideology)
        stats["food"] += food
        stats["money"] += money
        stats["medicine"] += medicine
        stats["energy"] += energy
        stats["shelter"] += shelter
        stats["health"] += health
        stats["ideology"] += ideology
        stats["taxes"] += float(person.taxes_paid)
        stats["welfare"] += float(person.welfare_received)
        if ideology < 0:
            stats["left"] += 1
        else:
            stats["right"] += 1

        lid = int(person.location_id)
        stats["loc_n"][lid] += 1
        stats["loc_food"][lid] += food
        stats["loc_money"][lid] += money
        stats["loc_health"][lid] += health

        bucket = stats["social"].get(person.social_class)
        if bucket is not None:
            bucket[0] += 1
            bucket[1] += money
            bucket[2] += food
            bucket[3] += shelter
            bucket[4] += health
            bucket[5] += ideology
            bucket[6] += float(person.work_experience)

        if person.is_working_age:
            stats["workforce"] += 1
            stats["employed"] += int(person.employer_id is not None)

    def _run_parallel_end_of_day(self):
        if not self.large_scale_mode or not self.shared_end_pool.enabled:
            return super()._run_parallel_end_of_day()

        for location in self.locations:
            self.crime_history[location.id].append(
                self.daily_crimes.get(location.id, 0)
            )
        rates = self.crime_rates()

        started = perf_counter()
        population_size = len(self.people)
        self.shared_end_buffers.begin_sync(population_size)
        for person in self.people:
            if not person.alive:
                continue
            self.demographics.support_dependent(person)
            location = self.locations[person.location_id]
            self.shared_end_buffers.write_person(
                person,
                location.shelter_decay_bonus,
                rates.get(person.location_id, 0.0),
            )
        self._record_phase("eod_shared_sync", started)

        started = perf_counter()
        self.shared_end_pool.run(self.current_day, population_size)
        self._record_phase("eod_dispatch", started)

        started = perf_counter()
        stats = self._new_population_accumulator()
        for person in self.people:
            if not person.alive or not self.shared_end_buffers.is_active(person.id):
                continue
            self._apply_end_delta(
                person,
                self.shared_end_buffers.read_delta(person.id),
            )
            if person.alive:
                self._accumulate_population_row(stats, person)
        self._eod_population_stats = (int(self.current_day), stats)
        self._record_phase("eod_apply", started)
        self.daily_crimes.clear()
        return stats

    def _scan_population_stats(self):
        stats = self._new_population_accumulator()
        for person in self.people:
            if person.alive:
                self._accumulate_population_row(stats, person)
        return stats

    def _write_population_stats_fast(self, day, police_snapshot):
        """Write population-derived daily statistics, reusing fused EOD stats when available."""
        started = perf_counter()
        cached = self._eod_population_stats
        if cached is not None and int(cached[0]) == int(day):
            stats = cached[1]
            self._eod_population_stats = None
        else:
            stats = self._scan_population_stats()

        alive = int(stats["alive"])
        n = max(1, alive)
        conn = self.store.conn
        sid = self.store.simulation_id
        conn.execute(
            "INSERT INTO daily_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sid, day, alive,
                stats["food"] / n, stats["money"] / n, stats["medicine"] / n,
                stats["energy"] / n, stats["shelter"] / n, stats["health"] / n,
                self.total_helps, self.total_thefts, self.total_attacks,
                self.total_observations, self.total_deaths,
                self.total_mobility_changes,
            ),
        )

        location_rows = []
        for location in self.locations:
            lid = location.id
            count = stats["loc_n"][lid]
            denom = max(1, count)
            location_rows.append(
                (
                    sid, day, lid, count,
                    stats["loc_food"][lid] / denom,
                    stats["loc_money"][lid] / denom,
                    stats["loc_health"][lid] / denom,
                    self.local_crime_rate(lid),
                )
            )
        conn.executemany(
            "INSERT INTO location_stats VALUES(?,?,?,?,?,?,?,?)",
            location_rows,
        )

        conn.execute(
            "INSERT INTO political_stats VALUES(?,?,?,?,?,?,?,?,?)",
            (
                sid, day, self.politics.government.id, self.politics.treasury,
                stats["ideology"] / n, stats["left"], stats["right"],
                stats["taxes"] / n, stats["welfare"] / n,
            ),
        )

        social_rows = []
        for name in _SOCIAL_CLASSES:
            bucket = stats["social"][name]
            count = bucket[0]
            denom = max(1, count)
            social_rows.append(
                (
                    sid, day, name, count,
                    bucket[1] / denom, bucket[2] / denom,
                    bucket[3] / denom, bucket[4] / denom,
                    bucket[5] / denom, bucket[6] / denom,
                )
            )
        conn.executemany(
            "INSERT INTO social_stats VALUES(?,?,?,?,?,?,?,?,?,?)",
            social_rows,
        )

        workforce = int(stats["workforce"])
        employed = int(stats["employed"])
        active = [e for e in self.labor_market.employers if e.alive]
        unemployed = max(0, workforce - employed)
        conn.execute(
            "INSERT INTO labor_stats VALUES(?,?,?,?,?,?,?,?)",
            (
                sid, day, employed, unemployed,
                unemployed / workforce if workforce else 0.0,
                sum(e.vacancies for e in active),
                len(active),
                sum(e.capacity for e in active),
            ),
        )

        self.store.write_market_stats(day, self)
        self.store.write_police_stats(day, police_snapshot)
        self._record_phase("population_stats_single_pass", started)

    def run_day(self, day):
        if not self.large_scale_mode:
            return super().run_day(day)

        self.current_day = day
        if day == 1 or (day - 1) % ELECTION_INTERVAL_DAYS == 0:
            self.run_election()

        self._run_parallel_actions(day)

        for shipment in self.transport.rebalance(day):
            self.store.shipment(shipment)

        if day % BUSINESS_INTERVAL_DAYS == 0:
            self.welfare_cycle()
            self.business_cycle()
            self.police.rebalance()

        self._run_parallel_end_of_day()
        self.goods_market.reprice()

        if day % MOBILITY_INTERVAL_DAYS == 0:
            self.mobility_cycle()
            self._eod_population_stats = None

        police_snapshot = self.police.end_day()
        self._write_population_stats_fast(day, police_snapshot)
        self.store.commit_day()

        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            self.demographics.cycle(day)
            self.demographics.write_stats(day)
            self.store.commit_day()

    def close_parallel(self):
        try:
            if self.large_scale_mode:
                summary = self.shared_end_pool.summary()
                if summary["days"]:
                    print(
                        "  aggressive shared end-of-day: "
                        f"days={summary['days']} tasks={summary['tasks']} "
                        f"items={summary['items']} "
                        f"worker_cpu={summary['worker_seconds']:.3f}s "
                        f"dispatch_wall={summary['dispatch_seconds']:.3f}s "
                        f"shm={summary['shared_bytes'] / (1024 * 1024):.1f}MiB"
                    )
            self.shared_end_pool.close()
        finally:
            try:
                self.shared_end_buffers.close(unlink=True)
            finally:
                super().close_parallel()

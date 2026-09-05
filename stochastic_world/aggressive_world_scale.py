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
        for person in self.people:
            if not person.alive or not self.shared_end_buffers.is_active(person.id):
                continue
            self._apply_end_delta(
                person,
                self.shared_end_buffers.read_delta(person.id),
            )
        self._record_phase("eod_apply", started)
        self.daily_crimes.clear()

    def _write_population_stats_fast(self, day, police_snapshot):
        """Write all population-derived daily statistics from one O(N) scan."""
        started = perf_counter()
        location_count = len(self.locations)
        loc_n = [0] * location_count
        loc_food = [0.0] * location_count
        loc_money = [0.0] * location_count
        loc_health = [0.0] * location_count

        alive = 0
        sum_food = sum_money = sum_medicine = 0.0
        sum_energy = sum_shelter = sum_health = 0.0
        sum_ideology = sum_taxes = sum_welfare = 0.0
        left_leaning = right_leaning = 0
        workforce = employed = 0

        social = {
            name: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for name in _SOCIAL_CLASSES
        }

        for person in self.people:
            if not person.alive:
                continue
            alive += 1
            food = float(person.food)
            money = float(person.money)
            medicine = float(person.medicine)
            energy = float(person.energy)
            shelter = float(person.shelter)
            health = float(person.health)
            ideology = float(person.ideology)

            sum_food += food
            sum_money += money
            sum_medicine += medicine
            sum_energy += energy
            sum_shelter += shelter
            sum_health += health
            sum_ideology += ideology
            sum_taxes += float(person.taxes_paid)
            sum_welfare += float(person.welfare_received)
            if ideology < 0:
                left_leaning += 1
            else:
                right_leaning += 1

            lid = int(person.location_id)
            loc_n[lid] += 1
            loc_food[lid] += food
            loc_money[lid] += money
            loc_health[lid] += health

            bucket = social.get(person.social_class)
            if bucket is not None:
                bucket[0] += 1
                bucket[1] += money
                bucket[2] += food
                bucket[3] += shelter
                bucket[4] += health
                bucket[5] += ideology
                bucket[6] += float(person.work_experience)

            if person.is_working_age:
                workforce += 1
                employed += int(person.employer_id is not None)

        n = max(1, alive)
        conn = self.store.conn
        sid = self.store.simulation_id
        conn.execute(
            "INSERT INTO daily_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                sid, day, alive,
                sum_food / n, sum_money / n, sum_medicine / n,
                sum_energy / n, sum_shelter / n, sum_health / n,
                self.total_helps, self.total_thefts, self.total_attacks,
                self.total_observations, self.total_deaths,
                self.total_mobility_changes,
            ),
        )

        location_rows = []
        for location in self.locations:
            lid = location.id
            count = loc_n[lid]
            denom = max(1, count)
            location_rows.append(
                (
                    sid, day, lid, count,
                    loc_food[lid] / denom,
                    loc_money[lid] / denom,
                    loc_health[lid] / denom,
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
                sum_ideology / n, left_leaning, right_leaning,
                sum_taxes / n, sum_welfare / n,
            ),
        )

        social_rows = []
        for name in _SOCIAL_CLASSES:
            bucket = social[name]
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

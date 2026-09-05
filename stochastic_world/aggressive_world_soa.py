"""SoA / Numba large-world fast lane layered over temporal BSP."""

from math import ceil
from time import perf_counter

import numpy as np

from .aggressive_jit import JIT_ENABLED
from .aggressive_soa import (
    B_ALIVE, F_CAREER, F_IDEOLOGY, F_LIFETIME_GROSS, F_MARKET_SPENDING,
    F_MONEY, F_TAXES, F_WELFARE,
    I_ARRESTS, I_CRIME_SUFFERED, I_DAYS_IN_CLASS, I_DETAINED, I_EMPLOYER,
    I_ENERGY, I_FOOD, I_HEALTH, I_JOBS_HELD, I_LID, I_LIFETIME_UNEMPLOYMENT,
    I_MED, I_PROFESSION, I_SHELTER, I_SHORTAGES, I_SOCIAL_CLASS,
    I_UNEMPLOYMENT, I_WORK_EXP,
    SharedCSRLocationIndex, SharedRelationMemory, SharedSoAAgentState,
    SoADomainPool, warmup as warmup_soa,
)
from .aggressive_economy import PROFESSION_NAMES, SOCIAL_CLASSES
from .aggressive_world_scale import AggressiveParallelAgentWorld as ScaleAggressiveWorld
from .aggressive_world_temporal import AggressiveParallelAgentWorld as TemporalAggressiveWorld
from .demographics import DEMOGRAPHIC_INTERVAL_DAYS
from .labor_market import BUSINESS_INTERVAL_DAYS, Employer
from .politics import ELECTION_INTERVAL_DAYS
from .professions import MOBILITY_INTERVAL_DAYS


_SOA_MIN_POPULATION = 100_000
_WELFARE_PHASE = 0x57454C46


class AggressiveParallelAgentWorld(TemporalAggressiveWorld):
    """Numba SoA engine for 100k+ agents with cold lifecycle barriers."""

    def __init__(self, *args, agent_workers=0, agent_worker_min_active=64, **kwargs):
        population_size = int(args[1]) if len(args) > 1 else int(kwargs.get("population_size", 0))
        if JIT_ENABLED and population_size >= _SOA_MIN_POPULATION:
            warmup_soa()
        super().__init__(*args, agent_workers=agent_workers, agent_worker_min_active=agent_worker_min_active, **kwargs)
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        self._soa_seed = int(seed)
        self.soa_mode = bool(JIT_ENABLED and self.temporal_mode and len(self.people) >= _SOA_MIN_POPULATION)
        self.soa_state = None
        self.soa_memory = None
        self.soa_index = None
        self.soa_pool = None
        self._soa_initialized = False
        if self.soa_mode:
            capacity = int(self.shared_buffers.population)
            self.soa_state = SharedSoAAgentState(capacity)
            self.soa_memory = SharedRelationMemory(capacity)
            self.soa_index = SharedCSRLocationIndex(capacity, len(self.locations))
            self.soa_pool = SoADomainPool(seed, self.soa_state, self.soa_memory, self.soa_index, len(self.locations), workers=agent_workers)

    def _initialize_soa(self):
        if self._soa_initialized:
            return
        started = perf_counter()
        self.soa_state.sync_world(self)
        self._soa_initialized = True
        self._record_phase("soa_initial_sync", started)

    def _rebuild_soa_index(self):
        started = perf_counter()
        count = self.soa_pool.rebuild_index(len(self.people))
        self._record_phase("soa_index_sync", started)
        return count

    def _run_soa_domain(self, day, *, fuse_eod):
        self._initialize_soa()
        active = self._rebuild_soa_index()
        if active <= 0 or self.actions_per_day <= 0:
            return [], None
        started = perf_counter()
        packet = self._build_temporal_packet()
        self._record_phase("soa_packet", started)
        started = perf_counter()
        results, stats = self.soa_pool.run_day(day, self.actions_per_day, self.encounter_sample, self.max_witnesses, self.visibility, packet, fuse_eod=fuse_eod)
        self._record_phase("soa_dispatch", started)
        started = perf_counter()
        self._apply_temporal_aggregates(results, stats, fused_eod=fuse_eod)
        self._record_phase("soa_aggregate", started)
        return results, stats

    def _soa_reconcile_to_world(self, *, count_new_deaths=False):
        if not self._soa_initialized:
            return
        started = perf_counter()
        old_alive = int(self.alive_count)
        location_count = len(self.locations)
        self.population_index.members = [[] for _ in range(location_count)]
        self.population_index.positions = [-1] * len(self.people)
        for employer in self.labor_market.employers:
            employer.employee_ids.clear()
        ints = self.soa_state.ints
        floats = self.soa_state.floats
        flags = self.soa_state.flags
        alive = 0
        workforce = 0
        for person in self.people:
            pid = int(person.id)
            person.location_id = int(ints[I_LID, pid])
            person.food = int(ints[I_FOOD, pid])
            person.medicine = int(ints[I_MED, pid])
            person.energy = int(ints[I_ENERGY, pid])
            person.health = int(ints[I_HEALTH, pid])
            person.shelter = int(ints[I_SHELTER, pid])
            person.money = float(floats[F_MONEY, pid])
            employer_id = int(ints[I_EMPLOYER, pid])
            person.employer_id = None if employer_id < 0 else employer_id
            profession_code = int(ints[I_PROFESSION, pid])
            if 0 <= profession_code < len(PROFESSION_NAMES):
                person.profession = PROFESSION_NAMES[profession_code]
            class_code = int(ints[I_SOCIAL_CLASS, pid])
            if 0 <= class_code < len(SOCIAL_CLASSES):
                person.social_class = SOCIAL_CLASSES[class_code]
            person.ideology = float(floats[F_IDEOLOGY, pid])
            person.detained_until_day = int(ints[I_DETAINED, pid])
            person.alive = bool(flags[B_ALIVE, pid])
            person.taxes_paid = float(floats[F_TAXES, pid])
            person.welfare_received = float(floats[F_WELFARE, pid])
            person.work_experience = int(ints[I_WORK_EXP, pid])
            person.career_progress = float(floats[F_CAREER, pid])
            person.lifetime_gross_income = float(floats[F_LIFETIME_GROSS, pid])
            person.market_spending = float(floats[F_MARKET_SPENDING, pid])
            person.shortage_experiences = int(ints[I_SHORTAGES, pid])
            person.crime_suffered = int(ints[I_CRIME_SUFFERED, pid])
            person.arrests = int(ints[I_ARRESTS, pid])
            person.unemployment_days = int(ints[I_UNEMPLOYMENT, pid])
            person.lifetime_unemployment_days = int(ints[I_LIFETIME_UNEMPLOYMENT, pid])
            person.days_in_class = int(ints[I_DAYS_IN_CLASS, pid])
            person.jobs_held = int(ints[I_JOBS_HELD, pid])
            if not person.alive:
                continue
            alive += 1
            self.population_index.add(pid, person.location_id)
            if person.is_working_age:
                workforce += 1
            if person.employer_id is not None:
                employer = self.labor_market.employer_any(person.employer_id)
                if employer is not None:
                    employer.employee_ids.add(pid)
        if count_new_deaths and alive < old_alive:
            self.total_deaths += old_alive - alive
        self.alive_count = alive
        self.demographics.working_age_count = workforce
        self.invalidate_living_cache()
        self._domain_location_population = {int(loc.id): self.population_index.population(loc.id) for loc in self.locations}
        for lid, count in self._domain_location_population.items():
            self.goods_market.set_population(lid, count)
        self._domain_employee_counts = {}
        for employer in self.labor_market.employers:
            count = len(employer.employee_ids)
            self._domain_employee_counts[int(employer.id)] = count
            employer._domain_employee_count = count
        self._record_phase("soa_cold_reconcile", started)

    def _resync_soa_from_world(self):
        started = perf_counter()
        self.soa_state.sync_world(self)
        self._soa_initialized = True
        self._domain_employee_counts = {int(e.id): len(e.employee_ids) for e in self.labor_market.employers}
        for e in self.labor_market.employers:
            e._domain_employee_count = self._domain_employee_counts[int(e.id)]
        self._domain_location_population = {int(loc.id): self.population_index.population(loc.id) for loc in self.locations}
        self._record_phase("soa_cold_resync", started)

    def _soa_welfare_cycle(self, day):
        """Apply monthly welfare directly to authoritative SoA state."""
        party = self.politics.government
        treasury = float(self.politics.treasury)
        base_cost = int(party.welfare_cash + party.welfare_food)
        if treasury < base_cost or base_cost <= 0:
            return 0
        n = len(self.people)
        alive = self.soa_state.flags[B_ALIVE, :n] != 0
        money = self.soa_state.floats[F_MONEY, :n]
        eligible = np.flatnonzero(alive & (money <= float(party.welfare_money_threshold)))
        if eligible.size == 0:
            return 0
        rng = np.random.default_rng((self._soa_seed ^ (int(day) << 17) ^ _WELFARE_PHASE) & ((1 << 63) - 1))
        medicine = rng.random(eligible.size) < float(party.welfare_medicine_chance)
        costs = np.full(eligible.size, base_cost, dtype=np.int64)
        costs += medicine.astype(np.int64) * 2
        cumulative = np.cumsum(costs, dtype=np.int64)
        count = int(np.searchsorted(cumulative, treasury, side="right"))
        if count <= 0:
            return 0
        ids = eligible[:count]
        med = medicine[:count]
        actual_costs = costs[:count]
        self.soa_state.floats[F_MONEY, ids] += float(party.welfare_cash)
        self.soa_state.ints[I_FOOD, ids] += int(party.welfare_food)
        if np.any(med):
            self.soa_state.ints[I_MED, ids[med]] += 1
        self.soa_state.floats[F_WELFARE, ids] += actual_costs.astype(np.float64)
        shifts = np.minimum(0.0035, actual_costs.astype(np.float64) * 0.00045)
        self.soa_state.floats[F_IDEOLOGY, ids] = np.maximum(
            -1.0,
            self.soa_state.floats[F_IDEOLOGY, ids] - shifts,
        )
        self.politics.treasury = max(0.0, treasury - float(cumulative[count - 1]))
        return count

    def _soa_business_cycle(self, day):
        """Run the firm review from aggregate employer counts, without Person scans."""
        n = len(self.people)
        ints = self.soa_state.ints
        flags = self.soa_state.flags
        alive_mask = flags[B_ALIVE, :n] != 0
        changes = []
        for employer in self.labor_market.employers:
            if not employer.alive:
                continue
            employee_count = int(self._domain_employee_counts.get(int(employer.id), 0))
            margin = float(employer.revenue_since_review - employer.payroll_since_review)
            if employer.cash < 4 or (
                margin < -22 and employer.cash < max(35, employer.capacity * 2)
            ):
                employer.alive = False
                mask = alive_mask & (ints[I_EMPLOYER, :n] == int(employer.id))
                laid_off = int(np.count_nonzero(mask))
                ints[I_EMPLOYER, :n][mask] = -1
                self._domain_employee_counts[int(employer.id)] = 0
                employer._domain_employee_count = 0
                changes.append(("closed", employer, {"laid_off_count": laid_off}))
            elif (
                margin > employer.capacity * 2.5
                and employer.cash > employer.capacity * 15
                and employer.capacity < 1000
            ):
                old = employer.capacity
                employer.capacity += max(1, min(20, employer.capacity // 20))
                changes.append(("expanded", employer, (old, employer.capacity)))
            elif margin < -8 and employer.capacity > 8 and employee_count < employer.capacity * 0.7:
                old = employer.capacity
                employer.capacity = max(8, int(employer.capacity * 0.95))
                changes.append(("contracted", employer, (old, employer.capacity)))
            employer.revenue_since_review = 0.0
            employer.payroll_since_review = 0.0
            employer.units_produced_since_review = 0.0

        alive_count = int(np.count_nonzero(alive_mask))
        employed = int(np.count_nonzero(alive_mask & (ints[I_EMPLOYER, :n] >= 0)))
        unemployment = (alive_count - employed) / alive_count if alive_count else 0.0
        if unemployment > 0.18 and self.rng.random() < min(0.65, unemployment):
            living_locations = list(self.labor_market.by_location)
            if living_locations:
                location_id = self.rng.choice(living_locations)
                good = "food" if self.rng.random() < 0.78 else "medicine"
                capacity = self.rng.randint(20, 80)
                employer = Employer(
                    id=self.labor_market.next_employer_id,
                    name=f"New Venture {self.labor_market.next_employer_id}",
                    location_id=location_id,
                    kind="new_venture",
                    capacity=capacity,
                    base_wage=self.rng.uniform(4.5, 6.5),
                    cash=capacity * self.rng.uniform(50, 90),
                    productivity=self.rng.uniform(0.95, 1.15),
                    preferred_professions=("laborer", "service_worker", "technician", "trader"),
                    output_good=good,
                    output_per_shift=self.rng.uniform(1.5, 3.0) if good == "food" else self.rng.uniform(0.45, 0.9),
                )
                self.labor_market.next_employer_id += 1
                self.labor_market._add_employer(employer)
                self._domain_employee_counts[int(employer.id)] = 0
                employer._domain_employee_count = 0
                changes.append(("created", employer, None))

        for action, employer, detail in changes:
            self.store.sync_employer(employer)
            self.store.event(
                day,
                self.next_sequence(),
                f"employer_{action}",
                employer_id=employer.id,
                employer=employer.name,
                location_id=employer.location_id,
                capacity=employer.capacity,
                cash=round(employer.cash, 2),
                detail=detail,
            )

        for lid, district in self.police.districts.items():
            population = int(self._domain_location_population.get(int(lid), 0))
            district.officers = (
                max(1, ceil(population * self.police.officers_per_1000 / 1000.0))
                if population else 0
            )
        return len(changes)

    def _apply_soa_eod_stats(self, stats):
        old_alive = int(self.alive_count)
        new_alive = int(stats.get("alive", old_alive))
        if new_alive < old_alive:
            self.total_deaths += old_alive - new_alive
        self.alive_count = new_alive
        self.demographics.working_age_count = int(stats.get("workforce", 0))
        self.invalidate_living_cache()
        loc_stats = stats.get("locations", {})
        for loc in self.locations:
            lid = int(loc.id)
            count = int(loc_stats.get(lid, (0,))[0]) if lid in loc_stats else 0
            self._domain_location_population[lid] = count
            self.goods_market.set_population(lid, count)
        authoritative_counts = {
            int(eid): int(count)
            for eid, count in stats.get("employer_counts", {}).items()
        }
        for employer in self.labor_market.employers:
            count = authoritative_counts.get(int(employer.id), 0)
            self._domain_employee_counts[int(employer.id)] = count
            employer._domain_employee_count = count
        for lid in self.crime_history:
            self.crime_history[lid].append(int(self.daily_crimes.get(lid, 0)))
        self.daily_crimes.clear()

    def _run_soa_eod_only(self, day):
        """Execute only end-of-day decay/statistics in the compiled domain kernel."""
        self._initialize_soa()
        active = self._rebuild_soa_index()
        if active <= 0:
            return None
        packet = self._build_temporal_packet()
        # The action pass placed today's crimes in daily_crimes. The EOD-only
        # kernel has no action counters, so fold that count into the history sum
        # while preserving the denominator expected by the normal fused kernel.
        histories = dict(packet.get("crime_history", {}))
        for lid, history in histories.items():
            row = list(history)
            todays = int(self.daily_crimes.get(int(lid), 0))
            if row:
                row[-1] = int(row[-1]) + todays
            histories[int(lid)] = tuple(row)
        packet["crime_history"] = histories
        started = perf_counter()
        _results, stats = self.soa_pool.run_day(
            day,
            0,
            self.encounter_sample,
            self.max_witnesses,
            self.visibility,
            packet,
            fuse_eod=True,
        )
        self._record_phase("soa_cold_eod_compiled", started)
        self._apply_soa_eod_stats(stats)
        return stats

    def _run_soa_cold_day_legacy(self, day):
        """Exact Python lifecycle fallback for rare mobility boundaries."""
        self.current_day = day
        self._run_soa_domain(day, fuse_eod=False)
        self._soa_reconcile_to_world(count_new_deaths=True)
        for shipment in self.transport.rebalance(day):
            self.store.shipment(shipment)
        if day % BUSINESS_INTERVAL_DAYS == 0:
            self.welfare_cycle()
            self.business_cycle()
            self.police.rebalance()
        ScaleAggressiveWorld._run_parallel_end_of_day(self)
        self.goods_market.reprice()
        if day % MOBILITY_INTERVAL_DAYS == 0:
            self.mobility_cycle()
        police_snapshot = self.police.end_day()
        ScaleAggressiveWorld._write_population_stats_fast(self, day, police_snapshot)
        self.store.commit_day()
        self.demographics.cycle(day)
        self.demographics.write_stats(day)
        self.store.commit_day()
        self._resync_soa_from_world()

    def _run_soa_cold_day(self, day):
        barrier_started = perf_counter()
        self.current_day = day
        if day % MOBILITY_INTERVAL_DAYS == 0:
            self._run_soa_cold_day_legacy(day)
            self._record_phase("soa_cold_barrier_total", barrier_started)
            return

        started = perf_counter()
        self._run_soa_domain(day, fuse_eod=False)
        self._record_phase("soa_cold_actions", started)

        started = perf_counter()
        for shipment in self.transport.rebalance(day):
            self.store.shipment(shipment)
        self._record_phase("soa_cold_transport", started)

        started = perf_counter()
        if day % BUSINESS_INTERVAL_DAYS == 0:
            self._soa_welfare_cycle(day)
            self._soa_business_cycle(day)
        self._record_phase("soa_cold_business_soa", started)

        stats = self._run_soa_eod_only(day)
        self.goods_market.reprice()
        if stats is None:
            self._record_phase("soa_cold_barrier_total", barrier_started)
            return

        started = perf_counter()
        police_snapshot = self._finish_temporal_police(stats)
        self._write_temporal_stats(day, stats, police_snapshot)
        self.store.commit_day()
        self._record_phase("soa_cold_stats_soa", started)

        # Demographics still owns households, pregnancies and Person creation.
        # Materialize exactly once after EOD, run that monthly lifecycle, then
        # return its changes to the authoritative SoA arrays.
        self._soa_reconcile_to_world(count_new_deaths=False)
        started = perf_counter()
        self.demographics.cycle(day)
        self.demographics.write_stats(day)
        self.store.commit_day()
        self._record_phase("soa_cold_demographics", started)
        self._resync_soa_from_world()
        self._record_phase("soa_cold_barrier_total", barrier_started)

    def run_day(self, day):
        if not self.soa_mode:
            return super().run_day(day)
        election_day = day == 1 or (day - 1) % ELECTION_INTERVAL_DAYS == 0
        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            return self._run_soa_cold_day(day)
        if election_day and day != 1 and self._soa_initialized:
            self._soa_reconcile_to_world()
        self.current_day = day
        if election_day:
            self.run_election()
        total_started = perf_counter()
        _results, stats = self._run_soa_domain(day, fuse_eod=True)
        self._record_phase("actions_total", total_started)
        if stats is None:
            return
        for shipment in self.transport.rebalance(day):
            self.store.shipment(shipment)
        self.goods_market.reprice()
        police_snapshot = self._finish_temporal_police(stats)
        self._write_temporal_stats(day, stats, police_snapshot)
        self.store.commit_day()

    def prepare_reporting(self):
        if self.soa_mode and self._soa_initialized:
            self._soa_reconcile_to_world()
        else:
            super().prepare_reporting()

    def close_parallel(self):
        try:
            if self.soa_mode and self.soa_pool is not None:
                summary = self.soa_pool.summary()
                if summary["days"]:
                    print("  aggressive SoA JIT pool: " f"days={summary['days']} tasks={summary['tasks']} " f"workers={summary['workers']} " f"worker_cpu={summary['worker_seconds']:.3f}s " f"dispatch_wall={summary['dispatch_seconds']:.3f}s " f"shm={summary['shared_bytes'] / (1024 * 1024):.1f}MiB")
                self.soa_pool.close()
        finally:
            try:
                if self.soa_index is not None:
                    self.soa_index.close(unlink=True)
                if self.soa_memory is not None:
                    self.soa_memory.close(unlink=True)
                if self.soa_state is not None:
                    self.soa_state.close(unlink=True)
            finally:
                super().close_parallel()

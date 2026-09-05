"""SoA / Numba large-world fast lane layered over temporal BSP."""

from time import perf_counter

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
from .labor_market import BUSINESS_INTERVAL_DAYS
from .politics import ELECTION_INTERVAL_DAYS
from .professions import MOBILITY_INTERVAL_DAYS


_SOA_MIN_POPULATION = 100_000


class AggressiveParallelAgentWorld(TemporalAggressiveWorld):
    """Numba SoA engine for 100k+ agents with cold lifecycle barriers."""

    def __init__(self, *args, agent_workers=0, agent_worker_min_active=64, **kwargs):
        population_size = int(args[1]) if len(args) > 1 else int(kwargs.get("population_size", 0))
        if JIT_ENABLED and population_size >= _SOA_MIN_POPULATION:
            warmup_soa()
        super().__init__(*args, agent_workers=agent_workers, agent_worker_min_active=agent_worker_min_active, **kwargs)
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
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

    def _run_soa_cold_day(self, day):
        if self._soa_initialized:
            self._soa_reconcile_to_world()
        self.current_day = day
        if day == 1 or (day - 1) % ELECTION_INTERVAL_DAYS == 0:
            self.run_election()
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
        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            self.demographics.cycle(day)
            self.demographics.write_stats(day)
            self.store.commit_day()
        self._resync_soa_from_world()

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

"""BSP / domain-decomposed large-scale aggressive world.

For 100k+ agents, action ownership is assigned by location. Workers mutate
shared hot state directly and main performs a single reconciliation pass after
the BSP barrier instead of replaying one intent per action.
"""

from collections import defaultdict
from time import perf_counter

from .aggressive_domain import (
    ALIVE,
    ARRESTS,
    CAREER,
    CRIME_SUFFERED,
    DAYS_IN_CLASS,
    DETAINED,
    EMPLOYER,
    FOOD,
    HEALTH,
    IDEOLOGY,
    JOBS_HELD,
    LID,
    LIFETIME_GROSS,
    LIFETIME_UNEMPLOYMENT,
    MARKET_SPENDING,
    MED,
    MONEY,
    SHELTER,
    SHORTAGES,
    TAXES,
    UNEMPLOYMENT,
    WELFARE,
    WORK_EXP,
    SharedDomainAgentState,
    DomainOwnerPool,
    build_domain_packet,
)
from .aggressive_world_scale import AggressiveParallelAgentWorld as ScaleAggressiveWorld


_DOMAIN_MIN_POPULATION = 100_000


class AggressiveParallelAgentWorld(ScaleAggressiveWorld):
    """Large-scale domain owner-computes fast lane."""

    def __init__(self, *args, agent_workers=0, agent_worker_min_active=64, **kwargs):
        super().__init__(
            *args,
            agent_workers=agent_workers,
            agent_worker_min_active=agent_worker_min_active,
            **kwargs,
        )
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        self.domain_mode = len(self.people) >= _DOMAIN_MIN_POPULATION
        self.domain_state = SharedDomainAgentState(self.shared_buffers.population)
        self.domain_pool = DomainOwnerPool(
            seed,
            self.domain_state,
            self.shared_social,
            len(self.locations),
            workers=agent_workers,
            min_active=_DOMAIN_MIN_POPULATION,
        )
        self._domain_action_counts = defaultdict(int)

    def _sync_domain_inputs(self, day):
        started = perf_counter()
        self.domain_state.sync_world(self)
        self._record_phase("domain_state_sync", started)

        started = perf_counter()
        eligible_count = self.shared_social.sync_world(self, day)
        self._record_phase("domain_social_sync", started)
        return eligible_count

    def _fast_domain_kill(self, person, cause="violence"):
        if not person.alive:
            return
        was_workforce = bool(person.is_working_age)
        partner = None
        if person.partner_id is not None and person.partner_id < len(self.people):
            partner = self.people[person.partner_id]
        employer = self.labor_market.employer_any(person.employer_id)
        if employer is not None:
            employer.employee_ids.discard(person.id)
        self.population_index.remove(person.id, person.location_id)
        person.employer_id = None
        person.alive = False
        self.alive_count -= 1
        self.total_deaths += 1
        self.invalidate_living_cache()
        if was_workforce and hasattr(self, "demographics"):
            self.demographics.working_age_count = max(
                0, self.demographics.working_age_count - 1
            )
        if partner is not None and partner.alive and partner.partner_id == person.id:
            partner.partner_id = None
        # Throughput-first domain mode intentionally omits per-death action
        # event rows here; lifecycle / natural deaths stay on the normal path.

    def _apply_domain_results(self, results):
        counters = defaultdict(int)
        treasury_delta = 0.0
        employer_rows = {}
        market_rows = {}
        police_rows = {}

        for result in results:
            for key, value in result.get("counters", {}).items():
                counters[key] += int(value)
            treasury_delta += float(result.get("treasury_delta", 0.0))
            for row in result.get("employers", ()):
                employer_rows[int(row[0])] = row
            market = result.get("market")
            if market is not None:
                market_rows[int(market[0])] = market
            police = result.get("police")
            if police is not None:
                police_rows[int(police[0])] = police

        started = perf_counter()
        for employer in self.labor_market.employers:
            employer.employee_ids.clear()

        dirty_locations = set()
        for person in self.people:
            pid = int(person.id)
            state = self.domain_state.read(pid)

            person.food = int(state[FOOD])
            person.medicine = int(state[MED])
            person.energy = int(state[3])
            person.health = int(state[HEALTH])
            person.shelter = int(state[SHELTER])
            person.money = float(state[MONEY])
            person.ideology = float(state[IDEOLOGY])
            person.detained_until_day = int(state[DETAINED])
            person.taxes_paid = float(state[TAXES])
            person.welfare_received = float(state[WELFARE])
            person.work_experience = int(state[WORK_EXP])
            person.career_progress = float(state[CAREER])
            person.lifetime_gross_income = float(state[LIFETIME_GROSS])
            person.market_spending = float(state[MARKET_SPENDING])
            person.shortage_experiences = int(state[SHORTAGES])
            person.crime_suffered = int(state[CRIME_SUFFERED])
            person.arrests = int(state[ARRESTS])
            person.unemployment_days = int(state[UNEMPLOYMENT])
            person.lifetime_unemployment_days = int(state[LIFETIME_UNEMPLOYMENT])
            person.days_in_class = int(state[DAYS_IN_CLASS])
            person.jobs_held = int(state[JOBS_HELD])

            if person.alive and not bool(state[ALIVE]):
                self._fast_domain_kill(person)
                continue
            if not person.alive:
                continue

            new_location = int(state[LID])
            if new_location != person.location_id:
                old_location = int(person.location_id)
                self.population_index.move(pid, old_location, new_location)
                person.location_id = new_location
                dirty_locations.add(old_location)
                dirty_locations.add(new_location)

            employer_id = int(state[EMPLOYER])
            person.employer_id = None if employer_id < 0 else employer_id
            if person.employer_id is not None:
                employer = self.labor_market.employer_any(person.employer_id)
                if employer is not None:
                    employer.employee_ids.add(pid)

        for location_id in dirty_locations:
            self.goods_market.set_population(
                location_id, self.population_index.population(location_id)
            )

        self.total_helps += counters["helps"]
        self.total_thefts += counters["thefts"]
        self.total_attacks += counters["attacks"]
        self.total_observations += counters["observations"]
        self.total_arrests += counters["arrests"]
        self.total_moves += counters["moves"]
        for result in results:
            lid = int(result["location"])
            self.daily_crimes[lid] += int(result.get("counters", {}).get("crimes", 0))

        self.politics.treasury += treasury_delta

        for eid, row in employer_rows.items():
            employer = self.labor_market.employer_any(eid)
            if employer is None:
                continue
            (
                _eid, _lid, capacity, _employees, cash, productivity,
                payroll, revenue, units, alive,
            ) = row
            employer.capacity = int(capacity)
            employer.cash = float(cash)
            employer.productivity = float(productivity)
            employer.payroll_since_review = float(payroll)
            employer.revenue_since_review = float(revenue)
            employer.units_produced_since_review = float(units)
            employer.alive = bool(alive)

        for lid, row in market_rows.items():
            (
                _lid, prices, food_suppliers, medicine_suppliers, demand, sold,
            ) = row
            market = self.goods_market.state(lid)
            market.prices.update(prices)
            market.supplier_stock["food"] = defaultdict(
                float, {int(k): float(v) for k, v in food_suppliers.items()}
            )
            market.supplier_stock["medicine"] = defaultdict(
                float, {int(k): float(v) for k, v in medicine_suppliers.items()}
            )
            market.demand["food"] = float(demand.get("food", 0.0))
            market.demand["medicine"] = float(demand.get("medicine", 0.0))
            market.sold["food"] = float(sold.get("food", 0.0))
            market.sold["medicine"] = float(sold.get("medicine", 0.0))

        for lid, row in police_rows.items():
            _lid, officers, incidents, responses, arrests = row
            district = self.police.districts[lid]
            district.officers = int(officers)
            district.incidents_today = int(incidents)
            district.responses_today = int(responses)
            district.arrests_today = int(arrests)

        for key, value in counters.items():
            if key.startswith("action_"):
                self._domain_action_counts[key[7:]] += int(value)

        self._record_phase("domain_reconcile", started)

    def _run_parallel_actions(self, day):
        if (
            not self.domain_mode
            or len(self.people) > self.domain_state.capacity
            or not self.domain_pool.should_parallelize(self.alive_count)
        ):
            return super()._run_parallel_actions(day)

        total_started = perf_counter()
        eligible_count = self._sync_domain_inputs(day)
        if eligible_count <= 0 or self.actions_per_day <= 0:
            self._record_phase("actions_total", total_started)
            return

        packet_started = perf_counter()
        packet = build_domain_packet(self)
        self._record_phase("domain_packet", packet_started)

        dispatch_started = perf_counter()
        results = self.domain_pool.run_day(
            day,
            self.actions_per_day,
            self.encounter_sample,
            self.max_witnesses,
            self.visibility,
            packet,
        )
        self._record_phase("domain_dispatch", dispatch_started)

        self._apply_domain_results(results)
        self._record_phase("actions_total", total_started)

    def close_parallel(self):
        try:
            if self.domain_mode:
                summary = self.domain_pool.summary()
                if summary["days"]:
                    print(
                        "  aggressive BSP domain pool: "
                        f"days={summary['days']} tasks={summary['tasks']} "
                        f"workers={summary['workers']} "
                        f"worker_cpu={summary['worker_seconds']:.3f}s "
                        f"dispatch_wall={summary['dispatch_seconds']:.3f}s "
                        f"shm={summary['shared_bytes'] / (1024 * 1024):.1f}MiB"
                    )
                    if self._domain_action_counts:
                        print("  aggressive BSP domain actions:")
                        for action, count in sorted(
                            self._domain_action_counts.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        ):
                            print(f"    {action:<16} calls={count:>10}")
            self.domain_pool.close()
        finally:
            try:
                self.domain_state.close(unlink=True)
            finally:
                super().close_parallel()

"""Persistent BSP domains with 30-day cold reconciliation barriers."""

from collections import defaultdict
from time import perf_counter

from .aggressive_domain import (
    ALIVE, ARRESTS, CAREER, CRIME_SUFFERED, DAYS_IN_CLASS, DETAINED,
    EMPLOYER, ENERGY, FOOD, HEALTH, IDEOLOGY, JOBS_HELD, LID,
    LIFETIME_GROSS, LIFETIME_UNEMPLOYMENT, MARKET_SPENDING, MED, MONEY,
    PROFESSION, SHELTER, SHORTAGES, SOCIAL_CLASS, TAXES, UNEMPLOYMENT,
    WELFARE, WORK_EXP, build_domain_packet,
)
from .aggressive_economy import PROFESSION_NAMES, SOCIAL_CLASSES
from .aggressive_temporal import TemporalDomainPool, sync_social_from_domain
from .aggressive_world_domain import AggressiveParallelAgentWorld as DomainAggressiveWorld
from .aggressive_world_scale import AggressiveParallelAgentWorld as ScaleAggressiveWorld
from .demographics import DEMOGRAPHIC_INTERVAL_DAYS
from .labor_market import BUSINESS_INTERVAL_DAYS
from .politics import ELECTION_INTERVAL_DAYS
from .professions import MOBILITY_INTERVAL_DAYS


class AggressiveParallelAgentWorld(DomainAggressiveWorld):
    """Large-world engine where domains remain authoritative between barriers."""

    def __init__(self, *args, agent_workers=0, agent_worker_min_active=64, **kwargs):
        super().__init__(
            *args,
            agent_workers=agent_workers,
            agent_worker_min_active=agent_worker_min_active,
            **kwargs,
        )
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        self.temporal_mode = bool(self.domain_mode)
        self.temporal_pool = TemporalDomainPool(
            seed, self.domain_state, self.shared_social, len(self.locations),
            workers=agent_workers,
        )
        self._domain_initialized = False
        self._domain_employee_counts = {
            int(e.id): len(e.employee_ids) for e in self.labor_market.employers
        }
        self._domain_location_population = {
            int(loc.id): self.population_index.population(loc.id)
            for loc in self.locations
        }

    def _initialize_domain_state(self):
        if self._domain_initialized:
            return
        started = perf_counter()
        self.domain_state.sync_world(self)
        self._domain_initialized = True
        self._record_phase("temporal_initial_sync", started)

    def _build_temporal_packet(self):
        packet = build_domain_packet(self)
        employer_rows = []
        for row in packet["employers"]:
            values = list(row)
            values[4] = int(self._domain_employee_counts.get(int(values[0]), values[4]))
            employer_rows.append(tuple(values))
        location_rows = []
        for row in packet["locations"]:
            values = list(row)
            values[-1] = int(self._domain_location_population.get(int(values[0]), values[-1]))
            location_rows.append(tuple(values))
        packet["employers"] = tuple(employer_rows)
        packet["locations"] = tuple(location_rows)
        packet["shelter_decay"] = {
            int(loc.id): int(loc.shelter_decay_bonus) for loc in self.locations
        }
        packet["crime_history"] = {
            int(lid): tuple(history) for lid, history in self.crime_history.items()
        }
        return packet

    def _sync_temporal_social(self, day):
        started = perf_counter()
        count = sync_social_from_domain(
            self.shared_social, self.domain_state, day, len(self.people)
        )
        self._record_phase("temporal_social_sync", started)
        return count

    def _apply_temporal_aggregates(self, results, stats, *, fused_eod):
        counters = defaultdict(int)
        treasury_delta = 0.0
        employer_rows = {}
        market_rows = {}
        police_rows = {}
        crimes_by_location = defaultdict(int)

        for result in results:
            rc = result.get("counters", {})
            for key, value in rc.items():
                counters[key] += int(value)
            lid = int(result["location"])
            crimes_by_location[lid] += int(rc.get("crimes", 0))
            treasury_delta += float(result.get("treasury_delta", 0.0))
            for row in result.get("employers", ()):
                employer_rows[int(row[0])] = row
            if result.get("market") is not None:
                market_rows[lid] = result["market"]
            if result.get("police") is not None:
                police_rows[lid] = result["police"]

        self.total_helps += counters["helps"]
        self.total_thefts += counters["thefts"]
        self.total_attacks += counters["attacks"]
        self.total_observations += counters["observations"]
        self.total_arrests += counters["arrests"]
        self.total_moves += counters["moves"]
        self.politics.treasury += treasury_delta

        for key, value in counters.items():
            if key.startswith("action_"):
                self._domain_action_counts[key[7:]] += int(value)

        for eid, row in employer_rows.items():
            employer = self.labor_market.employer_any(eid)
            if employer is None:
                continue
            (
                _eid, _lid, capacity, employee_count, cash, productivity,
                payroll, revenue, units, alive,
            ) = row
            employer.capacity = int(capacity)
            employer.cash = float(cash)
            employer.productivity = float(productivity)
            employer.payroll_since_review = float(payroll)
            employer.revenue_since_review = float(revenue)
            employer.units_produced_since_review = float(units)
            employer.alive = bool(alive)
            self._domain_employee_counts[eid] = int(employee_count)
            employer._domain_employee_count = int(employee_count)

        for lid, row in market_rows.items():
            _lid, prices, food_suppliers, medicine_suppliers, demand, sold = row
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

        if fused_eod:
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

            for lid, crimes in crimes_by_location.items():
                self.crime_history[lid].append(int(crimes))
            self.daily_crimes.clear()
        else:
            for lid, crimes in crimes_by_location.items():
                self.daily_crimes[lid] += int(crimes)

    def _cold_reconcile_from_domain(self, *, count_new_deaths=False):
        """Materialize authoritative domain memory back into Person objects."""
        started = perf_counter()
        old_alive = int(self.alive_count)
        location_count = len(self.locations)
        self.population_index.members = [[] for _ in range(location_count)]
        self.population_index.positions = [-1] * len(self.people)
        for employer in self.labor_market.employers:
            employer.employee_ids.clear()

        alive = 0
        workforce = 0
        for person in self.people:
            pid = int(person.id)
            state = self.domain_state.read(pid)
            person.location_id = int(state[LID])
            person.food = int(state[FOOD]); person.medicine = int(state[MED])
            person.energy = int(state[ENERGY]); person.health = int(state[HEALTH])
            person.shelter = int(state[SHELTER]); person.money = float(state[MONEY])
            employer_id = int(state[EMPLOYER])
            person.employer_id = None if employer_id < 0 else employer_id
            profession_code = int(state[PROFESSION])
            if 0 <= profession_code < len(PROFESSION_NAMES):
                person.profession = PROFESSION_NAMES[profession_code]
            class_code = int(state[SOCIAL_CLASS])
            if 0 <= class_code < len(SOCIAL_CLASSES):
                person.social_class = SOCIAL_CLASSES[class_code]
            person.ideology = float(state[IDEOLOGY])
            person.detained_until_day = int(state[DETAINED])
            person.alive = bool(state[ALIVE])
            person.taxes_paid = float(state[TAXES]); person.welfare_received = float(state[WELFARE])
            person.work_experience = int(state[WORK_EXP]); person.career_progress = float(state[CAREER])
            person.lifetime_gross_income = float(state[LIFETIME_GROSS])
            person.market_spending = float(state[MARKET_SPENDING])
            person.shortage_experiences = int(state[SHORTAGES])
            person.crime_suffered = int(state[CRIME_SUFFERED]); person.arrests = int(state[ARRESTS])
            person.unemployment_days = int(state[UNEMPLOYMENT])
            person.lifetime_unemployment_days = int(state[LIFETIME_UNEMPLOYMENT])
            person.days_in_class = int(state[DAYS_IN_CLASS]); person.jobs_held = int(state[JOBS_HELD])

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
        self._domain_location_population = {
            int(loc.id): self.population_index.population(loc.id) for loc in self.locations
        }
        for lid, count in self._domain_location_population.items():
            self.goods_market.set_population(lid, count)
        self._domain_employee_counts = {}
        for employer in self.labor_market.employers:
            count = len(employer.employee_ids)
            self._domain_employee_counts[int(employer.id)] = count
            employer._domain_employee_count = count
        self._record_phase("temporal_cold_reconcile", started)

    def _finish_temporal_police(self, stats):
        snapshots = {}
        loc_stats = stats.get("locations", {})
        for lid, district in self.police.districts.items():
            population = int(loc_stats.get(int(lid), (0,))[0]) if int(lid) in loc_stats else 0
            population = max(1, population)
            officers_per_1000 = district.officers * 1000.0 / population
            load = district.incidents_today / max(1, district.officers)
            coverage = max(0.02, min(0.92, 0.22 + officers_per_1000 * 0.12 - load * 0.035))
            snapshots[int(lid)] = {
                "officers": district.officers,
                "incidents": district.incidents_today,
                "responses": district.responses_today,
                "arrests": district.arrests_today,
                "coverage": coverage,
            }
            self.police.history[lid].append(district.incidents_today)
            district.incidents_today = 0; district.responses_today = 0; district.arrests_today = 0
        return snapshots

    def _write_temporal_stats(self, day, stats, police_snapshot):
        started = perf_counter()
        conn = self.store.conn; sid = self.store.simulation_id
        alive = int(stats.get("alive", 0)); n = max(1, alive)
        conn.execute(
            "INSERT INTO daily_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (sid, day, alive, stats["food"]/n, stats["money"]/n, stats["medicine"]/n,
             stats["energy"]/n, stats["shelter"]/n, stats["health"]/n,
             self.total_helps, self.total_thefts, self.total_attacks,
             self.total_observations, self.total_deaths, self.total_mobility_changes),
        )
        location_rows = []
        for loc in self.locations:
            lid = int(loc.id); row = stats["locations"].get(lid, [0,0.0,0.0,0.0])
            count = int(row[0]); denom = max(1, count)
            hist = self.crime_history[lid]
            crime_rate = sum(hist) / (denom * len(hist)) if hist else 0.0
            location_rows.append((sid, day, lid, count, row[1]/denom, row[2]/denom, row[3]/denom, crime_rate))
        conn.executemany("INSERT INTO location_stats VALUES(?,?,?,?,?,?,?,?)", location_rows)
        conn.execute(
            "INSERT INTO political_stats VALUES(?,?,?,?,?,?,?,?,?)",
            (sid, day, self.politics.government.id, self.politics.treasury,
             stats["ideology"]/n, int(stats["left"]), int(stats["right"]),
             stats["taxes"]/n, stats["welfare"]/n),
        )
        social_rows = []
        for index, name in enumerate(SOCIAL_CLASSES):
            row = stats["social"][index]; count = int(row[0]); denom = max(1, count)
            social_rows.append((sid, day, name, count, row[1]/denom, row[2]/denom,
                                row[3]/denom, row[4]/denom, row[5]/denom, row[6]/denom))
        conn.executemany("INSERT INTO social_stats VALUES(?,?,?,?,?,?,?,?,?,?)", social_rows)
        workforce = int(stats["workforce"]); employed = int(stats["employed"])
        active = [e for e in self.labor_market.employers if e.alive]
        conn.execute(
            "INSERT INTO labor_stats VALUES(?,?,?,?,?,?,?,?)",
            (sid, day, employed, max(0, workforce-employed),
             (workforce-employed)/workforce if workforce else 0.0,
             sum(max(0, e.capacity - self._domain_employee_counts.get(int(e.id), 0)) for e in active),
             len(active), sum(e.capacity for e in active)),
        )
        self.store.write_market_stats(day, self)
        self.store.write_police_stats(day, police_snapshot)
        self._record_phase("temporal_stats_commit", started)

    def _run_temporal_domain(self, day, *, fuse_eod):
        self._initialize_domain_state()
        eligible = self._sync_temporal_social(day)
        if eligible <= 0 or self.actions_per_day <= 0:
            return [], None
        packet_started = perf_counter(); packet = self._build_temporal_packet()
        self._record_phase("temporal_packet", packet_started)
        started = perf_counter()
        results, stats = self.temporal_pool.run_day(
            day, self.actions_per_day, self.encounter_sample,
            self.max_witnesses, self.visibility, packet, fuse_eod=fuse_eod,
        )
        self._record_phase("temporal_dispatch", started)
        started = perf_counter()
        self._apply_temporal_aggregates(results, stats, fused_eod=fuse_eod)
        self._record_phase("temporal_aggregate", started)
        return results, stats

    def _run_cold_barrier_day(self, day):
        # Materialize yesterday's domain state so slow global systems see current people.
        if self._domain_initialized:
            self._cold_reconcile_from_domain()
        self.current_day = day
        if day == 1 or (day - 1) % ELECTION_INTERVAL_DAYS == 0:
            self.run_election()

        self._run_temporal_domain(day, fuse_eod=False)
        self._cold_reconcile_from_domain(count_new_deaths=True)

        for shipment in self.transport.rebalance(day):
            self.store.shipment(shipment)
        if day % BUSINESS_INTERVAL_DAYS == 0:
            self.welfare_cycle(); self.business_cycle(); self.police.rebalance()

        ScaleAggressiveWorld._run_parallel_end_of_day(self)
        self.goods_market.reprice()
        if day % MOBILITY_INTERVAL_DAYS == 0:
            self.mobility_cycle()
        police_snapshot = self.police.end_day()
        ScaleAggressiveWorld._write_population_stats_fast(self, day, police_snapshot)
        self.store.commit_day()
        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            self.demographics.cycle(day); self.demographics.write_stats(day); self.store.commit_day()

        started = perf_counter()
        self.domain_state.sync_world(self)
        self._domain_initialized = True
        self._domain_employee_counts = {int(e.id): len(e.employee_ids) for e in self.labor_market.employers}
        for e in self.labor_market.employers:
            e._domain_employee_count = self._domain_employee_counts[int(e.id)]
        self._domain_location_population = {
            int(loc.id): self.population_index.population(loc.id) for loc in self.locations
        }
        self._record_phase("temporal_cold_resync", started)

    def run_day(self, day):
        if not self.temporal_mode:
            return super().run_day(day)

        # Monthly lifecycle/business/mobility work remains a cold barrier. An
        # election also needs current Person ideology before ballots are built.
        election_day = day == 1 or (day - 1) % ELECTION_INTERVAL_DAYS == 0
        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            return self._run_cold_barrier_day(day)
        if election_day and day != 1 and self._domain_initialized:
            self._cold_reconcile_from_domain()

        self.current_day = day
        if election_day:
            self.run_election()

        total_started = perf_counter()
        _results, stats = self._run_temporal_domain(day, fuse_eod=True)
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
        """Materialize the final domain state before end-of-run reporting."""
        if self.temporal_mode and self._domain_initialized:
            self._cold_reconcile_from_domain()

    def close_parallel(self):
        try:
            if self.temporal_mode:
                summary = self.temporal_pool.summary()
                if summary["days"]:
                    print(
                        "  aggressive temporal BSP pool: "
                        f"days={summary['days']} tasks={summary['tasks']} "
                        f"workers={summary['workers']} "
                        f"worker_cpu={summary['worker_seconds']:.3f}s "
                        f"dispatch_wall={summary['dispatch_seconds']:.3f}s "
                        f"shm={summary['shared_bytes'] / (1024 * 1024):.1f}MiB"
                    )
            self.temporal_pool.close()
        finally:
            super().close_parallel()

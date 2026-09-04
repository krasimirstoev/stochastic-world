from collections import defaultdict
from time import perf_counter

from .aggressive_economy import SharedEconomyState
from .aggressive_shared import SharedAgentBuffers
from .agent_shards_shared import SharedPersistentDayShardPool
from .agent_world import ParallelAgentWorld
from .population_index import permutation_ids
from .professions import PROFESSIONS


class AggressiveParallelAgentWorld(ParallelAgentWorld):
    """Opt-in throughput-first agent engine backed by shared-memory day shards."""

    def __init__(self, *args, agent_workers=0, agent_worker_min_active=64, **kwargs):
        super().__init__(
            *args,
            agent_workers=agent_workers,
            agent_worker_min_active=agent_worker_min_active,
            **kwargs,
        )
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        self.aggressive_parallel = True
        initial_population = max(1, len(self.people))
        shared_capacity = max(initial_population * 2, initial_population + 65_536)
        self.shared_buffers = SharedAgentBuffers(
            shared_capacity,
            self.actions_per_day,
            self.max_witnesses,
        )
        highest_employer_id = max((employer.id for employer in self.labor_market.employers), default=-1)
        employer_capacity = max(16_384, (highest_employer_id + 1) * 16)
        self.shared_economy = SharedEconomyState(
            shared_capacity,
            employer_capacity,
            self.locations,
        )
        self.shard_pool = SharedPersistentDayShardPool(
            seed,
            self.shared_buffers,
            self.shared_economy,
            workers=agent_workers,
            min_active=agent_worker_min_active,
        )
        self._aggressive_seconds = defaultdict(float)
        self._aggressive_calls = defaultdict(int)
        self._intent_seconds = defaultdict(float)
        self._intent_calls = defaultdict(int)

    def _record_phase(self, phase, started):
        self._aggressive_seconds[phase] += perf_counter() - started
        self._aggressive_calls[phase] += 1

    def _record_intent(self, action, started):
        self._intent_seconds[action] += perf_counter() - started
        self._intent_calls[action] += 1

    def _ensure_shared_capacity(self):
        if len(self.people) > self.shared_buffers.population:
            return False
        highest_employer_id = max((employer.id for employer in self.labor_market.employers), default=-1)
        return highest_employer_id < self.shared_economy.employer_capacity

    @staticmethod
    def _apply_local_state(person, final_state):
        (
            person.food,
            person.medicine,
            person.energy,
            person.health,
            person.shelter,
            person.money,
        ) = final_state

    def work(self, person):
        """Fallback compact work path for hiring/insolvency or stale prepared data."""
        if not person.is_working_age or person.energy < 8:
            person.energy = min(100, person.energy + self.rng.randint(12, 24))
            person.health = min(100, person.health + self.rng.randint(0, 2))
            self.next_sequence()
            return

        employer = self.labor_market.employer(person.employer_id)
        if employer is None:
            person.employer_id = None
            hired = self.labor_market.hire(person)
            if hired:
                self.store.employment_event(self.current_day, person, hired, "hired", "job_search")
                self.store.event(
                    self.current_day,
                    self.next_sequence(),
                    "job_found",
                    actor=person,
                    employer_id=hired.id,
                    employer=hired.name,
                )
            else:
                self.store.event(
                    self.current_day,
                    self.next_sequence(),
                    "job_search",
                    actor=person,
                    location_id=person.location_id,
                    vacancies=self.labor_market.vacancies(person.location_id),
                    success=0,
                )
            return

        if employer.location_id != person.location_id:
            return

        location = self.locations[person.location_id]
        profession = PROFESSIONS[person.profession]
        fit = 1.15 if location.kind in profession.workplace_kinds else 0.82
        scarcity = 1.10 if employer.vacancies > max(1, employer.capacity // 3) else 1.0
        preferred = 1.08 if person.profession in employer.preferred_professions else 0.92
        gross = max(
            1,
            round(
                employer.base_wage
                * profession.income_multiplier
                * fit
                * scarcity
                * preferred
            ),
        )

        if employer.cash < gross:
            self.labor_market.terminate(person, "insolvent")
            self.store.employment_event(self.current_day, person, employer, "laid_off", "insolvent")
            self.store.event(
                self.current_day,
                self.next_sequence(),
                "layoff",
                actor=person,
                employer_id=employer.id,
                reason="insolvent",
            )
            return

        employer.cash -= gross
        employer.payroll_since_review += gross
        produced_good = employer.output_good
        produced = 0.0
        if produced_good:
            produced = max(
                0.0,
                employer.output_per_shift
                * employer.productivity
                * fit
                * self.rng.uniform(0.85, 1.15),
            )
            employer.units_produced_since_review += produced
        elif employer.kind != "logistics":
            service_revenue = gross * employer.productivity * self.rng.uniform(1.12, 1.45)
            employer.cash += service_revenue
            employer.revenue_since_review += service_revenue

        energy = max(3, round(self.rng.randint(6, 12) * profession.energy_multiplier))
        person.money += gross
        person.lifetime_gross_income += gross
        person.work_experience += 1
        person.career_progress += profession.advancement_rate * fit
        self.politics.collect_tax(person, gross)
        person.energy = max(0, person.energy - energy)
        if produced_good and produced > 0:
            self.goods_market.add_supply(
                person.location_id,
                employer.id,
                produced_good,
                produced,
            )
        self.next_sequence()

    def _apply_prepared_work(self, person, plan):
        (
            _kind, employer_id, gross, energy_cost, output_good,
            produced, service_revenue, career_delta,
        ) = plan
        if not person.alive or not person.is_working_age or self.current_day < person.detained_until_day:
            return
        employer = self.labor_market.employer(employer_id)
        if (
            employer is None
            or person.employer_id != employer_id
            or employer.location_id != person.location_id
        ):
            return self.work(person)
        if person.energy < 8:
            return self.work(person)
        if employer.cash < gross:
            self.labor_market.terminate(person, "insolvent")
            self.store.employment_event(self.current_day, person, employer, "laid_off", "insolvent")
            self.store.event(
                self.current_day,
                self.next_sequence(),
                "layoff",
                actor=person,
                employer_id=employer.id,
                reason="insolvent",
            )
            return

        employer.cash -= gross
        employer.payroll_since_review += gross
        if output_good and produced > 0:
            employer.units_produced_since_review += produced
            self.goods_market.add_supply(
                person.location_id,
                employer.id,
                output_good,
                produced,
            )
        elif service_revenue > 0:
            employer.cash += service_revenue
            employer.revenue_since_review += service_revenue

        person.money += gross
        person.lifetime_gross_income += gross
        person.work_experience += 1
        person.career_progress += career_delta
        self.politics.collect_tax(person, gross)
        person.energy = max(0, person.energy - energy_cost)
        self.next_sequence()

    def _apply_prepared_move(self, person, plan):
        _kind, destination_id, energy_cost = plan
        if not person.alive or self.current_day < person.detained_until_day or person.energy < 4:
            return
        old_id = person.location_id
        if destination_id == old_id or destination_id < 0 or destination_id >= len(self.locations):
            return
        if destination_id not in self.locations[old_id].neighbors:
            return self.move(person)
        employer = self.labor_market.employer(person.employer_id)
        if employer and employer.location_id != destination_id:
            self.labor_market.terminate(person, "relocation")
            self.store.employment_event(self.current_day, person, employer, "ended", "relocation")
        person.energy = max(0, person.energy - int(energy_cost))
        self.population_index.move(person.id, old_id, destination_id)
        person.location_id = destination_id
        self.goods_market.set_population(old_id, self.population_index.population(old_id))
        self.goods_market.set_population(destination_id, self.population_index.population(destination_id))
        self.total_moves += 1
        self.store.event(
            self.current_day,
            self.next_sequence(),
            "move",
            actor=person,
            from_location=old_id,
            to_location=destination_id,
            energy_cost=int(energy_cost),
            profession=person.profession,
        )

    def _apply_prepared_buy(self, person, plan):
        _kind, good, requested = plan
        if (
            not person.alive
            or self.current_day < person.detained_until_day
            or good not in ("food", "medicine")
            or requested <= 0
        ):
            return
        result = self.goods_market.buy(person.location_id, good, requested, person.money)
        quantity = result["quantity"]
        cost = result["cost"]
        person.money -= cost
        person.market_spending += cost
        if good == "food":
            person.food += int(quantity)
        else:
            person.medicine += int(quantity)
        if result["shortage"]:
            person.shortage_experiences += 1
            person.shift_ideology(-0.00025)
        for employer_id, revenue in result["seller_revenue"].items():
            self.labor_market.credit_sale(employer_id, revenue)
        self.store.event(
            self.current_day,
            self.next_sequence(),
            "buy_supplies",
            actor=person,
            location_id=person.location_id,
            resource=good,
            requested=requested,
            amount=quantity,
            cost=cost,
            unit_price=result["unit_price"],
            shortage=int(result["shortage"]),
        )

    def _apply_prepared_social(self, person, prepared):
        """Help uses worker-owned planning memory; crime remains authoritative."""
        _pid, action, target_id, payload, witness_ids = prepared
        if action != "help":
            return super()._apply_prepared_social(person, prepared)
        if (
            target_id is None
            or not person.alive
            or self.current_day < person.detained_until_day
            or target_id >= len(self.people)
        ):
            return
        target = self.people[target_id]
        if not target.alive or target.location_id != person.location_id:
            return
        resource, proposed = payload
        amount = 0
        if resource == "medicine" and proposed and person.medicine > 0:
            amount = 1
            person.medicine -= 1
            target.medicine += 1
        elif resource == "food" and proposed and person.food > 1:
            amount = min(int(proposed), max(0, person.food - 1))
            person.food -= amount
            target.food += amount
        self.next_sequence()
        if not amount:
            return
        self.total_helps += 1
        for witness_id in witness_ids:
            if witness_id >= len(self.people):
                continue
            witness = self.people[witness_id]
            if (
                witness.alive
                and witness.location_id == person.location_id
                and witness.id not in (person.id, target.id)
            ):
                self.total_observations += 1

    def _run_parallel_actions(self, day):
        if (
            not self.shard_pool.should_parallelize(self.alive_count)
            or not self._ensure_shared_capacity()
        ):
            started = perf_counter()
            result = super()._run_parallel_actions(day)
            self._record_phase("fallback_actions", started)
            return result

        total_started = perf_counter()
        started = perf_counter()
        eligible_order = [
            pid
            for pid in permutation_ids(len(self.people), self.rng)
            if self.people[pid].alive
            and self.people[pid].is_adult
            and day >= self.people[pid].detained_until_day
        ]
        self._record_phase("eligible_order", started)
        if not eligible_order or self.actions_per_day <= 0:
            self._record_phase("actions_total", total_started)
            return

        started = perf_counter()
        self.shared_economy.sync_world(self)
        self._record_phase("shared_economy_sync", started)

        started = perf_counter()
        for pid in eligible_order:
            person = self.people[pid]
            self.shared_buffers.write_snapshot(self._action_snapshot(person))
            self.shared_economy.write_person(pid, person)
        self._record_phase("shared_input_sync", started)

        started = perf_counter()
        candidate_pools = self._social_candidate_pools()
        self._record_phase("candidate_pools", started)

        started = perf_counter()
        self.shard_pool.plan_day(
            day,
            eligible_order,
            candidate_pools,
            self.actions_per_day,
            self.encounter_sample,
            self.max_witnesses,
            self.visibility,
        )
        self._record_phase("shard_dispatch", started)

        started = perf_counter()
        for pid in eligible_order:
            person = self.people[pid]
            if not person.alive or day < person.detained_until_day:
                continue
            self._apply_local_state(person, self.shared_buffers.read_state(pid))
        self._record_phase("apply_shared_state", started)

        started = perf_counter()
        for round_index in range(self.actions_per_day):
            for pid in eligible_order:
                person = self.people[pid]
                if not person.alive or day < person.detained_until_day:
                    continue
                plan = self.shared_buffers.read_intent(pid, round_index)
                if plan is None:
                    continue
                kind = plan[0]
                intent_started = perf_counter()
                if kind == "social":
                    _, action, target_id, payload, witness_ids = plan
                    self._apply_prepared_social(
                        person,
                        (pid, action, target_id, payload, witness_ids),
                    )
                elif kind == "move_prepared":
                    action = "move"
                    self._apply_prepared_move(person, plan)
                elif kind == "work_prepared":
                    action = "work"
                    self._apply_prepared_work(person, plan)
                elif kind == "buy_prepared":
                    action = "buy_supplies"
                    self._apply_prepared_buy(person, plan)
                else:
                    _, action = plan
                    self._execute_shared_intent(person, action)
                self._record_intent(action, intent_started)
        self._record_phase("apply_intents", started)
        self._record_phase("actions_total", total_started)

    def aggressive_profile_summary(self):
        rows = []
        for phase, total in sorted(
            self._aggressive_seconds.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            calls = self._aggressive_calls[phase]
            rows.append((phase, calls, total, total / calls if calls else 0.0))
        return rows

    def aggressive_intent_summary(self):
        rows = []
        for action, total in sorted(
            self._intent_seconds.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            calls = self._intent_calls[action]
            rows.append((action, calls, total, total / calls if calls else 0.0))
        return rows

    def close_parallel(self):
        if self._aggressive_seconds:
            print("  aggressive phase profile (wall clock):")
            for phase, calls, total, avg in self.aggressive_profile_summary():
                print(
                    f"    {phase:<20} calls={calls:>5} "
                    f"total={total:>9.3f}s avg={avg:>8.4f}s"
                )
            if self._intent_seconds:
                print("  aggressive intent profile (main-process wall clock):")
                for action, calls, total, avg in self.aggressive_intent_summary():
                    print(
                        f"    {action:<20} calls={calls:>7} "
                        f"total={total:>9.3f}s avg={avg:>10.6f}s"
                    )
            shard = self.shard_pool.summary()
            if shard.get("started"):
                shared_mib = shard.get("shared_bytes", 0) / (1024 * 1024)
                print(
                    "  aggressive shared shard pool: "
                    f"days={shard.get('days', 0)} tasks={shard.get('tasks', 0)} "
                    f"items={shard.get('items_returned', 0)} "
                    f"worker_cpu={shard.get('worker_seconds', 0.0):.3f}s "
                    f"dispatch_wall={shard.get('dispatch_seconds', 0.0):.3f}s "
                    f"shm={shared_mib:.1f}MiB"
                )
        self.shard_pool.close()
        self.shared_economy.close(unlink=True)
        self.shared_buffers.close(unlink=True)
        super().close_parallel()

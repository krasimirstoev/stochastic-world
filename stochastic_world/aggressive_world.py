from collections import defaultdict
from time import perf_counter

from .agent_shards import PersistentDayShardPool
from .agent_world import ParallelAgentWorld
from .population_index import permutation_ids
from .professions import PROFESSIONS


class AggressiveParallelAgentWorld(ParallelAgentWorld):
    """Opt-in throughput-first agent engine with persistent day shards."""

    def __init__(self, *args, agent_workers=0, agent_worker_min_active=64, **kwargs):
        super().__init__(
            *args,
            agent_workers=agent_workers,
            agent_worker_min_active=agent_worker_min_active,
            **kwargs,
        )
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        self.aggressive_parallel = True
        self.shard_pool = PersistentDayShardPool(
            seed,
            workers=agent_workers,
            min_active=agent_worker_min_active,
        )
        self._aggressive_seconds = defaultdict(float)
        self._aggressive_calls = defaultdict(int)
        self._intent_seconds = defaultdict(float)
        self._intent_calls = defaultdict(int)
        self._help_memory_batch = None

    def _record_phase(self, phase, started):
        self._aggressive_seconds[phase] += perf_counter() - started
        self._aggressive_calls[phase] += 1

    def _record_intent(self, action, started):
        self._intent_seconds[action] += perf_counter() - started
        self._intent_calls[action] += 1

    def _shard_row(self, person):
        return self._action_snapshot(person)

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
        """Compact aggressive work path with one profession/fit/wage calculation."""
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
                self.store.employment_event(
                    self.current_day, person, hired, "hired", "job_search"
                )
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
            self.store.employment_event(
                self.current_day, person, employer, "laid_off", "insolvent"
            )
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

        energy = max(
            3,
            round(self.rng.randint(6, 12) * profession.energy_multiplier),
        )
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

    def _queue_help_memory(self, observer, other, role, magnitude):
        batch = self._help_memory_batch
        if batch is None:
            if role == "witness":
                observer.observe(other, self.current_day, "help", magnitude)
            else:
                observer.remember(other, self.current_day, "help", role, magnitude)
            return
        key = (observer.id, other.id, role)
        entry = batch.get(key)
        if entry is None:
            batch[key] = [1, float(magnitude)]
        else:
            entry[0] += 1
            entry[1] += float(magnitude)

    def _flush_help_memory_batch(self):
        batch = self._help_memory_batch
        if not batch:
            self._help_memory_batch = None
            return
        started = perf_counter()
        day = self.current_day
        people = self.people
        for (observer_id, other_id, role), (count, magnitude) in batch.items():
            if observer_id >= len(people) or other_id >= len(people):
                continue
            observer = people[observer_id]
            other = people[other_id]
            memory = observer.memory_of(other, day)
            old_affinity = memory.affinity
            old_conflict = memory.conflict_score
            memory.familiarity += count
            memory.last_day = day
            if role == "actor":
                memory.help_given += count
                memory.trust += 4.0 * magnitude
            elif role == "target":
                memory.help_received += count
                memory.trust += 8.0 * magnitude
                memory.grievance -= 3.0 * magnitude
            else:
                memory.observed_help += count
                memory.trust += 2.5 * magnitude
            memory._clamp()
            observer._update_contribution(memory, old_affinity, old_conflict)
        self._help_memory_batch = None
        self._record_phase("flush_help_memory", started)

    def _apply_prepared_social(self, person, prepared):
        """Fast help path; direct relationship bookkeeping is batched per day."""
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
        magnitude = float(amount)
        self._queue_help_memory(person, target, "actor", magnitude)
        self._queue_help_memory(target, person, "target", magnitude)
        for witness_id in witness_ids:
            if witness_id >= len(self.people):
                continue
            witness = self.people[witness_id]
            if (
                witness.alive
                and witness.location_id == person.location_id
                and witness.id not in (person.id, target.id)
            ):
                self._queue_help_memory(witness, person, "witness", magnitude)
                self.total_observations += 1

    def _run_parallel_actions(self, day):
        if not self.shard_pool.should_parallelize(self.alive_count):
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
        rows = [self._shard_row(self.people[pid]) for pid in eligible_order]
        self._record_phase("shard_rows", started)

        started = perf_counter()
        candidate_pools = self._social_candidate_pools()
        self._record_phase("candidate_pools", started)

        started = perf_counter()
        planned_rows = self.shard_pool.plan_day(
            day,
            rows,
            candidate_pools,
            self.actions_per_day,
            self.encounter_sample,
            self.max_witnesses,
            self.visibility,
        )
        self._record_phase("shard_dispatch", started)

        started = perf_counter()
        plans = {
            pid: (final_state, day_intents)
            for pid, final_state, day_intents in planned_rows
        }
        self._record_phase("plans_index", started)

        started = perf_counter()
        for pid in eligible_order:
            person = self.people[pid]
            planned = plans.get(pid)
            if planned is None or not person.alive or day < person.detained_until_day:
                continue
            self._apply_local_state(person, planned[0])
        self._record_phase("apply_local_state", started)

        self._help_memory_batch = {}
        started = perf_counter()
        for round_index in range(self.actions_per_day):
            for pid in eligible_order:
                person = self.people[pid]
                planned = plans.get(pid)
                if planned is None or not person.alive or day < person.detained_until_day:
                    continue
                day_intents = planned[1]
                if round_index >= len(day_intents):
                    continue
                plan = day_intents[round_index]
                if plan is None:
                    continue
                kind = plan[0]
                if kind == "social":
                    _, action, target_id, payload, witness_ids = plan
                    intent_started = perf_counter()
                    self._apply_prepared_social(
                        person,
                        (pid, action, target_id, payload, witness_ids),
                    )
                    self._record_intent(action, intent_started)
                else:
                    _, action = plan
                    intent_started = perf_counter()
                    self._execute_shared_intent(person, action)
                    self._record_intent(action, intent_started)
        self._record_phase("apply_intents", started)
        self._flush_help_memory_batch()
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
                print(
                    "  aggressive shard pool: "
                    f"days={shard.get('days', 0)} tasks={shard.get('tasks', 0)} "
                    f"items={shard.get('items_returned', 0)} "
                    f"worker_cpu={shard.get('worker_seconds', 0.0):.3f}s "
                    f"dispatch_wall={shard.get('dispatch_seconds', 0.0):.3f}s"
                )
        self.shard_pool.close()
        super().close_parallel()

"""RAM-first aggressive world.

This layer intentionally spends memory to remove Python object transport from
the hot action planner. The previous aggressive implementation remains in
aggressive_world_base.py as a fallback/reference implementation.
"""

from time import perf_counter

from .aggressive_economy import SharedEconomyState
from .aggressive_shared import SharedAgentBuffers
from .aggressive_social import SharedSocialState
from .aggressive_world_base import AggressiveParallelAgentWorld as BaseAggressiveWorld
from .agent_shards_shared_ram import RamPersistentDayShardPool
from .agent_world import ParallelAgentWorld
from .population_index import permutation_ids


class AggressiveParallelAgentWorld(BaseAggressiveWorld):
    """Maximum-throughput agent world with over-allocated shared hot state."""

    def __init__(self, *args, agent_workers=0, agent_worker_min_active=64, **kwargs):
        super().__init__(
            *args,
            agent_workers=agent_workers,
            agent_worker_min_active=agent_worker_min_active,
            **kwargs,
        )
        # BaseAggressiveWorld allocates the previous shared backend. Workers are
        # still lazy at this point, so retire those segments and replace them
        # with the RAM-first backend without leaving helper processes behind.
        self.shard_pool.close()
        self.shared_buffers.close(unlink=True)
        self.shared_economy.close(unlink=True)

        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        initial_population = max(1, len(self.people))
        shared_capacity = max(initial_population * 2, initial_population + 65_536)
        self.shared_buffers = SharedAgentBuffers(
            shared_capacity,
            self.actions_per_day,
            self.max_witnesses,
        )
        highest_employer_id = max(
            (employer.id for employer in self.labor_market.employers), default=-1
        )
        employer_capacity = max(16_384, (highest_employer_id + 1) * 16)
        self.shared_economy = SharedEconomyState(
            shared_capacity,
            employer_capacity,
            self.locations,
        )
        self.shared_social = SharedSocialState(
            shared_capacity,
            len(self.locations),
        )
        self.shard_pool = RamPersistentDayShardPool(
            seed,
            self.shared_buffers,
            self.shared_economy,
            self.shared_social,
            workers=agent_workers,
            min_active=agent_worker_min_active,
        )
        self._dirty_market_population = set()

    def _ensure_shared_capacity(self):
        if not super()._ensure_shared_capacity():
            return False
        return len(self.people) <= self.shared_social.population_capacity

    def _apply_prepared_move(self, person, plan):
        """Commit a prepared move and defer market-population refresh."""
        _kind, destination_id, energy_cost = plan
        if (
            not person.alive
            or self.current_day < person.detained_until_day
            or person.energy < 4
        ):
            return
        old_id = person.location_id
        if (
            destination_id == old_id
            or destination_id < 0
            or destination_id >= len(self.locations)
        ):
            return
        if destination_id not in self.locations[old_id].neighbors:
            return self.move(person)

        employer = self.labor_market.employer(person.employer_id)
        if employer and employer.location_id != destination_id:
            self.labor_market.terminate(person, "relocation")
            self.store.employment_event(
                self.current_day, person, employer, "ended", "relocation"
            )
        person.energy = max(0, person.energy - int(energy_cost))
        self.population_index.move(person.id, old_id, destination_id)
        person.location_id = destination_id
        self._dirty_market_population.add(old_id)
        self._dirty_market_population.add(destination_id)
        self.total_moves += 1
        # Aggressive+compact intentionally drops per-move JSON/event rows.
        # Keep sequence progression stable for the remaining compact events.
        self.next_sequence()

    def _apply_prepared_buy(self, person, plan):
        """Commit a prepared purchase without per-purchase event serialization."""
        _kind, good, requested = plan
        if (
            not person.alive
            or self.current_day < person.detained_until_day
            or good not in ("food", "medicine")
            or requested <= 0
        ):
            return
        result = self.goods_market.buy(
            person.location_id, good, requested, person.money
        )
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
        # State/market effects are authoritative; the high-volume JSON event is
        # deliberately omitted in throughput-first mode.
        self.next_sequence()

    def _apply_prepared_social(self, person, prepared):
        """Use worker-owned social memory; main resolves only real-world effects."""
        _pid, action, target_id, payload, witness_ids = prepared
        if action == "help":
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

        if action == "steal":
            resource, proposed = payload
            amount = 0
            if resource == "food" and proposed:
                amount = min(target.food, int(proposed))
                target.food -= amount
                person.food += amount
            elif resource == "money" and proposed:
                amount = min(target.money, float(proposed))
                target.money -= amount
                person.money += amount
            elif resource == "medicine" and proposed and target.medicine > 0:
                amount = 1
                target.medicine -= 1
                person.medicine += 1

            self.store.event(
                self.current_day,
                self.next_sequence(),
                "steal",
                actor=person,
                target=target,
                success=int(bool(amount)),
                location_id=person.location_id,
                resource=resource,
                amount=amount,
            )
            if not amount:
                return
            self.total_thefts += 1
            target.crime_suffered += 1
            self.daily_crimes[person.location_id] += 1
            magnitude = max(1.0, float(amount) / 2.0)
        else:
            damage, energy_cost = payload
            damage = int(damage)
            person.energy = max(0, person.energy - int(energy_cost))
            target.health -= damage
            self.total_attacks += 1
            target.crime_suffered += 1
            self.daily_crimes[person.location_id] += 1
            self.store.event(
                self.current_day,
                self.next_sequence(),
                "attack",
                actor=person,
                target=target,
                success=1,
                location_id=person.location_id,
                damage=damage,
                killed=target.health <= 0,
            )
            magnitude = float(damage) / 10.0

        # Relationship state used by aggressive planning already lives in the
        # persistent worker cache. Do not build a duplicate Python memory graph
        # in main. Preserve observation totals cheaply for statistics.
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

        self.police_response(action, person, target, magnitude)
        if action == "attack" and target.health <= 0:
            self.kill(target, "violence")

    def _flush_move_populations(self):
        if not self._dirty_market_population:
            return
        started = perf_counter()
        for location_id in self._dirty_market_population:
            self.goods_market.set_population(
                location_id,
                self.population_index.population(location_id),
            )
        self._dirty_market_population.clear()
        self._record_phase("flush_move_populations", started)

    def _run_parallel_actions(self, day):
        if (
            not self.shard_pool.should_parallelize(self.alive_count)
            or not self._ensure_shared_capacity()
        ):
            started = perf_counter()
            result = ParallelAgentWorld._run_parallel_actions(self, day)
            self._record_phase("fallback_actions", started)
            return result

        total_started = perf_counter()

        started = perf_counter()
        eligible_count = self.shared_social.sync_world(self, day)
        self._record_phase("shared_social_sync", started)
        if eligible_count <= 0 or self.actions_per_day <= 0:
            self._record_phase("actions_total", total_started)
            return

        started = perf_counter()
        eligible_order = [
            pid
            for pid in permutation_ids(len(self.people), self.rng)
            if self.shared_social.is_eligible(pid)
        ]
        self._record_phase("eligible_order", started)

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
        self.shard_pool.plan_day(
            day,
            len(self.people),
            eligible_count,
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

        self._dirty_market_population.clear()
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
        self._flush_move_populations()
        self._record_phase("actions_total", total_started)

    def close_parallel(self):
        try:
            super().close_parallel()
        finally:
            self.shared_social.close(unlink=True)

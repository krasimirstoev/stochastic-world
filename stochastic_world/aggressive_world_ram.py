"""RAM-first aggressive world.

This layer intentionally spends memory to remove Python object transport from
the hot action planner.  The previous aggressive implementation remains in
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
        # BaseAggressiveWorld allocates the previous shared backend.  Workers are
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
        """Commit a prepared move but defer market population refresh per location."""
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

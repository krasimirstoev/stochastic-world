from .agent_shards import PersistentDayShardPool
from .agent_world import ParallelAgentWorld
from .population_index import permutation_ids


class AggressiveParallelAgentWorld(ParallelAgentWorld):
    """Opt-in throughput-first agent engine with persistent day shards.

    Workers plan all action rounds for their fixed pid shard in one dispatch per
    simulated day. Main remains authoritative for shared-state mutations.
    """

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

    def _shard_row(self, person):
        memories = tuple(
            (
                other_id,
                (memory.trust, memory.grievance, memory.familiarity),
            )
            for other_id, memory in person.memories.items()
        )
        return (self._action_snapshot(person), memories)

    def _apply_shard_safe(self, person, action, event_data):
        if event_data is None:
            return
        if action == "rest":
            energy_gain, health_gain = event_data
            person.energy = min(100, person.energy + int(energy_gain))
            person.health = min(100, person.health + int(health_gain))
            self.store.event(
                self.current_day, self.next_sequence(), "rest",
                actor=person, location_id=person.location_id,
                energy_gain=energy_gain, health_gain=health_gain,
            )
            return
        if action == "heal":
            if person.medicine <= 0 or person.health >= 100:
                return
            (gain,) = event_data
            person.medicine -= 1
            person.health = min(100, person.health + int(gain))
            self.store.event(
                self.current_day, self.next_sequence(), "heal",
                actor=person, location_id=person.location_id, health_gain=gain,
            )
            return
        if action == "repair":
            if person.money < 3 or person.shelter >= 100:
                return
            (gain,) = event_data
            person.money -= 3
            person.shelter = min(100, person.shelter + int(gain))
            self.store.event(
                self.current_day, self.next_sequence(), "repair",
                actor=person, location_id=person.location_id, shelter_gain=gain,
            )
            return
        food_found, medicine_found, cost = event_data
        person.energy = max(0, person.energy - int(cost))
        person.food += int(food_found)
        person.medicine += int(medicine_found)
        self.store.event(
            self.current_day, self.next_sequence(), "scavenge",
            actor=person, location_id=person.location_id,
            food_found=food_found, medicine_found=medicine_found, energy_cost=cost,
        )

    def _run_parallel_actions(self, day):
        if not self.shard_pool.should_parallelize(self.alive_count):
            return super()._run_parallel_actions(day)

        eligible_order = [
            pid
            for pid in permutation_ids(len(self.people), self.rng)
            if self.people[pid].alive
            and self.people[pid].is_adult
            and day >= self.people[pid].detained_until_day
        ]
        if not eligible_order or self.actions_per_day <= 0:
            return

        rows = [self._shard_row(self.people[pid]) for pid in eligible_order]
        plans = {
            pid: day_plans
            for pid, day_plans in self.shard_pool.plan_day(
                day,
                rows,
                self._social_candidate_pools(),
                self.actions_per_day,
                self.encounter_sample,
                self.max_witnesses,
                self.visibility,
            )
        }

        for round_index in range(self.actions_per_day):
            for pid in eligible_order:
                person = self.people[pid]
                day_plans = plans.get(pid)
                if (
                    day_plans is None
                    or round_index >= len(day_plans)
                    or not person.alive
                    or day < person.detained_until_day
                ):
                    continue
                plan = day_plans[round_index]
                kind = plan[0]
                if kind == "safe":
                    _, action, event_data = plan
                    self._apply_shard_safe(person, action, event_data)
                elif kind == "social":
                    _, action, target_id, payload, witness_ids = plan
                    self._apply_prepared_social(
                        person,
                        (pid, action, target_id, payload, witness_ids),
                    )
                else:
                    _, action = plan
                    self._execute_shared_intent(person, action)

    def close_parallel(self):
        self.shard_pool.close()
        super().close_parallel()

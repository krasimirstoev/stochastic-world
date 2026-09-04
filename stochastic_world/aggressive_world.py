from collections import defaultdict
from time import perf_counter

from .agent_shards import PersistentDayShardPool
from .agent_world import ParallelAgentWorld
from .population_index import permutation_ids


class AggressiveParallelAgentWorld(ParallelAgentWorld):
    """Opt-in throughput-first agent engine with persistent day shards.

    Workers plan all action rounds for their fixed pid shard in one dispatch per
    simulated day. Main remains authoritative for shared-state mutations.

    Aggressive mode also records coarse wall-clock timings so expensive IPC and
    synchronization phases can be identified without cProfile distorting worker
    behavior.
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
        self._aggressive_seconds = defaultdict(float)
        self._aggressive_calls = defaultdict(int)

    def _record_phase(self, phase, started):
        self._aggressive_seconds[phase] += perf_counter() - started
        self._aggressive_calls[phase] += 1

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
        plans = {pid: day_plans for pid, day_plans in planned_rows}
        self._record_phase("plans_index", started)

        started = perf_counter()
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
        self._record_phase("apply_plans", started)
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

    def close_parallel(self):
        if self._aggressive_seconds:
            print("  aggressive phase profile (wall clock):")
            for phase, calls, total, avg in self.aggressive_profile_summary():
                print(
                    f"    {phase:<20} calls={calls:>5} "
                    f"total={total:>9.3f}s avg={avg:>8.4f}s"
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

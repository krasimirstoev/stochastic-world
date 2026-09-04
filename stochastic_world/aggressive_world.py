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
                    self._apply_prepared_social(
                        person,
                        (pid, action, target_id, payload, witness_ids),
                    )
                else:
                    _, action = plan
                    self._execute_shared_intent(person, action)
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

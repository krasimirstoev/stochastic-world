import os
from collections import defaultdict

from .demographics import DEMOGRAPHIC_INTERVAL_DAYS
from .labor_market import BUSINESS_INTERVAL_DAYS
from .life_world import LifeWorld
from .multiprocessing_engine import PersistentDistrictPool
from .politics import ELECTION_INTERVAL_DAYS
from .population_index import permutation_ids
from .professions import MOBILITY_INTERVAL_DAYS


class AgentWorkerPool(PersistentDistrictPool):
    """Persistent pool that shards full-agent work by person rather than district."""

    def __init__(self, master_seed, location_count, workers=0, min_active=1024):
        super().__init__(master_seed, location_count, workers=workers, min_active=min_active)
        requested = max(0, int(workers))
        if requested > 0:
            self.worker_count = min(requested, os.cpu_count() or requested)
            self.enabled = self.worker_count >= 2

    def _person_shards(self, snapshots, pid_index=0):
        rows_by_worker = defaultdict(list)
        for row in snapshots:
            pid = row[pid_index]
            rows_by_worker[int(pid) % self.worker_count].append(row)
        return rows_by_worker

    def plan_agent_actions(self, day, round_index, snapshots):
        return self._dispatch(
            "actions",
            day,
            round_index,
            self._person_shards(snapshots, pid_index=0),
        )

    def plan_agent_end_of_day(self, day, snapshots):
        rows_by_worker = defaultdict(list)
        for _district_id, row in snapshots:
            rows_by_worker[int(row[0]) % self.worker_count].append(row)
        return self._dispatch("end_of_day", day, 0, rows_by_worker)


class ParallelAgentWorld(LifeWorld):
    """Full-agent world with deterministic multiprocessing for planning-heavy phases.

    Workers never mutate the authoritative world. They plan one synchronous action
    round at a time and compute end-of-day deltas from primitive snapshots. The
    main process applies results in deterministic person order and remains the only
    SQLite writer.
    """

    def __init__(self, *args, agent_workers=0, agent_worker_min_active=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_mode = "agent"
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        self.district_pool = AgentWorkerPool(
            seed,
            len(self.locations),
            workers=agent_workers,
            min_active=agent_worker_min_active,
        )

    def _action_snapshot(self, person):
        memory = person.aggregate_memory()
        return (
            person.id,
            person.location_id,
            person.food,
            person.medicine,
            person.energy,
            person.health,
            person.shelter,
            person.money,
            person.employer_id is not None,
            self.location_of(person).kind,
            memory["positive_ties"],
            memory["hostile_ties"],
            memory["max_conflict"],
            memory["mean_affinity"],
        )

    def _execute_planned_action(self, person, action):
        if not person.alive or self.current_day < person.detained_until_day or not person.is_adult:
            return
        if not person.is_working_age:
            handler = {
                "scavenge": self.scavenge,
                "buy_supplies": self.buy_supplies,
                "rest": self.rest,
                "heal": self.heal,
                "repair": self.repair,
                "help": self.help,
            }.get(action, self.rest)
            handler(person)
            return
        handler = {
            "move": self.move,
            "work": self.work,
            "scavenge": self.scavenge,
            "buy_supplies": self.buy_supplies,
            "rest": self.rest,
            "heal": self.heal,
            "repair": self.repair,
            "help": self.help,
            "steal": self.steal,
            "attack": self.attack,
        }.get(action)
        if handler:
            handler(person)

    def _run_parallel_actions(self, day):
        eligible_order = [
            pid
            for pid in permutation_ids(len(self.people), self.rng)
            if self.people[pid].alive
            and self.people[pid].is_adult
            and day >= self.people[pid].detained_until_day
        ]
        if not eligible_order or self.actions_per_day <= 0:
            return

        for round_index in range(self.actions_per_day):
            snapshots = [
                self._action_snapshot(self.people[pid])
                for pid in eligible_order
                if self.people[pid].alive
                and day >= self.people[pid].detained_until_day
            ]
            if not snapshots:
                break
            planned = dict(self.district_pool.plan_agent_actions(day, round_index, snapshots))
            for pid in eligible_order:
                person = self.people[pid]
                action = planned.get(pid)
                if action is not None and person.alive and day >= person.detained_until_day:
                    self._execute_planned_action(person, action)

    def _end_snapshot(self, person, rates):
        location = self.location_of(person)
        return (
            person.id,
            person.food,
            person.energy,
            person.shelter,
            person.health,
            person.money,
            person.unemployment_days,
            person.employer_id is not None,
            person.is_working_age,
            person.is_dependent,
            person.is_adult,
            location.shelter_decay_bonus,
            rates.get(person.location_id, 0.0),
        )

    def _apply_end_delta(self, person, delta):
        (
            _pid,
            food,
            energy,
            shelter,
            health,
            unemployment_days,
            lifetime_unemployment_increment,
            ideology_shift,
            damage,
            causes,
        ) = delta
        if not person.alive:
            return
        person.days_in_class += 1
        person.food = food
        person.energy = energy
        person.shelter = shelter
        person.health = health
        person.unemployment_days = unemployment_days
        person.lifetime_unemployment_days += lifetime_unemployment_increment
        if ideology_shift:
            person.shift_ideology(ideology_shift)
        person.decay_memories()
        if damage:
            self.store.event(
                self.current_day,
                self.next_sequence(),
                "daily_harm",
                actor=person,
                location_id=person.location_id,
                damage=damage,
                causes=list(causes),
            )
        if person.health <= 0:
            self.kill(person, "+".join(causes) if causes else "injury")

    def _run_parallel_end_of_day(self):
        for location in self.locations:
            self.crime_history[location.id].append(self.daily_crimes.get(location.id, 0))
        rates = self.crime_rates()
        alive_ids = [p.id for p in self.people if p.alive]

        for pid in alive_ids:
            self.demographics.support_dependent(self.people[pid])

        snapshots = [
            (self.people[pid].location_id, self._end_snapshot(self.people[pid], rates))
            for pid in alive_ids
            if self.people[pid].alive
        ]
        deltas = dict(
            (row[0], row)
            for row in self.district_pool.plan_agent_end_of_day(self.current_day, snapshots)
        )
        for pid in alive_ids:
            person = self.people[pid]
            delta = deltas.get(pid)
            if delta is not None and person.alive:
                self._apply_end_delta(person, delta)
        self.daily_crimes.clear()

    def close_parallel(self):
        self.district_pool.close()

    def run_day(self, day):
        if not self.district_pool.should_parallelize(self.alive_count):
            return super().run_day(day)

        self.current_day = day
        if day == 1 or (day - 1) % ELECTION_INTERVAL_DAYS == 0:
            self.run_election()

        self._run_parallel_actions(day)

        for shipment in self.transport.rebalance(day):
            self.store.shipment(shipment)

        if day % BUSINESS_INTERVAL_DAYS == 0:
            self.welfare_cycle()
            self.business_cycle()
            self.police.rebalance()

        self._run_parallel_end_of_day()
        self.goods_market.reprice()

        if day % MOBILITY_INTERVAL_DAYS == 0:
            self.mobility_cycle()

        police_snapshot = self.police.end_day()
        self.store.write_daily_stats(day, self)
        self.store.write_location_stats(day, self)
        self.store.write_political_stats(day, self)
        self.store.write_social_stats(day, self)
        self.store.write_labor_stats(day, self)
        self.store.write_market_stats(day, self)
        self.store.write_police_stats(day, police_snapshot)
        self.store.commit_day()

        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            self.demographics.cycle(day)
            self.demographics.write_stats(day)
            self.store.commit_day()

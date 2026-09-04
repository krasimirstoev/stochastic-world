from .agent_coarse import AgentCoarsePool
from .demographics import DEMOGRAPHIC_INTERVAL_DAYS
from .labor_market import BUSINESS_INTERVAL_DAYS
from .life_world import LifeWorld
from .politics import ELECTION_INTERVAL_DAYS
from .population_index import permutation_ids
from .professions import MOBILITY_INTERVAL_DAYS


class ParallelAgentWorld(LifeWorld):
    """Full-agent world with deterministic coarse-grained multiprocessing.

    Workers execute agent-local actions completely and return compact state
    deltas. Shared-state actions remain intents applied by the authoritative main
    process in deterministic agent order. SQLite remains single-writer.
    """

    def __init__(self, *args, agent_workers=0, agent_worker_min_active=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_mode = "agent"
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        self.district_pool = AgentCoarsePool(
            seed,
            workers=agent_workers,
            min_active=agent_worker_min_active,
        )

    def remember_interaction(self, actor, target, action, magnitude=1.0):
        if self.store.event_mode == "full":
            return super().remember_interaction(actor, target, action, magnitude)
        actor.remember(target, self.current_day, action, "actor", magnitude)
        target.remember(actor, self.current_day, action, "target", magnitude)
        self.spread_reputation(actor, target, action, magnitude)

    def spread_reputation(self, actor, target, action, magnitude):
        if self.store.event_mode == "full":
            return super().spread_reputation(actor, target, action, magnitude)
        if not self.max_witnesses or self.rng.random() > self.visibility:
            return
        witnesses = self.population_index.sample_people(
            actor.location_id,
            self.rng,
            self.max_witnesses,
            exclude=(actor.id, target.id),
        )
        for witness in witnesses:
            witness.observe(actor, self.current_day, action, magnitude)
            self.total_observations += 1

    def _action_snapshot(self, person):
        memory = person.aggregate_memory()
        location = self.location_of(person)
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
            location.kind,
            memory["positive_ties"],
            memory["hostile_ties"],
            memory["max_conflict"],
            memory["mean_affinity"],
            location.scavenge_food_max,
            location.medicine_chance,
            person.is_working_age,
        )

    def _execute_shared_intent(self, person, action):
        if not person.alive or self.current_day < person.detained_until_day or not person.is_adult:
            return
        if not person.is_working_age:
            handler = {
                "buy_supplies": self.buy_supplies,
                "help": self.help,
            }.get(action, self.rest)
            handler(person)
            return
        handler = {
            "move": self.move,
            "work": self.work,
            "buy_supplies": self.buy_supplies,
            "help": self.help,
            "steal": self.steal,
            "attack": self.attack,
        }.get(action)
        if handler:
            handler(person)

    def _apply_safe_result(self, person, result):
        (
            _pid, action, _safe,
            food, medicine, energy, health, shelter, money, event_data,
        ) = result
        if not person.alive or self.current_day < person.detained_until_day:
            return

        person.food = food
        person.medicine = medicine
        person.energy = energy
        person.health = health
        person.shelter = shelter
        person.money = money

        if event_data is None:
            return

        if action == "rest":
            energy_gain, health_gain = event_data
            self.store.event(
                self.current_day, self.next_sequence(), "rest",
                actor=person, location_id=person.location_id,
                energy_gain=energy_gain, health_gain=health_gain,
            )
        elif action == "heal":
            (gain,) = event_data
            self.store.event(
                self.current_day, self.next_sequence(), "heal",
                actor=person, location_id=person.location_id, health_gain=gain,
            )
        elif action == "repair":
            (gain,) = event_data
            self.store.event(
                self.current_day, self.next_sequence(), "repair",
                actor=person, location_id=person.location_id, shelter_gain=gain,
            )
        elif action == "scavenge":
            food_found, medicine_found, cost = event_data
            self.store.event(
                self.current_day, self.next_sequence(), "scavenge",
                actor=person, location_id=person.location_id,
                food_found=food_found, medicine_found=medicine_found, energy_cost=cost,
            )

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

            planned = {
                row[0]: row
                for row in self.district_pool.plan_round(day, round_index, snapshots)
            }

            for pid in eligible_order:
                person = self.people[pid]
                result = planned.get(pid)
                if result is None or not person.alive or day < person.detained_until_day:
                    continue
                if result[2]:
                    self._apply_safe_result(person, result)
                else:
                    self._execute_shared_intent(person, result[1])

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
            _pid, food, energy, shelter, health, unemployment_days,
            lifetime_unemployment_increment, ideology_shift, damage, causes,
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
        # Agent mode keeps relationship decay fully lazy. Individual memories
        # materialize to the current day only when they are read or mutated.
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
            self.crime_history[location.id].append(
                self.daily_crimes.get(location.id, 0)
            )
        rates = self.crime_rates()
        alive_ids = [p.id for p in self.people if p.alive]

        for pid in alive_ids:
            self.demographics.support_dependent(self.people[pid])

        snapshots = [
            self._end_snapshot(self.people[pid], rates)
            for pid in alive_ids
            if self.people[pid].alive
        ]
        deltas = {
            row[0]: row
            for row in self.district_pool.plan_end_of_day(self.current_day, snapshots)
        }
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

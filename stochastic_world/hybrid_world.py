from .demographics import DEMOGRAPHIC_INTERVAL_DAYS
from .hybrid import PRIORITY_HIGH, PRIORITY_MANDATORY, PRIORITY_NORMAL
from .life_hybrid import LifeHybridEngine
from .life_world import LifeWorld
from .labor_market import BUSINESS_INTERVAL_DAYS
from .multiprocessing_engine import PersistentDistrictPool
from .performance import PhaseProfiler
from .politics import ELECTION_INTERVAL_DAYS
from .professions import MOBILITY_INTERVAL_DAYS


class HybridWorld(LifeWorld):
    """Priority-budgeted hybrid world with age-structured demographic turnover."""

    def __init__(self, *args, hybrid_sample_per_district=256, hybrid_interest_days=30,
                 hybrid_target_explicit=0.03, hybrid_max_explicit=0.05,
                 profile_periodic=False, hybrid_workers=-1, hybrid_worker_min_active=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine_mode = "hybrid"
        self.hybrid = LifeHybridEngine(
            self,
            sample_per_district=hybrid_sample_per_district,
            interest_days=hybrid_interest_days,
            target_explicit_fraction=hybrid_target_explicit,
            max_explicit_fraction=hybrid_max_explicit,
        )
        self.last_hybrid_stats = dict(self.hybrid.last_stats)
        self.profiler = PhaseProfiler(self, enabled=profile_periodic)
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        self.district_pool = PersistentDistrictPool(
            seed,
            len(self.locations),
            workers=hybrid_workers,
            min_active=hybrid_worker_min_active,
        )

    def remember_interaction(self, actor, target, action, magnitude=1.0):
        super().remember_interaction(actor, target, action, magnitude)
        if action == "attack":
            self.hybrid.mark_interesting(actor, days=14, reason="offender", priority=PRIORITY_HIGH)
            self.hybrid.mark_interesting(target, days=10, reason="victim", priority=PRIORITY_HIGH)
        elif action == "steal":
            self.hybrid.mark_interesting(actor, days=10, reason="offender", priority=PRIORITY_HIGH)
            self.hybrid.mark_interesting(target, days=7, reason="victim", priority=PRIORITY_HIGH)

    def police_response(self, crime_type, offender, victim, magnitude):
        result = super().police_response(crime_type, offender, victim, magnitude)
        if result["arrested"]:
            days = max(1, offender.detained_until_day - self.current_day)
            self.hybrid.mark_interesting(offender, days=days, reason="detained", priority=PRIORITY_MANDATORY)
        elif result["responded"]:
            self.hybrid.mark_interesting(offender, days=5, reason="police", priority=PRIORITY_HIGH)
        return result

    def buy_supplies(self, person):
        before = person.shortage_experiences
        super().buy_supplies(person)
        if person.shortage_experiences > before and person.shortage_experiences % 3 == 0:
            self.hybrid.mark_interesting(person, days=2, reason="repeated_shortage", priority=PRIORITY_NORMAL)

    def _end_selected_person(self, person, rates):
        self.apply_daily_person_effects(person, rates)

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

    def _prepare_explicit_people(self, active_ids, day):
        eligible = []
        for pid in active_ids:
            person = self.people[pid]
            if not person.alive:
                continue
            self.hybrid.catch_up(person, day)
            if person.health <= 0:
                self.kill(person, "hybrid_catchup")
                continue
            if day >= person.detained_until_day:
                eligible.append(pid)
        return eligible

    def _run_explicit_actions(self, active_ids, day):
        eligible = self._prepare_explicit_people(active_ids, day)
        if not eligible or self.actions_per_day <= 0:
            return

        if not self.district_pool.should_parallelize(len(eligible)):
            for pid in eligible:
                person = self.people[pid]
                for _ in range(self.actions_per_day):
                    if person.alive and day >= person.detained_until_day:
                        self.perform_action(person)
            return

        for round_index in range(self.actions_per_day):
            snapshots = [
                self._action_snapshot(self.people[pid])
                for pid in eligible
                if self.people[pid].alive
                and self.people[pid].is_adult
                and day >= self.people[pid].detained_until_day
            ]
            if not snapshots:
                break
            with self.profiler.phase(day, "mp_action_planning"):
                planned = dict(self.district_pool.plan_actions(day, round_index, snapshots))
            with self.profiler.phase(day, "mp_action_apply"):
                for pid in eligible:
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

    def _run_selected_end_of_day(self, active_ids):
        for location in self.locations:
            self.crime_history[location.id].append(self.daily_crimes.get(location.id, 0))
        rates = self.crime_rates()
        alive_ids = [pid for pid in active_ids if self.people[pid].alive]

        if not self.district_pool.should_parallelize(len(alive_ids)):
            for pid in alive_ids:
                person = self.people[pid]
                self._end_selected_person(person, rates)
                if person.alive:
                    self.hybrid.touch_after_day(person, self.current_day)
            self.daily_crimes.clear()
            return

        for pid in alive_ids:
            self.demographics.support_dependent(self.people[pid])

        snapshots = [
            (self.people[pid].location_id, self._end_snapshot(self.people[pid], rates))
            for pid in alive_ids
            if self.people[pid].alive
        ]
        with self.profiler.phase(self.current_day, "mp_end_of_day_planning"):
            deltas = dict(
                (row[0], row)
                for row in self.district_pool.plan_end_of_day(self.current_day, snapshots)
            )
        with self.profiler.phase(self.current_day, "mp_end_of_day_apply"):
            for pid in alive_ids:
                person = self.people[pid]
                delta = deltas.get(pid)
                if delta is None or not person.alive:
                    continue
                self._apply_end_delta(person, delta)
                if person.alive:
                    self.hybrid.touch_after_day(person, self.current_day)
        self.daily_crimes.clear()

    def close_parallel(self):
        self.district_pool.close()

    def run_day(self, day):
        self.current_day = day
        day_started = self.profiler.start_day()

        if day == 1 or (day - 1) % ELECTION_INTERVAL_DAYS == 0:
            with self.profiler.phase(day, "election"):
                self.run_election()

        with self.profiler.phase(day, "hybrid_select"):
            active_ids = self.hybrid.select_active(day)

        with self.profiler.phase(day, "hybrid_aggregate"):
            self.last_hybrid_stats = self.hybrid.aggregate_background(day, active_ids)

        with self.profiler.phase(day, "explicit_actions"):
            self._run_explicit_actions(active_ids, day)

        with self.profiler.phase(day, "selected_end_of_day"):
            self._run_selected_end_of_day(active_ids)

        with self.profiler.phase(day, "transport"):
            for shipment in self.transport.rebalance(day):
                self.store.shipment(shipment)

        if day % BUSINESS_INTERVAL_DAYS == 0:
            with self.profiler.phase(day, "welfare"):
                self.welfare_cycle()
            with self.profiler.phase(day, "business"):
                self.business_cycle()
            with self.profiler.phase(day, "police_rebalance"):
                self.police.rebalance()

        with self.profiler.phase(day, "market_reprice"):
            self.goods_market.reprice()

        if day % MOBILITY_INTERVAL_DAYS == 0:
            with self.profiler.phase(day, "mobility"):
                self.mobility_cycle()

        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            with self.profiler.phase(day, "demographics"):
                self.demographics.cycle(day)

        with self.profiler.phase(day, "statistics"):
            police_snapshot = self.police.end_day()
            self.store.write_daily_stats(day, self)
            self.store.write_location_stats(day, self)
            self.store.write_political_stats(day, self)
            self.store.write_social_stats(day, self)
            self.store.write_labor_stats(day, self)
            self.store.write_market_stats(day, self)
            self.store.write_police_stats(day, police_snapshot)
            self.store.write_hybrid_stats(day, self.last_hybrid_stats)
            if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
                self.demographics.write_stats(day)

        with self.profiler.phase(day, "commit"):
            self.store.commit_day()

        self.profiler.finish_day(day, day_started)

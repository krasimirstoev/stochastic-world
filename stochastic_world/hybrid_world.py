from .demographics import DEMOGRAPHIC_INTERVAL_DAYS
from .hybrid import PRIORITY_HIGH, PRIORITY_MANDATORY, PRIORITY_NORMAL
from .life_hybrid import LifeHybridEngine
from .life_world import LifeWorld
from .labor_market import BUSINESS_INTERVAL_DAYS
from .politics import ELECTION_INTERVAL_DAYS
from .professions import MOBILITY_INTERVAL_DAYS


class HybridWorld(LifeWorld):
    """Priority-budgeted hybrid world with age-structured demographic turnover."""

    def __init__(self, *args, hybrid_sample_per_district=256, hybrid_interest_days=30,
                 hybrid_target_explicit=0.03, hybrid_max_explicit=0.05, **kwargs):
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

    def run_day(self, day):
        self.current_day = day
        if day == 1 or (day - 1) % ELECTION_INTERVAL_DAYS == 0:
            self.run_election()
        active_ids = self.hybrid.select_active(day)
        self.last_hybrid_stats = self.hybrid.aggregate_background(day, active_ids)
        for pid in active_ids:
            person = self.people[pid]
            if not person.alive:
                continue
            self.hybrid.catch_up(person, day)
            if person.health <= 0:
                self.kill(person, "hybrid_catchup")
                continue
            if day >= person.detained_until_day:
                for _ in range(self.actions_per_day):
                    if person.alive and day >= person.detained_until_day:
                        self.perform_action(person)
        for location in self.locations:
            self.crime_history[location.id].append(self.daily_crimes.get(location.id, 0))
        rates = self.crime_rates()
        for pid in active_ids:
            person = self.people[pid]
            if not person.alive:
                continue
            self._end_selected_person(person, rates)
            if person.alive:
                self.hybrid.touch_after_day(person, day)
        self.daily_crimes.clear()
        for shipment in self.transport.rebalance(day):
            self.store.shipment(shipment)
        if day % BUSINESS_INTERVAL_DAYS == 0:
            self.welfare_cycle()
            self.business_cycle()
            self.police.rebalance()
        self.goods_market.reprice()
        if day % MOBILITY_INTERVAL_DAYS == 0:
            self.mobility_cycle()
        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            self.demographics.cycle(day)
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
        self.store.commit_day()

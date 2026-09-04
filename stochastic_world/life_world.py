from .demographics import DEMOGRAPHIC_INTERVAL_DAYS, DemographicSystem
from .professions import choose_profession, mobility_decision
from .world import World


class LifeWorld(World):
    """World with age, households, births, natural mortality and generational turnover."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labor_market.unemployment_rate = lambda people: self._workforce_unemployment_rate()
        seed = args[4] if len(args) > 4 else kwargs.get("population_seed", 0)
        locale = args[5] if len(args) > 5 else kwargs.get("faker_locale", "en_US")
        self.demographics = DemographicSystem(self, seed, locale)
        self.demographics.write_stats(0)
        self.store.conn.commit()

    def _workforce_unemployment_rate(self):
        workforce = [p for p in self.people if p.alive and p.is_working_age]
        if not workforce:
            return 0.0
        return sum(p.employer_id is None for p in workforce) / len(workforce)

    def _seed_employment(self):
        for person in self.people:
            if not person.is_working_age or self.rng.random() > 0.72:
                continue
            employer = self.labor_market.fast_hire(person)
            if employer:
                self.store.employment_event(0, person, employer, "hired", "initial_assignment")

    def add_person(self, person):
        if person.id != len(self.people):
            raise ValueError("new person_id must append contiguously")
        self.people.append(person)
        self.population_index.add(person.id, person.location_id)
        self.alive_count += 1
        self.invalidate_living_cache()
        self.goods_market.set_population(person.location_id, self.population_index.population(person.location_id))
        self.store.register_person(person)
        hybrid = getattr(self, "hybrid", None)
        if hybrid is not None:
            hybrid.ensure_person_capacity(len(self.people))

    def kill(self, person, cause):
        was_workforce = bool(person.alive and person.is_working_age)
        partner = None
        if person.partner_id is not None and person.partner_id < len(self.people):
            partner = self.people[person.partner_id]
        super().kill(person, cause)
        if was_workforce and hasattr(self, "demographics"):
            self.demographics.working_age_count = max(0, self.demographics.working_age_count - 1)
        if partner is not None and partner.alive and partner.partner_id == person.id:
            partner.partner_id = None
            if hasattr(self, "demographics"):
                self.demographics._persist_person(partner)

    def perform_action(self, person):
        if not person.is_adult:
            return
        if not person.is_working_age:
            # Retired adults do not use profession-dependent work/move paths and
            # do not participate in the routine crime action pool. They retain
            # household, care and consumption behavior.
            action = self.decision_engine.choose_action(person, self.rng, self.location_of(person))
            handler = {
                "scavenge": self.scavenge,
                "buy_supplies": self.buy_supplies,
                "rest": self.rest,
                "heal": self.heal,
                "repair": self.repair,
                "help": self.help,
            }.get(action)
            if handler is None:
                handler = self.rest
            return handler(person)
        return super().perform_action(person)

    def work(self, person):
        if not person.is_working_age:
            return self.rest(person)
        return super().work(person)

    def mobility_cycle(self):
        for person in self.people:
            if not person.alive or not person.is_working_age:
                continue
            old_class, old_profession = person.social_class, person.profession
            new_class, score, direction = mobility_decision(person)
            if new_class == old_class:
                continue
            person.social_class = new_class
            person.profession = choose_profession(new_class, self.rng)
            person.days_in_class = 0
            person.career_progress *= 0.55 if direction == "up" else 0.70
            self.total_mobility_changes += 1
            self.store.mobility_event(self.current_day, person, old_class, new_class, old_profession,
                                      person.profession, score, direction)
            self.store.event(self.current_day, self.next_sequence(), "social_mobility", actor=person,
                             old_class=old_class, new_class=new_class, old_profession=old_profession,
                             new_profession=person.profession, score=round(score, 3), direction=direction)

    def apply_daily_person_effects(self, person, rates):
        if not person.alive:
            return
        self.demographics.support_dependent(person)
        location = self.location_of(person)
        person.days_in_class += 1
        if person.is_working_age:
            if person.employer_id is None:
                person.unemployment_days += 1
                person.lifetime_unemployment_days += 1
            else:
                person.unemployment_days = 0
            if person.unemployment_days > 30:
                person.shift_ideology(-0.0004)
        else:
            person.unemployment_days = 0
        person.food -= 1
        person.energy = max(0, person.energy - (2 if person.is_dependent else 3))
        person.shelter = max(0, person.shelter - self.rng.randint(0, 2) - location.shelter_decay_bonus)
        person.decay_memories()
        if person.is_adult:
            self.politics.update_attitudes(person, rates.get(person.location_id, 0.0))
        damage, causes = 0, []
        if person.food < 0:
            person.food = 0
            damage += self.rng.randint(4, 10)
            causes.append("starvation")
        if person.energy == 0:
            damage += self.rng.randint(2, 6)
            causes.append("exhaustion")
        if person.shelter <= 20 and self.rng.random() < 0.35:
            damage += self.rng.randint(2, 7)
            causes.append("exposure")
        if damage:
            person.health -= damage
            self.store.event(self.current_day, self.next_sequence(), "daily_harm", actor=person,
                             location_id=location.id, damage=damage, causes=causes)
        if person.health <= 0:
            self.kill(person, "+".join(causes) if causes else "injury")

    def end_of_day(self):
        for location in self.locations:
            self.crime_history[location.id].append(self.daily_crimes.get(location.id, 0))
        rates = self.crime_rates()
        for person in self.people:
            if person.alive:
                self.apply_daily_person_effects(person, rates)
        self.daily_crimes.clear()

    def run_day(self, day):
        super().run_day(day)
        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            self.demographics.cycle(day)
            self.demographics.write_stats(day)
            self.store.commit_day()

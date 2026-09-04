from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import median

from faker import Faker

from .person import ADULT_AGE_DAYS, RETIREMENT_AGE_DAYS, Person
from .professions import choose_profession


DEMOGRAPHIC_INTERVAL_DAYS = 30
PREGNANCY_DAYS = 280


@dataclass(slots=True)
class Household:
    id: int
    location_id: int
    created_day: int = 0


class DemographicSystem:
    """Monthly demographic lifecycle with households, fertility, aging and natural mortality."""

    def __init__(self, world, seed: int, locale: str = "en_US"):
        self.world = world
        self.rng = world.rng
        self.fake = Faker(locale)
        self.fake.seed_instance(seed ^ 0xD3A06A9)
        self.households = {}
        self.members = defaultdict(set)
        self.next_household_id = 0
        self.total_births = 0
        self.total_natural_deaths = 0
        self.total_adulthoods = 0
        self.total_retirements = 0
        self.total_partnerships = 0
        self.working_age_count = sum(1 for p in world.people if p.alive and p.is_working_age)
        self._ensure_schema()
        self._initialize_households()
        self._persist_all_people()
        self._persist_households()

    def _ensure_schema(self):
        conn = self.world.store.conn
        existing = {r[1] for r in conn.execute("PRAGMA table_info(persons)")}
        additions = (
            ("age_days", "INTEGER NOT NULL DEFAULT 10950"),
            ("sex", "TEXT NOT NULL DEFAULT 'female'"),
            ("birth_day", "INTEGER NOT NULL DEFAULT 0"),
            ("mother_id", "INTEGER"),
            ("father_id", "INTEGER"),
            ("partner_id", "INTEGER"),
            ("household_id", "INTEGER"),
            ("generation", "INTEGER NOT NULL DEFAULT 0"),
            ("retired", "INTEGER NOT NULL DEFAULT 0"),
        )
        for name, definition in additions:
            if name not in existing:
                conn.execute(f"ALTER TABLE persons ADD COLUMN {name} {definition}")
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS households(
          simulation_id INTEGER NOT NULL,household_id INTEGER NOT NULL,
          created_day INTEGER NOT NULL,location_id INTEGER NOT NULL,
          PRIMARY KEY(simulation_id,household_id));
        CREATE TABLE IF NOT EXISTS household_members(
          simulation_id INTEGER NOT NULL,person_id INTEGER NOT NULL,
          household_id INTEGER NOT NULL,joined_day INTEGER NOT NULL,
          PRIMARY KEY(simulation_id,person_id));
        CREATE TABLE IF NOT EXISTS life_events(
          id INTEGER PRIMARY KEY AUTOINCREMENT,simulation_id INTEGER NOT NULL,
          day INTEGER NOT NULL,event_type TEXT NOT NULL,person_id INTEGER,
          related_person_id INTEGER,household_id INTEGER,data TEXT);
        CREATE TABLE IF NOT EXISTS demographic_stats(
          simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,population INTEGER NOT NULL,
          births INTEGER NOT NULL,natural_deaths INTEGER NOT NULL,total_deaths INTEGER NOT NULL,
          median_age REAL NOT NULL,age_0_14 INTEGER NOT NULL,age_15_24 INTEGER NOT NULL,
          age_25_44 INTEGER NOT NULL,age_45_64 INTEGER NOT NULL,age_65_plus INTEGER NOT NULL,
          households INTEGER NOT NULL,pregnant INTEGER NOT NULL,max_generation INTEGER NOT NULL,
          PRIMARY KEY(simulation_id,day));
        CREATE INDEX IF NOT EXISTS idx_life_events_type ON life_events(simulation_id,event_type,day);
        """)
        conn.commit()

    def _new_household(self, location_id, day=0):
        hid = self.next_household_id
        self.next_household_id += 1
        self.households[hid] = Household(hid, location_id, day)
        return hid

    def _attach(self, person, household_id):
        if person.household_id is not None:
            self.members[person.household_id].discard(person.id)
        person.household_id = household_id
        self.members[household_id].add(person.id)

    def _initialize_households(self):
        by_location = defaultdict(list)
        for p in self.world.people:
            by_location[p.location_id].append(p)

        for location_id, people in by_location.items():
            females = [p for p in people if p.is_adult and p.age_days < 60 * 365 and p.sex == "female"]
            males = [p for p in people if p.is_adult and p.age_days < 65 * 365 and p.sex == "male"]
            self.rng.shuffle(females)
            self.rng.shuffle(males)
            paired = set()
            for female, male in zip(females, males):
                if abs(female.age_days - male.age_days) > 18 * 365:
                    continue
                hid = self._new_household(location_id)
                self._attach(female, hid)
                self._attach(male, hid)
                female.partner_id = male.id
                male.partner_id = female.id
                paired.update((female.id, male.id))
            for p in people:
                if p.id in paired or p.is_dependent:
                    continue
                self._attach(p, self._new_household(location_id))

            family_households = [hid for hid in self.members if self.households[hid].location_id == location_id and len(self.members[hid]) >= 2]
            minors = [p for p in people if p.is_dependent]
            self.rng.shuffle(minors)
            for child in minors:
                if family_households:
                    hid = self.rng.choice(family_households)
                else:
                    hid = self._new_household(location_id)
                self._attach(child, hid)
                adults = [self.world.people[pid] for pid in self.members[hid] if self.world.people[pid].is_adult]
                mother = next((p for p in adults if p.sex == "female"), None)
                father = next((p for p in adults if p.sex == "male"), None)
                child.mother_id = mother.id if mother else None
                child.father_id = father.id if father else None

    def _persist_person(self, p):
        self.world.store.conn.execute("""
            UPDATE persons SET age_days=?,sex=?,birth_day=?,mother_id=?,father_id=?,partner_id=?,
                household_id=?,generation=?,retired=?
            WHERE simulation_id=? AND person_id=?
        """, (p.age_days,p.sex,p.birth_day,p.mother_id,p.father_id,p.partner_id,p.household_id,
              p.generation,int(p.retired),self.world.store.simulation_id,p.id))
        self.world.store.conn.execute("""
            INSERT INTO household_members(simulation_id,person_id,household_id,joined_day)
            VALUES(?,?,?,?) ON CONFLICT(simulation_id,person_id) DO UPDATE SET household_id=excluded.household_id
        """, (self.world.store.simulation_id,p.id,p.household_id if p.household_id is not None else -1,max(0,p.birth_day)))

    def _persist_all_people(self):
        for p in self.world.people:
            self._persist_person(p)

    def _persist_households(self):
        rows = [(self.world.store.simulation_id,h.id,h.created_day,h.location_id) for h in self.households.values()]
        self.world.store.conn.executemany("INSERT OR REPLACE INTO households VALUES(?,?,?,?)", rows)
        self.world.store.conn.commit()

    def _life_event(self, day, event_type, person=None, related=None, household_id=None, **data):
        import json
        self.world.store.conn.execute("""
            INSERT INTO life_events(simulation_id,day,event_type,person_id,related_person_id,household_id,data)
            VALUES(?,?,?,?,?,?,?)
        """, (self.world.store.simulation_id,day,event_type,getattr(person,"id",None),getattr(related,"id",None),
              household_id,json.dumps(data,ensure_ascii=False,sort_keys=True) if data else None))

    def support_dependent(self, person):
        if not person.alive or not person.is_dependent or person.household_id is None:
            return
        adults = [self.world.people[pid] for pid in self.members.get(person.household_id, ())
                  if pid < len(self.world.people) and self.world.people[pid].alive and self.world.people[pid].is_adult]
        if not adults:
            return
        donor = max(adults, key=lambda p: (p.food, p.money))
        if person.food <= 3 and donor.food > 5:
            amount = min(2, donor.food - 4)
            donor.food -= amount
            person.food += amount
        if person.health < 70 and person.medicine == 0:
            medic = max(adults, key=lambda p: p.medicine)
            if medic.medicine > 1:
                medic.medicine -= 1
                person.medicine += 1
        person.shelter = max(person.shelter, max(a.shelter for a in adults) - 8)

    def _annual_mortality(self, person):
        age = person.age_years
        if age < 1: base = 0.004
        elif age < 15: base = 0.0002
        elif age < 45: base = 0.0006
        elif age < 65: base = 0.004
        elif age < 75: base = 0.018
        elif age < 85: base = 0.055
        elif age < 95: base = 0.16
        else: base = min(0.65, 0.22 + (age - 95) * 0.035)
        health_factor = 1.0 + max(0, 70 - person.health) / 45.0
        return min(0.95, base * health_factor)

    def _fertility_annual(self, mother):
        age = mother.age_years
        if 18 <= age < 25: return 0.075
        if age < 35: return 0.115
        if age < 40: return 0.065
        if age <= 42: return 0.018
        return 0.0

    def _household_condition(self, person):
        ids = self.members.get(person.household_id, ())
        adults = [self.world.people[pid] for pid in ids if self.world.people[pid].alive and self.world.people[pid].is_adult]
        if not adults:
            return 0.4
        avg_money = sum(p.money for p in adults) / len(adults)
        avg_food = sum(p.food for p in adults) / len(adults)
        avg_shelter = sum(p.shelter for p in adults) / len(adults)
        return max(0.35, min(1.2, 0.55 + avg_money / 120 + avg_food / 45 + avg_shelter / 300))

    def _merge_households(self, keep_id, merge_id, day):
        if keep_id == merge_id:
            return keep_id
        for pid in list(self.members.get(merge_id, ())):
            self._attach(self.world.people[pid], keep_id)
            self._persist_person(self.world.people[pid])
        self.members.pop(merge_id, None)
        self.households.pop(merge_id, None)
        self.world.store.conn.execute("DELETE FROM households WHERE simulation_id=? AND household_id=?",
                                      (self.world.store.simulation_id, merge_id))
        return keep_id

    def _form_partnerships(self, day):
        by_location = defaultdict(lambda: {"female": [], "male": []})
        for p in self.world.people:
            if not p.alive or not (20 * 365 <= p.age_days <= 45 * 365) or p.partner_id is not None:
                continue
            by_location[p.location_id][p.sex].append(p)
        for pools in by_location.values():
            self.rng.shuffle(pools["female"]); self.rng.shuffle(pools["male"])
            for female, male in zip(pools["female"], pools["male"]):
                if self.rng.random() > 0.025 or abs(female.age_days - male.age_days) > 15 * 365:
                    continue
                female.partner_id = male.id; male.partner_id = female.id
                hid = self._merge_households(female.household_id, male.household_id, day)
                self._attach(female, hid); self._attach(male, hid)
                self._persist_person(female); self._persist_person(male)
                self.total_partnerships += 1
                self._life_event(day,"partnership",female,male,hid)

    def _start_pregnancies(self, day):
        child_counts = Counter(p.mother_id for p in self.world.people if p.alive and p.mother_id is not None)
        for mother in list(self.world.people):
            if not mother.alive or mother.sex != "female" or mother.pregnant_until_day or mother.partner_id is None:
                continue
            annual = self._fertility_annual(mother)
            if annual <= 0 or mother.health < 50:
                continue
            if mother.partner_id >= len(self.world.people):
                continue
            partner = self.world.people[mother.partner_id]
            if not partner.alive or partner.household_id != mother.household_id:
                continue
            existing = child_counts.get(mother.id, 0)
            parity = max(0.25, 1.0 - existing * 0.18)
            monthly_p = annual * (DEMOGRAPHIC_INTERVAL_DAYS / 365.0) * self._household_condition(mother) * parity
            if self.rng.random() < monthly_p:
                mother.pregnant_until_day = day + PREGNANCY_DAYS
                mother.pregnancy_partner_id = partner.id
                self._life_event(day,"pregnancy",mother,partner,mother.household_id,due_day=mother.pregnant_until_day)

    def _birth(self, mother, day):
        father = None
        if mother.pregnancy_partner_id is not None and mother.pregnancy_partner_id < len(self.world.people):
            candidate = self.world.people[mother.pregnancy_partner_id]
            if candidate.alive:
                father = candidate
        generation = max(mother.generation, father.generation if father else mother.generation) + 1
        ideology = mother.ideology if father is None else (mother.ideology + father.ideology) / 2
        child = Person(
            id=len(self.world.people), name=self.fake.name(), social_class=mother.social_class,
            profession="dependent", ideology=max(-1.0,min(1.0,ideology+self.rng.gauss(0,0.16))),
            location_id=mother.location_id, food=8, money=0.0, medicine=1, energy=90,
            shelter=max(35,mother.shelter), health=self.rng.randint(88,100), age_days=0,
            sex="female" if self.rng.random()<0.5 else "male", birth_day=day,
            mother_id=mother.id, father_id=father.id if father else None,
            household_id=mother.household_id, generation=generation,
        )
        self.world.add_person(child)
        self.members[child.household_id].add(child.id)
        self._persist_person(child)
        mother.pregnant_until_day = 0; mother.pregnancy_partner_id = None
        self._persist_person(mother)
        self.total_births += 1
        self._life_event(day,"birth",child,mother,child.household_id,father_id=child.father_id,generation=generation)

    def _complete_pregnancies(self, day):
        for mother in list(self.world.people):
            if mother.alive and mother.pregnant_until_day and mother.pregnant_until_day <= day:
                self._birth(mother, day)

    def _age_and_mortality(self, day):
        for p in list(self.world.people):
            if not p.alive:
                continue
            old_age = p.age_days
            p.age_days = max(p.age_days, day - p.birth_day)
            if old_age < ADULT_AGE_DAYS <= p.age_days:
                p.profession = choose_profession(p.social_class, self.rng)
                p.unemployment_days = 0
                self.total_adulthoods += 1
                self.working_age_count += 1
                self._persist_person(p)
                self._life_event(day,"coming_of_age",p,household_id=p.household_id,profession=p.profession)
            if old_age < RETIREMENT_AGE_DAYS <= p.age_days:
                employer = self.world.labor_market.employer_any(p.employer_id)
                if employer:
                    self.world.labor_market.terminate(p,"retirement")
                    self.world.store.employment_event(day,p,employer,"ended","retirement")
                p.retired = True; p.profession = "retired"
                self.working_age_count = max(0, self.working_age_count - 1)
                self.total_retirements += 1
                self._persist_person(p)
                self._life_event(day,"retirement",p,household_id=p.household_id)
            annual = self._annual_mortality(p)
            monthly = 1.0 - (1.0 - annual) ** (DEMOGRAPHIC_INTERVAL_DAYS / 365.0)
            if self.rng.random() < monthly:
                self.world.kill(p,"natural_age")
                self.total_natural_deaths += 1

    def cycle(self, day):
        self._age_and_mortality(day)
        self._complete_pregnancies(day)
        self._form_partnerships(day)
        self._start_pregnancies(day)
        for p in self.world.people:
            if p.alive and p.is_dependent:
                self.support_dependent(p)

    def write_stats(self, day):
        living = [p for p in self.world.people if p.alive]
        ages = [p.age_years for p in living]
        bands = [0,0,0,0,0]
        for p in living:
            age = p.age_years
            if age < 15: bands[0]+=1
            elif age < 25: bands[1]+=1
            elif age < 45: bands[2]+=1
            elif age < 65: bands[3]+=1
            else: bands[4]+=1
        active_households = len({p.household_id for p in living if p.household_id is not None})
        pregnant = sum(1 for p in living if p.pregnant_until_day)
        max_generation = max((p.generation for p in living), default=0)
        self.world.store.conn.execute("""
            INSERT OR REPLACE INTO demographic_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (self.world.store.simulation_id,day,len(living),self.total_births,self.total_natural_deaths,
              self.world.total_deaths,median(ages) if ages else 0.0,*bands,active_households,pregnant,max_generation))

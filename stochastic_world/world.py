from collections import defaultdict, deque

from .decisions import WeightedRandomDecision
from .geography import build_locations
from .labor_market import BUSINESS_INTERVAL_DAYS, LaborMarket, build_employers
from .market import GoodsMarket
from .police import PoliceSystem
from .politics import ELECTION_INTERVAL_DAYS, PoliticalSystem
from .population import build_population
from .population_index import PopulationIndex, permutation_ids
from .professions import MOBILITY_INTERVAL_DAYS, choose_profession, mobility_decision, profession_for, workplace_fit
from .transport import TransportSystem


LOCAL_ENCOUNTER_SAMPLE = 24


class World:
    def __init__(self,rng,population,actions_per_day,store,population_seed,faker_locale="en_US",decision_engine=None,visibility=0.65,max_witnesses=3,locations_count=0,target_neighborhood_size=20_000,police_per_1000=2.2):
        self.rng=rng;self.actions_per_day=actions_per_day;self.store=store;self.decision_engine=decision_engine or WeightedRandomDecision()
        self.locations=build_locations(locations_count,population_size=population,target_neighborhood_size=target_neighborhood_size)
        self.people=build_population(population,population_seed,faker_locale)
        offset=self.rng.randrange(len(self.locations))
        for person in self.people:person.location_id=(person.id+offset)%len(self.locations)
        self.population_index=PopulationIndex(self.people,len(self.locations));self.alive_count=len(self.people);self._living_cache=None
        self.politics=PoliticalSystem(population_seed,faker_locale)
        self.labor_market=LaborMarket(build_employers(self.locations,population,self.rng),self.rng)
        self.goods_market=GoodsMarket(self.locations,{loc.id:self.population_index.population(loc.id) for loc in self.locations})
        self.police=PoliceSystem(self.locations,self.population_index,self.rng,police_per_1000)
        self.transport=TransportSystem(self.locations,self.goods_market,self.labor_market,self.rng)
        self.visibility=max(0.0,min(1.0,visibility));self.max_witnesses=max(0,max_witnesses)
        self.current_day=0;self.sequence=0;self.total_moves=0;self.total_helps=0;self.total_thefts=0;self.total_attacks=0;self.total_observations=0;self.total_deaths=0;self.total_mobility_changes=0;self.total_arrests=0
        self.daily_crimes=defaultdict(int);self.crime_history={loc.id:deque(maxlen=30) for loc in self.locations}
        for location in self.locations:self.store.register_location(location)
        for person in self.people:self.store.register_person(person)
        self.store.register_parties(self.politics)
        for employer in self.labor_market.employers:self.store.register_employer(employer)
        self._seed_employment();self.store.conn.commit()

    def _seed_employment(self):
        for person in self.people:
            if self.rng.random()>0.72:continue
            employer=self.labor_market.fast_hire(person)
            if employer:self.store.employment_event(0,person,employer,"hired","initial_assignment")

    def invalidate_living_cache(self):self._living_cache=None
    def living_people(self):
        if self._living_cache is None:self._living_cache=[p for p in self.people if p.alive]
        return self._living_cache
    def people_in_location(self,location_id):return [self.people[pid] for pid in self.population_index.ids(location_id) if self.people[pid].alive]
    def location_of(self,person):return self.locations[person.location_id]
    def local_candidates(self,person,limit=LOCAL_ENCOUNTER_SAMPLE,exclude=()):return self.population_index.sample_people(person.location_id,self.rng,limit,tuple(exclude)+(person.id,))
    def local_crime_rate(self,location_id):
        population=max(1,self.population_index.population(location_id));history=self.crime_history[location_id]
        return sum(history)/(population*len(history)) if history else 0.0
    def crime_rates(self):return {loc.id:self.local_crime_rate(loc.id) for loc in self.locations}
    def next_sequence(self):self.sequence+=1;return self.sequence

    def weighted_target(self,actor,mode):
        candidates=self.local_candidates(actor)
        if not candidates:return None
        weights=[]
        for target in candidates:
            memory=actor.memory_of(target)
            if mode=="help":value=1+max(0,memory.affinity)/10+memory.familiarity/20
            elif mode=="attack":value=1+memory.conflict_score/5
            else:
                wealth=target.food+target.medicine*2+max(0,target.money)/4;value=1+memory.conflict_score/18+max(0,-memory.affinity)/30+wealth/25
            weights.append(max(0.1,value))
        return self.rng.choices(candidates,weights=weights,k=1)[0]

    def relation_snapshot(self,observer,other):
        m=observer.memory_of(other);return {"trust":round(m.trust,3),"grievance":round(m.grievance,3),"affinity":round(m.affinity,3),"conflict":round(m.conflict_score,3),"familiarity":m.familiarity,"observed_help":m.observed_help,"observed_theft":m.observed_theft,"observed_attack":m.observed_attack}

    def remember_interaction(self,actor,target,action,magnitude=1.0):
        actor_before=self.relation_snapshot(actor,target);target_before=self.relation_snapshot(target,actor);actor.remember(target,self.current_day,action,"actor",magnitude);target.remember(actor,self.current_day,action,"target",magnitude);actor_after=self.relation_snapshot(actor,target);target_after=self.relation_snapshot(target,actor)
        if self.store.event_mode=="full" or action in ("steal","attack"):
            self.store.relationship_event(self.current_day,self.sequence,action,actor,target,actor_before,actor_after,target_before,target_after);self.store.upsert_relationship(self.current_day,actor,target);self.store.upsert_relationship(self.current_day,target,actor)
        self.spread_reputation(actor,target,action,magnitude)

    def spread_reputation(self,actor,target,action,magnitude):
        if not self.max_witnesses or self.rng.random()>self.visibility:return
        witnesses=self.population_index.sample_people(actor.location_id,self.rng,self.max_witnesses,exclude=(actor.id,target.id))
        for witness in witnesses:
            before=self.relation_snapshot(witness,actor);witness.observe(actor,self.current_day,action,magnitude);after=self.relation_snapshot(witness,actor);self.total_observations+=1
            if self.store.event_mode=="full" or action in ("steal","attack"):
                self.store.observation_event(self.current_day,self.sequence,action,witness,actor,target,before,after,actor.location_id);self.store.upsert_relationship(self.current_day,witness,actor)

    def choose_destination(self,person):
        current=self.location_of(person);options=[self.locations[i] for i in current.neighbors]
        if not options:return current
        profession=profession_for(person);employer=self.labor_market.employer(person.employer_id);weights=[]
        for location in options:
            weight=1.0
            if employer and location.id==employer.location_id:weight*=5
            if location.kind in profession.workplace_kinds:weight*=2.2
            if not employer and self.labor_market.vacancies(location.id)>0:weight*=1.8
            if person.food<=3 and self.goods_market.total_stock(location.id,"food")>0:weight*=2.5
            if person.medicine==0 and location.kind=="clinic":weight*=3.0
            weights.append(weight)
        return self.rng.choices(options,weights=weights,k=1)[0]

    def move(self,person):
        if person.energy<4:return
        old=self.location_of(person);destination=self.choose_destination(person)
        if destination.id==old.id:return
        employer=self.labor_market.employer(person.employer_id)
        if employer and employer.location_id!=destination.id:self.labor_market.terminate(person,"relocation");self.store.employment_event(self.current_day,person,employer,"ended","relocation")
        cost=self.rng.randint(3,7);person.energy=max(0,person.energy-cost);self.population_index.move(person.id,old.id,destination.id);person.location_id=destination.id
        self.goods_market.set_population(old.id,self.population_index.population(old.id));self.goods_market.set_population(destination.id,self.population_index.population(destination.id));self.total_moves+=1
        self.store.event(self.current_day,self.next_sequence(),"move",actor=person,from_location=old.id,to_location=destination.id,energy_cost=cost,profession=person.profession)

    def work(self,person):
        if person.energy<8:return self.rest(person)
        employer=self.labor_market.employer(person.employer_id)
        if employer is None:
            person.employer_id=None;hired=self.labor_market.hire(person)
            if hired:self.store.employment_event(self.current_day,person,hired,"hired","job_search");self.store.event(self.current_day,self.next_sequence(),"job_found",actor=person,employer_id=hired.id,employer=hired.name)
            else:self.store.event(self.current_day,self.next_sequence(),"job_search",actor=person,location_id=person.location_id,vacancies=self.labor_market.vacancies(person.location_id),success=0)
            return
        if employer.location_id!=person.location_id:return
        shift=self.labor_market.work_shift(person,self.location_of(person))
        if shift["insolvent"]:
            self.labor_market.terminate(person,"insolvent");self.store.employment_event(self.current_day,person,employer,"laid_off","insolvent");self.store.event(self.current_day,self.next_sequence(),"layoff",actor=person,employer_id=employer.id,reason="insolvent");return
        gross=shift["gross"];profession=profession_for(person);fit=workplace_fit(person,self.location_of(person));energy=max(3,round(self.rng.randint(6,12)*profession.energy_multiplier));person.money+=gross;person.lifetime_gross_income+=gross;person.work_experience+=1;person.career_progress+=profession.advancement_rate*fit;tax=self.politics.collect_tax(person,gross);person.energy=max(0,person.energy-energy)
        if shift["produced_good"] and shift["produced"]>0:self.goods_market.add_supply(person.location_id,employer.id,shift["produced_good"],shift["produced"])
        self.store.event(self.current_day,self.next_sequence(),"work",actor=person,employer_id=employer.id,profession=person.profession,gross_income=gross,tax=tax,net_income=gross-tax,produced_good=shift["produced_good"],produced=round(shift["produced"],3),energy_cost=energy)

    def scavenge(self,person):
        location=self.location_of(person);cost=self.rng.randint(4,9);person.energy=max(0,person.energy-cost);food=self.rng.randint(0,location.scavenge_food_max);medicine=int(self.rng.random()<location.medicine_chance);person.food+=food;person.medicine+=medicine;self.store.event(self.current_day,self.next_sequence(),"scavenge",actor=person,location_id=location.id,food_found=food,medicine_found=medicine,energy_cost=cost)

    def buy_supplies(self,person):
        options=[]
        if person.food<=6:options.append(("food",3))
        if person.medicine<=1:options.append(("medicine",1))
        if not options:return
        good,requested=self.rng.choice(options);result=self.goods_market.buy(person.location_id,good,requested,person.money);quantity=result["quantity"];cost=result["cost"];person.money-=cost;person.market_spending+=cost
        if good=="food":person.food+=int(quantity)
        else:person.medicine+=int(quantity)
        if result["shortage"]:person.shortage_experiences+=1;person.shift_ideology(-0.00025)
        for employer_id,revenue in result["seller_revenue"].items():self.labor_market.credit_sale(employer_id,revenue)
        self.store.event(self.current_day,self.next_sequence(),"buy_supplies",actor=person,location_id=person.location_id,resource=good,requested=requested,amount=quantity,cost=cost,unit_price=result["unit_price"],shortage=int(result["shortage"]))

    def rest(self,person):
        energy_gain=self.rng.randint(12,24);health_gain=self.rng.randint(0,2);person.energy=min(100,person.energy+energy_gain);person.health=min(100,person.health+health_gain);self.store.event(self.current_day,self.next_sequence(),"rest",actor=person,location_id=person.location_id,energy_gain=energy_gain,health_gain=health_gain)
    def heal(self,person):
        if person.medicine<=0 or person.health>=100:return
        gain=self.rng.randint(8,18);person.medicine-=1;person.health=min(100,person.health+gain);self.store.event(self.current_day,self.next_sequence(),"heal",actor=person,location_id=person.location_id,health_gain=gain)
    def repair(self,person):
        if person.money<3 or person.shelter>=100:return
        person.money-=3;gain=self.rng.randint(8,16);person.shelter=min(100,person.shelter+gain);self.store.event(self.current_day,self.next_sequence(),"repair",actor=person,location_id=person.location_id,shelter_gain=gain)

    def help(self,person):
        target=self.weighted_target(person,"help")
        if not target:return
        resource,amount=None,0
        if target.health<70 and person.medicine>0:person.medicine-=1;target.medicine+=1;resource,amount="medicine",1
        elif person.food>2:amount=self.rng.randint(1,min(2,person.food-1));person.food-=amount;target.food+=amount;resource="food"
        self.store.event(self.current_day,self.next_sequence(),"help",actor=person,target=target,success=int(bool(amount)),location_id=person.location_id,resource=resource,amount=amount)
        if amount:self.total_helps+=1;self.remember_interaction(person,target,"help",amount)

    def police_response(self,crime_type,offender,victim,magnitude):
        result=self.police.respond(self.current_day,offender.location_id,crime_type,offender,victim,magnitude)
        if result["responded"]:self.store.event(self.current_day,self.next_sequence(),"police_response",actor=offender,target=victim,location_id=offender.location_id,crime_type=crime_type,**result)
        if result["arrested"]:self.total_arrests+=1
        return result

    def steal(self,person):
        target=self.weighted_target(person,"steal")
        if not target:return
        amount,resource=0,None
        if self.rng.random()<0.45:
            choices=(["food"]*4 if target.food else [])+(["money"]*2 if target.money else [])+(["medicine"] if target.medicine else [])
            if choices:
                resource=self.rng.choice(choices)
                if resource=="food":amount=min(target.food,self.rng.randint(1,3));target.food-=amount;person.food+=amount
                elif resource=="money":amount=min(target.money,self.rng.randint(1,5));target.money-=amount;person.money+=amount
                else:amount=1;target.medicine-=1;person.medicine+=1
                self.total_thefts+=1;target.crime_suffered+=1;self.daily_crimes[person.location_id]+=1
        self.store.event(self.current_day,self.next_sequence(),"steal",actor=person,target=target,success=int(bool(amount)),location_id=person.location_id,resource=resource,amount=amount)
        if amount:self.remember_interaction(person,target,"steal",max(1,amount/2));self.police_response("steal",person,target,max(1,amount/2))

    def attack(self,person):
        target=self.weighted_target(person,"attack")
        if not target:return
        damage=self.rng.randint(5,20);person.energy=max(0,person.energy-self.rng.randint(4,9));target.health-=damage;self.total_attacks+=1;target.crime_suffered+=1;self.daily_crimes[person.location_id]+=1
        self.store.event(self.current_day,self.next_sequence(),"attack",actor=person,target=target,success=1,location_id=person.location_id,damage=damage,killed=target.health<=0);self.remember_interaction(person,target,"attack",damage/10);self.police_response("attack",person,target,damage/10)
        if target.health<=0:self.kill(target,"violence")

    def kill(self,person,cause):
        if not person.alive:return
        employer=self.labor_market.employer_any(person.employer_id)
        if employer:self.labor_market.terminate(person,"death");self.store.employment_event(self.current_day,person,employer,"ended","death")
        self.population_index.remove(person.id,person.location_id);self.goods_market.set_population(person.location_id,self.population_index.population(person.location_id));person.alive=False;self.alive_count-=1;self.total_deaths+=1;self.invalidate_living_cache();self.store.event(self.current_day,self.next_sequence(),"death",actor=person,location_id=person.location_id,cause=cause)

    def perform_action(self,person):
        if self.current_day<person.detained_until_day:return
        action=self.decision_engine.choose_action(person,self.rng,self.location_of(person));handler={"move":self.move,"work":self.work,"scavenge":self.scavenge,"buy_supplies":self.buy_supplies,"rest":self.rest,"heal":self.heal,"repair":self.repair,"help":self.help,"steal":self.steal,"attack":self.attack}.get(action)
        if handler:handler(person)

    def run_election(self):
        votes,ballots,winner=self.politics.hold_election(self.current_day,self.people,self.crime_rates(),self.rng);self.store.election(self.current_day,self.politics,votes,ballots,winner);self.store.event(self.current_day,self.next_sequence(),"election",left_votes=votes["left"],right_votes=votes["right"],winner=winner.id,representative=self.politics.representatives[winner.id])
    def welfare_cycle(self):
        for person,cash,food,medicine,cost in self.politics.distribute_welfare(self.people,self.rng):self.store.event(self.current_day,self.next_sequence(),"welfare",actor=person,government=self.politics.government.id,cash=cash,food=food,medicine=medicine,treasury_cost=cost)
    def business_cycle(self):
        for action,employer,detail in self.labor_market.business_review(self.people):
            self.store.sync_employer(employer)
            if action=="closed":
                for pid in detail:self.store.employment_event(self.current_day,self.people[pid],employer,"laid_off","employer_closed")
            self.store.event(self.current_day,self.next_sequence(),f"employer_{action}",employer_id=employer.id,employer=employer.name,location_id=employer.location_id,capacity=employer.capacity,cash=round(employer.cash,2),detail=detail)
    def mobility_cycle(self):
        for person in self.people:
            if not person.alive:continue
            old_class,old_profession=person.social_class,person.profession;new_class,score,direction=mobility_decision(person)
            if new_class==old_class:continue
            person.social_class=new_class;person.profession=choose_profession(new_class,self.rng);person.days_in_class=0;person.career_progress*=0.55 if direction=="up" else 0.70;self.total_mobility_changes+=1;self.store.mobility_event(self.current_day,person,old_class,new_class,old_profession,person.profession,score,direction);self.store.event(self.current_day,self.next_sequence(),"social_mobility",actor=person,old_class=old_class,new_class=new_class,old_profession=old_profession,new_profession=person.profession,score=round(score,3),direction=direction)

    def end_of_day(self):
        for location in self.locations:self.crime_history[location.id].append(self.daily_crimes.get(location.id,0))
        rates=self.crime_rates()
        for person in self.people:
            if not person.alive:continue
            location=self.location_of(person);person.days_in_class+=1
            if person.employer_id is None:person.unemployment_days+=1;person.lifetime_unemployment_days+=1
            else:person.unemployment_days=0
            if person.unemployment_days>30:person.shift_ideology(-0.0004)
            person.food-=1;person.energy=max(0,person.energy-3);person.shelter=max(0,person.shelter-self.rng.randint(0,2)-location.shelter_decay_bonus);person.decay_memories();self.politics.update_attitudes(person,rates.get(person.location_id,0.0))
            damage,causes=0,[]
            if person.food<0:person.food=0;damage+=self.rng.randint(4,10);causes.append("starvation")
            if person.energy==0:damage+=self.rng.randint(2,6);causes.append("exhaustion")
            if person.shelter<=20 and self.rng.random()<0.35:damage+=self.rng.randint(2,7);causes.append("exposure")
            if damage:person.health-=damage;self.store.event(self.current_day,self.next_sequence(),"daily_harm",actor=person,location_id=location.id,damage=damage,causes=causes)
            if person.health<=0:self.kill(person,"+".join(causes) if causes else "injury")
        self.daily_crimes.clear()

    def run_day(self,day):
        self.current_day=day
        if day==1 or (day-1)%ELECTION_INTERVAL_DAYS==0:self.run_election()
        for pid in permutation_ids(len(self.people),self.rng):
            person=self.people[pid]
            if not person.alive or day<person.detained_until_day:continue
            for _ in range(self.actions_per_day):
                if person.alive and day>=person.detained_until_day:self.perform_action(person)
        for shipment in self.transport.rebalance(day):self.store.shipment(shipment)
        if day%BUSINESS_INTERVAL_DAYS==0:self.welfare_cycle();self.business_cycle();self.police.rebalance()
        self.end_of_day();self.goods_market.reprice()
        if day%MOBILITY_INTERVAL_DAYS==0:self.mobility_cycle()
        police_snapshot=self.police.end_day();self.store.write_daily_stats(day,self);self.store.write_location_stats(day,self);self.store.write_political_stats(day,self);self.store.write_social_stats(day,self);self.store.write_labor_stats(day,self);self.store.write_market_stats(day,self);self.store.write_police_stats(day,police_snapshot);self.store.commit_day()

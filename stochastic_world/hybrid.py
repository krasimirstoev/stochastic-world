from array import array
from collections import Counter, defaultdict
from math import ceil

from .professions import profession_for

PRIORITY_MANDATORY = 0
PRIORITY_HIGH = 1
PRIORITY_NORMAL = 2

class HybridEngine:
    def __init__(self, world, sample_per_district=256, interest_days=30,
                 target_explicit_fraction=0.03, max_explicit_fraction=0.05):
        self.world=world
        self.sample_per_district=max(16,int(sample_per_district))
        self.interest_days=max(1,int(interest_days))
        self.target_explicit_fraction=max(0.001,min(1.0,float(target_explicit_fraction)))
        self.max_explicit_fraction=max(self.target_explicit_fraction,min(1.0,float(max_explicit_fraction)))
        self.interests={}
        self.last_touched_day=array("I",[0])*len(world.people)
        self.last_active_ids=[]
        self.location_samples={}
        self.last_stats=self._empty_stats()

    def _empty_stats(self):
        return {"explicit_agents":0,"sampled_agents":0,"interesting_agents":0,"aggregated_agents":0,
                "aggregate_work_shifts":0,"aggregate_food_demand":0.0,"aggregate_medicine_demand":0.0,
                "mandatory_agents":0,"high_priority_agents":0,"normal_priority_agents":0,
                "pending_interesting":0,"budget_target":0,"budget_ceiling":0,"reason_counts":{}}

    def mark_interesting(self, person_or_id, day=None, days=None, *, reason="social", priority=PRIORITY_NORMAL):
        pid=person_or_id if isinstance(person_or_id,int) else person_or_id.id
        if pid<0 or pid>=len(self.world.people): return
        day=self.world.current_day if day is None else int(day)
        until=day+(self.interest_days if days is None else max(1,int(days)))
        priority=max(PRIORITY_MANDATORY,min(PRIORITY_NORMAL,int(priority)))
        current=self.interests.get(pid)
        if current is None:
            self.interests[pid]={"until":until,"priority":priority,"reason":reason}; return
        if priority<current["priority"]:
            current["priority"]=priority; current["reason"]=reason
        elif priority==current["priority"] and reason!=current["reason"]:
            current["reason"]=reason
        current["until"]=max(current["until"],until)

    def _prune_interesting(self,day):
        expired=[pid for pid,meta in self.interests.items()
                 if meta["until"]<day or not self.world.people[pid].alive]
        for pid in expired:self.interests.pop(pid,None)

    def _budgets(self):
        alive=max(0,self.world.alive_count)
        target=max(1,ceil(alive*self.target_explicit_fraction)) if alive else 0
        ceiling=max(target,ceil(alive*self.max_explicit_fraction)) if alive else 0
        return target,ceiling

    def _ranked_interest_ids(self,day,priority):
        rows=[]
        for pid,meta in self.interests.items():
            if meta["priority"]!=priority or meta["until"]<day: continue
            if not self.world.people[pid].alive: continue
            rows.append((self.world.rng.random(),-meta["until"],pid))
        rows.sort()
        return [pid for _,_,pid in rows]

    def _sample_to_budget(self,selected,budget):
        if budget<=len(selected): return set()
        remaining=budget-len(selected); alive=max(1,self.world.alive_count); sampled=set()
        locations=list(self.world.locations); self.world.rng.shuffle(locations)
        for location in locations:
            if remaining<=0: break
            pop=self.world.population_index.population(location.id)
            quota=min(self.sample_per_district,max(1,ceil(budget*pop/alive)),remaining)
            people=self.world.population_index.sample_people(location.id,self.world.rng,quota,
                                                               exclude=tuple(selected|sampled))
            sampled.update(p.id for p in people if p.alive)
            remaining=budget-len(selected)-len(sampled)
        return sampled

    def select_active(self,day):
        self._prune_interesting(day)
        target,ceiling=self._budgets()
        mandatory=self._ranked_interest_ids(day,PRIORITY_MANDATORY)
        selected=set(mandatory)
        high=[]
        for pid in self._ranked_interest_ids(day,PRIORITY_HIGH):
            if len(selected)>=ceiling: break
            selected.add(pid); high.append(pid)
        normal=[]
        for pid in self._ranked_interest_ids(day,PRIORITY_NORMAL):
            if len(selected)>=target: break
            selected.add(pid); normal.append(pid)
        sampled=self._sample_to_budget(selected,target); selected|=sampled
        ids=list(selected); self.world.rng.shuffle(ids); self.last_active_ids=ids
        location_samples=defaultdict(list)
        for pid in ids:
            p=self.world.people[pid]
            if p.alive: location_samples[p.location_id].append(pid)
        self.location_samples=dict(location_samples)
        reasons=Counter()
        for pid in selected:
            meta=self.interests.get(pid)
            if meta: reasons[meta["reason"]]+=1
        pending=sum(1 for pid,meta in self.interests.items()
                    if meta["until"]>=day and self.world.people[pid].alive and pid not in selected)
        self.last_stats=self._empty_stats()
        self.last_stats.update(explicit_agents=len(selected),sampled_agents=len(sampled),
            interesting_agents=sum(pid in self.interests for pid in selected),
            aggregated_agents=max(0,self.world.alive_count-len(selected)),
            mandatory_agents=len(mandatory),high_priority_agents=len(high),normal_priority_agents=len(normal),
            pending_interesting=pending,budget_target=target,budget_ceiling=ceiling,reason_counts=dict(reasons))
        return ids

    def catch_up(self,person,day):
        last=self.last_touched_day[person.id]
        skipped=max(0,day-int(last)-1) if last else max(0,day-1)
        if skipped<=0:
            self.last_touched_day[person.id]=day; return
        person.days_in_class+=skipped
        participation=min(0.92,0.32+0.12*max(0,self.world.actions_per_day-1))
        employer=self.world.labor_market.employer_any(person.employer_id)
        if employer is None or not employer.alive or employer.location_id!=person.location_id:
            if person.employer_id is not None: person.employer_id=None
            person.unemployment_days+=skipped; person.lifetime_unemployment_days+=skipped
            if person.unemployment_days>30: person.shift_ideology(-0.00015*min(skipped,60))
        else:
            person.unemployment_days=0
            shifts=skipped*participation; gross=shifts*max(1.0,employer.base_wage)
            tax_rate=getattr(self.world.politics.government,"tax_rate",0.0)
            net=gross*(1.0-tax_rate)
            person.money+=net; person.lifetime_gross_income+=gross
            person.work_experience+=int(shifts); person.career_progress+=shifts*profession_for(person).advancement_rate
        food_cost=skipped*0.92*self.world.goods_market.quote(person.location_id,"food")
        spent=min(max(0.0,person.money),food_cost); person.money-=spent; person.market_spending+=spent
        buffer_loss=skipped//14
        if buffer_loss: person.food=max(0,person.food-buffer_loss)
        person.energy=max(35,min(100,person.energy-skipped//10))
        location=self.world.location_of(person)
        shelter_loss=int(skipped*(0.06+0.04*location.shelter_decay_bonus))
        if shelter_loss: person.shelter=max(0,person.shelter-shelter_loss)
        if person.food==0 and skipped>=14: person.health-=min(30,skipped//7)
        if person.shelter<=15 and skipped>=30: person.health-=min(20,skipped//15)
        self.last_touched_day[person.id]=day

    def touch_after_day(self,person,day):
        self.last_touched_day[person.id]=day
        if person.detained_until_day>day:
            self.mark_interesting(person,day,days=max(1,person.detained_until_day-day),
                                  reason="detained",priority=PRIORITY_MANDATORY); return
        if person.health<35 or person.food<=1 or person.shelter<15:
            self.mark_interesting(person,day,days=3,reason="critical",priority=PRIORITY_MANDATORY); return
        conflict=person.aggregate_memory()["max_conflict"]
        if conflict>=45:
            self.mark_interesting(person,day,days=7,reason="conflict",priority=PRIORITY_HIGH)
        elif person.unemployment_days>90:
            self.mark_interesting(person,day,days=3,reason="unemployment",priority=PRIORITY_NORMAL)

    def aggregate_background(self,day,active_ids):
        world=self.world
        selected_by_employer=Counter(); selected_by_location=Counter()
        for pid in active_ids:
            p=world.people[pid]
            if not p.alive: continue
            selected_by_location[p.location_id]+=1
            if p.employer_id is not None:selected_by_employer[p.employer_id]+=1
        aggregate_work_shifts=0
        for employer in world.labor_market.employers:
            if not employer.alive: continue
            routine=max(0,len(employer.employee_ids)-selected_by_employer.get(employer.id,0))
            if routine<=0: continue
            participation=min(0.92,0.32+0.12*max(0,world.actions_per_day-1))
            shifts=int(routine*participation)
            if shifts<=0: continue
            wage=max(1.0,employer.base_wage); shifts=min(shifts,int(employer.cash//wage))
            if shifts<=0: continue
            payroll=shifts*wage; employer.cash-=payroll; employer.payroll_since_review+=payroll
            aggregate_work_shifts+=shifts
            tax_rate=getattr(world.politics.government,"tax_rate",0.0); world.politics.treasury+=payroll*tax_rate
            if employer.output_good:
                produced=shifts*employer.output_per_shift*employer.productivity*0.98
                employer.units_produced_since_review+=produced
                world.goods_market.add_supply(employer.location_id,employer.id,employer.output_good,produced)
            else:
                revenue=payroll*employer.productivity*1.18; employer.cash+=revenue; employer.revenue_since_review+=revenue
        food_demand=0.0; medicine_demand=0.0
        for location in world.locations:
            routine=max(0,world.population_index.population(location.id)-selected_by_location.get(location.id,0))
            if routine<=0: continue
            requested_food=routine*0.92; result=world.goods_market.buy_bulk(location.id,"food",requested_food)
            food_demand+=requested_food
            for eid,revenue in result["seller_revenue"].items():world.labor_market.credit_sale(eid,revenue)
            requested_medicine=routine*0.0035; result=world.goods_market.buy_bulk(location.id,"medicine",requested_medicine)
            medicine_demand+=requested_medicine
            for eid,revenue in result["seller_revenue"].items():world.labor_market.credit_sale(eid,revenue)
        self.last_stats.update(aggregate_work_shifts=aggregate_work_shifts,
                               aggregate_food_demand=food_demand,aggregate_medicine_demand=medicine_demand)
        return dict(self.last_stats)

    def sampled_people(self):
        return [self.world.people[pid] for pid in self.last_active_ids if self.world.people[pid].alive]

import json

from .storage import EventStore


class HybridEventStore(EventStore):
    """Sampling-aware statistics writer for HybridWorld."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS hybrid_stats(
          simulation_id INTEGER NOT NULL,
          day INTEGER NOT NULL,
          explicit_agents INTEGER NOT NULL,
          sampled_agents INTEGER NOT NULL,
          interesting_agents INTEGER NOT NULL,
          aggregated_agents INTEGER NOT NULL,
          aggregate_work_shifts INTEGER NOT NULL,
          aggregate_food_demand REAL NOT NULL,
          aggregate_medicine_demand REAL NOT NULL,
          mandatory_agents INTEGER NOT NULL DEFAULT 0,
          high_priority_agents INTEGER NOT NULL DEFAULT 0,
          normal_priority_agents INTEGER NOT NULL DEFAULT 0,
          pending_interesting INTEGER NOT NULL DEFAULT 0,
          budget_target INTEGER NOT NULL DEFAULT 0,
          budget_ceiling INTEGER NOT NULL DEFAULT 0,
          reason_counts_json TEXT NOT NULL DEFAULT '{}',
          PRIMARY KEY(simulation_id,day)
        )
        """)
        for definition in (
            "mandatory_agents INTEGER NOT NULL DEFAULT 0",
            "high_priority_agents INTEGER NOT NULL DEFAULT 0",
            "normal_priority_agents INTEGER NOT NULL DEFAULT 0",
            "pending_interesting INTEGER NOT NULL DEFAULT 0",
            "budget_target INTEGER NOT NULL DEFAULT 0",
            "budget_ceiling INTEGER NOT NULL DEFAULT 0",
            "reason_counts_json TEXT NOT NULL DEFAULT '{}'",
        ):
            self._add("hybrid_stats", definition)
        self.conn.commit()

    def _sample(self, w):
        return w.hybrid.sampled_people()

    def write_daily_stats(self, day, w):
        p = self._sample(w)
        self.conn.execute("INSERT INTO daily_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self.simulation_id, day, w.alive_count, self._avg(p,"food"), self._avg(p,"money"),
             self._avg(p,"medicine"), self._avg(p,"energy"), self._avg(p,"shelter"), self._avg(p,"health"),
             w.total_helps, w.total_thefts, w.total_attacks, w.total_observations, w.total_deaths, w.total_mobility_changes))

    def write_location_stats(self, day, w):
        rows=[]
        for location in w.locations:
            ids=w.hybrid.location_samples.get(location.id,())
            p=[w.people[pid] for pid in ids if w.people[pid].alive]
            rows.append((self.simulation_id,day,location.id,w.population_index.population(location.id),
                         self._avg(p,"food"),self._avg(p,"money"),self._avg(p,"health"),w.local_crime_rate(location.id)))
        self.conn.executemany("INSERT INTO location_stats VALUES(?,?,?,?,?,?,?,?)",rows)

    def write_political_stats(self, day, w):
        p=self._sample(w); n=max(1,len(p)); scale=w.alive_count/n if p else 0.0
        self.conn.execute("INSERT INTO political_stats VALUES(?,?,?,?,?,?,?,?,?)",
            (self.simulation_id,day,w.politics.government.id,w.politics.treasury,
             sum(x.ideology for x in p)/n,round(sum(x.ideology<0 for x in p)*scale),
             round(sum(x.ideology>=0 for x in p)*scale),sum(x.taxes_paid for x in p)/n,
             sum(x.welfare_received for x in p)/n))

    def write_social_stats(self, day, w):
        sample=self._sample(w); total=max(1,len(sample)); rows=[]
        for social_class in ("working","lower_middle","middle","upper_middle","affluent"):
            p=[x for x in sample if x.social_class==social_class]
            estimated=round(w.alive_count*len(p)/total) if sample else 0
            rows.append((self.simulation_id,day,social_class,estimated,self._avg(p,"money"),self._avg(p,"food"),
                         self._avg(p,"shelter"),self._avg(p,"health"),self._avg(p,"ideology"),self._avg(p,"work_experience")))
        self.conn.executemany("INSERT INTO social_stats VALUES(?,?,?,?,?,?,?,?,?,?)",rows)

    def write_labor_stats(self, day, w):
        active=[e for e in w.labor_market.employers if e.alive]
        employed=min(w.alive_count,sum(len(e.employee_ids) for e in active)); unemployed=max(0,w.alive_count-employed)
        self.conn.execute("INSERT INTO labor_stats VALUES(?,?,?,?,?,?,?,?)",
            (self.simulation_id,day,employed,unemployed,unemployed/w.alive_count if w.alive_count else 0.0,
             sum(e.vacancies for e in active),len(active),sum(e.capacity for e in active)))

    def write_hybrid_stats(self, day, stats):
        self.conn.execute("""INSERT INTO hybrid_stats(
            simulation_id,day,explicit_agents,sampled_agents,interesting_agents,aggregated_agents,
            aggregate_work_shifts,aggregate_food_demand,aggregate_medicine_demand,
            mandatory_agents,high_priority_agents,normal_priority_agents,pending_interesting,
            budget_target,budget_ceiling,reason_counts_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.simulation_id,day,stats["explicit_agents"],stats["sampled_agents"],stats["interesting_agents"],
             stats["aggregated_agents"],stats["aggregate_work_shifts"],stats["aggregate_food_demand"],
             stats["aggregate_medicine_demand"],stats["mandatory_agents"],stats["high_priority_agents"],
             stats["normal_priority_agents"],stats["pending_interesting"],stats["budget_target"],
             stats["budget_ceiling"],json.dumps(stats["reason_counts"],sort_keys=True)))

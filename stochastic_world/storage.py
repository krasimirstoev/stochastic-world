import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


COMPACT_EVENTS = {
    "election", "welfare", "steal", "attack", "police_response", "arrest",
    "death", "daily_harm", "social_mobility", "job_found", "job_search",
    "layoff", "employer_closed", "employer_created", "employer_expanded",
    "employer_contracted", "buy_supplies", "move",
}


class EventStore:
    def __init__(self, db_path, log_path, config, run_index=1, master_seed=None):
        self.db_path = Path(db_path)
        self.log_path = Path(log_path)
        self.event_mode = config.get("event_mode", "full")
        if self.event_mode == "auto":
            self.event_mode = "compact" if config["population"] >= 100_000 else "full"

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.create_schema()

        cur = self.conn.execute(
            """INSERT INTO simulations(
                started_at,master_seed,seed,run_index,population,actions_per_day,period,
                faker_locale,visibility,max_witnesses,locations_count,event_mode)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                str(master_seed if master_seed is not None else config["seed"]),
                str(config["seed"]),
                run_index,
                config["population"],
                config["actions_per_day"],
                config["period"],
                config["faker_locale"],
                config["visibility"],
                config["max_witnesses"],
                config["locations_count"],
                self.event_mode,
            ),
        )
        self.simulation_id = cur.lastrowid
        self.conn.commit()
        self.log_fh = self.log_path.open("a", encoding="utf-8")
        self.log_fh.write(
            f"\n=== Simulation {self.simulation_id} / run {run_index} ===\n"
            f"seed={config['seed']} population={config['population']} period={config['period']} "
            f"locations={config['locations_count']} event_mode={self.event_mode}\n"
        )

    def create_schema(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS simulations(
          id INTEGER PRIMARY KEY AUTOINCREMENT,started_at TEXT NOT NULL,ended_at TEXT,
          master_seed TEXT,seed TEXT NOT NULL,run_index INTEGER NOT NULL DEFAULT 1,
          population INTEGER NOT NULL,actions_per_day INTEGER NOT NULL,period INTEGER NOT NULL,
          faker_locale TEXT NOT NULL DEFAULT 'en_US',visibility REAL NOT NULL DEFAULT .65,
          max_witnesses INTEGER NOT NULL DEFAULT 3,locations_count INTEGER NOT NULL DEFAULT 5,
          event_mode TEXT NOT NULL DEFAULT 'full',collapse_day INTEGER);
        CREATE TABLE IF NOT EXISTS persons(
          simulation_id INTEGER NOT NULL,person_id INTEGER NOT NULL,name TEXT,
          social_class TEXT NOT NULL DEFAULT 'working',initial_profession TEXT NOT NULL DEFAULT 'laborer',
          initial_ideology REAL NOT NULL DEFAULT 0,initial_location_id INTEGER NOT NULL DEFAULT 0,
          initial_food INTEGER NOT NULL,initial_money REAL NOT NULL,initial_medicine INTEGER NOT NULL DEFAULT 2,
          initial_energy INTEGER NOT NULL DEFAULT 80,initial_shelter INTEGER NOT NULL DEFAULT 70,
          initial_health INTEGER NOT NULL,PRIMARY KEY(simulation_id,person_id));
        CREATE TABLE IF NOT EXISTS locations(
          simulation_id INTEGER NOT NULL,location_id INTEGER NOT NULL,name TEXT NOT NULL,kind TEXT NOT NULL,
          neighbors_json TEXT NOT NULL DEFAULT '[]',capacity_hint INTEGER NOT NULL DEFAULT 0,
          work_multiplier REAL,scavenge_food_max INTEGER,medicine_chance REAL,market INTEGER,
          shelter_decay_bonus INTEGER,PRIMARY KEY(simulation_id,location_id));
        CREATE TABLE IF NOT EXISTS events(
          id INTEGER PRIMARY KEY AUTOINCREMENT,simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,
          sequence INTEGER NOT NULL,actor_id INTEGER,target_id INTEGER,event_type TEXT NOT NULL,
          success INTEGER,data TEXT);
        CREATE TABLE IF NOT EXISTS relationships(
          simulation_id INTEGER NOT NULL,observer_id INTEGER NOT NULL,other_id INTEGER NOT NULL,
          last_day INTEGER,trust REAL NOT NULL DEFAULT 0,grievance REAL NOT NULL DEFAULT 0,
          affinity REAL NOT NULL DEFAULT 0,conflict REAL NOT NULL DEFAULT 0,familiarity INTEGER NOT NULL DEFAULT 0,
          help_given INTEGER NOT NULL DEFAULT 0,help_received INTEGER NOT NULL DEFAULT 0,
          thefts_committed INTEGER NOT NULL DEFAULT 0,thefts_suffered INTEGER NOT NULL DEFAULT 0,
          attacks_committed INTEGER NOT NULL DEFAULT 0,attacks_suffered INTEGER NOT NULL DEFAULT 0,
          observed_help INTEGER NOT NULL DEFAULT 0,observed_theft INTEGER NOT NULL DEFAULT 0,
          observed_attack INTEGER NOT NULL DEFAULT 0,recent_json TEXT,
          PRIMARY KEY(simulation_id,observer_id,other_id));
        CREATE TABLE IF NOT EXISTS relationship_history(
          id INTEGER PRIMARY KEY AUTOINCREMENT,simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,
          sequence INTEGER NOT NULL,action TEXT NOT NULL,actor_id INTEGER NOT NULL,target_id INTEGER NOT NULL,
          actor_before TEXT NOT NULL,actor_after TEXT NOT NULL,target_before TEXT NOT NULL,target_after TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations(
          id INTEGER PRIMARY KEY AUTOINCREMENT,simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,
          sequence INTEGER NOT NULL,action TEXT NOT NULL,witness_id INTEGER NOT NULL,
          actor_id INTEGER NOT NULL,target_id INTEGER NOT NULL,location_id INTEGER,
          before_json TEXT NOT NULL,after_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS parties(
          simulation_id INTEGER NOT NULL,party_id TEXT NOT NULL,name TEXT NOT NULL,representative TEXT NOT NULL,
          ideology REAL NOT NULL,tax_rate REAL NOT NULL,welfare_cash INTEGER NOT NULL,
          welfare_food INTEGER NOT NULL,welfare_money_threshold INTEGER NOT NULL,
          PRIMARY KEY(simulation_id,party_id));
        CREATE TABLE IF NOT EXISTS elections(
          id INTEGER PRIMARY KEY AUTOINCREMENT,simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,
          election_number INTEGER NOT NULL,left_votes INTEGER NOT NULL,right_votes INTEGER NOT NULL,
          winner TEXT NOT NULL,representative TEXT NOT NULL,treasury REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS votes(
          simulation_id INTEGER NOT NULL,election_number INTEGER NOT NULL,day INTEGER NOT NULL,
          person_id INTEGER NOT NULL,party_id TEXT NOT NULL,ideology REAL NOT NULL,
          social_class TEXT NOT NULL,location_id INTEGER NOT NULL,
          PRIMARY KEY(simulation_id,election_number,person_id));
        CREATE TABLE IF NOT EXISTS mobility_history(
          id INTEGER PRIMARY KEY AUTOINCREMENT,simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,
          person_id INTEGER NOT NULL,old_class TEXT NOT NULL,new_class TEXT NOT NULL,
          old_profession TEXT NOT NULL,new_profession TEXT NOT NULL,score REAL NOT NULL,direction TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS employers(
          simulation_id INTEGER NOT NULL,employer_id INTEGER NOT NULL,name TEXT NOT NULL,
          location_id INTEGER NOT NULL,kind TEXT NOT NULL,capacity INTEGER NOT NULL,
          base_wage REAL NOT NULL,cash REAL NOT NULL,productivity REAL NOT NULL,
          output_good TEXT,output_per_shift REAL NOT NULL DEFAULT 0,alive INTEGER NOT NULL,
          PRIMARY KEY(simulation_id,employer_id));
        CREATE TABLE IF NOT EXISTS employment_history(
          id INTEGER PRIMARY KEY AUTOINCREMENT,simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,
          person_id INTEGER NOT NULL,employer_id INTEGER,action TEXT NOT NULL,reason TEXT,
          profession TEXT,social_class TEXT,location_id INTEGER,wage REAL);
        CREATE TABLE IF NOT EXISTS daily_stats(
          simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,population_alive INTEGER NOT NULL,
          avg_food REAL,avg_money REAL,avg_medicine REAL,avg_energy REAL,avg_shelter REAL,avg_health REAL,
          helps INTEGER NOT NULL DEFAULT 0,thefts INTEGER NOT NULL,attacks INTEGER NOT NULL,
          observations INTEGER NOT NULL DEFAULT 0,deaths INTEGER NOT NULL,mobility_changes INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(simulation_id,day));
        CREATE TABLE IF NOT EXISTS location_stats(
          simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,location_id INTEGER NOT NULL,
          population_alive INTEGER NOT NULL,avg_food REAL,avg_money REAL,avg_health REAL,
          crime_rate REAL NOT NULL DEFAULT 0,PRIMARY KEY(simulation_id,day,location_id));
        CREATE TABLE IF NOT EXISTS political_stats(
          simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,government TEXT NOT NULL,
          treasury REAL NOT NULL,avg_ideology REAL NOT NULL,left_leaning INTEGER NOT NULL,
          right_leaning INTEGER NOT NULL,avg_taxes_paid REAL NOT NULL,avg_welfare_received REAL NOT NULL,
          PRIMARY KEY(simulation_id,day));
        CREATE TABLE IF NOT EXISTS social_stats(
          simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,social_class TEXT NOT NULL,
          population_alive INTEGER NOT NULL,avg_money REAL NOT NULL,avg_food REAL NOT NULL,
          avg_shelter REAL NOT NULL,avg_health REAL NOT NULL,avg_ideology REAL NOT NULL,
          avg_work_experience REAL NOT NULL,PRIMARY KEY(simulation_id,day,social_class));
        CREATE TABLE IF NOT EXISTS labor_stats(
          simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,employed INTEGER NOT NULL,
          unemployed INTEGER NOT NULL,unemployment_rate REAL NOT NULL,vacancies INTEGER NOT NULL,
          active_employers INTEGER NOT NULL,total_capacity INTEGER NOT NULL,
          PRIMARY KEY(simulation_id,day));
        CREATE TABLE IF NOT EXISTS market_stats(
          simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,location_id INTEGER NOT NULL,
          good TEXT NOT NULL,price REAL NOT NULL,stock REAL NOT NULL,inflation_index REAL NOT NULL,
          PRIMARY KEY(simulation_id,day,location_id,good));
        CREATE TABLE IF NOT EXISTS police_stats(
          simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,location_id INTEGER NOT NULL,
          officers INTEGER NOT NULL,incidents INTEGER NOT NULL,responses INTEGER NOT NULL,
          arrests INTEGER NOT NULL,coverage REAL NOT NULL,
          PRIMARY KEY(simulation_id,day,location_id));
        CREATE TABLE IF NOT EXISTS shipments(
          id INTEGER PRIMARY KEY AUTOINCREMENT,simulation_id INTEGER NOT NULL,day INTEGER NOT NULL,
          source_location INTEGER NOT NULL,target_location INTEGER NOT NULL,good TEXT NOT NULL,
          quantity REAL NOT NULL,transport_cost REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(simulation_id,event_type,day);
        CREATE INDEX IF NOT EXISTS idx_employment_person ON employment_history(simulation_id,person_id,day);
        CREATE INDEX IF NOT EXISTS idx_mobility_person ON mobility_history(simulation_id,person_id,day);
        CREATE INDEX IF NOT EXISTS idx_shipments_day ON shipments(simulation_id,day);
        """)
        self._migrate()
        self.conn.commit()

    def _columns(self, table):
        return {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}

    def _add(self, table, definition):
        if definition.split()[0] not in self._columns(table):
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")

    def _migrate(self):
        for table, definition in (
            ("simulations", "event_mode TEXT NOT NULL DEFAULT 'full'"),
            ("persons", "initial_profession TEXT NOT NULL DEFAULT 'laborer'"),
            ("daily_stats", "mobility_changes INTEGER NOT NULL DEFAULT 0"),
            ("locations", "neighbors_json TEXT NOT NULL DEFAULT '[]'"),
            ("locations", "capacity_hint INTEGER NOT NULL DEFAULT 0"),
            ("employers", "output_good TEXT"),
            ("employers", "output_per_shift REAL NOT NULL DEFAULT 0"),
        ):
            self._add(table, definition)

    def register_person(self, p):
        self.conn.execute("""INSERT INTO persons(simulation_id,person_id,name,social_class,initial_profession,initial_ideology,initial_location_id,initial_food,initial_money,initial_medicine,initial_energy,initial_shelter,initial_health) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",(self.simulation_id,p.id,p.name,p.social_class,p.profession,p.ideology,p.location_id,p.food,p.money,p.medicine,p.energy,p.shelter,p.health))

    def register_location(self, l):
        self.conn.execute("""INSERT INTO locations(simulation_id,location_id,name,kind,neighbors_json,capacity_hint,work_multiplier,scavenge_food_max,medicine_chance,market,shelter_decay_bonus) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",(self.simulation_id,l.id,l.name,l.kind,json.dumps(l.neighbors),l.capacity_hint,l.work_multiplier,l.scavenge_food_max,l.medicine_chance,int(l.market),l.shelter_decay_bonus))

    def register_parties(self, politics):
        from .politics import PARTIES
        for p in PARTIES:
            self.conn.execute("""INSERT INTO parties(simulation_id,party_id,name,representative,ideology,tax_rate,welfare_cash,welfare_food,welfare_money_threshold) VALUES(?,?,?,?,?,?,?,?,?)""",(self.simulation_id,p.id,p.name,politics.representatives[p.id],p.ideology,p.tax_rate,p.welfare_cash,p.welfare_food,p.welfare_money_threshold))

    def register_employer(self, e):
        self.conn.execute("""INSERT OR REPLACE INTO employers(simulation_id,employer_id,name,location_id,kind,capacity,base_wage,cash,productivity,output_good,output_per_shift,alive) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(self.simulation_id,e.id,e.name,e.location_id,e.kind,e.capacity,e.base_wage,e.cash,e.productivity,e.output_good,e.output_per_shift,int(e.alive)))

    sync_employer = register_employer

    def event(self, day, sequence, event_type, actor=None, target=None, success=None, **data):
        if self.event_mode == "compact" and event_type not in COMPACT_EVENTS:
            return
        self.conn.execute("""INSERT INTO events(simulation_id,day,sequence,actor_id,target_id,event_type,success,data) VALUES(?,?,?,?,?,?,?,?)""",(self.simulation_id,day,sequence,actor.id if actor else None,target.id if target else None,event_type,success,json.dumps(data,ensure_ascii=False,sort_keys=True) if data else None))
        self.log_fh.write(f"day={day:04d} seq={sequence:09d} type={event_type} actor={getattr(actor,'id',None)} target={getattr(target,'id',None)} data={data}\n")

    def employment_event(self, day, person, employer, action, reason=None, wage=None):
        self.conn.execute("""INSERT INTO employment_history(simulation_id,day,person_id,employer_id,action,reason,profession,social_class,location_id,wage) VALUES(?,?,?,?,?,?,?,?,?,?)""",(self.simulation_id,day,person.id,employer.id if employer else None,action,reason,person.profession,person.social_class,person.location_id,wage))

    def mobility_event(self, day, person, old_class, new_class, old_profession, new_profession, score, direction):
        self.conn.execute("""INSERT INTO mobility_history(simulation_id,day,person_id,old_class,new_class,old_profession,new_profession,score,direction) VALUES(?,?,?,?,?,?,?,?,?)""",(self.simulation_id,day,person.id,old_class,new_class,old_profession,new_profession,score,direction))

    def relationship_event(self, day, sequence, action, actor, target, ab, aa, tb, ta):
        self.conn.execute("""INSERT INTO relationship_history(simulation_id,day,sequence,action,actor_id,target_id,actor_before,actor_after,target_before,target_after) VALUES(?,?,?,?,?,?,?,?,?,?)""",(self.simulation_id,day,sequence,action,actor.id,target.id,json.dumps(ab),json.dumps(aa),json.dumps(tb),json.dumps(ta)))

    def observation_event(self, day, sequence, action, witness, actor, target, before, after, location_id):
        self.conn.execute("""INSERT INTO observations(simulation_id,day,sequence,action,witness_id,actor_id,target_id,location_id,before_json,after_json) VALUES(?,?,?,?,?,?,?,?,?,?)""",(self.simulation_id,day,sequence,action,witness.id,actor.id,target.id,location_id,json.dumps(before),json.dumps(after)))

    def upsert_relationship(self, day, observer, other):
        m=observer.memory_of(other)
        self.conn.execute("""INSERT INTO relationships VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(simulation_id,observer_id,other_id) DO UPDATE SET last_day=excluded.last_day,trust=excluded.trust,grievance=excluded.grievance,affinity=excluded.affinity,conflict=excluded.conflict,familiarity=excluded.familiarity,help_given=excluded.help_given,help_received=excluded.help_received,thefts_committed=excluded.thefts_committed,thefts_suffered=excluded.thefts_suffered,attacks_committed=excluded.attacks_committed,attacks_suffered=excluded.attacks_suffered,observed_help=excluded.observed_help,observed_theft=excluded.observed_theft,observed_attack=excluded.observed_attack,recent_json=excluded.recent_json""",(self.simulation_id,observer.id,other.id,day,m.trust,m.grievance,m.affinity,m.conflict_score,m.familiarity,m.help_given,m.help_received,m.thefts_committed,m.thefts_suffered,m.attacks_committed,m.attacks_suffered,m.observed_help,m.observed_theft,m.observed_attack,json.dumps(list(m.recent))))

    def election(self, day, politics, votes, ballots, winner):
        self.conn.execute("""INSERT INTO elections(simulation_id,day,election_number,left_votes,right_votes,winner,representative,treasury) VALUES(?,?,?,?,?,?,?,?)""",(self.simulation_id,day,politics.election_number,votes['left'],votes['right'],winner.id,politics.representatives[winner.id],politics.treasury))
        self.conn.executemany("""INSERT INTO votes(simulation_id,election_number,day,person_id,party_id,ideology,social_class,location_id) VALUES(?,?,?,?,?,?,?,?)""",((self.simulation_id,politics.election_number,day,person.id,party.id,person.ideology,person.social_class,person.location_id) for person,party in ballots))

    def shipment(self, shipment):
        self.conn.execute("""INSERT INTO shipments(simulation_id,day,source_location,target_location,good,quantity,transport_cost) VALUES(?,?,?,?,?,?,?)""",(self.simulation_id,shipment.day,shipment.source_location,shipment.target_location,shipment.good,shipment.quantity,shipment.transport_cost))

    def _avg(self, people, attr):
        return sum(getattr(p,attr) for p in people)/len(people) if people else 0.0

    def write_daily_stats(self, day, w):
        p=w.living_people();self.conn.execute("INSERT INTO daily_stats VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(self.simulation_id,day,len(p),self._avg(p,'food'),self._avg(p,'money'),self._avg(p,'medicine'),self._avg(p,'energy'),self._avg(p,'shelter'),self._avg(p,'health'),w.total_helps,w.total_thefts,w.total_attacks,w.total_observations,w.total_deaths,w.total_mobility_changes))

    def write_location_stats(self, day, w):
        rows=[]
        for l in w.locations:
            p=w.people_in_location(l.id);rows.append((self.simulation_id,day,l.id,len(p),self._avg(p,'food'),self._avg(p,'money'),self._avg(p,'health'),w.local_crime_rate(l.id)))
        self.conn.executemany("INSERT INTO location_stats VALUES(?,?,?,?,?,?,?,?)",rows)

    def write_political_stats(self, day, w):
        p=w.living_people();n=max(1,len(p));self.conn.execute("INSERT INTO political_stats VALUES(?,?,?,?,?,?,?,?,?)",(self.simulation_id,day,w.politics.government.id,w.politics.treasury,sum(x.ideology for x in p)/n,sum(x.ideology<0 for x in p),sum(x.ideology>=0 for x in p),sum(x.taxes_paid for x in p)/n,sum(x.welfare_received for x in p)/n))

    def write_social_stats(self, day, w):
        rows=[]
        for c in ('working','lower_middle','middle','upper_middle','affluent'):
            p=[x for x in w.living_people() if x.social_class==c];rows.append((self.simulation_id,day,c,len(p),self._avg(p,'money'),self._avg(p,'food'),self._avg(p,'shelter'),self._avg(p,'health'),self._avg(p,'ideology'),self._avg(p,'work_experience')))
        self.conn.executemany("INSERT INTO social_stats VALUES(?,?,?,?,?,?,?,?,?,?)",rows)

    def write_labor_stats(self, day, w):
        p=w.living_people();employed=sum(x.employer_id is not None for x in p);unemployed=len(p)-employed;active=[e for e in w.labor_market.employers if e.alive];self.conn.execute("INSERT INTO labor_stats VALUES(?,?,?,?,?,?,?,?)",(self.simulation_id,day,employed,unemployed,unemployed/len(p) if p else 0.0,sum(e.vacancies for e in active),len(active),sum(e.capacity for e in active)))

    def write_market_stats(self, day, w):
        inflation=w.goods_market.inflation_index();rows=[]
        for location_id,state in w.goods_market.states.items():
            for good in ('food','medicine'):rows.append((self.simulation_id,day,location_id,good,state.prices[good],state.stock(good),inflation))
        self.conn.executemany("INSERT INTO market_stats VALUES(?,?,?,?,?,?,?)",rows)

    def write_police_stats(self, day, snapshots):
        self.conn.executemany("INSERT INTO police_stats VALUES(?,?,?,?,?,?,?,?)",((self.simulation_id,day,location_id,s['officers'],s['incidents'],s['responses'],s['arrests'],s['coverage']) for location_id,s in snapshots.items()))

    def commit_day(self):
        self.conn.commit();self.log_fh.flush()

    def finish(self, collapse_day=None):
        self.conn.execute("UPDATE simulations SET ended_at=?,collapse_day=? WHERE id=?",(datetime.now(timezone.utc).isoformat(),collapse_day,self.simulation_id));self.conn.commit();self.log_fh.close();self.conn.close()

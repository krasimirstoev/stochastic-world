from time import perf_counter

from .fast_storage import BufferedEventMixin
from .hybrid_storage import HybridEventStore
from .storage import EventStore


_PERSON_INSERT = """INSERT INTO persons(
    simulation_id,person_id,name,social_class,initial_profession,initial_ideology,
    initial_location_id,initial_food,initial_money,initial_medicine,initial_energy,
    initial_shelter,initial_health
) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def _write_labor_row(store, day, w, employed, workforce_count):
    unemployed = max(0, workforce_count - employed)
    active = [e for e in w.labor_market.employers if e.alive]
    store.conn.execute("INSERT INTO labor_stats VALUES(?,?,?,?,?,?,?,?)",
        (store.simulation_id, day, employed, unemployed,
         unemployed / workforce_count if workforce_count else 0.0,
         sum(e.vacancies for e in active), len(active), sum(e.capacity for e in active)))


class LifeEventStore(BufferedEventMixin, EventStore):
    """Agent store with a startup fast path for very large compact simulations."""

    def __init__(self, *args, **kwargs):
        config = args[2] if len(args) > 2 else kwargs.get("config", {})
        super().__init__(*args, **kwargs)
        self._large_startup = (
            int(config.get("population", 0)) >= 100_000
            and self.event_mode == "compact"
        )
        self._person_batch = []
        self._person_batch_size = 20_000
        self.startup_profile = {
            "sqlite_persons": 0.0,
            "person_rows": 0,
            "initial_employment_rows_skipped": 0,
        }

    def _flush_person_batch(self):
        if not self._person_batch:
            return
        started = perf_counter()
        self.conn.executemany(_PERSON_INSERT, self._person_batch)
        self.startup_profile["sqlite_persons"] += perf_counter() - started
        self.startup_profile["person_rows"] += len(self._person_batch)
        self._person_batch.clear()

    def register_person(self, p):
        if not self._large_startup:
            return super().register_person(p)
        self._person_batch.append((
            self.simulation_id, p.id, p.name, p.social_class, p.profession,
            p.ideology, p.location_id, p.food, p.money, p.medicine,
            p.energy, p.shelter, p.health,
        ))
        if len(self._person_batch) >= self._person_batch_size:
            self._flush_person_batch()

    def register_parties(self, politics):
        self._flush_person_batch()
        return super().register_parties(politics)

    def employment_event(self, day, person, employer, action, reason=None, wage=None):
        if (
            self._large_startup
            and int(day) == 0
            and action == "hired"
            and reason == "initial_assignment"
        ):
            self.startup_profile["initial_employment_rows_skipped"] += 1
            return
        return super().employment_event(day, person, employer, action, reason, wage)

    def write_labor_stats(self, day, w):
        workforce = [p for p in w.people if p.alive and p.is_working_age]
        employed = sum(p.employer_id is not None for p in workforce)
        _write_labor_row(self, day, w, employed, len(workforce))


class LifeHybridEventStore(BufferedEventMixin, HybridEventStore):
    def write_labor_stats(self, day, w):
        workforce_count = w.demographics.working_age_count
        active = [e for e in w.labor_market.employers if e.alive]
        employed = min(workforce_count, sum(len(e.employee_ids) for e in active))
        _write_labor_row(self, day, w, employed, workforce_count)

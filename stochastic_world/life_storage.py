from .fast_storage import BufferedEventMixin
from .hybrid_storage import HybridEventStore
from .storage import EventStore


def _write_labor_row(store, day, w, employed, workforce_count):
    unemployed = max(0, workforce_count - employed)
    active = [e for e in w.labor_market.employers if e.alive]
    store.conn.execute("INSERT INTO labor_stats VALUES(?,?,?,?,?,?,?,?)",
        (store.simulation_id, day, employed, unemployed,
         unemployed / workforce_count if workforce_count else 0.0,
         sum(e.vacancies for e in active), len(active), sum(e.capacity for e in active)))


class LifeEventStore(BufferedEventMixin, EventStore):
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

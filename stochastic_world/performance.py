from collections import defaultdict
from contextlib import contextmanager
from statistics import mean
from time import perf_counter


class PhaseProfiler:
    """Low-overhead in-memory wall-clock profiler for hybrid simulation phases."""

    def __init__(self, world, enabled=False):
        self.world = world
        self.enabled = bool(enabled)
        self.rows = []
        self.samples = defaultdict(list)

    @contextmanager
    def phase(self, day, name):
        if not self.enabled:
            yield
            return
        started = perf_counter()
        try:
            yield
        finally:
            self.record(day, name, perf_counter() - started)

    def start_day(self):
        return perf_counter() if self.enabled else None

    def finish_day(self, day, started):
        if self.enabled and started is not None:
            self.record(day, "day_total", perf_counter() - started)

    def record(self, day, name, duration_seconds):
        if not self.enabled:
            return
        duration = max(0.0, float(duration_seconds))
        hybrid_stats = getattr(self.world, "last_hybrid_stats", {}) or {}
        explicit = int(hybrid_stats.get("explicit_agents", 0))
        alive = int(getattr(self.world, "alive_count", 0))
        self.rows.append((int(day), str(name), duration, alive, explicit))
        self.samples[str(name)].append(duration)

    def summary(self):
        result = []
        for phase, values in self.samples.items():
            ordered = sorted(values)
            p95_index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95))))
            result.append({
                "phase": phase,
                "calls": len(values),
                "total": sum(values),
                "avg": mean(values),
                "max": max(values),
                "p95": ordered[p95_index],
            })
        return sorted(result, key=lambda row: row["total"], reverse=True)

    def flush(self, store):
        if not self.enabled or not self.rows:
            return
        store.conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_timings(
              simulation_id INTEGER NOT NULL,
              day INTEGER NOT NULL,
              phase TEXT NOT NULL,
              duration_seconds REAL NOT NULL,
              population_alive INTEGER NOT NULL,
              explicit_agents INTEGER NOT NULL DEFAULT 0
            )
        """)
        store.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_performance_phase ON performance_timings(simulation_id,phase,day)"
        )
        store.conn.executemany(
            """INSERT INTO performance_timings(
                simulation_id,day,phase,duration_seconds,population_alive,explicit_agents
               ) VALUES(?,?,?,?,?,?)""",
            ((store.simulation_id, day, phase, duration, alive, explicit)
             for day, phase, duration, alive, explicit in self.rows),
        )
        store.conn.commit()

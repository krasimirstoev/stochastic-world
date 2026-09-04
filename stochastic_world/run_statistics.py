import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


def _safe_mean(values):
    values = list(values)
    return mean(values) if values else 0.0


class RunStatistics:
    """Build and append a detailed, human-readable run report."""

    def __init__(self, path="statistics.log"):
        self.path = Path(path) if path else None

    def _event_counts(self, store):
        return {
            row[0]: row[1]
            for row in store.conn.execute(
                "SELECT event_type,COUNT(*) FROM events WHERE simulation_id=? GROUP BY event_type ORDER BY COUNT(*) DESC",
                (store.simulation_id,),
            )
        }

    def _life_event_counts(self, store):
        try:
            return {
                row[0]: row[1]
                for row in store.conn.execute(
                    "SELECT event_type,COUNT(*) FROM life_events WHERE simulation_id=? GROUP BY event_type ORDER BY COUNT(*) DESC",
                    (store.simulation_id,),
                )
            }
        except Exception:
            return {}

    def _death_causes(self, store):
        causes = Counter()
        rows = store.conn.execute(
            "SELECT data FROM events WHERE simulation_id=? AND event_type='death'",
            (store.simulation_id,),
        )
        for (raw,) in rows:
            try:
                data = json.loads(raw or "{}")
            except (TypeError, json.JSONDecodeError):
                data = {}
            causes[data.get("cause", "unknown")] += 1
        return causes

    def _location_rows(self, world):
        rows = []
        for location in world.locations:
            ids = world.population_index.ids(location.id)
            people = [world.people[pid] for pid in ids if world.people[pid].alive]
            rows.append({
                "id": location.id,
                "name": location.name,
                "kind": location.kind,
                "population": len(people),
                "avg_food": _safe_mean(p.food for p in people),
                "avg_money": _safe_mean(p.money for p in people),
                "avg_health": _safe_mean(p.health for p in people),
                "crime_rate": world.local_crime_rate(location.id),
                "food_stock": world.goods_market.total_stock(location.id, "food"),
                "medicine_stock": world.goods_market.total_stock(location.id, "medicine"),
                "food_price": world.goods_market.quote(location.id, "food"),
                "medicine_price": world.goods_market.quote(location.id, "medicine"),
            })
        return rows

    def build(self, world, store, *, master_seed, run_seed, run_index, last_day, collapse_day, config):
        living = [p for p in world.people if p.alive]
        ages = [p.age_years for p in living]
        workforce = [p for p in living if p.is_working_age]
        employed = [p for p in workforce if p.employer_id is not None]
        employers = [e for e in world.labor_market.employers if e.alive]
        event_counts = self._event_counts(store)
        life_counts = self._life_event_counts(store)
        death_causes = self._death_causes(store)
        locations = self._location_rows(world)
        hybrid = getattr(world, "last_hybrid_stats", {}) or {}
        pool = getattr(world, "district_pool", None)
        pool_summary = pool.summary() if pool is not None else {"enabled": False}
        profiler = getattr(world, "profiler", None)
        profile_rows = profiler.summary() if profiler is not None else []

        age_bands = Counter()
        for person in living:
            age = person.age_years
            if age < 15:
                age_bands["0-14"] += 1
            elif age < 25:
                age_bands["15-24"] += 1
            elif age < 45:
                age_bands["25-44"] += 1
            elif age < 65:
                age_bands["45-64"] += 1
            else:
                age_bands["65+"] += 1

        class_counts = Counter(p.social_class for p in living)
        profession_counts = Counter(p.profession for p in living)
        generation_counts = Counter(p.generation for p in living)

        lines = []
        add = lines.append
        add("=" * 92)
        add(f"STOCHASTIC WORLD RUN REPORT | {datetime.now(timezone.utc).isoformat()}")
        add("=" * 92)
        add("")
        add("[IDENTITY]")
        add(f"simulation_id={store.simulation_id} run_index={run_index}")
        add(f"master_seed={master_seed}")
        add(f"run_seed={run_seed}")
        add(f"last_day={last_day} collapse_day={collapse_day}")
        add("")
        add("[CONFIGURATION]")
        for key in sorted(config):
            add(f"{key}={config[key]}")
        add("")
        add("[POPULATION AND DEMOGRAPHY]")
        add(f"initial_population={config.get('population', 0):,}")
        add(f"population_alive={world.alive_count:,}")
        add(f"people_ever_created={len(world.people):,}")
        add(f"total_births={world.demographics.total_births:,}")
        add(f"total_deaths={world.total_deaths:,}")
        add(f"natural_deaths={world.demographics.total_natural_deaths:,}")
        add(f"non_natural_deaths={max(0, world.total_deaths - world.demographics.total_natural_deaths):,}")
        add(f"median_age={median(ages) if ages else 0.0:.2f}")
        add(f"mean_age={_safe_mean(ages):.2f}")
        add(f"households={len(world.demographics.households):,}")
        add(f"working_age={len(workforce):,}")
        add(f"employed={len(employed):,}")
        add(f"unemployed={max(0, len(workforce)-len(employed)):,}")
        add(f"unemployment_rate={(1-len(employed)/len(workforce)) if workforce else 0.0:.2%}")
        add("age_bands=" + ", ".join(f"{k}:{age_bands[k]:,}" for k in ("0-14","15-24","25-44","45-64","65+")))
        add("generations=" + ", ".join(f"g{k}:{v:,}" for k, v in sorted(generation_counts.items())))
        add("")
        add("[DEATH CAUSES]")
        if death_causes:
            for cause, count in death_causes.most_common():
                add(f"{cause:<32} {count:>12,}")
        else:
            add("none")
        add("")
        add("[SOCIAL CLASS]")
        for name, count in class_counts.most_common():
            add(f"{name:<24} {count:>12,}")
        add("")
        add("[PROFESSIONS]")
        for name, count in profession_counts.most_common():
            add(f"{name:<24} {count:>12,}")
        add("")
        add("[POLITICS AND PUBLIC FINANCE]")
        add(f"government={world.politics.government.name}")
        add(f"elections={world.politics.election_number:,}")
        add(f"treasury={world.politics.treasury:,.2f}")
        add(f"tax_rate={world.politics.government.tax_rate:.2%}")
        add(f"avg_taxes_paid={_safe_mean(p.taxes_paid for p in living):,.2f}")
        add(f"avg_welfare_received={_safe_mean(p.welfare_received for p in living):,.2f}")
        add("")
        add("[ECONOMY]")
        add(f"active_employers={len(employers):,}")
        add(f"employer_capacity={sum(e.capacity for e in employers):,}")
        add(f"employer_cash={sum(e.cash for e in employers):,.2f}")
        add(f"avg_person_money={_safe_mean(p.money for p in living):,.2f}")
        add(f"avg_person_food={_safe_mean(p.food for p in living):,.2f}")
        add(f"avg_person_health={_safe_mean(p.health for p in living):,.2f}")
        add(f"avg_person_shelter={_safe_mean(p.shelter for p in living):,.2f}")
        add(f"market_spending={sum(p.market_spending for p in living):,.2f}")
        add(f"lifetime_gross_income={sum(p.lifetime_gross_income for p in living):,.2f}")
        add("")
        add("[CRIME, POLICE AND SOCIAL EVENTS]")
        add(f"arrests={world.total_arrests:,}")
        add(f"thefts={world.total_thefts:,}")
        add(f"attacks={world.total_attacks:,}")
        add(f"helps={world.total_helps:,}")
        add(f"observations={world.total_observations:,}")
        add(f"shipments={world.transport.shipments:,}")
        add("")
        add("[EVENT COUNTS]")
        for name, count in sorted(event_counts.items(), key=lambda item: (-item[1], item[0])):
            add(f"{name:<32} {count:>12,}")
        add("")
        add("[LIFE EVENT COUNTS]")
        if life_counts:
            for name, count in sorted(life_counts.items(), key=lambda item: (-item[1], item[0])):
                add(f"{name:<32} {count:>12,}")
        else:
            add("none")
        add("")
        add("[HYBRID ENGINE]")
        for key in (
            "explicit_agents","aggregated_agents","sampled_agents","interesting_agents",
            "mandatory_agents","high_priority_agents","normal_priority_agents",
            "pending_interesting","budget_target","budget_ceiling",
            "aggregate_work_shifts","aggregate_food_demand","aggregate_medicine_demand",
        ):
            if key in hybrid:
                add(f"{key}={hybrid[key]}")
        if hybrid.get("reason_counts"):
            add("reason_counts=" + json.dumps(hybrid["reason_counts"], sort_keys=True))
        add("")
        add("[MULTIPROCESSING]")
        for key, value in pool_summary.items():
            if isinstance(value, float):
                add(f"{key}={value:.6f}")
            else:
                add(f"{key}={value}")
        add("")
        add("[PERFORMANCE]")
        if profile_rows:
            for row in profile_rows:
                add(
                    f"{row['phase']:<24} calls={row['calls']:>6} "
                    f"total={row['total']:>10.4f}s avg={row['avg']:>9.6f}s "
                    f"p95={row['p95']:>9.6f}s max={row['max']:>9.6f}s"
                )
        else:
            add("profiling disabled")
        add("")
        add("[DISTRICTS]")
        for row in locations:
            add(
                f"id={row['id']:>3} name={row['name']!r} kind={row['kind']:<11} "
                f"pop={row['population']:>8,} crime={row['crime_rate']:.6f} "
                f"avg_food={row['avg_food']:.2f} avg_money={row['avg_money']:.2f} "
                f"avg_health={row['avg_health']:.2f} "
                f"food_stock={row['food_stock']:.2f} food_price={row['food_price']:.3f} "
                f"medicine_stock={row['medicine_stock']:.2f} medicine_price={row['medicine_price']:.3f}"
            )
        add("")
        add("[END]")
        add("")
        return "\n".join(lines)

    def write(self, text):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")

    @staticmethod
    def cli_summary(world, path):
        non_natural = max(0, world.total_deaths - world.demographics.total_natural_deaths)
        pool = getattr(world, "district_pool", None)
        pool_summary = pool.summary() if pool is not None else {"enabled": False, "workers": 0}
        return (
            f"  detailed statistics: {path or 'disabled'}\n"
            f"  population: alive={world.alive_count:,} births={world.demographics.total_births:,} "
            f"deaths={world.total_deaths:,} natural={world.demographics.total_natural_deaths:,} "
            f"non_natural={non_natural:,}\n"
            f"  multiprocessing: enabled={pool_summary.get('enabled', False)} "
            f"workers={pool_summary.get('workers', 0)} tasks={pool_summary.get('tasks', 0):,} "
            f"items={pool_summary.get('items_returned', 0):,}"
        )

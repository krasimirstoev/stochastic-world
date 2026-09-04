# Stochastic World

A reproducible Python society simulation driven by entropy-seeded randomness. The model combines geography, local markets, transport, firms, employment, social mobility, memory/reputation, crime, police, politics and elections.

## Run

```bash
python3 world.py \
  --population 300 \
  --actions-per-day 4 \
  --period 4000 \
  --locale bg_BG
```

`--actions` remains a compatibility alias for `--actions-per-day`.

The interactive `tqdm` progress bar shows day, overall completion, alive population and arrests. Hybrid mode additionally shows `explicit`, `aggregated`, mandatory `p0`, high-priority agents and pending interesting agents.

## Execution engines

```bash
--engine auto
--engine agent
--engine hybrid
```

`auto` uses full agent mode below 100,000 people and hybrid mode at 100,000 or more.

### Agent mode

Every living person is explicitly evaluated every simulated day.

### Hybrid mode

Hybrid mode keeps socially important activity agent-level while routine economic activity is aggregated by district and employer.

The explicit pool is now priority-budgeted instead of growing without bound.

Default controls:

```bash
--hybrid-target-explicit 0.03
--hybrid-max-explicit 0.05
--hybrid-sample-per-district 256
--hybrid-interest-days 30
```

The target is 3% of the living population. The soft ceiling is 5%. Mandatory P0 agents may exceed the ceiling.

Priority model:

```text
P0  mandatory
    detention
    critical health / food / shelter

P1  high priority
    recent offender
    recent victim
    police response
    severe active conflict

P2  normal priority
    prolonged unemployment
    repeated shortage

P3  routine district sample
```

Selection order is:

```text
P0 -> always explicit
P1 -> admitted up to soft ceiling
P2 -> admitted up to target budget
P3 -> fills remaining target budget
```

`help` still changes direct memory and reputation, but no longer pins both participants in the persistent explicit pool.

Long-lived conditions no longer restart a 30-day retention clock every day. Acute critical states use short retention windows and are reevaluated when sampled again.

For 10,000 living people the default target is about 300 explicit agents and the soft ceiling about 500. For 2,000,000 people they are about 60,000 and 100,000.

## Automatic districts

`--locations 0` is the default:

```text
districts = max(5, ceil(population / target_neighborhood_size))
```

With the default target of 20,000 people per district, 2,000,000 people produce 100 districts.

## Hybrid statistics

Hybrid runs write `hybrid_stats`.

Alongside the original aggregation metrics, it now records:

```text
mandatory_agents
high_priority_agents
normal_priority_agents
pending_interesting
budget_target
budget_ceiling
reason_counts_json
```

Useful query:

```sql
SELECT
    day,
    explicit_agents,
    aggregated_agents,
    mandatory_agents,
    high_priority_agents,
    normal_priority_agents,
    pending_interesting,
    budget_target,
    budget_ceiling,
    reason_counts_json
FROM hybrid_stats
WHERE simulation_id = 1
ORDER BY day;
```

Explicit percentage:

```sql
SELECT
    day,
    ROUND(
      100.0 * explicit_agents /
      NULLIF(explicit_agents + aggregated_agents, 0),
      2
    ) AS explicit_percent,
    pending_interesting
FROM hybrid_stats
WHERE simulation_id = 1
ORDER BY day;
```

The pending count is intentional: it shows how much lower-priority social state is waiting outside the explicit budget.

## Recommended benchmark

After the earlier 10k run showed roughly 98% explicit population, rerun the same scale after this change:

```bash
time python3 world.py \
  --population 10000 \
  --actions-per-day 1 \
  --period 100 \
  --engine hybrid \
  --seed 12345
```

Expected healthy default behavior is approximately 3-5% explicit unless mandatory P0 states genuinely become widespread.

Then inspect:

```sql
SELECT
    AVG(explicit_agents),
    MAX(explicit_agents),
    AVG(aggregated_agents),
    AVG(mandatory_agents),
    AVG(high_priority_agents),
    MAX(pending_interesting)
FROM hybrid_stats;
```

If explicit stays near target and performance improves, the next benchmark should scale to 100k and then 2M.

## Reproducibility

The same seed and CLI configuration reproduce a run within the same engine mode. `agent` and `hybrid` intentionally define different stochastic models.

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The hybrid budget tests verify the 3% target, 5% high-priority ceiling and mandatory P0 overflow behavior.

## Next scaling step

Once the hybrid explicit pool stays bounded, profile the remaining periodic O(N) work: elections, welfare and mobility. Only after that should district-local persistent multiprocessing workers be introduced.

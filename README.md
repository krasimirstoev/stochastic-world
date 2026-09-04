# Stochastic World

Stochastic World is a reproducible, entropy-seeded society simulation written in Python. It models a population that lives in districts, works for firms, buys scarce goods, receives wages and welfare, changes social class, moves, forms households, has children, ages, commits crimes, interacts with police, votes in elections, remembers other people, and can survive or collapse over long simulated periods.

The project is intentionally stochastic, but not "random without memory". Randomness chooses outcomes; persistent world state changes the probabilities of later outcomes.

```text
random choice
    |
    v
world state changes
    |
    v
future probabilities change
    |
    v
new random choice
```

The same seed and the same configuration reproduce the same run within the same execution model.

---

## 1. Quick start

A small agent-level run:

```bash
python3 world.py \
  --population 300 \
  --actions-per-day 4 \
  --period 4000 \
  --seed 12345
```

A larger hybrid run with persistent multiprocessing:

```bash
python3 world.py \
  --population 100000 \
  --actions-per-day 1 \
  --period 180 \
  --engine hybrid \
  --seed 12345 \
  --profile-periodic
```

Important output files:

```text
simulation.sqlite   structured simulation database
simulation.log      event stream
statistics.log      detailed end-of-run human-readable report
```

`statistics.log` is enabled by default. Disable it with:

```bash
--no-statistics-log
```

or change its path:

```bash
--statistics-log reports/run-001.log
```

---

## 2. High-level architecture

```text
                         +----------------------+
                         |      CLI / run       |
                         | stochastic_world.cli |
                         +----------+-----------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
          +---------------+                  +----------------+
          |  Agent World  |                  |  Hybrid World  |
          |   LifeWorld   |                  |  HybridWorld   |
          +-------+-------+                  +--------+-------+
                  |                                   |
                  |                                   +----------------------+
                  |                                                          |
                  v                                                          v
        explicit person loop                                    priority explicit pool
                                                                           |
                                                                           v
                                                            persistent district workers
                                                                           |
                                                                           v
                                                               authoritative main process
                                                                           |
                                 +-----------------------------------------+-----------------------------------+
                                 |                 |                |                   |                       |
                                 v                 v                v                   v                       v
                              markets           firms          transport             police                 politics
                                 |                 |                |                   |                       |
                                 +-----------------+----------------+-------------------+-----------------------+
                                                                           |
                                                                           v
                                                                  demographics / households
                                                                           |
                                                                           v
                                                                    SQLite + logs
```

The main process is always authoritative. Workers never own SQLite, never mutate the canonical world directly, and never receive the complete `World` object.

---

## 3. Reproducibility and randomness

### 3.1 Master seed

When `--seed` is omitted, the simulation creates a master seed from OS entropy.

When a seed is supplied:

```bash
--seed 12345
```

the run is reproducible.

For multiple runs:

```bash
--runs 10
```

deterministic child seeds are derived from the master seed.

### 3.2 Serial random stream

The classic agent/hybrid paths use the simulation RNG for stochastic decisions and effects.

### 3.3 Parallel deterministic streams

The multiprocessing planner does not depend on process scheduling or worker identity.

A deterministic per-person stream is derived from:

```text
master seed
+ simulated day
+ person id
+ phase
+ action round
```

Workers are therefore free to finish in any order. Results are merged and applied by the main process in stable agent order.

This is important because OS scheduling must not become part of the simulation model.

---

## 4. Population and persons

Every individual is represented by `Person`.

Important state includes:

```text
identity
  id
  name

demographics
  age_days
  sex
  birth_day
  mother_id
  father_id
  partner_id
  household_id
  generation
  pregnant_until_day
  retired

economic state
  money
  food
  medicine
  shelter
  employer_id
  profession
  social_class
  work_experience
  lifetime_gross_income
  unemployment_days
  market_spending

physical state
  health
  energy
  alive

social / political state
  ideology
  taxes_paid
  welfare_received
  crime_suffered
  memories

justice state
  detained_until_day
  arrests
```

`person_id` is canonical. Names are descriptive and do not need to be unique.

---

## 5. Age and life stages

The model distinguishes:

```text
0-17       dependent
18-66      working age
67+        retired
```

Children do not use the ordinary adult decision loop.

At adulthood, a person can enter the labour force and receive an adult profession.

At retirement, a person leaves the working-age labour force and no longer uses profession-dependent `work` and `move` paths.

---

## 6. Households and families

Households are persistent social-economic units.

They provide a basic dependency model:

```text
adult resources
      |
      v
dependent child
  food support
  medicine support
  shelter support
```

Children can therefore survive through household resources rather than behaving as isolated economic agents with completely independent survival budgets.

Household membership is stored in:

```text
households
household_members
```

Life transitions are stored in:

```text
life_events
```

including partnership, pregnancy, birth, adulthood and retirement events.

---

## 7. Demography

The demographic system runs periodically rather than every day.

Default interval:

```text
30 simulated days
```

The cycle performs:

```text
aging / lifecycle checks
        |
        +--> natural mortality
        |
        +--> completed pregnancies -> births
        |
        +--> partnership formation
        |
        +--> new pregnancies
        |
        +--> household maintenance
```

### 7.1 Fertility

Fertility depends on age and household condition.

The model intentionally treats fertility coefficients as experiment assumptions rather than demographic truth.

### 7.2 Natural mortality

Natural mortality is stochastic and age-dependent.

Young people have a very small mortality hazard. The hazard increases substantially in older age groups and is modified by poor health.

### 7.3 Generations

Newborns receive:

```text
generation = max(parent generations) + 1
```

This allows multi-generation simulations and future inheritance mechanics.

### 7.4 Demographic statistics

`demographic_stats` records:

```text
population
births
natural_deaths
total_deaths
median_age
age_0_14
age_15_24
age_25_44
age_45_64
age_65_plus
households
pregnant
max_generation
```

---

## 8. Geography

The world is divided into districts.

Automatic district count:

```text
max(5, ceil(population / target_neighborhood_size))
```

Default target neighbourhood size:

```text
20,000 people
```

Examples:

```text
100,000 people   -> 5 districts
500,000 people   -> 25 districts
2,000,000 people -> 100 districts
```

District kinds repeat through a simple functional topology:

```text
Residential ---- Market ---- Industrial ---- Outskirts
                    |
                  Clinic
```

Locations carry characteristics such as:

```text
work multiplier
food scavenging capacity
medicine chance
market availability
shelter decay modifier
neighbours
capacity hint
```

---

## 9. Population index

`PopulationIndex` provides efficient district membership.

It maintains:

```text
district -> list of person ids
person id -> position inside district bucket
```

Removal uses swap-delete, making local membership updates O(1).

The index is also dynamically expandable so births can append new person IDs without rebuilding the complete population structure.

---

## 10. Decision engine

Adult explicit agents use a weighted random decision model.

Base actions:

```text
move
work
scavenge
buy_supplies
rest
heal
repair
help
steal
attack
idle
```

Weights are modified by current state.

Examples:

```text
low food
  -> more scavenge
  -> more purchasing
  -> more stealing

low energy
  -> much more rest
  -> less work
  -> less movement
  -> less attack

poor health
  -> more healing if medicine exists

low money
  -> more work
  -> less purchasing

hostile memories
  -> more theft / attack

positive social ties
  -> more help
```

The decision engine does not script stories. Stories emerge from state transitions and weighted choices.

---

## 11. Social memory and reputation

Interactions create directional memory.

A person can remember:

```text
trust
grievance
familiarity
help
theft
attack
observed behaviour
affinity
conflict
```

Memory affects later target selection and action probabilities.

Example:

```text
theft
  |
  v
victim grievance increases
  |
  v
future conflict probability increases
  |
  v
attack becomes more likely
  |
  v
police response / retaliation / new memories
```

Witnesses can also observe actions and update reputation.

Memories decay gradually.

---

## 12. Professions and social class

Social classes:

```text
working
lower_middle
middle
upper_middle
affluent
```

Professions include roles such as labourer, service worker, technician, clerk, teacher, nurse, trader, engineer, manager, entrepreneur and executive.

Profession influences:

```text
workplace fit
energy cost
career advancement
preferred district types
```

Social mobility is evaluated periodically.

Mobility can change both class and profession.

---

## 13. Labour market

Firms have:

```text
location
industry/kind
capacity
base wage
cash
productivity
output good
output per shift
employee roster
```

Initial employment is seeded stochastically.

Workers can:

```text
find jobs
work shifts
receive wages
pay taxes
lose jobs
move and terminate incompatible employment
```

Firms can:

```text
expand
contract
close
hire
produce goods
receive sales revenue
```

The target firm size is intentionally bounded so large populations produce many firms rather than a few enormous objects.

---

## 14. Goods market

The market currently models:

```text
food
medicine
```

Each district has local:

```text
price
stock
supplier ownership
demand
sold quantity
unmet demand
```

Purchases reduce stock and credit seller revenue.

Hybrid background demand uses bulk purchases for routine aggregated residents.

Prices are periodically repriced according to local market state.

---

## 15. Transport and logistics

Transport moves goods between neighbouring districts.

The system compares surplus and shortage conditions and creates shipments.

Simplified flow:

```text
supplier surplus
      |
      v
neighbour shortage
      |
      v
shipment capacity
      |
      v
food / medicine transferred
```

Shipments record:

```text
day
source
target
good
quantity
transport cost
```

Transport remains a global reconciliation phase because goods can cross district boundaries.

---

## 16. Crime and police

Crime actions:

```text
steal
attack
```

Crime changes:

```text
victim resources
victim health
social memory
crime rate
police load
arrest counts
detention
fines
```

Police are represented as district-level aggregate capacity rather than one explicit police agent per officer.

Coverage is influenced by:

```text
officers per population
current incident load
crime severity
```

Police can respond and arrest offenders.

Detained agents skip normal actions until their detention ends.

---

## 17. Politics

The baseline political system contains:

```text
Civic Left
Civic Right
```

A party defines:

```text
ideology
tax rate
welfare cash
welfare food
medicine welfare chance
welfare eligibility threshold
```

Voting depends on:

```text
person ideology
poverty / food insecurity
tax burden
welfare experience
local crime
personal crime victimisation
stochastic preference noise
```

Children do not vote.

Election interval:

```text
1460 days
```

The government controls the active tax and welfare policy.

---

## 18. Treasury and taxes

Working people pay taxes from gross income.

Taxes accumulate in the government treasury.

Welfare is limited by treasury funds.

Current simplified public-finance flow:

```text
work
 |
 v
gross wage
 |
 +--> household/person net income
 |
 +--> tax
       |
       v
   treasury
       |
       v
    welfare
```

### Planned public-sector extension

The next major economic layer is public employment.

A future public-service system will allocate part of the population to roles such as:

```text
administration
health
education
police / justice
infrastructure
```

Public salaries will be paid from collected tax revenue rather than created ex nihilo.

Planned flow:

```text
tax collection
      |
      v
treasury
      |
      +--> welfare
      |
      +--> public payroll
      |
      +--> future public investment
```

This extension should use explicit budget constraints so an underfunded government cannot maintain unlimited public employment.

---

## 19. Agent execution engine

`--engine agent`

Every living eligible person is processed explicitly.

Advantages:

```text
maximum individual detail
simple semantics
useful for small populations
```

Disadvantages:

```text
O(N) every day
expensive for very large populations
```

---

## 20. Hybrid execution engine

`--engine hybrid`

Hybrid mode divides the population into:

```text
explicit agents
aggregated routine residents
```

Routine economic activity is approximated in bulk while socially important agents remain explicit.

### Explicit budget

Defaults:

```text
target explicit fraction: 3%
soft ceiling:            5%
```

CLI:

```bash
--hybrid-target-explicit 0.03
--hybrid-max-explicit 0.05
```

### Priority classes

```text
P0 mandatory
   detention
   critical health / food / shelter

P1 high
   recent offender
   recent victim
   police response
   severe conflict

P2 normal
   prolonged unemployment
   repeated shortage

P3 routine sample
```

Selection:

```text
P0 always
  |
  v
P1 up to ceiling
  |
  v
P2 up to target
  |
  v
P3 fills remaining target
```

Mandatory P0 agents may exceed the soft ceiling.

---

## 21. Hybrid background economy

Routine workers who are not explicit are processed in aggregate.

Per employer:

```text
routine employees
      |
      v
participation estimate
      |
      v
aggregate shifts
      |
      +--> payroll
      +--> taxes
      +--> production / service revenue
```

Per district:

```text
routine residents
      |
      +--> aggregate food demand
      +--> aggregate medicine demand
```

When a previously dormant person becomes explicit again, catch-up approximates skipped economic and survival state.

Hybrid mode is therefore intentionally an approximation, not a bit-for-bit compressed version of agent mode.

---

## 22. Persistent district multiprocessing

Profiling showed that ordinary explicit actions and selected end-of-day work dominate runtime once the explicit pool is bounded.

The multiprocessing layer therefore parallelises **local planning**, not global mutation.

### 22.1 Why not send the World object?

Sending or copying `World` would be expensive because it contains:

```text
population
memories
firms
markets
police
transport
SQLite store
indexes
```

Instead, workers receive compact primitive snapshots.

### 22.2 Worker lifecycle

Workers use Python's `spawn` multiprocessing context and are created lazily once the explicit pool is large enough.

```text
run starts
   |
   v
no worker cost yet
   |
   v
explicit pool reaches threshold
   |
   v
create persistent workers
   |
   +--> worker 0 owns district shard
   +--> worker 1 owns district shard
   +--> ...
   |
   v
reuse workers every simulated day
   |
   v
run ends
   |
   v
graceful shutdown
```

No process is created per person or per simulated day.

### 22.3 Stable district sharding

District assignment:

```text
worker = district_id % worker_count
```

The worker count can change performance but does not define the random seed.

### 22.4 Action planning

For each action round:

```text
main process snapshots explicit agents
                 |
                 v
        shard snapshots by district
                 |
      +----------+----------+
      |          |          |
      v          v          v
   worker 0   worker 1   worker N
      |          |          |
      +----------+----------+
                 |
                 v
       planned action per agent
                 |
                 v
       main applies in stable order
```

Workers choose actions. The main process executes the action handlers because handlers can mutate global state such as markets, firms, crime, police, relationships and transport-sensitive state.

This design avoids conflicting writes and preserves deterministic event ordering.

### 22.5 End-of-day planning

Local survival arithmetic is also planned in workers.

Workers calculate deterministic per-person:

```text
food decay
energy decay
shelter decay
unemployment counter
ideology drift
starvation damage
exhaustion damage
exposure damage
```

The main process then applies the returned deltas, records events and performs deaths.

Household support remains authoritative in the main process because a child can mutate a donor's resources.

### 22.6 Automatic worker count

Default:

```bash
--hybrid-workers -1
```

means automatic:

```text
min(number of districts, max(1, cpu_count - 1))
```

Disable multiprocessing:

```bash
--hybrid-workers 0
```

Force a count:

```bash
--hybrid-workers 4
```

### 22.7 Small-workload fallback

Multiprocessing has IPC overhead.

The engine therefore falls back to the serial path when the explicit pool is too small.

Default:

```bash
--hybrid-worker-min-active 1024
```

Example:

```bash
--hybrid-worker-min-active 5000
```

This threshold is a performance parameter, not a simulation rule.

### 22.8 Authoritative state rule

Workers never:

```text
write SQLite
change markets
change employers
change police
change treasury
move people
kill people
create events
```

Only the main process performs those mutations.

This keeps the concurrency model deliberately conservative.

---

## 23. Daily hybrid execution flow

```text
START DAY
   |
   +--> election if due
   |
   +--> select explicit pool
   |
   +--> aggregate routine economy
   |
   +--> catch up explicit agents
   |
   +--> [parallel] action planning
   |          |
   |          v
   |      main action application
   |
   +--> crime history snapshot
   |
   +--> household dependent support
   |
   +--> [parallel] end-of-day planning
   |          |
   |          v
   |      main delta application
   |
   +--> transport
   |
   +--> welfare/business/police rebalance if due
   |
   +--> market repricing
   |
   +--> mobility if due
   |
   +--> demographics if due
   |
   +--> statistics
   |
   +--> SQLite commit
   |
END DAY
```

---

## 24. Performance profiler

Enable:

```bash
--profile-periodic
```

The profiler records wall-clock time for phases such as:

```text
day_total
election
hybrid_select
hybrid_aggregate
explicit_actions
mp_action_planning
mp_action_apply
selected_end_of_day
mp_end_of_day_planning
mp_end_of_day_apply
transport
welfare
business
police_rebalance
market_reprice
mobility
demographics
statistics
commit
```

CLI prints:

```text
calls
total
average
p95
maximum
```

Measurements are buffered and written to:

```text
performance_timings
```

after the run so profiling writes do not contaminate normal phase measurements.

---

## 25. Detailed statistics log

Every completed run appends a report to `statistics.log`.

The report contains:

```text
run identity and seeds
full configuration
population and demographics
age bands
generations
death causes
social classes
professions
government
treasury
tax rate
welfare
labour force
firm totals
personal resources
crime
police-related counts
event-type counts
life-event counts
hybrid explicit/aggregate state
multiprocessing metrics
performance profile
per-district population
per-district crime
per-district food/medicine stock and prices
```

The CLI also prints a compact final summary.

The statistics log is intended for humans. SQLite remains the canonical structured data source for deeper analysis.

---

## 26. Multiprocessing statistics

`statistics.log` records:

```text
enabled
started
workers
min_active
tasks
items_sent
items_returned
action_calls
end_of_day_calls
action_worker_seconds
action_dispatch_seconds
end_of_day_worker_seconds
end_of_day_dispatch_seconds
```

Interpretation:

```text
worker_seconds
    sum of work measured inside workers

dispatch_seconds
    main-process wall time from dispatch until all shard results return
```

A high `worker_seconds / dispatch_seconds` ratio indicates useful concurrency.

If dispatch time stays close to worker time, the workload may be IPC-bound or insufficiently parallel.

---

## 27. SQLite storage

Major tables include:

```text
simulations
persons
locations
events
relationships
relationship_history
observations
parties
elections
votes
mobility_history
employers
employment_history
daily_stats
location_stats
political_stats
social_stats
labor_stats
market_stats
police_stats
shipments
households
household_members
life_events
demographic_stats
hybrid_stats
performance_timings
```

SQLite runs with WAL and normal synchronous mode.

Workers do not access the connection.

---

## 28. Event modes

```bash
--event-mode full
--event-mode compact
--event-mode auto
```

`auto` selects compact mode for large populations.

Full mode stores detailed routine events.

Compact mode keeps higher-value events and avoids excessive database growth.

---

## 29. CLI reference

Core:

```text
--seed N
--population N
--actions-per-day N
--period DAYS
--runs N
--engine auto|agent|hybrid
```

Storage:

```text
--db PATH
--log PATH
--statistics-log PATH
--no-statistics-log
--event-mode auto|full|compact
```

World:

```text
--locale LOCALE
--visibility FLOAT
--max-witnesses N
--locations N
--target-neighborhood-size N
--police-per-1000 FLOAT
```

Hybrid:

```text
--hybrid-sample-per-district N
--hybrid-interest-days N
--hybrid-target-explicit FLOAT
--hybrid-max-explicit FLOAT
--hybrid-workers -1|0|N
--hybrid-worker-min-active N
```

Diagnostics:

```text
--profile-periodic
--no-progress
--quiet
```

---

## 30. Recommended performance benchmark

Use:

```bash
time python3 world.py \
  --population 100000 \
  --actions-per-day 1 \
  --period 180 \
  --engine hybrid \
  --seed 12345 \
  --profile-periodic
```

Then compare with multiprocessing disabled:

```bash
time python3 world.py \
  --population 100000 \
  --actions-per-day 1 \
  --period 180 \
  --engine hybrid \
  --seed 12345 \
  --hybrid-workers 0 \
  --profile-periodic
```

The two runs use different decision RNG execution models, so use them for performance comparison rather than exact world-state equivalence.

For larger scale:

```bash
time python3 world.py \
  --population 500000 \
  --actions-per-day 1 \
  --period 30 \
  --engine hybrid \
  --seed 12345 \
  --profile-periodic
```

and later:

```bash
time python3 world.py \
  --population 2000000 \
  --actions-per-day 1 \
  --period 10 \
  --engine hybrid \
  --seed 12345 \
  --profile-periodic
```

---

## 31. Useful SQL

Hybrid pool:

```sql
SELECT
    day,
    explicit_agents,
    aggregated_agents,
    mandatory_agents,
    high_priority_agents,
    pending_interesting
FROM hybrid_stats
WHERE simulation_id = 1
ORDER BY day;
```

Performance:

```sql
SELECT
    phase,
    COUNT(*) AS calls,
    ROUND(SUM(duration_seconds), 3) AS total_seconds,
    ROUND(AVG(duration_seconds), 5) AS avg_seconds,
    ROUND(MAX(duration_seconds), 5) AS max_seconds
FROM performance_timings
WHERE simulation_id = 1
GROUP BY phase
ORDER BY total_seconds DESC;
```

Demography:

```sql
SELECT
    day,
    population,
    births,
    natural_deaths,
    total_deaths,
    median_age,
    age_0_14,
    age_15_24,
    age_25_44,
    age_45_64,
    age_65_plus,
    max_generation
FROM demographic_stats
WHERE simulation_id = 1
ORDER BY day;
```

Death events:

```sql
SELECT day, actor_id, data
FROM events
WHERE simulation_id = 1
  AND event_type = 'death'
ORDER BY day;
```

---

## 32. Tests

Run:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

The test suite covers hybrid budgeting, demographic basics, profiling and deterministic multiprocessing planning.

---

## 33. Scaling philosophy

The project follows this order:

```text
1. establish correct model semantics
2. bound explicit population
3. measure bottlenecks
4. parallelise only measured local work
5. keep global reconciliation central
6. re-profile
7. optimise the next dominant phase
```

This avoids adding multiprocessing where it cannot help.

---

## 34. Known model simplifications

The simulation is experimental.

Current simplifications include:

```text
small set of goods
simplified firm finance
simplified welfare
simplified household finance
simplified fertility
simplified natural mortality
aggregate police officers
aggregate background economy
approximate hybrid catch-up
simple two-party politics
simple transport
no full public sector yet
```

Coefficients are assumptions for experimentation, not empirical claims.

---

## 35. Future development

Near-term priorities:

```text
public-sector employment and tax-funded salaries
public budget allocation
health / education / administration services
better accounting conservation
district-worker profiling at 500k and 2M
reduce remaining serial explicit action cost
hybridise selected periodic O(N) phases only if measurements justify it
more detailed generational inheritance
```

The intended public-sector architecture is:

```text
citizens / firms
      |
      v
     taxes
      |
      v
   treasury
      |
  +---+--------------------+
  |                        |
  v                        v
welfare               public payroll
                           |
               +-----------+-----------+
               |           |           |
               v           v           v
            health      education   administration
```

The goal is to keep public employment economically constrained by collected revenue and policy, rather than treating public salaries as unlimited money creation.

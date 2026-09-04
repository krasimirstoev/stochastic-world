# Demographic lifecycle

The simulation now models a population that can reproduce and turn over across generations rather than a fixed cohort that can only shrink.

## Lifecycle

- Initial people receive a deterministic age distribution and reproductive sex.
- Under-18 residents are dependents and do not take normal economic/crime actions or vote.
- Adults can work and vote; residents retire at 67.
- Households are initialized locally and can merge when new partnerships form.
- A monthly demographic cycle handles aging, partnership formation, pregnancy, birth and age-dependent natural mortality.
- Pregnancy lasts 280 simulated days. Fertility responds to age, health, household resources and existing children.
- Newborns inherit household, social class and a noisy blend of parental ideology. Their `generation` is parent generation + 1.
- Dependents receive food/medicine/shelter support from adults in their household.
- At 18 a dependent receives a profession from their social class and joins the working-age population.

## Storage

`persons` gains age/sex/parent/partner/household/generation fields. Additional tables:

- `households`
- `household_members`
- `life_events`
- `demographic_stats`

Example:

```sql
SELECT day,population,births,natural_deaths,total_deaths,
       ROUND(median_age,1) AS median_age,
       age_0_14,age_15_24,age_25_44,age_45_64,age_65_plus,
       households,pregnant,max_generation
FROM demographic_stats
WHERE simulation_id = 1
ORDER BY day;
```

`labor_stats` now treats only living working-age residents as the labor force, so children and retirees are not counted as unemployed.

## Hybrid mode

Births append new contiguous `person_id` values. `PopulationIndex` and hybrid touch-state grow dynamically. Routine explicit sampling favors adults; dependent children become explicit when risk or other priority rules select them.

This is still a toy demographic model. Fertility and mortality coefficients are experimental parameters, not demographic forecasts.

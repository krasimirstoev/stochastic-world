import argparse

from tqdm import tqdm

from .geography import DEFAULT_TARGET_NEIGHBORHOOD_SIZE, recommended_location_count
from .hybrid_world import HybridWorld
from .life_storage import LifeEventStore, LifeHybridEventStore
from .life_world import LifeWorld
from .rng import derive_run_seeds, make_rng


def parse_args():
    p = argparse.ArgumentParser(description="Entropy-seeded stochastic society simulation")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--population", type=int, default=100)
    p.add_argument("--actions-per-day", "--actions", dest="actions_per_day", type=int, default=3)
    p.add_argument("--period", type=int, default=100)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--db", default="simulation.sqlite")
    p.add_argument("--log", default="simulation.log")
    p.add_argument("--locale", default="en_US")
    p.add_argument("--visibility", type=float, default=.65)
    p.add_argument("--max-witnesses", type=int, default=3)
    p.add_argument("--locations", type=int, default=0)
    p.add_argument("--target-neighborhood-size", type=int, default=DEFAULT_TARGET_NEIGHBORHOOD_SIZE)
    p.add_argument("--police-per-1000", type=float, default=2.2)
    p.add_argument("--event-mode", choices=("auto", "full", "compact"), default="auto")
    p.add_argument("--engine", choices=("auto", "agent", "hybrid"), default="auto")
    p.add_argument("--hybrid-sample-per-district", type=int, default=256)
    p.add_argument("--hybrid-interest-days", type=int, default=30)
    p.add_argument("--hybrid-target-explicit", type=float, default=0.03)
    p.add_argument("--hybrid-max-explicit", type=float, default=0.05)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def resolve_engine(args):
    if args.engine != "auto":
        return args.engine
    return "hybrid" if args.population >= 100_000 else "agent"


def run_once(args, master_seed, run_seed, run_index, progress, engine):
    _, rng = make_rng(run_seed)
    locations_count = args.locations if args.locations > 0 else recommended_location_count(args.population, args.target_neighborhood_size)
    config = {"seed": run_seed, "population": args.population, "actions_per_day": args.actions_per_day,
              "period": args.period, "faker_locale": args.locale, "visibility": args.visibility,
              "max_witnesses": args.max_witnesses, "locations_count": locations_count,
              "event_mode": args.event_mode}
    store_cls = LifeHybridEventStore if engine == "hybrid" else LifeEventStore
    world_cls = HybridWorld if engine == "hybrid" else LifeWorld
    store = store_cls(args.db, args.log, config, run_index=run_index, master_seed=master_seed)
    world_kwargs = dict(visibility=args.visibility, max_witnesses=args.max_witnesses,
                        locations_count=locations_count, target_neighborhood_size=args.target_neighborhood_size,
                        police_per_1000=args.police_per_1000)
    if engine == "hybrid":
        world_kwargs.update(hybrid_sample_per_district=args.hybrid_sample_per_district,
                            hybrid_interest_days=args.hybrid_interest_days,
                            hybrid_target_explicit=args.hybrid_target_explicit,
                            hybrid_max_explicit=args.hybrid_max_explicit)
    world = world_cls(rng, args.population, args.actions_per_day, store, run_seed, args.locale, **world_kwargs)
    if engine == "agent":
        world.engine_mode = "agent"
    if not args.quiet:
        tqdm.write(f"Run {run_index}/{args.runs} | simulation_id={store.simulation_id} | seed={run_seed} | districts={len(world.locations)} | engine={engine} | event_mode={store.event_mode}")
    collapse = None; last = 0
    for day in range(1, args.period + 1):
        last = day; world.run_day(day)
        if progress is not None:
            progress.update(1)
            postfix = {"run": f"{run_index}/{args.runs}", "day": f"{day}/{args.period}",
                       "alive": world.alive_count, "births": world.demographics.total_births,
                       "arrests": world.total_arrests}
            if engine == "hybrid":
                s = world.last_hybrid_stats
                postfix.update(explicit=s["explicit_agents"], aggregated=s["aggregated_agents"],
                               p0=s["mandatory_agents"], high=s["high_priority_agents"], pending=s["pending_interesting"])
            progress.set_postfix(**postfix, refresh=False)
        if world.alive_count == 0:
            collapse = day; break
    if progress is not None and last < args.period:
        progress.update(args.period - last)
    store.finish(collapse)
    if not args.quiet:
        tqdm.write(f"  finished day={last} alive={world.alive_count} births={world.demographics.total_births} "
                   f"natural_deaths={world.demographics.total_natural_deaths} government={world.politics.government.name} "
                   f"elections={world.politics.election_number} deaths={world.total_deaths} arrests={world.total_arrests} "
                   f"shipments={world.transport.shipments} collapse_day={collapse}")
    return collapse


def main():
    args = parse_args()
    if args.population < 1 or args.period < 1 or args.runs < 1 or args.actions_per_day < 0:
        raise SystemExit("invalid numeric arguments")
    if args.locations not in (0,) and args.locations < 5:
        raise SystemExit("locations must be 0 (auto) or >= 5")
    if args.target_neighborhood_size < 100:
        raise SystemExit("target-neighborhood-size must be >= 100")
    if not 0 <= args.visibility <= 1 or args.max_witnesses < 0:
        raise SystemExit("invalid visibility settings")
    if args.police_per_1000 < 0:
        raise SystemExit("police-per-1000 must be >= 0")
    if args.hybrid_sample_per_district < 16 or args.hybrid_interest_days < 1:
        raise SystemExit("invalid hybrid settings")
    if not 0 < args.hybrid_target_explicit <= 1:
        raise SystemExit("hybrid-target-explicit must be in (0, 1]")
    if not args.hybrid_target_explicit <= args.hybrid_max_explicit <= 1:
        raise SystemExit("hybrid-max-explicit must be >= target and <= 1")
    engine = resolve_engine(args)
    master, _ = make_rng(args.seed); seeds = derive_run_seeds(master, args.runs)
    resolved_locations = args.locations if args.locations > 0 else recommended_location_count(args.population, args.target_neighborhood_size)
    print(f"Master seed: {master}\nRuns: {args.runs}\nPopulation: {args.population:,}\nActions/day: {args.actions_per_day}\nDistricts: {resolved_locations} ({'manual' if args.locations else 'auto'})\nEngine: {engine}\nDemographics: enabled\n")
    progress = None
    if not args.no_progress:
        progress = tqdm(total=args.runs * args.period, desc="Simulation", unit="day", dynamic_ncols=True, smoothing=0.1)
    collapses = []
    try:
        for i, seed in enumerate(seeds, 1):
            collapse = run_once(args, master, seed, i, progress, engine)
            if collapse is not None:
                collapses.append(collapse)
    finally:
        if progress is not None:
            progress.close()
    print(f"\nBatch summary\n  runs={args.runs}\n  collapsed={len(collapses)}\n  collapse_rate={len(collapses)/args.runs:.2%}")


if __name__ == "__main__":
    main()

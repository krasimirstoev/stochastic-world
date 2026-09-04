import argparse

from tqdm import tqdm

from .agent_world import ParallelAgentWorld
from .geography import DEFAULT_TARGET_NEIGHBORHOOD_SIZE, recommended_location_count
from .hybrid_world import HybridWorld
from .life_storage import LifeEventStore, LifeHybridEventStore
from .rng import derive_run_seeds, make_rng
from .run_statistics import RunStatistics


def parse_args():
    p = argparse.ArgumentParser(description="Entropy-seeded stochastic society simulation")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--population", type=int, default=100)
    p.add_argument("--actions-per-day", "--actions", dest="actions_per_day", type=int, default=3)
    p.add_argument("--period", type=int, default=100)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--db", default="simulation.sqlite")
    p.add_argument("--log", default="simulation.log")
    p.add_argument("--statistics-log", default="statistics.log",
                   help="Append detailed end-of-run reports to this file (default: statistics.log).")
    p.add_argument("--no-statistics-log", action="store_true",
                   help="Disable the detailed statistics log.")
    p.add_argument("--locale", default="en_US")
    p.add_argument("--visibility", type=float, default=.65)
    p.add_argument("--max-witnesses", type=int, default=3)
    p.add_argument("--locations", type=int, default=0)
    p.add_argument("--target-neighborhood-size", type=int, default=DEFAULT_TARGET_NEIGHBORHOOD_SIZE)
    p.add_argument("--police-per-1000", type=float, default=2.2)
    p.add_argument("--event-mode", choices=("auto", "full", "compact"), default="auto")
    p.add_argument("--engine", choices=("auto", "agent", "hybrid"), default="auto")
    p.add_argument("--initial-government", choices=("auto", "left", "right"), default="auto",
                   help="Force the day-1 government; auto preserves election-driven startup.")
    p.add_argument("--memory-cap", type=int, default=64,
                   help="Maximum detailed social memories per person; 0 means unlimited (default: 64).")
    p.add_argument("--encounter-sample", type=int, default=16,
                   help="Candidate people sampled for social target selection (default: 16).")
    p.add_argument("--hybrid-sample-per-district", type=int, default=256)
    p.add_argument("--hybrid-interest-days", type=int, default=30)
    p.add_argument("--hybrid-target-explicit", type=float, default=0.03)
    p.add_argument("--hybrid-max-explicit", type=float, default=0.05)
    p.add_argument(
        "--workers", "--hybrid-workers", dest="hybrid_workers", type=int, default=-1,
        help=(
            "Worker processes: agent uses 0 when omitted; hybrid uses auto when omitted. "
            "0=off, N=exact workers. --hybrid-workers remains a compatibility alias."
        ),
    )
    p.add_argument("--hybrid-worker-min-active", type=int, default=1024,
                   help="Use multiprocessing only when at least this many active agents are present.")
    p.add_argument("--profile-periodic", action="store_true",
                   help="Profile hybrid phases and persist timings to performance_timings.")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def resolve_engine(args):
    if args.engine != "auto":
        return args.engine
    return "hybrid" if args.population >= 100_000 else "agent"


def _print_profile_summary(world):
    rows = world.profiler.summary()
    if not rows:
        return
    tqdm.write("  performance profile (wall clock):")
    for row in rows:
        tqdm.write(
            f"    {row['phase']:<24} calls={row['calls']:>5} "
            f"total={row['total']:>8.3f}s avg={row['avg']:>8.4f}s "
            f"p95={row['p95']:>8.4f}s max={row['max']:>8.4f}s"
        )


def _resolved_worker_arg(args, engine):
    if engine == "agent" and args.hybrid_workers < 0:
        return 0
    return args.hybrid_workers


def run_once(args, master_seed, run_seed, run_index, progress, engine):
    _, rng = make_rng(run_seed)
    locations_count = (
        args.locations
        if args.locations > 0
        else recommended_location_count(args.population, args.target_neighborhood_size)
    )
    workers = _resolved_worker_arg(args, engine)
    config = {
        "seed": run_seed,
        "population": args.population,
        "actions_per_day": args.actions_per_day,
        "period": args.period,
        "faker_locale": args.locale,
        "visibility": args.visibility,
        "max_witnesses": args.max_witnesses,
        "locations_count": locations_count,
        "event_mode": args.event_mode,
        "engine": engine,
        "initial_government": args.initial_government,
        "memory_cap": args.memory_cap,
        "encounter_sample": args.encounter_sample,
        "workers": workers,
        "hybrid_workers": args.hybrid_workers,
        "hybrid_worker_min_active": args.hybrid_worker_min_active,
        "hybrid_target_explicit": args.hybrid_target_explicit,
        "hybrid_max_explicit": args.hybrid_max_explicit,
        "hybrid_sample_per_district": args.hybrid_sample_per_district,
        "hybrid_interest_days": args.hybrid_interest_days,
    }
    store_cls = LifeHybridEventStore if engine == "hybrid" else LifeEventStore
    world_cls = HybridWorld if engine == "hybrid" else ParallelAgentWorld
    store = store_cls(args.db, args.log, config, run_index=run_index, master_seed=master_seed)
    world_kwargs = dict(
        visibility=args.visibility,
        max_witnesses=args.max_witnesses,
        locations_count=locations_count,
        target_neighborhood_size=args.target_neighborhood_size,
        police_per_1000=args.police_per_1000,
    )
    if engine == "hybrid":
        world_kwargs.update(
            hybrid_sample_per_district=args.hybrid_sample_per_district,
            hybrid_interest_days=args.hybrid_interest_days,
            hybrid_target_explicit=args.hybrid_target_explicit,
            hybrid_max_explicit=args.hybrid_max_explicit,
            profile_periodic=args.profile_periodic,
            hybrid_workers=workers,
            hybrid_worker_min_active=args.hybrid_worker_min_active,
        )
    else:
        world_kwargs.update(
            agent_workers=workers,
            agent_worker_min_active=args.hybrid_worker_min_active,
        )

    world = world_cls(
        rng,
        args.population,
        args.actions_per_day,
        store,
        run_seed,
        args.locale,
        **world_kwargs,
    )
    world.memory_cap = args.memory_cap
    world.encounter_sample = args.encounter_sample
    for person in world.people:
        person.memory_cap = args.memory_cap
    if args.initial_government != "auto":
        world.politics.force_initial_government(args.initial_government)

    if not args.quiet:
        pool = getattr(world, "district_pool", None)
        worker_text = ""
        if pool is not None:
            worker_text = (
                f" | workers={pool.worker_count if pool.enabled else 0}"
                f" | mp_min_active={pool.min_active}"
            )
        tqdm.write(
            f"Run {run_index}/{args.runs} | simulation_id={store.simulation_id} | seed={run_seed} "
            f"| districts={len(world.locations)} | engine={engine} | event_mode={store.event_mode}{worker_text}"
        )

    collapse = None
    last = 0
    completed = False
    try:
        for day in range(1, args.period + 1):
            last = day
            world.run_day(day)
            if progress is not None:
                progress.update(1)
                postfix = {
                    "run": f"{run_index}/{args.runs}",
                    "day": f"{day}/{args.period}",
                    "alive": world.alive_count,
                    "births": world.demographics.total_births,
                    "arrests": world.total_arrests,
                }
                pool = getattr(world, "district_pool", None)
                if pool is not None and pool.enabled:
                    postfix["mp"] = pool.worker_count
                if engine == "hybrid":
                    s = world.last_hybrid_stats
                    postfix.update(
                        explicit=s["explicit_agents"],
                        aggregated=s["aggregated_agents"],
                        p0=s["mandatory_agents"],
                        high=s["high_priority_agents"],
                        pending=s["pending_interesting"],
                    )
                progress.set_postfix(**postfix, refresh=False)
            if world.alive_count == 0:
                collapse = day
                break

        if progress is not None and last < args.period:
            progress.update(args.period - last)

        if engine == "hybrid" and args.profile_periodic:
            world.profiler.flush(store)
            if not args.quiet:
                _print_profile_summary(world)

        statistics_path = None if args.no_statistics_log else args.statistics_log
        reporter = RunStatistics(statistics_path)
        report = reporter.build(
            world,
            store,
            master_seed=master_seed,
            run_seed=run_seed,
            run_index=run_index,
            last_day=last,
            collapse_day=collapse,
            config=config,
        )
        reporter.write(report)
        if not args.quiet:
            tqdm.write(RunStatistics.cli_summary(world, statistics_path))

        store.finish(collapse)
        completed = True
        if not args.quiet:
            tqdm.write(
                f"  finished day={last} alive={world.alive_count} births={world.demographics.total_births} "
                f"natural_deaths={world.demographics.total_natural_deaths} government={world.politics.government.name} "
                f"elections={world.politics.election_number} deaths={world.total_deaths} arrests={world.total_arrests} "
                f"shipments={world.transport.shipments} collapse_day={collapse}"
            )
        return collapse
    finally:
        close_parallel = getattr(world, "close_parallel", None)
        if close_parallel is not None:
            close_parallel()
        if not completed:
            try:
                store.log_fh.flush()
            except Exception:
                pass


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
    if args.memory_cap < 0:
        raise SystemExit("memory-cap must be >= 0")
    if args.encounter_sample < 1:
        raise SystemExit("encounter-sample must be >= 1")
    if args.hybrid_sample_per_district < 16 or args.hybrid_interest_days < 1:
        raise SystemExit("invalid hybrid settings")
    if not 0 < args.hybrid_target_explicit <= 1:
        raise SystemExit("hybrid-target-explicit must be in (0, 1]")
    if not args.hybrid_target_explicit <= args.hybrid_max_explicit <= 1:
        raise SystemExit("hybrid-max-explicit must be >= target and <= 1")
    if args.hybrid_workers < -1:
        raise SystemExit("workers must be -1 (auto/default), 0 (off), or a positive integer")
    if args.hybrid_worker_min_active < 1:
        raise SystemExit("hybrid-worker-min-active must be >= 1")

    engine = resolve_engine(args)
    if args.profile_periodic and engine != "hybrid":
        raise SystemExit("--profile-periodic currently requires --engine hybrid (or auto at population >= 100000)")

    master, _ = make_rng(args.seed)
    seeds = derive_run_seeds(master, args.runs)
    resolved_locations = (
        args.locations
        if args.locations > 0
        else recommended_location_count(args.population, args.target_neighborhood_size)
    )
    workers = _resolved_worker_arg(args, engine)
    if engine == "hybrid" and workers < 0:
        worker_mode = "auto"
    elif workers == 0:
        worker_mode = "off"
    else:
        worker_mode = str(workers)

    print(
        f"Master seed: {master}\n"
        f"Runs: {args.runs}\n"
        f"Population: {args.population:,}\n"
        f"Actions/day: {args.actions_per_day}\n"
        f"Districts: {resolved_locations} ({'manual' if args.locations else 'auto'})\n"
        f"Engine: {engine}\n"
        f"Demographics: enabled\n"
        f"Initial government: {args.initial_government}\n"
        f"Memory cap: {'unlimited' if args.memory_cap == 0 else args.memory_cap}\n"
        f"Encounter sample: {args.encounter_sample}\n"
        f"Workers: {worker_mode}\n"
        f"Statistics log: {'disabled' if args.no_statistics_log else args.statistics_log}\n"
        f"Profiling: {'periodic phases' if args.profile_periodic else 'off'}\n"
    )

    progress = None
    if not args.no_progress:
        progress = tqdm(
            total=args.runs * args.period,
            desc="Simulation",
            unit="day",
            dynamic_ncols=True,
            smoothing=0.1,
        )
    collapses = []
    try:
        for i, seed in enumerate(seeds, 1):
            collapse = run_once(args, master, seed, i, progress, engine)
            if collapse is not None:
                collapses.append(collapse)
    finally:
        if progress is not None:
            progress.close()

    print(
        f"\nBatch summary\n"
        f"  runs={args.runs}\n"
        f"  collapsed={len(collapses)}\n"
        f"  collapse_rate={len(collapses)/args.runs:.2%}"
    )


if __name__ == "__main__":
    main()

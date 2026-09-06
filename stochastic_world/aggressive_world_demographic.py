"""Demographic-authoritative layer for the 100k+ aggressive SoA engine.

Normal monthly lifecycle boundaries stay entirely in array state. Full Person
materialization is reserved for reporting and explicit fallback boundaries.
"""

from time import perf_counter

from .aggressive_balanced_pool import BalancedSoADomainPool
from .aggressive_demographics import SoADemographicState
from .aggressive_mobility import run_soa_mobility
from .aggressive_soa import B_WORKING_AGE
from .aggressive_world_soa import AggressiveParallelAgentWorld as SoAAggressiveWorld
from .demographics import DEMOGRAPHIC_INTERVAL_DAYS
from .labor_market import BUSINESS_INTERVAL_DAYS
from .politics import ELECTION_INTERVAL_DAYS
from .professions import MOBILITY_INTERVAL_DAYS


class AggressiveParallelAgentWorld(SoAAggressiveWorld):
    """SoA world with vectorized monthly demographics and mobility."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soa_demographics = None
        self._soa_demographics_initialized = False
        if self.soa_mode:
            # The base SoA pool has not started yet at construction time. Replace
            # its fixed location_id % workers ownership with a persistent pool
            # whose owner sets are greedily rebalanced from live CSR counts on
            # every BSP superstep.
            old_pool = self.soa_pool
            worker_count = int(getattr(old_pool, "worker_count", 0))
            self.soa_pool = BalancedSoADomainPool(
                self._soa_seed,
                self.soa_state,
                self.soa_memory,
                self.soa_index,
                len(self.locations),
                workers=worker_count,
            )
            self.soa_demographics = SoADemographicState(
                self.soa_state.capacity,
                self._soa_seed,
            )

    def _initialize_soa(self):
        super()._initialize_soa()
        if (
            not self.soa_mode
            or self.soa_demographics is None
            or self._soa_demographics_initialized
        ):
            return
        started = perf_counter()
        self.soa_demographics.sync_world(self)
        self._soa_demographics_initialized = True
        self._record_phase("soa_demographic_initial_sync", started)

    def _soa_reconcile_to_world(self, *, count_new_deaths=False):
        super()._soa_reconcile_to_world(count_new_deaths=count_new_deaths)
        if (
            not self.soa_mode
            or self.soa_demographics is None
            or not self._soa_demographics_initialized
        ):
            return
        started = perf_counter()
        for person in self.people:
            self.soa_demographics.apply_to_person(person)
        self.demographics.working_age_count = int(
            self.soa_state.flags[B_WORKING_AGE, : len(self.people)].sum()
        )
        self._record_phase("soa_demographic_reconcile", started)

    def _resync_soa_from_world(self):
        super()._resync_soa_from_world()
        if not self.soa_mode or self.soa_demographics is None:
            return
        started = perf_counter()
        self.soa_demographics.sync_world(self)
        self._soa_demographics_initialized = True
        self._record_phase("soa_demographic_resync", started)

    def _run_soa_cold_day(self, day):
        barrier_started = perf_counter()
        self.current_day = day

        started = perf_counter()
        self._run_soa_domain(day, fuse_eod=False)
        self._record_phase("soa_cold_actions", started)

        started = perf_counter()
        for shipment in self.transport.rebalance(day):
            self.store.shipment(shipment)
        self._record_phase("soa_cold_transport", started)

        started = perf_counter()
        if day % BUSINESS_INTERVAL_DAYS == 0:
            self._soa_welfare_cycle(day)
            self._soa_business_cycle(day)
        self._record_phase("soa_cold_business_soa", started)

        stats = self._run_soa_eod_only(day)
        self.goods_market.reprice()
        if stats is None:
            self._record_phase("soa_cold_barrier_total", barrier_started)
            return

        if day % MOBILITY_INTERVAL_DAYS == 0:
            started = perf_counter()
            run_soa_mobility(self, day, stats)
            self._record_phase("soa_cold_mobility_soa", started)

        started = perf_counter()
        police_snapshot = self._finish_temporal_police(stats)
        self._write_temporal_stats(day, stats, police_snapshot)
        self.store.commit_day()
        self._record_phase("soa_cold_stats_soa", started)

        started = perf_counter()
        self._initialize_soa()
        self.soa_demographics.cycle(self, day)
        self.soa_demographics.write_stats(self, day)
        self.store.commit_day()
        self._record_phase("soa_cold_demographics_soa", started)

        self._record_phase("soa_cold_barrier_total", barrier_started)

    def run_day(self, day):
        if not self.soa_mode:
            return super().run_day(day)
        election_day = day == 1 or (day - 1) % ELECTION_INTERVAL_DAYS == 0
        if day % DEMOGRAPHIC_INTERVAL_DAYS == 0:
            return self._run_soa_cold_day(day)

        self.current_day = day
        if election_day:
            # Day 1 deliberately uses the original Person election because the
            # initial objects are still authoritative. Later elections are
            # handled by the public aggressive world's SoA aggregate path.
            if day != 1:
                self._initialize_soa()
            self.run_election()

        total_started = perf_counter()
        _results, stats = self._run_soa_domain(day, fuse_eod=True)
        self._record_phase("actions_total", total_started)
        if stats is None:
            return
        for shipment in self.transport.rebalance(day):
            self.store.shipment(shipment)
        self.goods_market.reprice()
        police_snapshot = self._finish_temporal_police(stats)
        self._write_temporal_stats(day, stats, police_snapshot)
        self.store.commit_day()

    def prepare_reporting(self):
        super().prepare_reporting()
        if (
            self.soa_mode
            and self.soa_demographics is not None
            and self._soa_demographics_initialized
        ):
            started = perf_counter()
            self.soa_demographics.persist_people(self)
            self.store.conn.commit()
            self._record_phase("soa_demographic_persist", started)

    def close_parallel(self):
        if self.soa_mode and self.soa_demographics is not None:
            print(
                "  aggressive demographic SoA: "
                f"local={self.soa_demographics.allocated_bytes / (1024 * 1024):.1f}MiB"
            )
            if self.soa_pool is not None:
                summary = self.soa_pool.summary()
                if summary.get("days"):
                    print(
                        "  aggressive SoA load balance: "
                        f"avg_max/mean={summary.get('balance_avg_ratio', 1.0):.3f} "
                        f"max={summary.get('balance_max_ratio', 1.0):.3f}"
                    )
        super().close_parallel()


__all__ = ["AggressiveParallelAgentWorld"]

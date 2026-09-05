"""Demographic-authoritative layer for the 100k+ aggressive SoA engine.

Normal monthly lifecycle boundaries stay entirely in array state. Full Person
materialization is reserved for mobility, election/reporting and explicit
fallback boundaries.
"""

from time import perf_counter

from .aggressive_demographics import SoADemographicState
from .aggressive_soa import B_WORKING_AGE
from .aggressive_world_soa import AggressiveParallelAgentWorld as SoAAggressiveWorld
from .labor_market import BUSINESS_INTERVAL_DAYS
from .professions import MOBILITY_INTERVAL_DAYS


class AggressiveParallelAgentWorld(SoAAggressiveWorld):
    """SoA world with vectorized monthly demographics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.soa_demographics = None
        self._soa_demographics_initialized = False
        if self.soa_mode:
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
        # Mobility still mutates Person social/profession state and therefore
        # keeps the exact legacy boundary. All ordinary monthly boundaries stay
        # array-authoritative.
        if day % MOBILITY_INTERVAL_DAYS == 0:
            return super()._run_soa_cold_day(day)

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
        super().close_parallel()


__all__ = ["AggressiveParallelAgentWorld"]

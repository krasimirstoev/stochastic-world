"""Public entry point for the aggressive engine."""

from .aggressive_jit import install as _install_jit
from .aggressive_world_temporal import AggressiveParallelAgentWorld as _TemporalWorld


class AggressiveParallelAgentWorld(_TemporalWorld):
    """Temporal BSP world with optional Numba hot kernels."""

    def __init__(self, *args, **kwargs):
        self._aggressive_jit_enabled = bool(_install_jit())
        super().__init__(*args, **kwargs)

    def close_parallel(self):
        if getattr(self, "temporal_mode", False):
            mode = "numba=on" if self._aggressive_jit_enabled else "numba=off (python fallback)"
            print(f"  aggressive JIT kernels: {mode}")
        super().close_parallel()


__all__ = ["AggressiveParallelAgentWorld"]

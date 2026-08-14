"""3D acoustic FDTD: room responses from a staggered pressure–velocity scheme."""

from .analytic import discrete_mode_frequency, mode_frequency, mode_initial_state
from .grid import Grid, max_stable_time_step
from .medium import AIR, Medium
from .scheme import AcousticFDTD

__all__ = [
    "AIR",
    "AcousticFDTD",
    "Grid",
    "Medium",
    "discrete_mode_frequency",
    "max_stable_time_step",
    "mode_frequency",
    "mode_initial_state",
]

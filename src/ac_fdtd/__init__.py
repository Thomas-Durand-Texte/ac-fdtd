"""3D acoustic FDTD: room responses from a staggered pressure–velocity scheme."""

from .air import AirAbsorption
from .analytic import discrete_mode_frequency, mode_frequency, mode_initial_state
from .boundaries import (
    AbsorbingLayer,
    WallAdmittances,
    admittance_from_absorption,
    reflection_coefficient,
)
from .grid import Grid, max_stable_time_step
from .medium import AIR, Medium
from .scheme import AcousticFDTD

__all__ = [
    "AIR",
    "AbsorbingLayer",
    "AirAbsorption",
    "AcousticFDTD",
    "Grid",
    "Medium",
    "WallAdmittances",
    "admittance_from_absorption",
    "discrete_mode_frequency",
    "max_stable_time_step",
    "mode_frequency",
    "mode_initial_state",
    "reflection_coefficient",
]

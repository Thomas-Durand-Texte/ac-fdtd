"""Acoustic FDTD: room responses from a staggered pressure–velocity scheme, in 3D and 2D."""

from .air import AirAbsorption
from .analytic import discrete_mode_frequency, mode_frequency, mode_initial_state
from .boundaries import (
    AbsorbingLayer,
    WallAdmittances,
    admittance_from_absorption,
    random_incidence_absorption,
    reflection_coefficient,
)
from .boundaries2d import AbsorbingLayer2D, WallAdmittances2D
from .cpml import AbsorbingBoundary
from .cpml2d import AbsorbingBoundary2D
from .grid import Grid, max_stable_time_step
from .grid2d import Grid2D, max_stable_time_step_2d
from .io import resample, write_wav
from .medium import AIR, Medium
from .room import ImpulseResponse, Room, simulate_impulse_response
from .scalar import ScalarFDTD
from .scalar2d import ScalarFDTD2D
from .scheme import AcousticFDTD
from .scheme2d import AcousticFDTD2D
from .sources import band_limited_impulse, deconvolve, gaussian_pulse

__all__ = [
    "AIR",
    "AbsorbingBoundary",
    "AbsorbingBoundary2D",
    "AbsorbingLayer",
    "AbsorbingLayer2D",
    "AirAbsorption",
    "AcousticFDTD",
    "AcousticFDTD2D",
    "Grid",
    "Grid2D",
    "ImpulseResponse",
    "Medium",
    "Room",
    "ScalarFDTD",
    "ScalarFDTD2D",
    "WallAdmittances",
    "WallAdmittances2D",
    "admittance_from_absorption",
    "band_limited_impulse",
    "deconvolve",
    "discrete_mode_frequency",
    "max_stable_time_step",
    "max_stable_time_step_2d",
    "mode_frequency",
    "gaussian_pulse",
    "mode_initial_state",
    "resample",
    "simulate_impulse_response",
    "write_wav",
    "random_incidence_absorption",
    "reflection_coefficient",
]

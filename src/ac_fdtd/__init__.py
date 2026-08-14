"""3D acoustic FDTD: room responses from a staggered pressure–velocity scheme."""

from .air import AirAbsorption
from .analytic import discrete_mode_frequency, mode_frequency, mode_initial_state
from .boundaries import (
    AbsorbingLayer,
    WallAdmittances,
    admittance_from_absorption,
    random_incidence_absorption,
    reflection_coefficient,
)
from .grid import Grid, max_stable_time_step
from .io import resample, write_wav
from .medium import AIR, Medium
from .room import ImpulseResponse, Room, simulate_impulse_response
from .scheme import AcousticFDTD
from .sources import band_limited_impulse, deconvolve, gaussian_pulse

__all__ = [
    "AIR",
    "AbsorbingLayer",
    "AirAbsorption",
    "AcousticFDTD",
    "Grid",
    "ImpulseResponse",
    "Medium",
    "Room",
    "WallAdmittances",
    "admittance_from_absorption",
    "band_limited_impulse",
    "deconvolve",
    "discrete_mode_frequency",
    "max_stable_time_step",
    "mode_frequency",
    "gaussian_pulse",
    "mode_initial_state",
    "resample",
    "simulate_impulse_response",
    "write_wav",
    "random_incidence_absorption",
    "reflection_coefficient",
]

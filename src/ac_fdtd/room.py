"""The short path: a room, a source, some receivers, and an impulse response.

Everything here can be done with :class:`~ac_fdtd.scheme.AcousticFDTD` directly, and anything
unusual should be. What this module adds is the set of decisions that are the same every time —
how fine the grid has to be for a stated bandwidth, how long to run, which excitation, and
dividing that excitation back out — so that they are made once, in one place, with their
reasons written down.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .air import AirAbsorption
from .boundaries import (
    WallAdmittances,
    admittance_from_absorption,
    random_incidence_absorption,
)
from .grid import Grid
from .medium import AIR, Medium
from .metrics import air_absorption_coefficient, eyring_time, sabine_time
from .scheme import AcousticFDTD
from .sources import band_limited_impulse, deconvolve

__all__ = ["Room", "ImpulseResponse", "simulate_impulse_response"]

#: Points per wavelength at the stated bandwidth limit. Ten is the conventional figure for a
#: second-order scheme; `PROGRESS.md` records what the error actually is at that resolution.
DEFAULT_POINTS_PER_WAVELENGTH = 10.0

#: Run length as a multiple of the estimated reverberation time. T30 needs 35 dB of clean
#: decay, and a fit is only as good as its tail.
DEFAULT_DURATION_FACTOR = 1.5


@dataclass(frozen=True)
class Room:
    """A rectangular room with uniform wall absorption.

    Args:
        dimensions: Interior dimensions in metres. What is simulated is rounded to whole cells;
            :attr:`ImpulseResponse.grid` reports what was actually built.
        absorption: Normal-incidence absorption coefficient of every wall.
        air: Atmospheric conditions, or ``None`` for lossless air.
        medium: Fluid properties. Defaults to air at 20 °C.
    """

    dimensions: tuple[float, float, float]
    absorption: float = 0.1
    air: AirAbsorption | None = field(default_factory=AirAbsorption)
    medium: Medium = AIR

    @property
    def volume(self) -> float:
        return float(np.prod(self.dimensions))

    @property
    def surface(self) -> float:
        x, y, z = self.dimensions
        return 2.0 * (x * y + y * z + z * x)

    @property
    def wall_admittance(self) -> float:
        """Normalised admittance of the walls, from the quoted absorption coefficient."""
        return admittance_from_absorption(self.absorption)

    @property
    def diffuse_absorption(self) -> float:
        """The random-incidence absorption coefficient — what the statistical formulae want.

        :attr:`absorption` is quoted at normal incidence, because that is what a wall
        admittance means. The two differ by a factor approaching two for a live room, and
        mixing them up is the single easiest way to make a correct simulation look wrong.
        """
        return random_incidence_absorption(self.wall_admittance)

    def air_coefficient(self, frequency: float) -> float:
        """The ``m`` in Sabine's ``4 m V`` term at one frequency, 1/m. Zero without air losses."""
        if self.air is None:
            return 0.0
        return air_absorption_coefficient(float(self.air.attenuation(frequency)))

    def sabine_reverberation_time(self, frequency: float = 1000.0) -> float:
        """Sabine's estimate at one frequency, in seconds — a prediction, not a result."""
        return sabine_time(
            self.volume, self.surface, self.diffuse_absorption, self.air_coefficient(frequency)
        )

    def eyring_reverberation_time(self, frequency: float = 1000.0) -> float:
        """Eyring's estimate at one frequency, in seconds."""
        return eyring_time(
            self.volume, self.surface, self.diffuse_absorption, self.air_coefficient(frequency)
        )


@dataclass(frozen=True)
class ImpulseResponse:
    """Impulse responses at one or more receivers, and what produced them.

    Attributes:
        signals: ``(receiver, sample)``, deconvolved, at :attr:`sample_rate`.
        recorded: The raw pressure before deconvolution, same shape.
        excitation: The signal that was injected.
        sample_rate: In Hz, and not a round number — the grid chose it.
        max_frequency: Above this the response is zero by construction, because above this the
            grid was not asked to be right.
        grid: The grid that was actually built.
    """

    signals: np.ndarray
    recorded: np.ndarray
    excitation: np.ndarray
    sample_rate: float
    max_frequency: float
    grid: Grid


def simulate_impulse_response(
    room: Room,
    source: tuple[float, float, float],
    receivers: list[tuple[float, float, float]],
    max_frequency: float,
    duration: float | None = None,
    points_per_wavelength: float = DEFAULT_POINTS_PER_WAVELENGTH,
    courant: float = 1.0,
) -> ImpulseResponse:
    """Excite the room at one point and record the impulse response at others.

    Args:
        room: What to simulate.
        source: Position of the monopole, in metres.
        receivers: Positions to record at.
        max_frequency: Bandwidth to resolve. This sets the grid spacing, and through it the
            time step, the memory and most of the run time — cost scales as its fourth power.
        duration: Seconds to simulate. Defaults to 1.5 times Sabine's estimate, which is enough
            decay for a T30 fit.
        points_per_wavelength: Spatial resolution at ``max_frequency``.
        courant: Fraction of the stability limit. Leave at 1.
    """
    grid = Grid.from_max_frequency(
        room.dimensions,
        max_frequency=max_frequency,
        sound_speed=room.medium.sound_speed,
        points_per_wavelength=points_per_wavelength,
    )
    solver = AcousticFDTD(
        grid,
        medium=room.medium,
        courant=courant,
        walls=WallAdmittances.from_absorption(room.absorption),
        air_absorption=room.air,
    )

    if duration is None:
        duration = DEFAULT_DURATION_FACTOR * room.sabine_reverberation_time()
    n_steps = int(round(duration / solver.dt))

    excitation = band_limited_impulse(n_steps, solver.dt, max_frequency)
    solver.add_volume_source(source, excitation)
    for receiver in receivers:
        solver.add_pressure_receiver(receiver)

    solver.run(n_steps)

    recorded = solver.recorded_pressure
    return ImpulseResponse(
        signals=deconvolve(recorded, excitation, solver.dt, max_frequency),
        recorded=recorded,
        excitation=excitation,
        sample_rate=solver.sample_rate,
        max_frequency=max_frequency,
        grid=grid,
    )

"""Air absorption: the ISO 9613-1 attenuation curve, and a time-domain medium that has it.

The target
----------
:meth:`AirAbsorption.attenuation` is the standard's own formula — classical/rotational losses
plus the oxygen and nitrogen relaxation terms, as a function of temperature, humidity and
static pressure. It is the reference this module is judged against, not part of the solver.

The medium
----------
A time-stepping scheme cannot apply a frequency-dependent attenuation directly; it needs a
material whose *equations* have that attenuation. A relaxing fluid does, and it is the physics
the ISO formula was derived from in the first place, so nothing is being approximated by
choosing it — only re-expressed.

For each relaxation process, add one auxiliary field driven by the divergence:

    dp/dt = -rho c_inf^2 div v + sum_nu psi_nu
    tau_nu dpsi_nu/dt + psi_nu = rho c_0^2 Delta_nu div v

In the frequency domain that is a bulk modulus ``K(w) = rho c_0^2 [1 + sum Delta_nu iw tau_nu /
(1 + iw tau_nu)]``, whose attenuation works out to

    alpha_nu(f) = (pi Delta_nu / c) f^2 F_nu / (F_nu^2 + f^2),     F_nu = 1 / (2 pi tau_nu)

which is exactly the shape of each ISO relaxation term. Matching them coefficient by
coefficient gives ``Delta_nu = C_nu c / pi`` with ``C_nu`` read straight out of the standard —
so the model is *fitted to nothing*. There is no curve fitting step and no free parameter.

The classical ``f^2`` term is the same shape with its relaxation frequency at infinity. Infinity
is not available on a grid, so it is placed at the Nyquist frequency of the time step: below
that the term is indistinguishable from ``f^2``, and above it the model rolls off instead of
growing — which is the right way to be wrong in a band the scheme cannot represent anyway.

Two consequences worth stating
------------------------------
**Sound speed becomes frequency dependent**, because that is what relaxation does: the
high-frequency limit is faster than the low-frequency one, here by about 0.05 %. So
:attr:`Medium.sound_speed` is taken to be the *unrelaxed* speed, the one the time step is
derived from. Reading it as the low-frequency speed instead would make the Courant number
0.05 % larger than requested, which at the stability limit is not a rounding difference.

**This is not a stability mechanism.** At 20 °C and 50 % relative humidity the attenuation is
about 0.02 dB/m at 4 kHz: a fraction of a decibel over a room. It is here because it is
audible in a long decay and because it bounds the accumulation of high-frequency numerical
energy — not to rescue a run that would otherwise diverge. Stability comes from the Courant
condition and from passive boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["AirAbsorption", "RelaxationProcess"]

_REFERENCE_PRESSURE = 101325.0
_REFERENCE_TEMPERATURE = 293.15
_TRIPLE_POINT = 273.16
_KELVIN = 273.15
#: Neper to decibel. The standard states attenuation in dB/m; the physics is in Np/m.
_DECIBELS_PER_NEPER = 8.685889638065035


@dataclass(frozen=True)
class RelaxationProcess:
    """One relaxation mechanism, as the solver needs it.

    Attributes:
        strength: ``Delta``, the fractional increase in ``c^2`` this process contributes
            between its low- and high-frequency limits. Of order 1e-3 for air.
        relaxation_time: ``tau``, in seconds.
    """

    strength: float
    relaxation_time: float

    @property
    def frequency(self) -> float:
        """Relaxation frequency ``1 / (2 pi tau)``, in Hz."""
        return 1.0 / (2.0 * math.pi * self.relaxation_time)

    def attenuation(self, frequencies: np.ndarray, sound_speed: float) -> np.ndarray:
        """Attenuation this process alone produces, in Np/m."""
        relaxation = self.frequency
        return (
            self.strength
            * math.pi
            / sound_speed
            * frequencies**2
            * relaxation
            / (relaxation**2 + frequencies**2)
        )


@dataclass(frozen=True)
class AirAbsorption:
    """Air absorption at given atmospheric conditions, per ISO 9613-1.

    Args:
        temperature: Degrees Celsius.
        relative_humidity: Percent.
        static_pressure: Pascals.
        classical: Include the classical (viscous, thermal, rotational) ``f^2`` term.
        relaxation: Include the oxygen and nitrogen relaxation terms. These dominate the
            audio band by one to two orders of magnitude; without them the model is not
            wrong in form, it is wrong in size.
    """

    temperature: float = 20.0
    relative_humidity: float = 50.0
    static_pressure: float = _REFERENCE_PRESSURE
    classical: bool = True
    relaxation: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.relative_humidity <= 100.0:
            raise ValueError(f"relative_humidity is a percentage, got {self.relative_humidity}")
        if self.temperature <= -_KELVIN:
            raise ValueError(f"temperature must be above absolute zero, got {self.temperature}")

    @property
    def kelvin(self) -> float:
        return self.temperature + _KELVIN

    @property
    def _pressure_ratio(self) -> float:
        return self.static_pressure / _REFERENCE_PRESSURE

    @property
    def _temperature_ratio(self) -> float:
        return self.kelvin / _REFERENCE_TEMPERATURE

    @property
    def molar_water_concentration(self) -> float:
        """Water vapour concentration ``h``, in molar percent — the variable that drives it all.

        Absorption is far more sensitive to humidity than to temperature: this number moves the
        nitrogen relaxation frequency by two orders of magnitude across the ordinary range.
        """
        exponent = -6.8346 * (_TRIPLE_POINT / self.kelvin) ** 1.261 + 4.6151
        saturation_ratio = 10.0**exponent
        return self.relative_humidity * saturation_ratio / self._pressure_ratio

    @property
    def oxygen_relaxation_frequency(self) -> float:
        """``F_rO``, in Hz — tens of kilohertz at ordinary humidity."""
        h = self.molar_water_concentration
        return self._pressure_ratio * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))

    @property
    def nitrogen_relaxation_frequency(self) -> float:
        """``F_rN``, in Hz — hundreds of hertz, which is why it matters for rooms."""
        h = self.molar_water_concentration
        return (
            self._pressure_ratio
            * self._temperature_ratio**-0.5
            * (9.0 + 280.0 * h * math.exp(-4.170 * (self._temperature_ratio ** (-1 / 3) - 1.0)))
        )

    @property
    def _classical_coefficient(self) -> float:
        """``B`` in ``alpha = B f^2``, in Np/(m Hz^2)."""
        return 1.84e-11 / self._pressure_ratio * self._temperature_ratio**0.5

    @property
    def _oxygen_coefficient(self) -> float:
        return self._temperature_ratio**-2.5 * 0.01275 * math.exp(-2239.1 / self.kelvin)

    @property
    def _nitrogen_coefficient(self) -> float:
        return self._temperature_ratio**-2.5 * 0.1068 * math.exp(-3352.0 / self.kelvin)

    def attenuation(self, frequencies: np.ndarray) -> np.ndarray:
        """ISO 9613-1 attenuation, in dB/m. The reference, not the model."""
        frequencies = np.asarray(frequencies, dtype=float)
        total = np.zeros_like(frequencies)
        if self.classical:
            total += self._classical_coefficient * frequencies**2
        if self.relaxation:
            for coefficient, relaxation in (
                (self._oxygen_coefficient, self.oxygen_relaxation_frequency),
                (self._nitrogen_coefficient, self.nitrogen_relaxation_frequency),
            ):
                total += (
                    coefficient * frequencies**2 * relaxation / (relaxation**2 + frequencies**2)
                )
        return _DECIBELS_PER_NEPER * total

    def processes(self, sound_speed: float, dt: float) -> tuple[RelaxationProcess, ...]:
        """The auxiliary fields the solver has to carry, for this air and this time step.

        The classical term's surrogate relaxation frequency is the Nyquist frequency of ``dt``;
        see the module docstring for why it cannot simply be infinite.
        """
        processes = []
        if self.relaxation:
            for coefficient, relaxation in (
                (self._oxygen_coefficient, self.oxygen_relaxation_frequency),
                (self._nitrogen_coefficient, self.nitrogen_relaxation_frequency),
            ):
                processes.append(
                    RelaxationProcess(
                        strength=coefficient * sound_speed / math.pi,
                        relaxation_time=1.0 / (2.0 * math.pi * relaxation),
                    )
                )
        if self.classical:
            nyquist = 0.5 / dt
            processes.append(
                RelaxationProcess(
                    strength=self._classical_coefficient * nyquist * sound_speed / math.pi,
                    relaxation_time=1.0 / (2.0 * math.pi * nyquist),
                )
            )
        return tuple(processes)

    def model_attenuation(
        self, frequencies: np.ndarray, sound_speed: float, dt: float
    ) -> np.ndarray:
        """Attenuation of the *implemented* medium, in dB/m, before discretisation.

        Differs from :meth:`attenuation` only where the classical surrogate rolls off, which is
        why both are plotted: it separates "the model does not match the standard" from "the
        grid does not resolve the model".
        """
        frequencies = np.asarray(frequencies, dtype=float)
        total = np.zeros_like(frequencies)
        for process in self.processes(sound_speed, dt):
            total += process.attenuation(frequencies, sound_speed)
        return _DECIBELS_PER_NEPER * total

    def total_strength(self, sound_speed: float, dt: float) -> float:
        """``sum Delta`` — how much faster the unrelaxed speed is than the relaxed one, squared."""
        return sum(process.strength for process in self.processes(sound_speed, dt))

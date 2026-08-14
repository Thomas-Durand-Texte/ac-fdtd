"""Room acoustics parameters, and the statistical formulae they are usually checked against.

One thing here is easier than in a real measurement and one is harder.

**Easier: there is no noise floor.** Schroeder's backward integration assumes the tail of the
impulse response is decay and not background noise; in a measurement it is not, which is why
real analysis needs a truncation point and a compensating estimate of what was cut off. A
simulated impulse response decays into round-off, so the integration can simply run to the end
and is exact. None of that machinery is here, and its absence is a property of the input rather
than a simplification.

**Harder: the statistical formulae are not ground truth.** Sabine and Eyring assume a diffuse
field: many modes overlapping, energy uniformly distributed, absorption spread evenly. Below
the Schroeder frequency a room has none of that — it has individual modes with individual decay
rates, and no amount of simulation accuracy will make a measured T60 agree with a formula that
does not apply there. So :func:`schroeder_frequency` is part of this module, and any comparison
against Sabine that does not mention it is reporting an opinion.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

__all__ = [
    "clarity",
    "early_decay_time",
    "eyring_time",
    "octave_band",
    "reverberation_time",
    "sabine_time",
    "schroeder_decay",
    "schroeder_frequency",
    "air_absorption_coefficient",
]

#: 24 ln(10) / c at 343 m/s — the constant in Sabine's formula, spelled out rather than 0.161.
_SABINE_NUMERATOR = 24.0 * np.log(10.0) / 343.0


def octave_band(response: np.ndarray, sample_rate: float, centre: float) -> np.ndarray:
    """Filter to one octave band, zero-phase.

    Zero-phase (forward-backward) filtering is used because the quantity of interest is an
    energy decay, and a filter's own group delay would otherwise be indistinguishable from the
    start of the room's.
    """
    nyquist = 0.5 * sample_rate
    low = centre / np.sqrt(2.0) / nyquist
    high = min(centre * np.sqrt(2.0) / nyquist, 0.999)
    if low >= high:
        raise ValueError(f"the {centre} Hz octave band does not fit below {nyquist} Hz")
    sections = scipy_signal.butter(4, (low, high), btype="bandpass", output="sos")
    return scipy_signal.sosfiltfilt(sections, response, axis=-1)


def schroeder_decay(response: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    """Backward-integrated energy decay curve, in dB, starting at 0.

    Runs to the end of the response: see the module docstring for why that is legitimate here
    and would not be on a measurement.
    """
    energy = np.asarray(response, dtype=float) ** 2
    integrated = np.cumsum(energy[..., ::-1], axis=-1)[..., ::-1]
    total = integrated[..., :1]
    with np.errstate(divide="ignore"):
        decay = 10.0 * np.log10(np.maximum(integrated / total, 1e-300))
    return np.maximum(decay, floor_db)


def _decay_slope(decay: np.ndarray, sample_rate: float, upper: float, lower: float) -> float:
    """Seconds per 60 dB, from a straight-line fit between two levels on the decay curve."""
    start = int(np.argmax(decay <= upper))
    end = int(np.argmax(decay <= lower))
    if end <= start:
        raise ValueError(
            f"the decay never reaches {lower} dB: the response is too short, or the room is "
            "too live for the run length"
        )
    instants = np.arange(start, end) / sample_rate
    slope, _ = np.polyfit(instants, decay[start:end], 1)
    return -60.0 / slope


def reverberation_time(
    response: np.ndarray, sample_rate: float, evaluation_range: float = 20.0
) -> float:
    """T20 or T30, extrapolated to 60 dB, in seconds.

    The fit starts 5 dB below the peak, as the convention requires: the first few decibels are
    the direct sound and the early reflections, which are not the reverberant decay and would
    bias the slope if included.
    """
    decay = schroeder_decay(response)
    return _decay_slope(decay, sample_rate, -5.0, -5.0 - evaluation_range)


def early_decay_time(response: np.ndarray, sample_rate: float) -> float:
    """EDT: the first 10 dB, extrapolated to 60. Closer to what a listener hears than T30."""
    decay = schroeder_decay(response)
    return _decay_slope(decay, sample_rate, 0.0, -10.0)


def clarity(response: np.ndarray, sample_rate: float, split: float = 0.05) -> float:
    """C50 (or C80 with ``split=0.08``): early-to-late energy ratio, in dB."""
    boundary = int(round(split * sample_rate))
    energy = np.asarray(response, dtype=float) ** 2
    early = float(np.sum(energy[..., :boundary]))
    late = float(np.sum(energy[..., boundary:]))
    return 10.0 * np.log10(early / late)


def air_absorption_coefficient(attenuation_db_per_metre: float) -> float:
    """Convert an attenuation in dB/m to the energy coefficient ``m`` the formulae want, 1/m.

    Sabine's ``4 m V`` term is in nepers of *energy* per metre, while ISO 9613-1 states
    attenuation in decibels of *pressure*. The factor is ``10 log10 e``, and forgetting it is a
    factor of 4.34.
    """
    return attenuation_db_per_metre / (10.0 * np.log10(np.e))


def sabine_time(
    volume: float, surface: float, absorption: float, air_coefficient: float = 0.0
) -> float:
    """Sabine's estimate of T60, in seconds.

    Derived assuming absorption is low and evenly spread. It does not go to zero as the
    absorption goes to one, which is the well-known sign that it is an approximation.
    """
    return _SABINE_NUMERATOR * volume / (surface * absorption + 4.0 * air_coefficient * volume)


def eyring_time(
    volume: float, surface: float, absorption: float, air_coefficient: float = 0.0
) -> float:
    """Eyring's estimate of T60, in seconds — the one that behaves at high absorption."""
    return (
        _SABINE_NUMERATOR
        * volume
        / (-surface * np.log(1.0 - absorption) + 4.0 * air_coefficient * volume)
    )


def schroeder_frequency(volume: float, reverberation: float) -> float:
    """Above this, modes overlap enough for the statistical formulae to mean anything.

    ``2000 sqrt(T60 / V)``. Below it, a room is a set of individual modes and a measured decay
    is under no obligation to match Sabine.
    """
    return 2000.0 * np.sqrt(reverberation / volume)

"""Getting impulse responses out of the grid's sample rate and into a file.

The grid picks the sample rate, and it is never the one anyone wants: the time step comes from
the Courant condition, so a 2 cm grid samples at 29.7 kHz and a 1 cm grid at 59.4 kHz. Neither
is 48 kHz. Resampling is therefore not an optional convenience but part of producing a usable
result, and it is done with a polyphase filter at an exact rational ratio rather than by
interpolation, which would be an unspecified lowpass with unspecified ripple.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile

__all__ = ["resample", "write_wav"]

#: Largest denominator allowed when approximating the resampling ratio. A ratio this fine is
#: exact to a part in ten thousand, which is far below anything audible, while keeping the
#: polyphase filter short enough to run quickly.
_MAX_DENOMINATOR = 1000


def resample(signal: np.ndarray, source_rate: float, target_rate: float) -> np.ndarray:
    """Resample along the last axis, from the grid's rate to an audio one."""
    if source_rate <= 0.0 or target_rate <= 0.0:
        raise ValueError(f"sample rates must be positive, got {source_rate} and {target_rate}")
    ratio = Fraction(target_rate / source_rate).limit_denominator(_MAX_DENOMINATOR)
    return scipy_signal.resample_poly(
        np.asarray(signal, dtype=float), ratio.numerator, ratio.denominator, axis=-1
    )


def write_wav(
    path: str | Path,
    signal: np.ndarray,
    sample_rate: float,
    target_rate: float | None = 48000.0,
    normalise: bool = True,
) -> Path:
    """Write one or more impulse responses to a 32-bit float WAV file.

    Float rather than integer, because an impulse response has a peak-to-tail ratio of 60 dB or
    more and the point of the file is usually to be convolved with something else, not to be
    played. Normalisation is on by default for the same reason it is usually wrong to leave off:
    absolute pressures out of a simulation are in whatever units the source was given.

    Args:
        signal: One response, or a stack with time along the last axis, written as channels.
        sample_rate: Rate the signal is at now — ``1 / dt`` for a raw solver recording.
        target_rate: Rate to resample to, or ``None`` to write at the grid's own rate.
    """
    signal = np.atleast_2d(np.asarray(signal, dtype=float))
    if target_rate is not None:
        signal = resample(signal, sample_rate, target_rate)
        sample_rate = target_rate

    if normalise:
        peak = np.max(np.abs(signal))
        if peak > 0.0:
            signal = signal / peak

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, int(round(sample_rate)), signal.T.astype(np.float32))
    return path

"""Excitation signals, and getting the impulse response back out of what they produce.

Why not just inject an impulse
------------------------------
A single-sample impulse is flat all the way to the Nyquist frequency, and the top of that range
is exactly where the grid is at its worst: at four points per wavelength the scheme's phase
velocity is wrong by tens of percent. The response would not be wrong in a way anyone could
see — it would just have a smeared, wrong-sounding top end, with nothing in the output to say
so. So excitation is band-limited to the range the grid can carry, and the band is a stated
parameter rather than an accident of the time step.

Why not a sine sweep
--------------------
Swept sines exist to buy signal-to-noise ratio against background noise and to push harmonic
distortion out of the measurement window. A simulation has neither: it is silent between
arrivals and exactly linear. A short band-limited pulse plus deconvolution gives the same
impulse response for a fraction of the run time, so the sweep would be ceremony.

Deconvolution
-------------
The recorded pressure is the impulse response convolved with the excitation, so the excitation
has to be divided back out. Dividing spectra is only well posed where the excitation actually
has energy, which is what the regularisation and the band limit in :func:`deconvolve` are for.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "band_limited_impulse",
    "deconvolve",
    "gaussian_pulse",
]

#: Level, relative to the peak of the spectrum, defining a Gaussian pulse's stated bandwidth.
_BANDWIDTH_LEVEL_DB = 40.0


def gaussian_pulse(n_steps: int, dt: float, max_frequency: float) -> np.ndarray:
    """A Gaussian pulse whose spectrum is 40 dB down at ``max_frequency``.

    The smooth choice: no ringing, no cutoff to place, and nothing above the stated band worth
    worrying about. Its drawback is that the spectrum is not flat, so levels within the band
    have to be deconvolved rather than read directly.
    """
    decay = np.log(10.0 ** (_BANDWIDTH_LEVEL_DB / 20.0))
    width = np.sqrt(decay) / (np.pi * max_frequency)
    centre = 4.0 * width
    instants = (np.arange(n_steps) + 0.5) * dt
    return np.exp(-(((instants - centre) / width) ** 2))


def band_limited_impulse(n_steps: int, dt: float, max_frequency: float) -> np.ndarray:
    """A windowed sinc: flat to ``max_frequency``, then nothing.

    The flat spectrum is what makes this the better excitation for an impulse response — every
    frequency in the band is driven equally hard, so the deconvolution has no weak spots.
    """
    if max_frequency >= 0.5 / dt:
        raise ValueError(
            f"max_frequency {max_frequency} is at or above the Nyquist frequency {0.5 / dt}"
        )
    half_length = int(np.ceil(4.0 / (max_frequency * dt)))
    if 2 * half_length + 1 > n_steps:
        raise ValueError(
            f"a pulse band-limited to {max_frequency} Hz needs {2 * half_length + 1} steps, "
            f"but only {n_steps} were asked for"
        )

    offsets = np.arange(-half_length, half_length + 1)
    kernel = 2.0 * max_frequency * dt * np.sinc(2.0 * max_frequency * dt * offsets)
    kernel *= np.hanning(kernel.size)

    signal = np.zeros(n_steps)
    signal[: kernel.size] = kernel
    return signal


def deconvolve(
    recorded: np.ndarray,
    excitation: np.ndarray,
    dt: float,
    max_frequency: float,
    regularisation: float = 1e-6,
) -> np.ndarray:
    """Divide the excitation back out, returning the impulse response.

    Args:
        recorded: Pressure at a receiver, one sample per step. May be a stack of receivers,
            with time along the last axis.
        excitation: The signal that was injected, same length.
        dt: Time step.
        max_frequency: Everything above this is zeroed. The excitation has no energy there by
            construction, so anything the division produces there is the ratio of two very
            small numbers and belongs to no one.
        regularisation: Floor on the excitation's power spectrum, relative to its peak. Guards
            the same division inside the band.
    """
    recorded = np.asarray(recorded, dtype=float)
    spectrum = np.fft.rfft(recorded, axis=-1)
    source = np.fft.rfft(np.asarray(excitation, dtype=float))
    frequencies = np.fft.rfftfreq(recorded.shape[-1], dt)

    power = np.abs(source) ** 2
    response = spectrum * np.conjugate(source) / (power + regularisation * power.max())
    response[..., frequencies > max_frequency] = 0.0
    return np.fft.irfft(response, n=recorded.shape[-1], axis=-1)

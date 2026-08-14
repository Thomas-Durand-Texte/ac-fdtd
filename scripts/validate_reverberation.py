"""M4 — impulse responses, and reverberation time against Sabine and Eyring.

This is the first study whose output is a file someone can listen to, and the first whose
reference is not exact. Rungs 0 to 3 of the validation ladder compared against closed-form
answers; Sabine and Eyring are not that. They assume a diffuse field — many overlapping modes,
energy spread evenly — and a small room does not have one below its Schroeder frequency. So the
question this answers is not "does the simulation match the formula" but "does it match it
where the formula applies, and depart from it where the formula does not". Agreement everywhere
would be the suspicious result.

Two traps are worth naming, because both were live in this repository before this script:

* **The absorption coefficient in the formulae is the random-incidence one.** A wall admittance
  is defined at normal incidence, and a locally reacting surface absorbs far more obliquely: a
  wall quoted at 0.15 heads-on is 0.25 averaged over angle, and the predicted reverberation time
  differs by a factor of 1.7. Feeding Sabine the normal-incidence figure made a correct
  simulation look twice as absorbent as it should be.
* **Decay rates are measured on the raw recording, not the deconvolved response.** Deconvolution
  divides two spectra and leaves a small broadband residue where the excitation had little
  energy; that residue never decays, so backward integration reaches it and flattens out, and
  the T30 fit then reports the noise floor as a reverberation time. The excitation is a few
  milliseconds long, so it affects the first few milliseconds of the decay and nothing else.

Run: ``uv run python scripts/validate_reverberation.py``  (a few minutes)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotstyle  # noqa: E402

from ac_fdtd import Room, simulate_impulse_response, write_wav  # noqa: E402
from ac_fdtd.metrics import (  # noqa: E402
    clarity,
    early_decay_time,
    octave_band,
    reverberation_time,
    schroeder_decay,
    schroeder_frequency,
)

FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m4_reverberation.svg"
AUDIO_PATH = Path(__file__).resolve().parents[1] / "outputs" / "room_impulse_response.wav"

ROOM = Room(dimensions=(3.2, 2.7, 2.3), absorption=0.15)
MAX_FREQUENCY = 1450.0
POINTS_PER_WAVELENGTH = 8.0
DURATION = 0.45

#: Two source positions, because one is an anecdote: a single source-receiver pair sees its own
#: modal accidents, and a reverberation time is a property of the room rather than of a pair.
SOURCES = ((0.9, 0.7, 1.2), (2.4, 1.9, 0.8))
RECEIVERS = ((2.3, 1.9, 1.4), (1.6, 2.1, 0.8), (2.7, 0.9, 1.7), (1.1, 1.3, 1.6))
OCTAVE_CENTRES = (125.0, 250.0, 500.0, 1000.0)


def run() -> tuple[list[np.ndarray], float, object]:
    """Every source-receiver pair's raw recording, the sample rate, and the last response."""
    recordings = []
    sample_rate = 0.0
    for source in SOURCES:
        response = simulate_impulse_response(
            ROOM,
            source=source,
            receivers=list(RECEIVERS),
            max_frequency=MAX_FREQUENCY,
            duration=DURATION,
            points_per_wavelength=POINTS_PER_WAVELENGTH,
        )
        recordings.extend(response.recorded)
        sample_rate = response.sample_rate
        last = response
    write_wav(AUDIO_PATH, last.signals, last.sample_rate)
    return recordings, sample_rate, last


def main() -> None:
    plotstyle.apply()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    recordings, sample_rate, response = run()

    per_band = {}
    for centre in OCTAVE_CENTRES:
        times = []
        for recording in recordings:
            filtered = octave_band(recording, sample_rate, centre)
            times.append(reverberation_time(filtered, sample_rate, 20.0))
        per_band[centre] = np.array(times)

    predicted_sabine = np.array([ROOM.sabine_reverberation_time(f) for f in OCTAVE_CENTRES])
    predicted_eyring = np.array([ROOM.eyring_reverberation_time(f) for f in OCTAVE_CENTRES])
    crossover = schroeder_frequency(ROOM.volume, float(np.mean(predicted_eyring)))

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # --- the impulse response ----------------------------------------------------------
    ax = axes[0]
    instants = np.arange(response.signals.shape[1]) / response.sample_rate * 1e3
    normalised = response.signals[0] / np.max(np.abs(response.signals[0]))
    ax.plot(instants, normalised, color=plotstyle.SERIES[0], lw=0.8)
    ax.set_xlim(0, 120)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("time  [ms]")
    ax.set_ylabel("pressure  [normalised]")
    ax.set_title(
        f"Impulse response, {ROOM.dimensions[0]}x{ROOM.dimensions[1]}x{ROOM.dimensions[2]} m, "
        f"C50 = {clarity(response.signals[0], response.sample_rate):.1f} dB"
    )

    # --- decay curves ------------------------------------------------------------------
    ax = axes[1]
    for centre, colour in zip((125.0, 500.0, 1000.0), plotstyle.SERIES, strict=True):
        filtered = octave_band(recordings[0], sample_rate, centre)
        decay = schroeder_decay(filtered)
        ax.plot(
            np.arange(decay.size) / sample_rate * 1e3,
            decay,
            color=colour,
            label=f"{centre:.0f} Hz",
        )
    ax.axhspan(-25.0, -5.0, color=plotstyle.GRID, alpha=0.6, zorder=0)
    ax.text(5, -15, "T20 fit range", fontsize=8, color=plotstyle.MUTED)
    # Stops short of the end of the recording: backward integration always falls off a cliff
    # in its last samples, and that cliff is the integration's own edge, not the room's.
    ax.set_xlim(0, 1e3 * 0.6 * DURATION)
    ax.set_ylim(-60, 2)
    ax.set_xlabel("time  [ms]")
    ax.set_ylabel("energy decay  [dB]")
    ax.set_title("Backward-integrated decay, one receiver")
    ax.legend(loc="upper right")

    # --- against the formulae ----------------------------------------------------------
    ax = axes[2]
    means = np.array([per_band[c].mean() for c in OCTAVE_CENTRES])
    spread = np.array([per_band[c].std() for c in OCTAVE_CENTRES])
    ax.errorbar(
        OCTAVE_CENTRES,
        means,
        yerr=spread,
        fmt="o-",
        color=plotstyle.SERIES[0],
        capsize=4,
        label=f"simulated, {len(recordings)} positions",
    )
    ax.plot(OCTAVE_CENTRES, predicted_sabine, "s--", color=plotstyle.SERIES[1], label="Sabine")
    ax.plot(OCTAVE_CENTRES, predicted_eyring, "^--", color=plotstyle.SERIES[2], label="Eyring")
    ax.axvspan(OCTAVE_CENTRES[0] * 0.7, crossover, color=plotstyle.GRID, alpha=0.6, zorder=0)
    ax.text(
        135,
        0.10,
        "below the Schroeder frequency:\nindividual modes, not a diffuse field",
        fontsize=8,
        color=plotstyle.MUTED,
    )
    ax.set_xscale("log")
    ax.set_xticks(OCTAVE_CENTRES)
    ax.set_xticklabels([f"{c:.0f}" for c in OCTAVE_CENTRES])
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("octave band centre  [Hz]")
    ax.set_ylabel("T20 extrapolated to 60 dB  [s]")
    ax.set_title(f"Schroeder frequency {crossover:.0f} Hz")
    ax.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, bbox_inches="tight")

    print(f"room {ROOM.dimensions} m, V = {ROOM.volume:.1f} m3, S = {ROOM.surface:.1f} m2")
    print(
        f"absorption {ROOM.absorption} at normal incidence, "
        f"{ROOM.diffuse_absorption:.3f} random incidence"
    )
    print(
        f"grid {response.grid.shape} at {response.grid.dx * 1e3:.1f} mm, "
        f"{response.sample_rate / 1e3:.1f} kHz, {len(recordings)} source-receiver pairs"
    )
    print(f"Schroeder frequency {crossover:.0f} Hz\n")
    print("  band     simulated T20        Sabine   Eyring   vs Eyring")
    for centre, sabine, eyring in zip(
        OCTAVE_CENTRES, predicted_sabine, predicted_eyring, strict=True
    ):
        values = per_band[centre]
        print(
            f"  {centre:5.0f} Hz  {values.mean():.3f} +/- {values.std():.3f} s   "
            f"{sabine:.3f} s  {eyring:.3f} s   {100 * (values.mean() / eyring - 1):+6.1f} %"
        )

    print(
        f"\nbroadband EDT {early_decay_time(recordings[0], sample_rate):.3f} s, "
        f"C50 {clarity(response.signals[0], response.sample_rate):.1f} dB"
    )
    print(f"wrote {FIGURE_PATH} and {AUDIO_PATH}  ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()

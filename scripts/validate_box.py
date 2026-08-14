"""M1 — does the lossless scheme reproduce the one 3D case with an exact answer?

Three questions, one figure:

1. **Does the discretisation error behave?** The scheme's own modal frequency should approach
   the analytical one as ``dx^2``. Measured from the dispersion relation, so this panel costs
   nothing and covers spacings a simulation could not.
2. **Does the half step between p and v actually matter?** Yes, and this quantifies it: the
   correctly staggered energy is conserved to round-off, while the same run scored with the
   naive ``p^2 + v^2`` form is not conserved at all. Its error scales as ``omega*dt/2`` — a
   fraction of a percent for a well-resolved low mode, order one for content near the grid
   limit — so it oscillates rather than drifts, and it can never serve as a bug detector.
3. **Do the modes land where they should in a real broadband run?** A pulse, a probe, an FFT,
   and the analytical frequencies drawn on top.

Run: ``uv run python scripts/validate_box.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotstyle  # noqa: E402

from ac_fdtd import AIR, AcousticFDTD, Grid  # noqa: E402
from ac_fdtd.analytic import (  # noqa: E402
    discrete_mode_frequency,
    mode_frequency,
    mode_initial_state,
)

FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m1_box_validation.svg"

ROOM = (1.6, 1.2, 1.0)
CONVERGENCE_MODES = ((1, 0, 0), (2, 1, 1), (3, 3, 2))
SPACINGS = (0.1, 0.05, 0.025, 0.0125, 0.00625)

#: Broadband run: fine enough that discretisation error is far below the FFT resolution, so a
#: peak landing off the line means a bug rather than a known approximation.
BROADBAND_DX = 0.025
BROADBAND_DURATION = 0.6
SOURCE_POINT = (0.11, 0.09, 0.07)
PROBE_POINT = (1.49, 1.13, 0.91)
SPECTRUM_MAX_FREQUENCY = 400.0


def convergence() -> dict[tuple[int, int, int], list[float]]:
    """Relative error of the scheme's modal frequency, per mode, over the spacing sweep."""
    errors = {}
    for mode in CONVERGENCE_MODES:
        per_mode = []
        for dx in SPACINGS:
            grid = Grid.from_lengths(ROOM, dx)
            dt = grid.dx / (AIR.sound_speed * np.sqrt(3.0))
            exact = mode_frequency(grid.lengths, mode, AIR.sound_speed)
            numerical = discrete_mode_frequency(grid, mode, AIR, dt)
            per_mode.append(abs(numerical - exact) / exact)
        errors[mode] = per_mode
    return errors


def energy_histories(n_steps: int = 4000, sample_every: int = 25):
    """Correct and naive energy of one lossless run, as deviations from their own start.

    The run starts from a superposition of three exact modes rather than from noise. That
    matters for fairness: a noise field with zero velocity is not a consistently staggered
    state, and scoring it would flatter the correct form by handing the naive one a wrong
    starting value. Started from a legitimate state, the naive form oscillates instead — which
    is the real failure mode and the harder one to notice.
    """
    solver = AcousticFDTD(Grid.from_lengths((0.8, 0.65, 0.55), 0.05), courant=1.0)
    for mode, amplitude in (((1, 0, 0), 1.0), ((2, 1, 1), 0.6), ((3, 3, 2), 0.35)):
        pressure, velocity = mode_initial_state(
            solver.grid, mode, solver.medium, solver.dt, amplitude
        )
        solver.p += pressure
        for axis in range(3):
            solver.velocity[axis] += velocity[axis]

    medium, grid = solver.medium, solver.grid
    scale = grid.dx**3

    def naive_energy() -> float:
        potential = np.sum(solver.p**2) / (2 * medium.density * medium.sound_speed**2)
        kinetic = sum(0.5 * medium.density * np.sum(v**2) for v in solver.velocity)
        return float(scale * (potential + kinetic))

    steps, correct, naive = [0], [solver.energy()], [naive_energy()]
    while solver.step_index < n_steps:
        solver.run(sample_every)
        steps.append(solver.step_index)
        correct.append(solver.energy())
        naive.append(naive_energy())

    def deviation(values):
        values = np.asarray(values)
        # Clamped at 1e-17 only so exact zeros are drawable on a log axis; double precision
        # cannot resolve below ~1e-16 anyway.
        return np.maximum(np.abs(values / values[0] - 1.0), 1e-17)

    return np.array(steps), deviation(correct), deviation(naive)


def broadband_response():
    """Impulse response at a probe, plus the analytical mode frequencies to compare against."""
    grid = Grid.from_lengths(ROOM, BROADBAND_DX)
    solver = AcousticFDTD(grid, medium=AIR, courant=1.0)
    n_steps = int(BROADBAND_DURATION / solver.dt)

    # A Gaussian pulse whose spectrum is flat well past the band we look at, and smooth enough
    # that the grid is not asked to carry wavelengths it cannot represent.
    width_steps = max(4.0, 1.0 / (4.0 * SPECTRUM_MAX_FREQUENCY * solver.dt))
    time_index = np.arange(n_steps)
    centre = 4.0 * width_steps
    signal = np.exp(-(((time_index - centre) / width_steps) ** 2))
    solver.add_volume_source(SOURCE_POINT, signal)

    recorded = np.empty(n_steps)
    for step in range(n_steps):
        solver.step()
        recorded[step] = solver.probe_pressure(PROBE_POINT)

    spectrum = np.fft.rfft(recorded * np.hanning(n_steps))
    frequencies = np.fft.rfftfreq(n_steps, solver.dt)

    modes = []
    for l in range(6):  # noqa: E741 — l, m, n is the standard modal notation
        for m in range(6):
            for n in range(6):
                if (l, m, n) == (0, 0, 0):
                    continue
                frequency = mode_frequency(grid.lengths, (l, m, n), AIR.sound_speed)
                if frequency <= SPECTRUM_MAX_FREQUENCY:
                    modes.append(frequency)
    return frequencies, np.abs(spectrum), sorted(modes), solver, n_steps


def peak_agreement(frequencies, magnitude, modes):
    """Distance from each analytical mode to the nearest simulated peak, in Hz.

    Peaks are parabolically interpolated on the log spectrum, which resolves them far below the
    FFT bin spacing — otherwise this would measure the record length rather than the physics.

    A caveat the summary statistic cannot express: a mode with a node at the source or at the
    probe is not excited at all, and its nearest peak is then some *other* mode's. Those
    entries are meaningless rather than bad, which is why the median is quoted alongside the
    count of modes matched to within one bin, and not the maximum on its own.
    """
    log_magnitude = np.log(magnitude + 1e-30)
    interior = np.arange(1, len(magnitude) - 1)
    is_peak = (log_magnitude[interior] > log_magnitude[interior - 1]) & (
        log_magnitude[interior] > log_magnitude[interior + 1]
    )
    peak_bins = interior[is_peak]
    bin_width = frequencies[1] - frequencies[0]

    refined = []
    for b in peak_bins:
        left, centre, right = log_magnitude[b - 1], log_magnitude[b], log_magnitude[b + 1]
        offset = 0.5 * (left - right) / (left - 2 * centre + right)
        refined.append(frequencies[b] + offset * bin_width)
    refined = np.array(refined)

    return np.array([np.min(np.abs(refined - mode)) for mode in modes])


def main() -> None:
    plotstyle.apply()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    errors = convergence()
    steps, correct_drift, naive_drift = energy_histories()
    frequencies, magnitude, modes, solver, n_steps = broadband_response()
    deviations = peak_agreement(frequencies, magnitude, modes)

    figure = plt.figure(figsize=(10.5, 7.2))
    layout = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15), hspace=0.42, wspace=0.28)

    # --- convergence -------------------------------------------------------------------
    ax = figure.add_subplot(layout[0, 0])
    for (mode, values), marker in zip(errors.items(), ("o", "s", "^"), strict=True):
        ax.loglog(SPACINGS, 100 * np.array(values), marker + "-", label=f"mode {mode}")
    reference = 100 * errors[CONVERGENCE_MODES[-1]][0] * (np.array(SPACINGS) / SPACINGS[0]) ** 2
    ax.loglog(SPACINGS, reference, "--", color=plotstyle.MUTED, lw=1.2, label="$dx^2$")
    ax.set_xlabel("grid spacing dx  [m]")
    ax.set_ylabel("modal frequency error  [%]")
    ax.set_title("Second-order convergence")
    ax.legend(loc="lower right")

    # --- energy ------------------------------------------------------------------------
    ax = figure.add_subplot(layout[0, 1])
    ax.semilogy(steps, correct_drift, "-", color=plotstyle.SERIES[0], label="staggered (correct)")
    ax.semilogy(steps, naive_drift, "-", color=plotstyle.SERIES[1], label="naive $p^2+v^2$")
    ax.axhline(2.2e-16, color=plotstyle.MUTED, lw=1.2, ls="--")
    ax.text(
        steps[1],
        3.5e-16,
        "double precision eps",
        ha="left",
        va="bottom",
        fontsize=8,
        color=plotstyle.MUTED,
    )
    ax.set_ylim(1e-18, 1e1)
    ax.set_xlabel("time step")
    ax.set_ylabel("|E/E(0) − 1|")
    ax.set_title("Energy conservation needs the half-step term")
    ax.legend(loc="center right")

    # --- spectrum ----------------------------------------------------------------------
    ax = figure.add_subplot(layout[1, :])
    band = frequencies <= SPECTRUM_MAX_FREQUENCY
    decibels = 20 * np.log10(magnitude[band] / np.max(magnitude[band]) + 1e-12)
    for index, mode_frequency_hz in enumerate(modes):
        ax.axvline(
            mode_frequency_hz,
            color=plotstyle.SERIES[1],
            lw=0.9,
            alpha=0.75,
            zorder=1,
            label="analytical modes" if index == 0 else None,
        )
    ax.plot(
        frequencies[band],
        decibels,
        color=plotstyle.SERIES[0],
        lw=1.4,
        zorder=2,
        label="simulated",
    )
    ax.set_xlim(0, SPECTRUM_MAX_FREQUENCY)
    ax.set_ylim(-70, 5)
    ax.set_xlabel("frequency  [Hz]")
    ax.set_ylabel("level  [dB re. max]")
    matched = int(np.sum(deviations <= frequencies[1]))
    ax.set_title(
        f"Rigid box {ROOM[0]}x{ROOM[1]}x{ROOM[2]} m, dx = {BROADBAND_DX * 1e3:.0f} mm — "
        f"{matched}/{len(modes)} modes below {SPECTRUM_MAX_FREQUENCY:.0f} Hz matched "
        f"within one FFT bin ({frequencies[1]:.2f} Hz)"
    )
    ax.legend(loc="upper left", ncol=2)

    figure.savefig(FIGURE_PATH, bbox_inches="tight")

    print(f"grid            : {solver.grid.shape}, dx = {solver.grid.dx} m")
    print(f"simulated room  : {tuple(round(v, 4) for v in solver.grid.lengths)} m")
    print(f"time step       : {solver.dt * 1e6:.2f} us, {n_steps} steps")
    print(f"convergence     : {[f'{e:.2e}' for e in errors[CONVERGENCE_MODES[1]]]}")
    print(
        f"energy drift    : max {correct_drift.max():.2e} (correct), "
        f"{naive_drift.max():.2e} (naive)"
    )
    print(
        f"peak offset     : median {np.median(deviations):.3f} Hz, "
        f"max {deviations.max():.3f} Hz, "
        f"{int(np.sum(deviations <= frequencies[1]))}/{len(modes)} within one bin"
    )
    print(f"FFT resolution  : {frequencies[1]:.3f} Hz")
    print(f"wrote {FIGURE_PATH}")


if __name__ == "__main__":
    main()

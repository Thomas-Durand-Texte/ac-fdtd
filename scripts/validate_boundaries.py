"""M2 — do the boundaries do what they claim, and how well?

Three measurements, one figure:

1. **Absorbing walls.** A pulse down a one-cell duct, reflected off a wall of known admittance.
   The measured reflection coefficient should be ``(1 - xi) / (1 + xi)`` at every frequency,
   because a real admittance has no frequency dependence to hide behind.
2. **The absorbing layer.** At normal incidence the layer is not approximately transparent but
   *exactly* so: damping both equations by the same sigma keeps the local impedance at rho c,
   which decouples the two characteristics, so a right-going wave stays right-going whatever
   sigma does. The duct measurement confirms it at around -100 dB, and that is the least
   interesting thing about it — obliquity is not matched, and a real 3D field arrives at every
   angle. So the panel measures what actually limits it: the error the layer injects into a
   free-field simulation, against distance from the layer, isolated by re-running the same
   problem in a domain large enough that its own walls cannot answer within the window.
3. **Free field.** A monopole in a domain lined with the layer, against the analytical
   spherical Green's function ``p = rho Q'(t - r/c) / (4 pi r)``. This checks the propagation,
   the source calibration and the layer at once, which is also why it is the last one: it only
   isolates something if the first two already passed.

Run: ``uv run python scripts/validate_boundaries.py``  (about two minutes)
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

from ac_fdtd import (  # noqa: E402
    AIR,
    AbsorbingLayer,
    AcousticFDTD,
    Grid,
    WallAdmittances,
    reflection_coefficient,
)

FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m2_boundaries.svg"

# --- duct, shared by the wall and layer measurements -------------------------------------
DUCT_DX = 0.01
DUCT_CELLS = 600
DUCT_STEPS = 2600
PULSE_WIDTH = 20.0
SOURCE_X = 1.0
PROBE_X = 2.0
#: Start of the reflected-arrival gate, in steps. The incident pulse has gone by, the echo has
#: not yet reached the probe, and both windows are the same length so their spectra compare.
GATE_START = 1300
GATE_LENGTH = 1200
BAND = (60.0, 2000.0)
#: Incident level, relative to the peak of the excitation, below which a bin says nothing.
EXCITATION_FLOOR = 10 ** (-40 / 20)

TESTED_ADMITTANCES = (0.25, 0.5, 1.0)
TESTED_THICKNESSES = (8, 16, 32)

# --- free field ---------------------------------------------------------------------------
FREE_FIELD_DX = 0.01
REFERENCE_CELLS = 240
FREE_FIELD_STEPS = 400
#: Comparison stops before the reference box's own walls answer back.
FREE_FIELD_GATE = 380
#: Receiver distances, in cells rather than metres. A probe snaps to the cell containing it,
#: so a radius of 17.5 cells is measured at 17 and compared against the analytical solution at
#: 17.5 — which cost an apparent 7 % error before the separations were taken from the indices.
RADIUS_CELLS = tuple(range(5, 36, 3))
#: The three drawn as waveforms; the rest only carry the error-against-distance curve.
SHOWN_RADIUS_CELLS = (5, 20, 35)
PULSE_WIDTHS = (5.0, 14.0, 30.0)
LAYER_THICKNESSES = (16, 32)


def duct_response(walls: WallAdmittances, layer: AbsorbingLayer | None) -> np.ndarray:
    """Pressure at the probe of a one-cell-wide duct, for the whole run."""
    solver = AcousticFDTD(
        Grid(dx=DUCT_DX, nx=DUCT_CELLS, ny=1, nz=1), walls=walls, absorbing_layer=layer
    )
    index = np.arange(DUCT_STEPS)
    signal = np.exp(-(((index - 4 * PULSE_WIDTH) / PULSE_WIDTH) ** 2))
    solver.add_volume_source((SOURCE_X, 0.5 * DUCT_DX, 0.5 * DUCT_DX), signal)

    recorded = np.empty(DUCT_STEPS)
    for step in range(DUCT_STEPS):
        solver.step()
        recorded[step] = solver.probe_pressure((PROBE_X, 0.5 * DUCT_DX, 0.5 * DUCT_DX))
    return recorded, solver.dt


def reflection_spectrum(recorded: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Reflection magnitude against frequency, from the two time gates.

    Bins where the incident pulse carries nothing are dropped rather than plotted. A Gaussian
    pulse is down by 60 dB well before the Nyquist frequency, and dividing one numerical zero
    by another produced a "reflection coefficient" of 4000 the first time this ran — a reminder
    that the band a measurement is valid over is set by the excitation, not by the axis limits.
    """
    incident = np.fft.rfft(recorded[:GATE_LENGTH])
    reflected = np.fft.rfft(recorded[GATE_START : GATE_START + GATE_LENGTH])
    frequencies = np.fft.rfftfreq(GATE_LENGTH, dt)
    magnitude = np.abs(incident)
    band = (
        (frequencies >= BAND[0])
        & (frequencies <= BAND[1])
        & (magnitude >= EXCITATION_FLOOR * magnitude.max())
    )
    return frequencies[band], np.abs(reflected[band] / incident[band])


def free_field_run(cells: int, layer: AbsorbingLayer | None, width: float):
    """Monopole at the centre of a cubic domain, recorded at :data:`RADIUS_CELLS` along +x."""
    grid = Grid(dx=FREE_FIELD_DX, nx=cells, ny=cells, nz=cells)
    solver = AcousticFDTD(grid, absorbing_layer=layer)
    centre = 0.5 * grid.lengths[0]

    index = np.arange(FREE_FIELD_STEPS)
    signal = np.exp(-(((index - 4 * width) / width) ** 2))
    solver.add_volume_source((centre, centre, centre), signal)

    probes = [(centre + cells_away * grid.dx, centre, centre) for cells_away in RADIUS_CELLS]
    source_index = grid.cell_index((centre, centre, centre))[0]
    radii = np.array([(grid.cell_index(probe)[0] - source_index) * grid.dx for probe in probes])

    recorded = np.zeros((len(probes), FREE_FIELD_STEPS))
    for step in range(FREE_FIELD_STEPS):
        solver.step()
        for k, probe in enumerate(probes):
            recorded[k, step] = solver.probe_pressure(probe)
    return recorded, solver.dt, signal, radii


def analytical_monopole(radius: float, signal: np.ndarray, dt: float) -> np.ndarray:
    """``p = rho Q'(t - r/c) / (4 pi r)`` sampled on the recording instants.

    Two conventions have to line up here, and getting either wrong costs several percent that
    looks exactly like a physics error. The injected volume velocity is ``Q = q * dx^3``, since
    the source term adds ``rho c^2 dt q`` to one cell of volume ``dx^3``. And sample ``n`` of
    the signal acts at ``(n + 1/2) dt`` while the recording is at ``(n + 1) dt``, which is the
    half-step shift applied below — the *opposite* sign to the intuitive one.
    """
    strength = AIR.density * FREE_FIELD_DX**3 / (4.0 * np.pi * radius)
    derivative = np.gradient(signal, dt)
    instants = (np.arange(len(signal)) + 1) * dt
    retarded = instants - radius / AIR.sound_speed + 0.5 * dt
    return strength * np.interp(retarded, instants, derivative, left=0.0, right=0.0)


def main() -> None:
    plotstyle.apply()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # --- walls -------------------------------------------------------------------------
    ax = axes[0]
    rigid, dt = duct_response(WallAdmittances(x_min=1.0, x_max=0.0), None)
    frequencies, floor = reflection_spectrum(rigid, dt)
    for xi, colour in zip(TESTED_ADMITTANCES, plotstyle.SERIES, strict=True):
        recorded, dt = duct_response(WallAdmittances(x_min=1.0, x_max=xi), None)
        frequencies, measured = reflection_spectrum(recorded, dt)
        ax.plot(frequencies, measured, color=colour, label=f"$\\xi$ = {xi}")
        ax.axhline(
            abs(reflection_coefficient(xi)), color=plotstyle.MUTED, lw=1.0, ls="--", zorder=1
        )
    ax.set_xlim(*BAND)
    ax.set_ylim(0, 0.9)
    ax.set_xlabel("frequency  [Hz]")
    ax.set_ylabel("|R|")
    ax.set_title("Wall reflection vs theory")
    ax.legend(loc="upper right")

    # --- layer -------------------------------------------------------------------------
    reference, dt, signal, radii = free_field_run(REFERENCE_CELLS, None, PULSE_WIDTHS[-1])
    reference_peaks = np.max(np.abs(reference[:, :FREE_FIELD_GATE]), axis=1)

    ax = axes[1]
    layer_errors = {}
    for thickness, colour in zip(LAYER_THICKNESSES, plotstyle.SERIES, strict=False):
        layer = AbsorbingLayer(thickness=thickness, target_reflection=1e-5)
        cells = 2 * (thickness + max(RADIUS_CELLS) + 10)
        layered, _, _, _ = free_field_run(cells, layer, PULSE_WIDTHS[-1])
        difference = np.max(
            np.abs(layered[:, :FREE_FIELD_GATE] - reference[:, :FREE_FIELD_GATE]), axis=1
        )
        levels = 20 * np.log10(difference / reference_peaks)
        clearance = 0.5 * cells * FREE_FIELD_DX - thickness * FREE_FIELD_DX - radii
        layer_errors[thickness] = (clearance, levels)
        ax.plot(clearance / FREE_FIELD_DX, levels, "o-", color=colour, label=f"{thickness} cells")
    ax.set_xlabel("clear cells between receiver and layer")
    ax.set_ylabel("error injected by the layer  [dB]")
    ax.set_title("Obliquity, not thickness, is the limit")
    ax.legend(loc="upper right")

    # --- free field --------------------------------------------------------------------
    ax = axes[2]
    instants = (np.arange(FREE_FIELD_STEPS) + 1) * dt * 1e3
    errors = {}
    for k, radius in enumerate(radii):
        analytical = analytical_monopole(radius, signal, dt)
        errors[RADIUS_CELLS[k]] = np.max(
            np.abs(reference[k][:FREE_FIELD_GATE] - analytical[:FREE_FIELD_GATE])
        ) / np.max(np.abs(analytical[:FREE_FIELD_GATE]))
        if RADIUS_CELLS[k] not in SHOWN_RADIUS_CELLS:
            continue
        colour = plotstyle.SERIES[SHOWN_RADIUS_CELLS.index(RADIUS_CELLS[k])]
        # Scaled by r, so the 1/r decay is divided out and the three collapse onto one
        # waveform — which is what makes a discrepancy visible rather than merely small.
        ax.plot(
            instants,
            1e3 * radius * reference[k],
            color=colour,
            label=f"r = {radius * 100:.0f} cm",
        )
        ax.plot(
            instants,
            1e3 * radius * analytical,
            color=plotstyle.MUTED,
            lw=1.0,
            ls="--",
            zorder=1,
        )
    ax.plot([], [], color=plotstyle.MUTED, lw=1.0, ls="--", label="analytical")
    ax.set_xlim(0, instants[FREE_FIELD_GATE])
    ax.set_xlabel("time  [ms]")
    ax.set_ylabel("r x pressure  [mPa m]")
    ax.set_title(
        f"Free field: {100 * errors[max(RADIUS_CELLS)]:.2f} % error at {max(RADIUS_CELLS)} cells"
    )
    ax.legend(loc="upper left")

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, bbox_inches="tight")

    # --- numbers that do not fit in a picture ------------------------------------------
    print("wall reflection, mean over the band:")
    print(f"  rigid (measurement floor, should be 1.000): {floor.mean():.4f}")
    for xi in TESTED_ADMITTANCES:
        recorded, dt = duct_response(WallAdmittances(x_min=1.0, x_max=xi), None)
        _, measured = reflection_spectrum(recorded, dt)
        print(
            f"  xi = {xi:4.2f}: measured {measured.mean():.4f}, "
            f"theory {abs(reflection_coefficient(xi)):.4f}"
        )

    print("\nabsorbing layer at normal incidence, worst reflection over the band:")
    for thickness in TESTED_THICKNESSES:
        layer = AbsorbingLayer(thickness=thickness, target_reflection=1e-5, faces=((0, 1),))
        recorded, dt_duct = duct_response(WallAdmittances(x_min=1.0), layer)
        _, measured = reflection_spectrum(recorded, dt_duct)
        print(f"  {thickness:2d} cells: {20 * np.log10(measured.max()):6.1f} dB")

    print("\nlayer error in 3D, isolated against a large reference domain:")
    for thickness, (clearance, levels) in layer_errors.items():
        worst = int(np.argmin(clearance))
        best = int(np.argmax(clearance))
        print(
            f"  {thickness:2d} cells: {levels[worst]:.1f} dB at "
            f"{clearance[worst] / FREE_FIELD_DX:.0f} clear cells, "
            f"{levels[best]:.1f} dB at {clearance[best] / FREE_FIELD_DX:.0f}"
        )

    print("\nfree field vs the analytical Green's function:")
    for width in PULSE_WIDTHS:
        recorded, dt, signal, radii = free_field_run(REFERENCE_CELLS, None, width)
        spectrum = np.abs(np.fft.rfft(np.gradient(signal, dt)))
        frequencies = np.fft.rfftfreq(len(signal), dt)
        centroid = float((frequencies * spectrum).sum() / spectrum.sum())
        per_radius = []
        for k, radius in enumerate(radii):
            analytical = analytical_monopole(radius, signal, dt)
            per_radius.append(
                np.max(np.abs(recorded[k][:FREE_FIELD_GATE] - analytical[:FREE_FIELD_GATE]))
                / np.max(np.abs(analytical[:FREE_FIELD_GATE]))
            )
        points_per_wavelength = AIR.sound_speed / (centroid * FREE_FIELD_DX)
        print(
            f"  centroid {centroid:6.0f} Hz ({points_per_wavelength:5.1f} points/wavelength): "
            + ", ".join(
                f"r={cells_away} cells {100 * per_radius[RADIUS_CELLS.index(cells_away)]:.2f} %"
                for cells_away in SHOWN_RADIUS_CELLS
            )
        )

    print(f"\nwrote {FIGURE_PATH}  ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()

"""M3 — does the simulated air attenuate the way ISO 9613-1 says it should?

The measurement is a pulse down a long one-cell duct, recorded at two stations and differenced.
It works because the two error sources separate cleanly: numerical dispersion changes the
*phase* of a spectrum and not its magnitude, so a ratio of magnitude spectra sees the
attenuation and nothing else. What it cannot escape is resolution — the top of the band is
where the grid, not the physics, sets the answer, and the middle panel is there to show where
that starts.

Three humidities, because humidity is what the curve is most sensitive to: it moves the
nitrogen relaxation frequency by an order of magnitude and the 8 kHz attenuation by a factor
of four. Getting that dependence right is a stronger claim than matching one curve.

Run: ``uv run python scripts/validate_air_absorption.py``  (under a minute)
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

from ac_fdtd import AIR, AcousticFDTD, AirAbsorption, Grid, WallAdmittances  # noqa: E402

FIGURE_PATH = Path(__file__).resolve().parents[1] / "docs" / "figures" / "m3_air_absorption.svg"

DX = 0.002
DUCT_CELLS = 10_000
STEPS = 18_000
PULSE_WIDTH = 6.0
SOURCE_X = 1.0
NEAR_X = 2.0
FAR_X = 18.0

TEMPERATURE = 20.0
HUMIDITIES = (20.0, 50.0, 80.0)
#: Where the measurement is trustworthy: below this the loss over 16 m is under a hundredth of
#: a decibel and the ratio is measuring round-off, above it the grid runs out of resolution.
BAND = (200.0, 16000.0)
#: Loss over the baseline, in dB, below which the ratio measures round-off rather than air.
MEASURABLE_LOSS = 0.05
MARKER_FREQUENCIES = (250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, 16000.0)
DISTANCES = (5.0, 20.0, 100.0)


def measure_attenuation(air: AirAbsorption) -> tuple[np.ndarray, np.ndarray, float]:
    """Attenuation against frequency, in dB/m, from two stations along a duct."""
    solver = AcousticFDTD(
        Grid(dx=DX, nx=DUCT_CELLS, ny=1, nz=1),
        walls=WallAdmittances(x_min=1.0, x_max=1.0),
        air_absorption=air,
    )
    index = np.arange(STEPS)
    signal = np.exp(-(((index - 4 * PULSE_WIDTH) / PULSE_WIDTH) ** 2))
    solver.add_volume_source((SOURCE_X, DX / 2, DX / 2), signal)

    recorded = np.zeros((2, STEPS))
    for step in range(STEPS):
        solver.step()
        recorded[0, step] = solver.probe_pressure((NEAR_X, DX / 2, DX / 2))
        recorded[1, step] = solver.probe_pressure((FAR_X, DX / 2, DX / 2))

    spectra = np.abs(np.fft.rfft(recorded, axis=1))
    frequencies = np.fft.rfftfreq(STEPS, solver.dt)
    attenuation = -20.0 * np.log10(spectra[1] / spectra[0]) / (FAR_X - NEAR_X)
    return frequencies, attenuation, solver.dt


def at_frequencies(frequencies: np.ndarray, values: np.ndarray, wanted) -> np.ndarray:
    """Nearest available bin for each requested frequency."""
    return np.array([values[int(np.argmin(np.abs(frequencies - f)))] for f in wanted])


def main() -> None:
    plotstyle.apply()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    measurements = {}
    for humidity in HUMIDITIES:
        air = AirAbsorption(temperature=TEMPERATURE, relative_humidity=humidity)
        frequencies, attenuation, dt = measure_attenuation(air)
        measurements[humidity] = (air, frequencies, attenuation, dt)

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    smooth = np.geomspace(*BAND, 300)

    # --- the curve ---------------------------------------------------------------------
    ax = axes[0]
    for (humidity, (air, frequencies, attenuation, _)), colour in zip(
        measurements.items(), plotstyle.SERIES, strict=True
    ):
        ax.loglog(smooth, 1e3 * air.attenuation(smooth), color=colour, label=f"{humidity:.0f} % RH")
        ax.loglog(
            MARKER_FREQUENCIES,
            1e3 * at_frequencies(frequencies, attenuation, MARKER_FREQUENCIES),
            "o",
            color=colour,
            markerfacecolor="white",
            markeredgewidth=1.6,
        )
    ax.plot([], [], "o", color=plotstyle.MUTED, markerfacecolor="white", label="simulated")
    ax.set_xlabel("frequency  [Hz]")
    ax.set_ylabel("attenuation  [dB/km]")
    ax.set_title(f"ISO 9613-1 at {TEMPERATURE:.0f} °C, lines; simulation, circles")
    ax.legend(loc="upper left")

    # --- where the grid gives up -------------------------------------------------------
    ax = axes[1]
    worst_in_band = 0.0
    for (_humidity, (air, frequencies, attenuation, _)), colour in zip(
        measurements.items(), plotstyle.SERIES, strict=True
    ):
        # Restricted to where there is something to measure: at 250 Hz the loss over the
        # 16 m baseline is under a hundredth of a decibel, and a percentage error on that is a
        # statement about round-off, not about the medium.
        band = (
            (frequencies >= BAND[0])
            & (frequencies <= BAND[1])
            & (air.attenuation(frequencies) * (FAR_X - NEAR_X) > MEASURABLE_LOSS)
        )
        reference = air.attenuation(frequencies[band])
        ratio = attenuation[band] / reference
        points_per_wavelength = AIR.sound_speed / (frequencies[band] * DX)
        ax.semilogx(points_per_wavelength, 100 * (ratio - 1.0), color=colour, lw=1.4)
        resolved = points_per_wavelength >= 20.0
        worst_in_band = max(worst_in_band, np.max(np.abs(100 * (ratio[resolved] - 1.0))))
    ax.axhline(0.0, color=plotstyle.MUTED, lw=1.0, ls="--")
    ax.axvspan(4.0, 20.0, color=plotstyle.GRID, alpha=0.5, zorder=0)
    ax.text(
        14.0,
        4.5,
        "under 20 points\nper wavelength",
        fontsize=8,
        color=plotstyle.MUTED,
        ha="center",
    )
    ax.set_xlim(400, 9)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("points per wavelength")
    ax.set_ylabel("error against ISO  [%]")
    ax.set_title(f"Within {worst_in_band:.1f} % wherever the grid resolves it")
    ax.text(
        0.03,
        0.04,
        "all three humidities, indistinguishable:\nwhat is left is the grid, not the air",
        transform=ax.transAxes,
        fontsize=8,
        color=plotstyle.MUTED,
    )

    # --- what it is worth --------------------------------------------------------------
    ax = axes[2]
    air = AirAbsorption(temperature=TEMPERATURE, relative_humidity=50.0)
    for distance, colour in zip(DISTANCES, plotstyle.SERIES, strict=True):
        ax.semilogx(
            smooth, distance * air.attenuation(smooth), color=colour, label=f"{distance:.0f} m"
        )
    ax.axhline(1.0, color=plotstyle.MUTED, lw=1.0, ls="--")
    ax.text(BAND[0] * 1.2, 1.15, "1 dB", fontsize=8, color=plotstyle.MUTED)
    ax.set_ylim(0, 12)
    ax.set_xlabel("frequency  [Hz]")
    ax.set_ylabel("level lost along the path  [dB]")
    ax.set_title("Why it is worth two extra fields")
    ax.legend(loc="upper left")

    figure.tight_layout()
    figure.savefig(FIGURE_PATH, bbox_inches="tight")

    print(
        f"grid {DX * 1e3:.0f} mm, {DUCT_CELLS} cells, {STEPS} steps, "
        f"{FAR_X - NEAR_X:.0f} m between stations\n"
    )
    for humidity, (air, frequencies, attenuation, _dt) in measurements.items():
        print(
            f"{humidity:.0f} % RH   (F_rN = {air.nitrogen_relaxation_frequency:.0f} Hz, "
            f"F_rO = {air.oxygen_relaxation_frequency / 1e3:.1f} kHz)"
        )
        print("     f [Hz]   measured    ISO 9613-1   ratio   points/wavelength")
        measured = at_frequencies(frequencies, attenuation, MARKER_FREQUENCIES)
        for frequency, value in zip(MARKER_FREQUENCIES, measured, strict=True):
            reference = float(air.attenuation(frequency))
            print(
                f"  {frequency:9.0f} {1e3 * value:10.2f} {1e3 * reference:12.2f} "
                f"{value / reference:8.3f} {AIR.sound_speed / (frequency * DX):15.1f}"
            )
        print()

    print(f"wrote {FIGURE_PATH}  ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()

"""Air absorption: the ISO formula, the medium built from it, and what the grid does to it."""

import numpy as np
import pytest

from ac_fdtd.air import AirAbsorption
from ac_fdtd.boundaries import WallAdmittances
from ac_fdtd.grid import Grid
from ac_fdtd.medium import AIR
from ac_fdtd.scheme import AcousticFDTD

FREQUENCIES = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])


def test_relaxation_frequencies_land_where_they_should():
    """Nitrogen in the hundreds of hertz, oxygen in the tens of kilohertz.

    This is the whole reason air absorption matters for rooms and not just for long-range
    outdoor propagation: the nitrogen process sits inside the audio band.
    """
    air = AirAbsorption(temperature=20.0, relative_humidity=50.0)
    assert 100.0 < air.nitrogen_relaxation_frequency < 1000.0
    assert 10e3 < air.oxygen_relaxation_frequency < 100e3


def test_attenuation_matches_the_familiar_landmarks():
    """20 °C, 50 % RH: about 5 dB/km at 1 kHz and about 100 dB/km at 8 kHz."""
    air = AirAbsorption(temperature=20.0, relative_humidity=50.0)
    per_kilometre = 1000.0 * air.attenuation(np.array([1000.0, 8000.0]))
    assert 4.0 < per_kilometre[0] < 6.0
    assert 80.0 < per_kilometre[1] < 130.0


def test_attenuation_rises_with_frequency_and_falls_with_humidity():
    air_dry = AirAbsorption(temperature=20.0, relative_humidity=10.0)
    air_damp = AirAbsorption(temperature=20.0, relative_humidity=80.0)
    dry = air_dry.attenuation(FREQUENCIES)
    damp = air_damp.attenuation(FREQUENCIES)

    assert np.all(np.diff(dry) > 0.0)
    assert np.all(np.diff(damp) > 0.0)
    # Dry air is the lossy case in the upper band, by a wide margin.
    assert dry[-1] > 2.0 * damp[-1]


def test_relaxation_dominates_the_audio_band():
    """The classical term alone is not a cheaper approximation, it is the wrong size."""
    full = AirAbsorption(temperature=20.0, relative_humidity=50.0)
    classical_only = AirAbsorption(temperature=20.0, relative_humidity=50.0, relaxation=False)
    ratio = full.attenuation(FREQUENCIES) / classical_only.attenuation(FREQUENCIES)
    assert np.all(ratio > 5.0)


def test_the_implemented_medium_reproduces_the_standard():
    """No fitting step: the relaxation strengths come straight from the ISO coefficients."""
    air = AirAbsorption(temperature=20.0, relative_humidity=50.0)
    dt = 3.4e-6
    frequencies = np.geomspace(50.0, 16000.0, 60)
    reference = air.attenuation(frequencies)
    model = air.model_attenuation(frequencies, AIR.sound_speed, dt)
    assert np.max(np.abs(model / reference - 1.0)) < 0.01


def test_sound_speed_dispersion_is_small_but_present():
    air = AirAbsorption(temperature=20.0, relative_humidity=50.0)
    strength = air.total_strength(AIR.sound_speed, 3.4e-6)
    assert 1e-4 < strength < 1e-2


def test_no_absorption_leaves_the_scheme_untouched():
    grid = Grid.from_lengths((0.4, 0.35, 0.3), 0.05)
    plain = AcousticFDTD(grid)
    disabled = AcousticFDTD(grid, air_absorption=AirAbsorption(classical=False, relaxation=False))
    for solver in (plain, disabled):
        solver.p[:] = np.random.default_rng(5).standard_normal(grid.shape)
    plain.run(40)
    disabled.run(40)
    assert np.array_equal(plain.p, disabled.p)


def test_absorbing_air_removes_energy():
    grid = Grid.from_lengths((0.4, 0.35, 0.3), 0.02)
    solver = AcousticFDTD(grid, air_absorption=AirAbsorption())
    solver.p[:] = np.random.default_rng(6).standard_normal(grid.shape)

    initial = solver.energy()
    solver.run(400)
    assert 0.0 < solver.energy() < initial


@pytest.mark.parametrize("frequency", [2000.0, 4000.0])
def test_measured_attenuation_matches_iso(frequency):
    """A pulse down a long duct, measured at two stations and differenced.

    Pure numerical dispersion changes the phase of the spectrum and not its magnitude, so the
    ratio of two magnitude spectra isolates the attenuation from it. What the grid does limit
    is the top of the band, which is why this stops at 4 kHz on a 3 mm grid.
    """
    dx = 0.003
    air = AirAbsorption(temperature=20.0, relative_humidity=50.0)
    solver = AcousticFDTD(
        Grid(dx=dx, nx=3000, ny=1, nz=1),
        walls=WallAdmittances(x_min=1.0, x_max=1.0),
        air_absorption=air,
    )

    n_steps, width = 5200, 4.0
    index = np.arange(n_steps)
    solver.add_volume_source((0.5, dx / 2, dx / 2), np.exp(-(((index - 4 * width) / width) ** 2)))

    near, far = 1.5, 8.0
    recorded = np.zeros((2, n_steps))
    for step in range(n_steps):
        solver.step()
        recorded[0, step] = solver.probe_pressure((near, dx / 2, dx / 2))
        recorded[1, step] = solver.probe_pressure((far, dx / 2, dx / 2))

    spectra = np.abs(np.fft.rfft(recorded, axis=1))
    frequencies = np.fft.rfftfreq(n_steps, solver.dt)
    bin_index = int(np.argmin(np.abs(frequencies - frequency)))
    measured = -20.0 * np.log10(spectra[1, bin_index] / spectra[0, bin_index]) / (far - near)

    expected = float(air.attenuation(frequencies[bin_index]))
    assert measured == pytest.approx(expected, rel=0.05)

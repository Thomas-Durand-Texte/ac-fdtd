"""Absorbing walls and the absorbing layer: do they absorb, and do they stay passive?"""

import numpy as np
import pytest

from ac_fdtd.boundaries import (
    AbsorbingLayer,
    WallAdmittances,
    admittance_from_absorption,
    layer_factors,
    reflection_coefficient,
    wall_update_coefficients,
)
from ac_fdtd.grid import Grid
from ac_fdtd.medium import AIR
from ac_fdtd.scheme import AcousticFDTD


def _noisy_box(walls=None, layer=None, dx=0.05, seed=3):
    solver = AcousticFDTD(
        Grid.from_lengths((0.8, 0.65, 0.55), dx), walls=walls, absorbing_layer=layer
    )
    solver.p[:] = np.random.default_rng(seed).standard_normal(solver.grid.shape)
    return solver


def test_absorption_and_admittance_are_inverse():
    for absorption in (0.0, 0.1, 0.5, 0.9, 1.0):
        xi = admittance_from_absorption(absorption)
        reflection = reflection_coefficient(xi)
        assert 1.0 - reflection**2 == pytest.approx(absorption)


def test_a_rigid_wall_is_full_reflection_and_a_matched_wall_is_none():
    assert reflection_coefficient(0.0) == pytest.approx(1.0)
    assert reflection_coefficient(1.0) == pytest.approx(0.0)


def test_negative_admittance_is_refused():
    with pytest.raises(ValueError, match="generates sound"):
        WallAdmittances(x_min=-0.1)


def test_zero_admittance_reproduces_the_rigid_solver_exactly():
    rigid = _noisy_box()
    explicit = _noisy_box(walls=WallAdmittances.uniform(0.0))
    rigid.run(50)
    explicit.run(50)
    assert np.array_equal(rigid.p, explicit.p)


def test_corner_cells_sum_the_contribution_of_every_face_they_touch():
    grid = Grid.from_lengths((0.3, 0.3, 0.3), 0.1)
    walls = WallAdmittances.uniform(1.0)
    indices, coefficient = wall_update_coefficients(grid, AIR, 1e-5, walls)

    # Of the 27 cells only the middle one touches nothing, and no cell is listed twice.
    assert indices.size == 26
    assert np.array_equal(indices, np.unique(indices))

    # A corner touches three faces, an edge two, a face one; the face centre is the unit.
    unit = coefficient.min()
    counts = np.round(coefficient / unit).astype(int)
    assert list(np.bincount(counts)[1:]) == [6, 12, 8]  # 6 faces, 12 edges, 8 corners


@pytest.mark.parametrize("xi", [0.1, 0.5, 1.0, 3.0])
def test_absorbing_walls_are_passive(xi):
    """Energy may only decrease. This is the property the time-centred form was chosen for."""
    solver = _noisy_box(walls=WallAdmittances.uniform(xi))
    energies = [solver.energy()]
    for _ in range(60):
        solver.run(10)
        energies.append(solver.energy())

    energies = np.array(energies)
    assert np.all(np.diff(energies) <= 0.0)
    assert energies[-1] < 0.5 * energies[0]


def test_the_absorbing_layer_is_passive_and_empties_the_domain():
    solver = _noisy_box(layer=AbsorbingLayer(thickness=6, target_reflection=1e-4))
    energies = [solver.energy()]
    for _ in range(60):
        solver.run(10)
        energies.append(solver.energy())

    energies = np.array(energies)
    assert np.all(np.diff(energies) <= 0.0)
    assert energies[-1] < 1e-6 * energies[0]


def test_the_layer_leaves_the_interior_alone():
    grid = Grid.from_lengths((1.0, 1.0, 1.0), 0.05)
    layer = AbsorbingLayer(thickness=4)
    pressure_factors, velocity_factors = layer_factors(grid, AIR, 1e-5, layer)
    for axis in range(3):
        assert np.all(pressure_factors[axis][4:-4] == 1.0)
        assert np.any(pressure_factors[axis][:4] < 1.0)
        assert np.all(velocity_factors[axis][axis][5:-5] == 1.0)


def test_a_layer_can_line_a_subset_of_the_walls():
    grid = Grid.from_lengths((1.0, 1.0, 1.0), 0.05)
    layer = AbsorbingLayer(thickness=4, faces=((0, 1),))
    pressure_factors, _ = layer_factors(grid, AIR, 1e-5, layer)
    assert np.all(pressure_factors[0][:-4] == 1.0)
    assert np.any(pressure_factors[0][-4:] < 1.0)
    assert np.all(pressure_factors[1] == 1.0)
    assert np.all(pressure_factors[2] == 1.0)


@pytest.mark.parametrize(
    ("xi", "expected"), [(0.0, 1.0), (0.25, 0.6), (0.5, 1.0 / 3.0), (1.0, 0.0)]
)
def test_normal_incidence_reflection_matches_theory(xi, expected):
    """A pulse down a one-cell-wide duct, reflected off the far wall and measured.

    The near wall is impedance-matched so that the leftward half of the source's radiation is
    swallowed instead of coming back as a second echo through the measurement window.
    """
    dx = 0.01
    solver = AcousticFDTD(
        Grid(dx=dx, nx=600, ny=1, nz=1), walls=WallAdmittances(x_min=1.0, x_max=xi)
    )
    n_steps, width = 2600, 60.0
    index = np.arange(n_steps)
    solver.add_volume_source((1.0, 0.005, 0.005), np.exp(-(((index - 4 * width) / width) ** 2)))

    recorded = np.empty(n_steps)
    for step in range(n_steps):
        solver.step()
        recorded[step] = solver.probe_pressure((2.0, 0.005, 0.005))

    window = 1200
    incident = np.fft.rfft(recorded[:window])
    reflected = np.fft.rfft(recorded[1300 : 1300 + window])
    frequencies = np.fft.rfftfreq(window, solver.dt)
    band = (frequencies > 80.0) & (frequencies < 800.0)
    measured = np.mean(np.abs(reflected[band] / incident[band]))

    # The rigid case measures 0.993 rather than 1.000, which is the accuracy of the time
    # gating itself, not of the boundary; 3 % covers that floor for every case.
    assert measured == pytest.approx(expected, abs=0.03)

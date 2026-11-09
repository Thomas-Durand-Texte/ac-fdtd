"""Correctness of the 2D lossless scheme, measured against things with exact answers."""

import math

import numpy as np
import pytest

from ac_fdtd.analytic2d import (
    discrete_mode_frequency,
    mode_frequency,
    mode_initial_state,
    mode_pressure_field,
)
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.medium import AIR
from ac_fdtd.scheme2d import AcousticFDTD2D


def _box(dx=0.05, lengths=(0.8, 0.65), courant=1.0):
    return AcousticFDTD2D(Grid2D.from_lengths(lengths, dx), medium=AIR, courant=courant)


def test_rigid_walls_are_never_written_to():
    solver = _box()
    solver.p[:] = np.random.default_rng(0).standard_normal(solver.grid.shape)
    solver.run(20)
    for axis in range(2):
        wall = solver.velocity[axis]
        lower = tuple(0 if a == axis else slice(None) for a in range(2))
        upper = tuple(-1 if a == axis else slice(None) for a in range(2))
        assert np.all(wall[lower] == 0.0)
        assert np.all(wall[upper] == 0.0)


@pytest.mark.parametrize("courant", [1.0, 0.9, 0.5])
def test_energy_is_conserved_to_round_off(courant):
    """The sharpest available check: drift here means a bug, not a modelling choice."""
    solver = _box(courant=courant)
    solver.p[:] = np.random.default_rng(1).standard_normal(solver.grid.shape)

    initial = solver.energy()
    assert initial > 0.0

    drift = []
    for _ in range(40):
        solver.run(50)
        drift.append(abs(solver.energy() - initial) / initial)

    assert max(drift) < 1e-11


@pytest.mark.parametrize("mode", [(1, 0), (1, 1), (2, 1)])
def test_a_single_mode_stays_exactly_sinusoidal(mode):
    """Started from one discrete eigenvector, the field stays that mode at its own frequency."""
    grid = Grid2D.from_lengths((0.8, 0.65), 0.02)
    solver = AcousticFDTD2D(grid, medium=AIR)
    pressure, velocity = mode_initial_state(grid, mode, AIR, solver.dt)
    solver.p[:] = pressure
    for axis in range(2):
        solver.velocity[axis][:] = velocity[axis]

    omega = 2.0 * math.pi * discrete_mode_frequency(grid, mode, AIR, solver.dt)
    field = mode_pressure_field(grid, mode)
    for _ in range(300):
        solver.step()
        expected = math.cos(omega * solver.time) * field
        assert np.max(np.abs(solver.p - expected)) < 1e-8


def test_discrete_frequency_approaches_the_analytic_one_at_second_order():
    mode = (2, 1)
    lengths = (1.6, 1.2)
    errors = []
    for dx in (0.05, 0.025, 0.0125):
        grid = Grid2D.from_lengths(lengths, dx)
        dt = grid.dx / (AIR.sound_speed * math.sqrt(2))
        exact = mode_frequency(grid.lengths, mode, AIR.sound_speed)
        numerical = discrete_mode_frequency(grid, mode, AIR, dt)
        errors.append(abs(numerical - exact) / exact)

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(3.5 < ratio < 4.5 for ratio in ratios), (errors, ratios)


def test_a_soft_source_conserves_nothing_but_stays_bounded():
    solver = _box()
    n_steps = 400
    signal = np.zeros(n_steps)
    signal[:20] = np.hanning(20)
    solver.add_volume_source((0.3, 0.3), signal)
    solver.run(n_steps)

    assert np.isfinite(solver.p).all()
    assert solver.energy() > 0.0
    assert np.max(np.abs(solver.p)) < 1e6


def test_a_prescribed_pressure_source_holds_its_node_then_frees_it():
    solver = _box()
    signal = np.random.default_rng(3).standard_normal(15)
    point = (0.4, 0.3)
    solver.add_prescribed_source(point, signal, kind="pressure")
    index = solver.grid.cell_index(point)

    for k in range(signal.size):
        solver.step()
        assert solver.p[index] == pytest.approx(signal[k]), k

    held = solver.p[index]
    values = []
    for _ in range(5):
        solver.step()
        values.append(solver.p[index])
    assert any(v != held for v in values)


def test_a_prescribed_velocity_source_drives_the_field_from_rest():
    solver = _box()
    signal = np.full(10, 0.01)
    point = (0.4, 0.3)
    axis = 0
    solver.add_prescribed_source(point, signal, kind="velocity", axis=axis)
    index = solver.grid.cell_index(point)

    assert solver.energy() == 0.0
    solver.step()
    assert solver.velocity[axis][index] == pytest.approx(signal[0])
    assert np.any(solver.p != 0.0)

    solver.run(20)
    assert np.isfinite(solver.p).all()
    assert solver.energy() > 0.0


def test_a_prescribed_source_is_inert_outside_its_window():
    solver = _box()
    signal = np.array([1.0, 1.0, 1.0])
    point = (0.4, 0.3)
    solver.add_prescribed_source(point, signal, start=5, kind="pressure")
    index = solver.grid.cell_index(point)

    solver.run(5)
    assert solver.energy() == 0.0
    assert solver.p[index] == 0.0

    solver.step()
    assert solver.p[index] == pytest.approx(1.0)


def test_prescribed_source_rejects_contradictory_arguments():
    solver = _box()
    signal = np.ones(4)
    point = (0.1, 0.1)
    with pytest.raises(ValueError):
        solver.add_prescribed_source(point, signal, kind="velocity")
    with pytest.raises(ValueError):
        solver.add_prescribed_source(point, signal, kind="pressure", axis=0)
    with pytest.raises(ValueError):
        solver.add_prescribed_source(point, signal, kind="nonsense")
    with pytest.raises(ValueError):
        solver.add_prescribed_source(point, signal, start=-1)

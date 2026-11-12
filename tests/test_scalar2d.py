"""The pressure-only 2D scheme is the first-order one with velocity gone — so it must prove it.

Every physical case here is checked bit-for-bit against :class:`AcousticFDTD2D`: eliminating the
velocity is algebra, not approximation. The correspondence is exact for a matching start —
``p^{-1} = p^0`` is the first-order ``v^{-1/2} = 0`` — so the free-decay cases seed a pressure at
rest and the wall cases drive from silence (a nonzero pressure held against an *absorbing* wall is
not a state the three-level wall recurrence is consistent with, and would show an O(a) one-step
transient on the wall ring only; irrelevant to any driven run).
"""

import numpy as np
import pytest

from ac_fdtd import analytic2d
from ac_fdtd.boundaries2d import WallAdmittances2D
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scalar2d import ScalarFDTD2D
from ac_fdtd.scheme2d import AcousticFDTD2D

GRID = Grid2D.from_lengths((1.3, 1.1), 0.02)


def _ricker(n: int) -> np.ndarray:
    t = np.arange(n)
    return (1.0 - 0.5 * ((t - 60) / 20) ** 2) * np.exp(-(((t - 60) / 20) ** 2) / 2.0)


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


@pytest.mark.parametrize("steps", [1, 2, 7, 300])
def test_rigid_box_reproduces_the_first_order_scheme(steps):
    field = np.random.default_rng(0).standard_normal(GRID.shape)
    reference = AcousticFDTD2D(GRID)
    reference.p[:] = field
    variant = ScalarFDTD2D(GRID)
    variant.set_initial_pressure(field)

    reference.run(steps)
    variant.run(steps)
    np.testing.assert_allclose(variant.p, reference.p, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    "walls",
    [
        WallAdmittances2D.matched(),
        WallAdmittances2D.from_absorption(0.3),
        WallAdmittances2D(x_min=0.0, x_max=0.8, y_min=0.2, y_max=1.0),
    ],
)
def test_impedance_walls_match_when_driven_from_rest(walls):
    """Driven from silence, the three-level wall relation is bit-identical to the first-order."""
    signal = _ricker(400)
    reference = AcousticFDTD2D(GRID, walls=walls)
    variant = ScalarFDTD2D(GRID, walls=walls)
    reference.add_volume_source((0.4, 0.5), signal)
    variant.add_volume_source((0.4, 0.5), signal)

    reference.run(400)
    variant.run(400)
    assert _max_rel(variant.p, reference.p) < 1e-10


def test_rigid_obstacle_matches():
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[20:30, 15:18] = True
    field = np.random.default_rng(1).standard_normal(GRID.shape) * (~rigid)
    reference = AcousticFDTD2D(GRID, rigid_cells=rigid)
    reference.p[:] = field
    variant = ScalarFDTD2D(GRID, rigid_cells=rigid)
    variant.set_initial_pressure(field)

    reference.run(300)
    variant.run(300)
    assert _max_rel(variant.p, reference.p) < 1e-10
    assert np.all(variant.p[rigid] == 0.0)


def test_obstacle_with_absorbing_walls_driven():
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[40:44, 20:40] = True
    walls = WallAdmittances2D.from_absorption(0.4)
    signal = _ricker(400)
    reference = AcousticFDTD2D(GRID, walls=walls, rigid_cells=rigid)
    variant = ScalarFDTD2D(GRID, walls=walls, rigid_cells=rigid)
    reference.add_volume_source((0.3, 0.6), signal)
    variant.add_volume_source((0.3, 0.6), signal)

    reference.run(400)
    variant.run(400)
    assert _max_rel(variant.p, reference.p) < 1e-10


def test_prescribed_pressure_matches():
    signal = _ricker(400)
    reference = AcousticFDTD2D(GRID)
    variant = ScalarFDTD2D(GRID)
    reference.add_prescribed_source((0.5, 0.5), signal, start=5, kind="pressure")
    variant.add_prescribed_source((0.5, 0.5), signal, start=5, kind="pressure")

    reference.run(400)
    variant.run(400)
    assert _max_rel(variant.p, reference.p) < 1e-10


def test_moving_start_single_eigenmode_matches():
    mode = (2, 1)
    reference = AcousticFDTD2D(GRID)
    pressure, velocity = analytic2d.mode_initial_state(GRID, mode, reference.medium, reference.dt)
    reference.p[:] = pressure
    reference.velocity[0][:] = velocity[0]
    reference.velocity[1][:] = velocity[1]
    variant = ScalarFDTD2D(GRID)
    variant.set_initial_state(pressure, velocity)

    reference.run(300)
    variant.run(300)
    assert _max_rel(variant.p, reference.p) < 1e-10


def test_eigenmode_stays_a_pure_sinusoid():
    """A cosine mode is an exact eigenvector of the discrete rigid Laplacian, so the scheme must
    hold it as a single sinusoid at exactly the discrete eigenfrequency — the sharpest physics test.
    """
    grid = Grid2D.from_lengths((1.37, 1.09), 0.015)
    variant = ScalarFDTD2D(grid)
    mode = (2, 3)
    frequency = analytic2d.discrete_mode_frequency(grid, mode, variant.medium, variant.dt)
    pressure = analytic2d.mode_pressure_field(grid, mode)
    _, velocity = analytic2d.mode_initial_state(grid, mode, variant.medium, variant.dt)
    variant.set_initial_state(pressure, velocity)
    channel = variant.add_pressure_receiver((0.0075, 0.0075))

    steps = 3000
    variant.run(steps)
    recording = variant.recorded_pressure[channel]
    time = (np.arange(1, steps + 1)) * variant.dt
    omega = 2.0 * np.pi * frequency
    basis = np.column_stack([np.cos(omega * time), np.sin(omega * time)])
    coefficients, *_ = np.linalg.lstsq(basis, recording, rcond=None)
    residual = np.max(np.abs(recording - basis @ coefficients)) / np.hypot(*coefficients)
    assert residual < 1e-9


def test_velocity_receiver_matches_the_reference_probe():
    signal = _ricker(300)
    reference = AcousticFDTD2D(GRID)
    variant = ScalarFDTD2D(GRID)
    reference.add_volume_source((0.4, 0.5), signal)
    variant.add_volume_source((0.4, 0.5), signal)
    point = (0.7, 0.45)
    channel = variant.add_velocity_receiver(point, 0)

    expected = []
    for _ in range(250):
        expected.append(reference.probe_velocity(point, 0))
        reference.step()
        variant.step()
    assert _max_rel(variant.recorded_velocity[channel], np.array(expected)) < 1e-10


def test_volume_source_needs_the_difference_not_the_sample():
    """Guard the three-level source rule: injecting S_n (not S_n - S_{n-1}) drifts badly."""
    signal = _ricker(200)
    reference = AcousticFDTD2D(GRID)
    variant = ScalarFDTD2D(GRID)
    reference.add_volume_source((0.5, 0.55), signal)
    variant.add_volume_source((0.5, 0.55), signal)
    reference.run(200)
    variant.run(200)
    assert _max_rel(variant.p, reference.p) < 1e-10


def test_stores_two_arrays_not_four():
    variant = ScalarFDTD2D(GRID)
    cell_sized = [
        value
        for value in vars(variant).values()
        if isinstance(value, np.ndarray) and value.shape == GRID.shape
    ]
    assert len(cell_sized) == 2  # p and p_old; no velocity grid

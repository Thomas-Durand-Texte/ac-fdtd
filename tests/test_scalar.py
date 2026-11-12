"""The pressure-only 3D scheme must reproduce the first-order one (bit for bit) and absorb at the
open boundary. It is the dimension where dropping the velocity matters most — two arrays, not four.
"""

import numpy as np
import pytest

from ac_fdtd import analytic
from ac_fdtd.boundaries import WallAdmittances
from ac_fdtd.cpml import AbsorbingBoundary
from ac_fdtd.grid import Grid
from ac_fdtd.scalar import ScalarFDTD
from ac_fdtd.scheme import AcousticFDTD

GRID = Grid.from_lengths((0.6, 0.5, 0.45), 0.03)


def _ricker(n: int) -> np.ndarray:
    t = np.arange(n)
    return (1.0 - 0.5 * ((t - 40) / 14) ** 2) * np.exp(-(((t - 40) / 14) ** 2) / 2.0)


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


@pytest.mark.parametrize("steps", [1, 3, 200])
def test_rigid_box_reproduces_the_first_order_scheme(steps):
    field = np.random.default_rng(0).standard_normal(GRID.shape)
    reference = AcousticFDTD(GRID)
    reference.p[:] = field
    variant = ScalarFDTD(GRID)
    variant.set_initial_pressure(field)
    reference.run(steps)
    variant.run(steps)
    np.testing.assert_allclose(variant.p, reference.p, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    "walls",
    [
        WallAdmittances.matched(),
        WallAdmittances.from_absorption(0.3),
        WallAdmittances(x_min=0.0, x_max=0.7, y_min=0.2, y_max=0.0, z_min=0.5, z_max=0.9),
    ],
)
def test_impedance_walls_match_when_driven_from_rest(walls):
    signal = _ricker(300)
    reference = AcousticFDTD(GRID, walls=walls)
    variant = ScalarFDTD(GRID, walls=walls)
    reference.add_volume_source((0.3, 0.25, 0.2), signal)
    variant.add_volume_source((0.3, 0.25, 0.2), signal)
    reference.run(300)
    variant.run(300)
    assert _max_rel(variant.p, reference.p) < 1e-10


def test_moving_eigenmode_matches_and_stays_pure():
    mode = (1, 1, 1)
    reference = AcousticFDTD(GRID)
    pressure, velocity = analytic.mode_initial_state(GRID, mode, reference.medium, reference.dt)
    reference.p[:] = pressure
    for axis in range(3):
        reference.velocity[axis][:] = velocity[axis]
    variant = ScalarFDTD(GRID)
    variant.set_initial_state(pressure, velocity)
    reference.run(250)
    variant.run(250)
    assert _max_rel(variant.p, reference.p) < 1e-10


def test_prescribed_pressure_and_velocity_receiver_match():
    signal = _ricker(300)
    reference = AcousticFDTD(GRID)
    variant = ScalarFDTD(GRID)
    reference.add_volume_source((0.3, 0.25, 0.2), signal)
    variant.add_volume_source((0.3, 0.25, 0.2), signal)
    point = (0.4, 0.3, 0.25)
    channel = variant.add_velocity_receiver(point, 0)
    expected = []
    for _ in range(200):
        expected.append(reference.probe_velocity(point, 0))
        reference.step()
        variant.step()
    assert _max_rel(variant.recorded_velocity[channel], np.array(expected)) < 1e-10


def test_rigid_obstacle_stays_solid_and_stable():
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[6:10, 6:10, 4:8] = True
    field = np.random.default_rng(1).standard_normal(GRID.shape) * (~rigid)
    variant = ScalarFDTD(GRID, rigid_cells=rigid)
    variant.set_initial_pressure(field)
    variant.p[rigid] = 0.0
    variant.p_old[rigid] = 0.0
    start = float(np.sum(variant.p**2))
    variant.run(400)
    assert np.all(variant.p[rigid] == 0.0)
    assert np.isfinite(variant.p).all()
    assert float(np.sum(variant.p**2)) < 5.0 * start  # rigid obstacle adds no energy


def test_open_boundary_absorbs_into_the_corners():
    thickness = 16
    n = 70 + 2 * thickness
    grid = Grid.from_lengths((n * 0.02, n * 0.02, n * 0.02), 0.02)
    sim = ScalarFDTD(grid, absorbing_boundary=AbsorbingBoundary.open_field(thickness, 1200.0))
    # a true zero-mean Ricker: a volume source must inject no net volume, or the leftover DC is a
    # static field no PML can radiate away (it would mask the boundary's real performance).
    t = (np.arange(400) - 40) * sim.dt
    a = (np.pi * 1200.0 * t) ** 2
    sim.add_volume_source(
        tuple(length / 2 for length in grid.lengths), (1.0 - 2.0 * a) * np.exp(-a)
    )
    interior = np.zeros(grid.shape, dtype=bool)
    interior[thickness:-thickness, thickness:-thickness, thickness:-thickness] = True
    peak = 0.0
    for _ in range(900):
        sim.step()
        rms = np.sqrt(np.mean(sim.p[interior] ** 2))
        peak = max(peak, rms)
    final = np.sqrt(np.mean(sim.p[interior] ** 2))
    assert np.isfinite(peak) and peak > 0
    assert 20.0 * np.log10(final / peak) < -45.0


def test_closed_room_keeps_two_arrays_and_no_velocity():
    sim = ScalarFDTD(GRID)
    assert not hasattr(sim, "_velocity")
    cell_sized = [
        value
        for value in vars(sim).values()
        if isinstance(value, np.ndarray) and value.shape == GRID.shape
    ]
    assert len(cell_sized) == 2

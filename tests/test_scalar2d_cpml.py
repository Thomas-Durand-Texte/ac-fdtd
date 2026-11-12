"""The CFS-CPML open boundary on the pressure-only 2D scheme: it must actually absorb, at all angles
and into the corners, and a tuned skin must beat a locally-reacting wall where that wall saturates.

Reflection is measured, not assumed (the repository's standing rule for boundaries): the field in a
truncated domain minus the same field in a domain large enough that the far wall cannot answer in
the window is exactly the spurious reflection. The monitor sits *before* the layer, never inside it.
"""

import numpy as np

from ac_fdtd.boundaries2d import WallAdmittances2D
from ac_fdtd.cpml2d import AbsorbingBoundary2D
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scalar2d import ScalarFDTD2D

DX = 0.01


def _ricker(n: int, dt: float, f0: float, n0: int = 40) -> np.ndarray:
    t = (np.arange(n) - n0) * dt
    a = (np.pi * f0 * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def _reflection_db(boundary_factory, nx_small: int, ny: int, steps: int = 700) -> float:
    """Worst reflection over the column just inside the x_max boundary, in dB re incident."""
    sy, sx = ny // 2, 30
    thickness = getattr(boundary_factory(nx_small, ny).absorbing_boundary, "thickness", 1)
    monitor = nx_small - thickness - 3
    dt_probe = ScalarFDTD2D(Grid2D(DX, nx_small, ny)).dt
    f0 = 1500.0
    clearance = (
        int(
            1.01 * ScalarFDTD2D(Grid2D(DX, nx_small, ny)).medium.sound_speed * steps * dt_probe / DX
        )
        + 20
    )

    big = ScalarFDTD2D(Grid2D(DX, monitor + clearance, ny))
    big.add_volume_source(((sx + 0.5) * DX, (sy + 0.5) * DX), _ricker(steps, big.dt, f0))
    ref = np.zeros((steps, ny))
    for n in range(steps):
        big.step()
        ref[n] = big.p[monitor, :]
    incident = np.max(np.abs(ref))

    test = boundary_factory(nx_small, ny)
    test.add_volume_source(((sx + 0.5) * DX, (sy + 0.5) * DX), _ricker(steps, test.dt, f0))
    rec = np.zeros((steps, ny))
    for n in range(steps):
        test.step()
        rec[n] = test.p[monitor, :]
    return 20.0 * np.log10(max(np.max(np.abs(rec - ref)), 1e-12) / incident)


def test_open_boundary_absorbs_a_central_pulse_into_the_corners():
    thickness = 24
    grid = Grid2D(DX, 120 + 2 * thickness, 120 + 2 * thickness)
    sim = ScalarFDTD2D(grid, absorbing_boundary=AbsorbingBoundary2D.open_field(thickness, 1200.0))
    centre = (grid.lengths[0] / 2, grid.lengths[1] / 2)
    sim.add_volume_source(centre, _ricker(800, sim.dt, 1200.0))

    interior = np.zeros(grid.shape, dtype=bool)
    interior[thickness:-thickness, thickness:-thickness] = True
    peak = 0.0
    for _ in range(2500):
        sim.step()
        rms = np.sqrt(np.mean(sim.p[interior] ** 2))
        peak = max(peak, rms)
    final = np.sqrt(np.mean(sim.p[interior] ** 2))
    assert np.isfinite(peak) and peak > 0
    assert 20.0 * np.log10(final / peak) < -45.0  # all-angle + corner absorption


def test_reflection_deepens_with_thickness():
    def factory(thickness):
        return lambda nx, ny: ScalarFDTD2D(
            Grid2D(DX, nx, ny),
            absorbing_boundary=AbsorbingBoundary2D.open_field(thickness, 1000.0, faces=((0, 1),)),
        )

    db16 = _reflection_db(factory(16), 2 * 16 + 160, 200)
    db32 = _reflection_db(factory(32), 2 * 32 + 160, 200)
    assert db16 < -25.0
    assert db32 < -42.0
    assert db32 < db16 - 8.0  # deeper layer, markedly quieter


def test_partial_skin_beats_a_locally_reacting_wall_at_high_absorption():
    """A locally-reacting wall saturates near -13 dB however high alpha; a skin breaks that."""
    alpha = 0.99

    def wall(nx, ny):
        return ScalarFDTD2D(Grid2D(DX, nx, ny), walls=WallAdmittances2D(x_max=_xi(alpha)))

    def skin(nx, ny):
        return ScalarFDTD2D(
            Grid2D(DX, nx, ny),
            absorbing_boundary=AbsorbingBoundary2D.partial_skin(
                alpha, thickness=8, faces=((0, 1),)
            ),
        )

    db_wall = _reflection_db(wall, 200, 200)
    db_skin = _reflection_db(skin, 200, 200)
    assert db_skin < db_wall - 6.0


def test_open_on_one_axis_rigid_on_the_other_is_stable_and_absorbs():
    thickness = 20
    grid = Grid2D(DX, 160 + 2 * thickness, 160)
    sim = ScalarFDTD2D(
        grid,
        absorbing_boundary=AbsorbingBoundary2D.open_field(
            thickness, 1000.0, faces=((0, 0), (0, 1))
        ),
    )
    sim.add_volume_source((grid.lengths[0] / 2, grid.lengths[1] / 2), _ricker(600, sim.dt, 1000.0))
    for _ in range(4000):
        sim.step()
    assert np.isfinite(sim.p).all()
    assert np.max(np.abs(sim.p)) < 1.0  # the x-open energy has left; only slow y-rigid modes remain


def test_closed_room_keeps_two_arrays_and_no_velocity():
    sim = ScalarFDTD2D(Grid2D(DX, 50, 50))
    assert not hasattr(sim, "vx")
    cell_sized = [
        value
        for value in vars(sim).values()
        if isinstance(value, np.ndarray) and value.shape == sim.grid.shape
    ]
    assert len(cell_sized) == 2


def _xi(absorption: float) -> float:
    from ac_fdtd.boundaries2d import admittance_from_absorption

    return admittance_from_absorption(absorption)

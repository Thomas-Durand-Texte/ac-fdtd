"""2D edges: absorbing walls stay dissipative, corners sum, the matched layer quiets the field."""

import numpy as np
import pytest

from ac_fdtd.boundaries2d import (
    AbsorbingLayer2D,
    WallAdmittances2D,
    admittance_from_absorption,
    wall_update_coefficients_2d,
)
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.medium import AIR
from ac_fdtd.scheme2d import AcousticFDTD2D


def test_negative_admittance_is_rejected():
    with pytest.raises(ValueError):
        WallAdmittances2D(x_min=-0.1)


def test_absorbing_walls_only_remove_energy():
    """The closed-form wall solve is unconditionally dissipative: energy never climbs, and falls."""
    grid = Grid2D.from_lengths((0.8, 0.65), 0.02)
    solver = AcousticFDTD2D(grid, medium=AIR, walls=WallAdmittances2D.matched())
    solver.p[:] = np.random.default_rng(5).standard_normal(grid.shape)

    energies = [solver.energy()]
    for _ in range(300):
        solver.step()
        energies.append(solver.energy())

    e = np.array(energies)
    assert np.all(np.diff(e) <= 1e-9 * e[0])  # never increases beyond round-off
    assert e[-1] < 0.5 * e[0]  # and genuinely decays


def test_a_corner_cell_carries_both_walls_coefficients():
    """A cell on two absorbing walls appears once, with their summed coefficient."""
    grid = Grid2D.from_lengths((0.5, 0.5), 0.05)  # 10 x 10
    dt = 1e-5
    walls = WallAdmittances2D(x_min=0.3, y_min=0.7)
    indices, coefficient = wall_update_coefficients_2d(grid, AIR, dt, walls)

    scale = AIR.sound_speed * dt / (2.0 * grid.dx)
    corner_flat = 0  # cell (0, 0), on both x_min and y_min
    where = np.where(indices == corner_flat)[0]
    assert where.size == 1
    assert coefficient[where[0]] == pytest.approx(scale * (0.3 + 0.7))


def test_the_absorbing_layer_damps_an_outgoing_pulse():
    grid = Grid2D.from_lengths((1.0, 1.0), 0.01)
    solver = AcousticFDTD2D(grid, medium=AIR, absorbing_layer=AbsorbingLayer2D(thickness=16))

    x = grid.cell_centres(0)
    y = grid.cell_centres(1)
    coords_x, coords_y = np.meshgrid(x, y, indexing="ij")
    r2 = (coords_x - 0.5) ** 2 + (coords_y - 0.5) ** 2
    solver.p[:] = np.exp(-r2 / 0.03**2)

    start = solver.energy()
    solver.run(600)
    assert np.isfinite(solver.p).all()
    assert solver.energy() < 5e-2 * start


def test_a_more_absorbent_wall_decays_a_mode_faster():
    """Sanity on the admittance scale: more absorption, faster decay of the same initial field."""
    grid = Grid2D.from_lengths((0.8, 0.65), 0.02)
    field = np.random.default_rng(6).standard_normal(grid.shape)

    def remaining(absorption: float) -> float:
        walls = WallAdmittances2D.uniform(admittance_from_absorption(absorption))
        solver = AcousticFDTD2D(grid, medium=AIR, walls=walls)
        solver.p[:] = field
        start = solver.energy()
        solver.run(300)
        return solver.energy() / start

    assert remaining(0.3) < remaining(0.05)

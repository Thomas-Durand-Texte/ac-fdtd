"""Rigid interior obstacles in the 2D scheme: lossless, sealing, and shape-checked."""

import numpy as np
import pytest

from ac_fdtd.grid2d import Grid2D
from ac_fdtd.medium import AIR
from ac_fdtd.scheme2d import AcousticFDTD2D


def test_rigid_cells_shape_must_match_the_grid():
    grid = Grid2D.from_lengths((0.5, 0.5), 0.05)
    with pytest.raises(ValueError, match="does not match the grid"):
        AcousticFDTD2D(grid, rigid_cells=np.zeros((3, 3), dtype=bool))


@pytest.mark.parametrize("courant", [1.0, 0.9])
def test_energy_is_conserved_with_an_interior_obstacle(courant):
    """A rigid obstacle only pins v.n = 0, which does no work, so the domain stays lossless."""
    grid = Grid2D.from_lengths((0.8, 0.65), 0.05)
    nx, ny = grid.shape
    mask = np.zeros(grid.shape, dtype=bool)
    mask[nx // 3 : nx // 3 + 3, ny // 4 : ny // 2] = True  # a solid block off to one side

    solver = AcousticFDTD2D(grid, medium=AIR, courant=courant, rigid_cells=mask)
    solver.p[:] = np.random.default_rng(1).standard_normal(grid.shape)
    solver.p[mask] = 0.0  # start consistent: a solid cell stores no energy

    initial = solver.energy()
    assert initial > 0.0
    drift = []
    for _ in range(60):
        solver.run(1)
        assert np.all(solver.p[mask] == 0.0)  # solids never pick up pressure
        drift.append(abs(solver.energy() - initial) / initial)
    assert max(drift) < 1e-11


def test_faces_touching_a_solid_cell_stay_at_zero():
    grid = Grid2D.from_lengths((0.6, 0.6), 0.05)
    nx, ny = grid.shape
    mask = np.zeros(grid.shape, dtype=bool)
    mask[nx // 2, ny // 2] = True  # a single solid cell

    solver = AcousticFDTD2D(grid, rigid_cells=mask)
    solver.p[:] = np.random.default_rng(2).standard_normal(grid.shape)
    solver.p[mask] = 0.0
    solver.run(15)

    i, j = nx // 2, ny // 2
    assert solver.velocity[0][i, j] == 0.0 and solver.velocity[0][i + 1, j] == 0.0  # both x faces
    assert solver.velocity[1][i, j] == 0.0 and solver.velocity[1][i, j + 1] == 0.0  # both y faces


def test_a_full_rigid_barrier_seals_off_the_far_side():
    """A solid wall spanning the domain leaves the far side an untouched, silent region."""
    grid = Grid2D.from_lengths((1.0, 0.6), 0.02)
    nx, ny = grid.shape
    column = nx // 2
    mask = np.zeros(grid.shape, dtype=bool)
    mask[column, :] = True  # one cell thick is enough to block all flux across it

    solver = AcousticFDTD2D(grid, rigid_cells=mask)
    n_steps = 400
    signal = np.zeros(n_steps)
    signal[:4] = 1.0  # a short pulse on the near side
    solver.add_volume_source((0.25, 0.3), signal)
    solver.add_pressure_receiver((0.75, 0.3))  # on the sealed far side
    solver.run(n_steps)

    recorded = np.asarray(solver.recorded_pressure)
    assert np.all(recorded == 0.0)  # nothing crosses a rigid wall


def test_an_obstacle_changes_the_field_versus_an_empty_domain():
    grid = Grid2D.from_lengths((0.8, 0.8), 0.02)
    nx, ny = grid.shape
    mask = np.zeros(grid.shape, dtype=bool)
    mask[nx // 2 : nx // 2 + 2, ny // 3 : 2 * ny // 3] = True  # a short barrier mid-domain

    signal = np.zeros(300)
    signal[:4] = 1.0

    def run(rigid):
        solver = AcousticFDTD2D(grid, rigid_cells=rigid)
        solver.add_volume_source((0.2, 0.4), signal)
        solver.add_pressure_receiver((0.6, 0.4))  # behind the barrier
        solver.run(300)
        return np.asarray(solver.recorded_pressure)

    empty = run(None)
    blocked = run(mask)
    assert not np.allclose(empty, blocked)  # the obstacle casts a shadow

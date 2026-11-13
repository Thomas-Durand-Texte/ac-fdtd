"""Absorbing masked surfaces in the pressure-only 2D scheme (T008).

A solid cell of normalised admittance ``xi`` makes each of its faces that borders a fluid cell a
locally-reacting surface, ``v.n = xi p``. The fluid cell behind such a face already uses the closed
(Neumann) flux there; it now also carries the three-level wall term of the outer walls,

    (1 + a) p^{n+1} = 2 p^n - (1 - a) p^{n-1} + lam^2 L(p^n),
    a = (c dt / 2 dx) * (sum of xi over its absorbing solid neighbours + its outer-wall xi),

so the oracles here are: that update itself on one step; the discrete energy identity it implies
(``E^{n+1/2} - E^{n-1/2} = -sum a (p^{n+1} - p^{n-1})^2``); and the fact that the fluid region in
front of an absorbing solid slab obeys *exactly* the equations of a domain whose outer wall has the
same ``xi`` (the slab's cells are p = 0 and decoupled), so the two runs agree to round-off.
"""

import numpy as np
import pytest

from ac_fdtd.boundaries2d import WallAdmittances2D
from ac_fdtd.cpml2d import AbsorbingBoundary2D
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scalar2d import ScalarFDTD2D

DX = 0.02
# 31 x 25 cells: two different counts, so a swapped axis cannot pass by symmetry.
GRID = Grid2D(dx=DX, nx=31, ny=25)
PAD = 3  # thickness, in cells, of the solid slab that stands in for an outer wall


def _ricker(n: int, centre: float = 40.0, width: float = 10.0) -> np.ndarray:
    t = (np.arange(n) - centre) / width
    return (1.0 - 0.5 * t**2) * np.exp(-(t**2) / 2.0)


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def _cell(i: int, j: int) -> tuple[float, float]:
    """Centre of cell (i, j), so the point -> cell mapping is never in doubt."""
    return ((i + 0.5) * DX, (j + 0.5) * DX)


def _laplacian(p: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Independent masked-flux Laplacian: Neumann outer walls, closed faces next to a solid."""
    lap = np.zeros_like(p)
    fx = (p[1:, :] - p[:-1, :]) * ~(solid[1:, :] | solid[:-1, :])
    lap[:-1, :] += fx
    lap[1:, :] -= fx
    fy = (p[:, 1:] - p[:, :-1]) * ~(solid[:, 1:] | solid[:, :-1])
    lap[:, :-1] += fy
    lap[:, 1:] -= fy
    return lap


def _summed_neighbour_admittance(solid: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Per fluid cell, the sum of ``xi`` over its in-domain solid 4-neighbours (0 on solids)."""
    weight = np.where(solid, xi, 0.0)
    total = np.zeros(solid.shape)
    total[:-1, :] += weight[1:, :]
    total[1:, :] += weight[:-1, :]
    total[:, :-1] += weight[:, 1:]
    total[:, 1:] += weight[:, :-1]
    return np.where(solid, 0.0, total)


def _outer_admittance(shape: tuple[int, int], walls: WallAdmittances2D) -> np.ndarray:
    total = np.zeros(shape)
    total[0, :] += walls.x_min
    total[-1, :] += walls.x_max
    total[:, 0] += walls.y_min
    total[:, -1] += walls.y_max
    return total


def _notched_obstacles() -> tuple[np.ndarray, np.ndarray]:
    """Solids exercising every case: a fluid cell with three absorbing faces of different xi and
    one rigid face; a thick absorbing block (solid-solid faces must not count); an absorbing solid
    whose fluid neighbour sits on the absorbing x_min ring; a rigid-only block."""
    solid = np.zeros(GRID.shape, dtype=bool)
    xi = np.zeros(GRID.shape)
    for (i, j), value in {(10, 10): 0.3, (12, 10): 1.5, (11, 9): 2.0, (11, 11): 0.0}.items():
        solid[i, j] = True
        xi[i, j] = value  # fluid cell (11, 10) is the notch
    solid[20:23, 15:18] = True
    xi[20:23, 15:18] = 0.9
    solid[1, 5] = True
    xi[1, 5] = 0.7  # its neighbour (0, 5) is on the x_min ring
    solid[5:8, 20:22] = True  # rigid: xi stays 0
    return solid, xi


# -- None / zero is today's rigid mask, bit for bit --------------------------------------------
def test_none_and_all_zero_admittance_are_bit_identical_to_the_rigid_mask():
    solid, _ = _notched_obstacles()
    walls = WallAdmittances2D(x_min=0.4, x_max=0.0, y_min=0.2, y_max=1.0)
    field = np.random.default_rng(3).standard_normal(GRID.shape) * ~solid
    signal = _ricker(150)

    def run(**kwargs) -> np.ndarray:
        solver = ScalarFDTD2D(GRID, walls=walls, rigid_cells=solid, **kwargs)
        solver.set_initial_pressure(field)
        solver.add_volume_source(_cell(15, 12), signal)
        solver.run(150)
        return solver.p

    rigid = run()
    np.testing.assert_array_equal(run(obstacle_admittance=None), rigid)
    np.testing.assert_array_equal(run(obstacle_admittance=np.zeros(GRID.shape)), rigid)


def test_values_at_fluid_cells_are_ignored():
    solid, xi = _notched_obstacles()
    noisy = xi.copy()
    noisy[~solid] = -5.0  # read only at solid cells: even an invalid value is ignored there
    noisy[0, 0] = np.nan

    def run(admittance: np.ndarray) -> np.ndarray:
        solver = ScalarFDTD2D(GRID, rigid_cells=solid, obstacle_admittance=admittance)
        solver.set_initial_pressure(np.random.default_rng(4).standard_normal(GRID.shape) * ~solid)
        solver.run(40)
        return solver.p

    np.testing.assert_array_equal(run(noisy), run(xi))


# -- the update itself, one step, from a moving state -----------------------------------------
def test_one_step_is_the_three_level_wall_relation_with_summed_face_coefficients():
    solid, xi = _notched_obstacles()
    walls = WallAdmittances2D(x_min=0.4)
    solver = ScalarFDTD2D(GRID, walls=walls, rigid_cells=solid, obstacle_admittance=xi)
    rng = np.random.default_rng(5)
    p = rng.standard_normal(GRID.shape) * ~solid
    p_old = rng.standard_normal(GRID.shape) * ~solid
    solver.p[:] = p
    solver.p_old[:] = p_old

    s = solver.medium.sound_speed * solver.dt / (2.0 * DX)
    lam2 = (solver.medium.sound_speed * solver.dt / DX) ** 2
    a = s * (_summed_neighbour_admittance(solid, xi) + _outer_admittance(GRID.shape, walls))
    assert a[11, 10] == pytest.approx(s * (0.3 + 1.5 + 2.0))  # the test's own bookkeeping
    expected = (2.0 * p - (1.0 - a) * p_old + lam2 * _laplacian(p, solid)) / (1.0 + a)
    expected[solid] = 0.0

    solver.step()
    np.testing.assert_allclose(solver.p, expected, rtol=1e-13, atol=1e-13)
    assert np.all(solver.p[solid] == 0.0)


# -- energy: the scheme's discrete energy can only decrease -----------------------------------
def _energy(p: np.ndarray, p_old: np.ndarray, solid: np.ndarray, lam2: float) -> float:
    """``E^{n-1/2} = |p^n - p^{n-1}|^2 - lam^2 <L p^n, p^{n-1}>``, conserved when lossless."""
    return float(np.sum((p - p_old) ** 2) - lam2 * np.sum(_laplacian(p, solid) * p_old))


@pytest.mark.parametrize("value", [0.0, 0.2, 1.0, 1e3])
def test_energy_follows_the_dissipation_identity_and_never_grows(value):
    solid = np.zeros(GRID.shape, dtype=bool)
    solid[10:14, 6:18] = True
    solid[22, 3:9] = True
    xi = np.where(solid, value, 0.0)
    xi[22, 3:9] = 0.0  # one rigid strip next to the absorbing block

    solver = ScalarFDTD2D(GRID, rigid_cells=solid, obstacle_admittance=xi)
    solver.set_initial_pressure(np.random.default_rng(6).standard_normal(GRID.shape) * ~solid)
    s = solver.medium.sound_speed * solver.dt / (2.0 * DX)
    lam2 = (solver.medium.sound_speed * solver.dt / DX) ** 2
    a = s * _summed_neighbour_admittance(solid, xi)

    initial = _energy(solver.p, solver.p_old, solid, lam2)
    assert initial > 0.0
    energy = initial
    for _ in range(300):
        p_before = solver.p_old.copy()  # p^{n-1}
        solver.step()
        new = _energy(solver.p, solver.p_old, solid, lam2)
        loss = float(np.sum(a * (solver.p - p_before) ** 2))
        assert new - energy == pytest.approx(-loss, abs=1e-11 * initial)
        assert new <= energy + 1e-12 * initial
        energy = new
    if value == 0.0:
        assert abs(energy - initial) < 1e-10 * initial  # rigid: lossless
    else:
        assert energy < 0.99 * initial  # an absorbing surface actually absorbs


# -- an absorbing solid slab is the outer locally-reacting wall of the same xi -----------------
@pytest.mark.parametrize("side", ["x_max", "x_min", "y_max"])
@pytest.mark.parametrize("absorption", [0.3, 1.0])
def test_absorbing_slab_equals_the_outer_wall(side, absorption):
    """Normal (and every other) incidence: the fluid region in front of a solid slab of admittance
    xi follows the same equations as a domain whose ``side`` wall has that xi. The other walls are
    absorbing too, so the slab's end cells also sum an outer-wall term at the corners."""
    from ac_fdtd.boundaries2d import admittance_from_absorption

    xi = admittance_from_absorption(absorption) if absorption < 1.0 else 1.0
    axis = 0 if side.startswith("x") else 1
    high = side.endswith("max")
    other = {"x_min": 0.25, "x_max": 0.0, "y_min": 0.5, "y_max": 0.1}
    walls_outer = WallAdmittances2D(**{**other, side: xi})
    walls_padded = WallAdmittances2D(**{**other, side: 0.0})

    shape = list(GRID.shape)
    shape[axis] += PAD
    padded = Grid2D(dx=DX, nx=shape[0], ny=shape[1])
    fluid = [slice(None), slice(None)]
    fluid[axis] = slice(0, GRID.shape[axis]) if high else slice(PAD, None)
    fluid = tuple(fluid)
    solid = np.ones(padded.shape, dtype=bool)
    solid[fluid] = False
    admittance = np.where(solid, xi, 0.0)
    offset = [0, 0]
    offset[axis] = 0 if high else PAD

    reference = ScalarFDTD2D(GRID, walls=walls_outer)
    masked = ScalarFDTD2D(
        padded, walls=walls_padded, rigid_cells=solid, obstacle_admittance=admittance
    )
    signal = _ricker(400)
    receivers = [(26, 12), (30, 3), (4, 22), (15, 24), (0, 0)]
    reference.add_volume_source(_cell(20, 9), signal)
    masked.add_volume_source(_cell(20 + offset[0], 9 + offset[1]), signal)
    for i, j in receivers:
        reference.add_pressure_receiver(_cell(i, j))
        masked.add_pressure_receiver(_cell(i + offset[0], j + offset[1]))

    reference.run(400)
    masked.run(400)
    assert np.max(np.abs(reference.p)) > 0.0
    assert _max_rel(masked.p[fluid], reference.p) < 1e-12
    assert _max_rel(masked.recorded_pressure, reference.recorded_pressure) < 1e-12
    assert np.all(masked.p[solid] == 0.0)


def test_driven_mixed_rigid_and_absorbing_obstacle():
    """A rigid interior block plus an absorbing slab on x_max equals the rigid block alone in a
    domain with an absorbing x_max wall: the mask mixes rigid and absorbing solids correctly."""
    xi = 0.6
    block = np.zeros(GRID.shape, dtype=bool)
    block[12:15, 8:16] = True
    reference = ScalarFDTD2D(GRID, walls=WallAdmittances2D(x_max=xi), rigid_cells=block)

    padded = Grid2D(dx=DX, nx=GRID.nx + PAD, ny=GRID.ny)
    solid = np.zeros(padded.shape, dtype=bool)
    solid[: GRID.nx][block] = True
    solid[GRID.nx :, :] = True
    admittance = np.zeros(padded.shape)
    admittance[GRID.nx :, :] = xi  # the block stays rigid
    masked = ScalarFDTD2D(padded, rigid_cells=solid, obstacle_admittance=admittance)

    signal = _ricker(500)
    for solver in (reference, masked):
        solver.add_volume_source(_cell(6, 12), signal)
        solver.add_pressure_receiver(_cell(25, 12))  # between the block and the slab
        solver.add_pressure_receiver(_cell(18, 11))  # in the block's shadow
    reference.run(500)
    masked.run(500)
    assert _max_rel(masked.p[: GRID.nx], reference.p) < 1e-12
    assert _max_rel(masked.recorded_pressure, reference.recorded_pressure) < 1e-12

    # and the absorbing slab is not a rigid one: making it rigid changes the field
    rigid = ScalarFDTD2D(padded, rigid_cells=solid)
    rigid.add_volume_source(_cell(6, 12), signal)
    rigid.run(500)
    assert _max_rel(masked.p, rigid.p) > 1e-2


# -- combined with the CPML open boundary ------------------------------------------------------
def test_absorbing_obstacle_inside_a_cpml_box_is_stable_and_decays():
    grid = Grid2D(dx=DX, nx=48, ny=40)
    solid = np.zeros(grid.shape, dtype=bool)
    solid[20:26, 14:26] = True
    solid[30, 10:30] = True
    xi = np.zeros(grid.shape)
    xi[20:26, 14:26] = 2.0
    xi[30, 10:30] = 0.5
    solver = ScalarFDTD2D(
        grid,
        rigid_cells=solid,
        obstacle_admittance=xi,
        absorbing_boundary=AbsorbingBoundary2D(thickness=8),
    )
    solver.add_volume_source(_cell(14, 20), _ricker(120))
    peak = 0.0
    for _ in range(2000):
        solver.step()
        assert np.all(np.isfinite(solver.p))
        peak = max(peak, float(np.max(np.abs(solver.p))))
    assert peak > 0.0
    assert np.max(np.abs(solver.p)) < 1e-3 * peak
    assert np.all(solver.p[solid] == 0.0)


# -- input checking ------------------------------------------------------------------------------
def test_admittance_shape_must_match_the_grid():
    solid = np.zeros(GRID.shape, dtype=bool)
    solid[5, 5] = True
    with pytest.raises(ValueError, match="does not match the grid"):
        ScalarFDTD2D(GRID, rigid_cells=solid, obstacle_admittance=np.zeros((3, 3)))


def test_admittance_without_a_solid_mask_is_rejected():
    with pytest.raises(ValueError, match="rigid_cells"):
        ScalarFDTD2D(GRID, obstacle_admittance=np.zeros(GRID.shape))


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
def test_negative_or_non_finite_admittance_at_a_solid_cell_is_rejected(bad):
    solid = np.zeros(GRID.shape, dtype=bool)
    solid[5:8, 5:8] = True
    xi = np.where(solid, 0.5, 0.0)
    xi[6, 6] = bad
    with pytest.raises(ValueError, match="admittance"):
        ScalarFDTD2D(GRID, rigid_cells=solid, obstacle_admittance=xi)

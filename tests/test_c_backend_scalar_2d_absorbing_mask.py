"""Absorbing masked surfaces in the compiled pressure-only 2D kernel must be the reference's (T009).

The reference is ``ScalarFDTD2D(..., rigid_cells=solid, obstacle_admittance=xi)`` (T008): each fluid
cell behind an absorbing solid face gains the outer walls' three-level term, with
``a = (c dt / 2 dx) * (sum of xi over its absorbing solid neighbours + its outer-wall xi)``. In fp64
the compiled kernel must reproduce both stored levels with ``assert_array_equal``; any change in how
``a`` is rounded or applied shows up as a last-bit difference: a kernel bug, never a tolerance.
"""

import numpy as np
import pytest

from ac_fdtd.boundaries2d import WallAdmittances2D
from ac_fdtd.cpml2d import AbsorbingBoundary2D
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scalar2d import ScalarFDTD2D

c_backend = pytest.importorskip("ac_fdtd.c_backend")

pytestmark = pytest.mark.skipif(not c_backend.is_available(), reason="no C compiler available")

DX = 0.02
GRID = Grid2D(dx=DX, nx=41, ny=33)  # not square, so a swapped axis cannot pass by symmetry
WALLS = WallAdmittances2D(x_min=0.4, x_max=0.0, y_min=0.2, y_max=1.0)


def _ricker(n: int, centre: float = 40.0, width: float = 10.0) -> np.ndarray:
    t = (np.arange(n) - centre) / width
    return (1.0 - 0.5 * t**2) * np.exp(-(t**2) / 2.0)


def _cell(i: int, j: int) -> tuple[float, float]:
    return ((i + 0.5) * DX, (j + 0.5) * DX)


def _obstacles() -> tuple[np.ndarray, np.ndarray]:
    """Every case of the reference's tests: a fluid notch with three absorbing faces of different xi
    and one rigid face; a thick absorbing block (solid-solid faces must not count); absorbing solids
    whose fluid neighbours sit on the outer-wall ring (x_min edge and the (0, 0)-adjacent corner);
    a rigid-only block."""
    solid = np.zeros(GRID.shape, dtype=bool)
    xi = np.zeros(GRID.shape)
    for (i, j), value in {(10, 10): 0.3, (12, 10): 1.5, (11, 9): 2.0, (11, 11): 0.0}.items():
        solid[i, j] = True
        xi[i, j] = value  # fluid cell (11, 10) is the notch
    solid[20:24, 15:19] = True
    xi[20:24, 15:19] = 0.9
    solid[1, 5] = True
    xi[1, 5] = 0.7  # its neighbour (0, 5) is on the x_min ring
    solid[1, 1] = True
    xi[1, 1] = 1.2  # neighbours (0, 1) and (1, 0) are wall cells, next to the corner
    solid[30:33, 25:27] = True  # rigid: xi stays 0
    return solid, xi


def _field(solid: np.ndarray, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(GRID.shape) * ~solid


def _assert_identical(compiled, reference) -> None:
    np.testing.assert_array_equal(compiled.p, reference.p)
    np.testing.assert_array_equal(compiled.previous, reference.p_old)


# -- fp64: bit-identical to the reference ------------------------------------------------------
@pytest.mark.parametrize("walls", [None, WALLS], ids=["rigid-walls", "impedance-walls"])
def test_mixed_mask_from_a_random_start_is_bit_identical(walls):
    solid, xi = _obstacles()
    field = _field(solid, 0)
    kwargs = dict(walls=walls, rigid_cells=solid, obstacle_admittance=xi)
    reference = ScalarFDTD2D(GRID, **kwargs)
    compiled = c_backend.CScalarFDTD2D(GRID, **kwargs)
    reference.set_initial_pressure(field)
    compiled.set_initial_pressure(field)
    reference.run(250)
    compiled.run(250)
    _assert_identical(compiled, reference)
    assert np.all(compiled.p[solid] == 0.0)


@pytest.mark.parametrize("value", [0.25, 1.0, 1e3])
def test_absorbing_only_mask_driven_with_receivers_is_bit_identical(value):
    """Every solid absorbs (including a very stiff-damped 1e3); source and receivers recorded."""
    solid = np.zeros(GRID.shape, dtype=bool)
    solid[14:20, 8:24] = True
    solid[30, 3:12] = True
    xi = np.where(solid, value, 0.0)
    signal = _ricker(300)
    reference = ScalarFDTD2D(GRID, walls=WALLS, rigid_cells=solid, obstacle_admittance=xi)
    compiled = c_backend.CScalarFDTD2D(GRID, walls=WALLS, rigid_cells=solid, obstacle_admittance=xi)
    for solver in (reference, compiled):
        solver.add_volume_source(_cell(6, 16), signal)
        solver.add_pressure_receiver(_cell(13, 16))  # in front of the absorbing face
        solver.add_pressure_receiver(_cell(26, 16))  # in its shadow
        solver.run(300)
    _assert_identical(compiled, reference)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    assert np.max(np.abs(reference.recorded_pressure)) > 0.0


def test_velocity_source_on_a_face_next_to_an_absorbing_solid_is_bit_identical():
    """A driven face whose cells carry a merged obstacle coefficient must divide by that 1 + a:
    the face reads the same (merged) wall table the update uses, as in the reference."""
    solid, xi = _obstacles()
    signal = _ricker(200)
    reference = ScalarFDTD2D(GRID, walls=WALLS, rigid_cells=solid, obstacle_admittance=xi)
    compiled = c_backend.CScalarFDTD2D(GRID, walls=WALLS, rigid_cells=solid, obstacle_admittance=xi)
    for solver in (reference, compiled):
        # lower face of (19, 16): cell (19, 16) borders the absorbing block at (20, 16)
        solver.add_prescribed_source(_cell(19, 16), signal, kind="velocity", axis=1)
        # lower x face of (11, 10), the notch with three absorbing faces: lower cell (10, 10)
        # is solid, so this face is closed and must stay inert
        solver.add_prescribed_source(_cell(11, 10), signal, kind="velocity", axis=0)
        solver.add_prescribed_source(_cell(0, 6), signal, start=30, kind="pressure")
        solver.run(200)
    _assert_identical(compiled, reference)


@pytest.mark.parametrize(
    "boundary",
    [AbsorbingBoundary2D(thickness=6), AbsorbingBoundary2D(thickness=5, faces=((0, 0), (1, 1)))],
    ids=["all-walls", "two-walls"],
)
def test_absorbing_mask_inside_a_cpml_shell_is_bit_identical(boundary):
    """Solids in the interior, one absorbing solid straddling the shell edge (its shell-side faces
    must be dropped as the reference drops shell wall cells), plus impedance on unlined walls."""
    solid = np.zeros(GRID.shape, dtype=bool)
    xi = np.zeros(GRID.shape)
    solid[16:22, 12:20] = True
    xi[16:22, 12:20] = 2.0
    solid[26, 6:28] = True
    xi[26, 6:28] = 0.5
    solid[5:8, 8:10] = True  # straddles the 6- and 5-cell shells on x_min
    xi[5:8, 8:10] = 0.8
    signal = _ricker(300)
    kwargs = dict(
        walls=WALLS, rigid_cells=solid, obstacle_admittance=xi, absorbing_boundary=boundary
    )
    reference = ScalarFDTD2D(GRID, **kwargs)
    compiled = c_backend.CScalarFDTD2D(GRID, **kwargs)
    for solver in (reference, compiled):
        solver.add_volume_source(_cell(12, 16), signal)
        solver.add_pressure_receiver(_cell(24, 16))
        solver.run(300)
    _assert_identical(compiled, reference)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)


# -- None / all-zero: the rigid path is untouched ----------------------------------------------
def test_none_and_all_zero_admittance_are_bit_identical_to_the_rigid_mask():
    solid, _ = _obstacles()
    field = _field(solid, 1)
    signal = _ricker(150)

    def run(**kwargs) -> np.ndarray:
        solver = c_backend.CScalarFDTD2D(GRID, walls=WALLS, rigid_cells=solid, **kwargs)
        solver.set_initial_pressure(field)
        solver.add_volume_source(_cell(15, 12), signal)
        solver.run(150)
        return solver.p

    rigid = run()
    np.testing.assert_array_equal(run(obstacle_admittance=None), rigid)
    np.testing.assert_array_equal(run(obstacle_admittance=np.zeros(GRID.shape)), rigid)


def test_values_at_fluid_cells_are_ignored():
    solid, xi = _obstacles()
    noisy = xi.copy()
    noisy[~solid] = -5.0  # read only at solid cells: even invalid values are ignored there
    noisy[0, 0] = np.nan

    def run(admittance: np.ndarray) -> np.ndarray:
        solver = c_backend.CScalarFDTD2D(GRID, rigid_cells=solid, obstacle_admittance=admittance)
        solver.set_initial_pressure(_field(solid, 2))
        solver.run(60)
        return solver.p

    np.testing.assert_array_equal(run(noisy), run(xi))


# -- chunking, threads, precision --------------------------------------------------------------
def test_chunked_runs_and_thread_counts_do_not_change_the_answer():
    solid, xi = _obstacles()
    field = _field(solid, 3)
    signal = _ricker(80)
    results = []
    for threads, chunks in ((1, (80,)), (3, (5, 1, 33, 41)), (8, (40, 40))):
        compiled = c_backend.CScalarFDTD2D(
            GRID, walls=WALLS, rigid_cells=solid, obstacle_admittance=xi
        )
        compiled.set_threads(threads)
        compiled.set_initial_pressure(field)
        compiled.add_volume_source(_cell(30, 8), signal)
        for chunk in chunks:
            compiled.run(chunk)
        assert compiled.step_index == 80
        results.append((compiled.p.copy(), compiled.previous.copy()))
    compiled.set_threads(0)
    for p, previous in results[1:]:
        np.testing.assert_array_equal(p, results[0][0])
        np.testing.assert_array_equal(previous, results[0][1])


def test_single_precision_tracks_the_float32_reference():
    solid, xi = _obstacles()
    signal = _ricker(300)
    kwargs = dict(walls=WALLS, rigid_cells=solid, obstacle_admittance=xi, dtype=np.float32)
    reference = ScalarFDTD2D(GRID, **kwargs)
    compiled = c_backend.CScalarFDTD2D(GRID, **kwargs)
    for solver in (reference, compiled):
        solver.add_volume_source(_cell(6, 16), signal)
        solver.run(300)
    assert compiled.p.dtype == np.float32
    scale = float(np.max(np.abs(reference.p)))
    assert scale > 0.0
    assert float(np.max(np.abs(compiled.p - reference.p))) < 1e-4 * scale
    assert np.all(compiled.p[solid] == 0.0)


# -- errors match the reference ----------------------------------------------------------------
def test_admittance_shape_must_match_the_grid():
    solid = np.zeros(GRID.shape, dtype=bool)
    solid[5, 5] = True
    with pytest.raises(ValueError, match="does not match the grid"):
        c_backend.CScalarFDTD2D(GRID, rigid_cells=solid, obstacle_admittance=np.zeros((3, 3)))


def test_admittance_without_a_solid_mask_is_rejected():
    with pytest.raises(ValueError, match="rigid_cells"):
        c_backend.CScalarFDTD2D(GRID, obstacle_admittance=np.zeros(GRID.shape))


@pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
def test_negative_or_non_finite_admittance_at_a_solid_cell_is_rejected(bad):
    solid = np.zeros(GRID.shape, dtype=bool)
    solid[5:8, 5:8] = True
    xi = np.where(solid, 0.5, 0.0)
    xi[6, 6] = bad
    with pytest.raises(ValueError, match="admittance"):
        c_backend.CScalarFDTD2D(GRID, rigid_cells=solid, obstacle_admittance=xi)


def test_the_docstring_documents_the_new_keyword():
    assert "obstacle_admittance" in (c_backend.CScalarFDTD2D.__doc__ or "")

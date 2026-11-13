"""The CFS-CPML shell of the compiled pressure-only 2D kernel must be the reference's, bit for bit.

The reference is ``ScalarFDTD2D(..., absorbing_boundary=...)``: a p-v + CPML update on the shell
cells (overwriting the scalar update there), the scalar update in the interior, impedance walls only
on interior wall cells, then sources, prescribed pressure and rigid cells. In fp64 the compiled
kernel must reproduce both stored levels with ``assert_array_equal`` — any reordering of the CPML
arithmetic shows up here as a last-bit difference: a kernel bug, never a tolerance to widen.
"""

import numpy as np
import pytest

from ac_fdtd.boundaries2d import WallAdmittances2D
from ac_fdtd.cpml2d import AbsorbingBoundary2D
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scalar2d import ScalarFDTD2D

c_backend = pytest.importorskip("ac_fdtd.c_backend")

pytestmark = pytest.mark.skipif(not c_backend.is_available(), reason="no C compiler available")

GRID = Grid2D.from_lengths((1.3, 1.1), 0.02)  # 65 x 55: not square, so axes cannot be swapped
THICKNESS = 8


def _ricker(n: int) -> np.ndarray:
    t = np.arange(n)
    return (1.0 - 0.5 * ((t - 40) / 12) ** 2) * np.exp(-(((t - 40) / 12) ** 2) / 2.0)


def _pair(**kwargs):
    return ScalarFDTD2D(GRID, **kwargs), c_backend.CScalarFDTD2D(GRID, **kwargs)


def _assert_identical(compiled, reference) -> None:
    np.testing.assert_array_equal(compiled.p, reference.p)
    np.testing.assert_array_equal(compiled.previous, reference.p_old)


def test_open_field_all_faces_is_bit_identical():
    """All four walls lined, corners by overlapping slabs; a random start excites every angle."""
    boundary = AbsorbingBoundary2D.open_field(THICKNESS, 1000.0)
    field = np.random.default_rng(0).standard_normal(GRID.shape)
    reference, compiled = _pair(absorbing_boundary=boundary)
    reference.set_initial_pressure(field)
    compiled.set_initial_pressure(field)
    reference.run(300)
    compiled.run(300)
    _assert_identical(compiled, reference)
    assert compiled.absorbing_boundary is boundary


@pytest.mark.parametrize(
    "faces",
    [
        ((0, 1),),  # one wall
        ((1, 0),),  # one wall on the other axis
        ((0, 0), (1, 1)),  # two walls meeting at one corner
        ((0, 0), (0, 1), (1, 0)),  # three walls, two corners
    ],
)
def test_wall_subsets_with_impedance_on_the_rest(faces):
    """The unlined walls keep their admittance; lined ones and shell corner cells ignore it."""
    boundary = AbsorbingBoundary2D.open_field(THICKNESS, 1200.0, faces=faces)
    walls = WallAdmittances2D.from_absorption(0.3)
    signal = _ricker(250)
    reference, compiled = _pair(walls=walls, absorbing_boundary=boundary)
    for sim in (reference, compiled):
        sim.add_volume_source((0.45, 0.5), signal)
        sim.add_pressure_receiver((0.8, 0.6))
        sim.add_pressure_receiver((0.05, 0.05))  # inside the shell when a low wall is lined
        sim.run(250)
    _assert_identical(compiled, reference)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)


def test_partial_skin_is_bit_identical():
    """kappa_max = 1, order 2, shallow: the other CPML parameter regime."""
    boundary = AbsorbingBoundary2D.partial_skin(0.8, thickness=4)
    reference, compiled = _pair(absorbing_boundary=boundary)
    signal = _ricker(250)
    for sim in (reference, compiled):
        sim.add_volume_source((0.6, 0.5), signal)
        sim.run(250)
    _assert_identical(compiled, reference)


def test_thickness_one_and_source_inside_the_shell():
    """The thinnest shell, with a soft source and a held node in it (applied after the CPML)."""
    boundary = AbsorbingBoundary2D(thickness=1, alpha_max=2000.0)
    reference, compiled = _pair(absorbing_boundary=boundary)
    signal = _ricker(200)
    for sim in (reference, compiled):
        sim.add_volume_source((0.01, 0.5), signal)  # cell (0, 25): in the x_min shell
        sim.add_prescribed_source((1.29, 0.3), signal[:60], start=10, kind="pressure")
        sim.run(200)
    _assert_identical(compiled, reference)


def test_everything_together():
    """CPML on x, impedance on y, a rigid obstacle and a driven interior face, all at once."""
    boundary = AbsorbingBoundary2D.open_field(THICKNESS, 1000.0, faces=((0, 0), (0, 1)))
    walls = WallAdmittances2D.from_absorption(0.5)
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[30:34, 12:30] = True
    signal = _ricker(300)
    reference, compiled = _pair(walls=walls, rigid_cells=rigid, absorbing_boundary=boundary)
    for sim in (reference, compiled):
        sim.add_volume_source((0.4, 0.7), signal)
        sim.add_prescribed_source((0.9, 0.4), 1e-3 * signal[:80], kind="velocity", axis=0)
        sim.add_pressure_receiver((0.5, 0.2))
        sim.run(300)
    _assert_identical(compiled, reference)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    assert np.all(compiled.p[rigid] == 0.0)


def test_chunked_runs_carry_the_shell_state():
    """Velocity and memory fields persist across run calls, not just the two pressure levels."""
    boundary = AbsorbingBoundary2D.open_field(THICKNESS, 1000.0)
    field = np.random.default_rng(1).standard_normal(GRID.shape)
    whole = c_backend.CScalarFDTD2D(GRID, absorbing_boundary=boundary)
    pieces = c_backend.CScalarFDTD2D(GRID, absorbing_boundary=boundary)
    whole.set_initial_pressure(field)
    pieces.set_initial_pressure(field)
    whole.run(41)
    for chunk in (5, 1, 17, 18):
        pieces.run(chunk)
    np.testing.assert_array_equal(pieces.p, whole.p)
    np.testing.assert_array_equal(pieces.previous, whole.previous)
    assert pieces.step_index == whole.step_index == 41


def test_thread_count_does_not_change_the_answer():
    boundary = AbsorbingBoundary2D.open_field(THICKNESS, 1000.0, faces=((0, 1), (1, 0)))
    signal = _ricker(150)
    results = []
    for threads in (1, 3, 8):
        compiled = c_backend.CScalarFDTD2D(GRID, absorbing_boundary=boundary)
        compiled.set_threads(threads)
        compiled.add_volume_source((0.4, 0.5), signal)
        compiled.run(150)
        results.append(compiled.p.copy())
    compiled.set_threads(0)
    for other in results[1:]:
        np.testing.assert_array_equal(results[0], other)


def test_single_precision_matches_the_single_precision_reference():
    boundary = AbsorbingBoundary2D.open_field(THICKNESS, 1000.0)
    signal = _ricker(250)
    reference, compiled = _pair(absorbing_boundary=boundary, dtype=np.float32)
    for sim in (reference, compiled):
        sim.add_volume_source((0.4, 0.5), signal)
        sim.run(250)
    assert compiled.p.dtype == np.float32
    scale = float(np.max(np.abs(reference.p)))
    assert scale > 0.0
    assert float(np.max(np.abs(compiled.p - reference.p))) / scale < 1e-4


def test_shell_state_is_stored_in_the_shell_only():
    """The interior stays two arrays: no velocity or memory field the size of the grid."""
    grid = Grid2D(0.02, 200, 160)
    compiled = c_backend.CScalarFDTD2D(
        grid, absorbing_boundary=AbsorbingBoundary2D.open_field(6, 1000.0)
    )
    buffers = {id(compiled.p), id(compiled.previous)}
    extra = []
    for value in vars(compiled).values():
        items = value.values() if isinstance(value, dict) else value
        items = items if isinstance(items, (list, tuple)) else [value]
        extra += [a for a in items if isinstance(a, np.ndarray) and id(a) not in buffers]
    assert not any(a.shape == grid.shape for a in extra)
    # Six full-grid shell fields (the NumPy reference's layout) would be 6 * grid.size; slabs of
    # depth 6 on a 200 x 160 grid are about 4.3k cells per field.
    assert sum(a.size for a in extra) < 2 * grid.size
    compiled.run(3)  # and it runs
    assert np.isfinite(compiled.p).all()


def test_a_shell_that_does_not_fit_is_refused():
    with pytest.raises(ValueError):
        c_backend.CScalarFDTD2D(GRID, absorbing_boundary=AbsorbingBoundary2D(thickness=28))


def test_a_velocity_source_bordering_the_shell_is_refused():
    compiled = c_backend.CScalarFDTD2D(
        GRID, absorbing_boundary=AbsorbingBoundary2D.open_field(THICKNESS, 1000.0)
    )
    # cell (8, 20): first interior cell along x; its lower x face borders shell cell (7, 20).
    with pytest.raises(ValueError):
        compiled.add_prescribed_source((0.17, 0.41), np.ones(5), kind="velocity", axis=0)


def test_no_boundary_is_unchanged():
    """``absorbing_boundary=None`` (the default) is the closed-room kernel, bit for bit."""
    field = np.random.default_rng(2).standard_normal(GRID.shape)
    explicit = c_backend.CScalarFDTD2D(GRID, absorbing_boundary=None)
    default = c_backend.CScalarFDTD2D(GRID)
    for sim in (explicit, default):
        sim.set_initial_pressure(field)
        sim.run(50)
    np.testing.assert_array_equal(explicit.p, default.p)

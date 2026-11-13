"""The CFS-CPML shell of the compiled pressure-only 3D kernel must be the reference's, bit for bit.

The reference is ``ScalarFDTD(..., absorbing_boundary=...)``: a p-v + CPML update on the shell
cells (overwriting the scalar update there), the scalar update in the interior, impedance walls only
on interior wall cells, then sources and rigid cells. In fp64 the compiled kernel must reproduce
both stored levels with ``assert_array_equal``: any reordering of the CPML arithmetic shows up here
as a last-bit difference, which is a kernel bug, never a tolerance to widen. The 3D shell has edges
(two slabs overlap) and corners (three), which the face subsets below exercise one by one.
"""

import numpy as np
import pytest

from ac_fdtd.boundaries import WallAdmittances
from ac_fdtd.cpml import AbsorbingBoundary
from ac_fdtd.grid import Grid
from ac_fdtd.scalar import ScalarFDTD

c_backend = pytest.importorskip("ac_fdtd.c_backend")

pytestmark = pytest.mark.skipif(not c_backend.is_available(), reason="no C compiler available")

# 22 x 19 x 17: three different counts, so a swapped axis cannot pass by symmetry.
GRID = Grid.from_lengths((0.66, 0.57, 0.51), 0.03)
THICKNESS = 5
CENTRE = (0.33, 0.28, 0.25)  # cell (11, 9, 8): interior for every boundary below
ASYMMETRIC = WallAdmittances(x_min=0.1, x_max=0.7, y_min=0.25, y_max=1.0, z_min=0.5, z_max=0.9)


def _ricker(n: int) -> np.ndarray:
    t = np.arange(n)
    return (1.0 - 0.5 * ((t - 30) / 8) ** 2) * np.exp(-(((t - 30) / 8) ** 2) / 2.0)


def _pair(**kwargs):
    return ScalarFDTD(GRID, **kwargs), c_backend.CScalarFDTD(GRID, **kwargs)


def _assert_identical(compiled, reference) -> None:
    np.testing.assert_array_equal(compiled.p, reference.p)
    np.testing.assert_array_equal(compiled.previous, reference.p_old)


def test_open_field_all_faces_is_bit_identical():
    """All six walls lined, edges and corners by overlapping slabs; a random start excites all."""
    boundary = AbsorbingBoundary.open_field(THICKNESS, 1000.0)
    field = np.random.default_rng(0).standard_normal(GRID.shape)
    reference, compiled = _pair(absorbing_boundary=boundary)
    reference.set_initial_pressure(field)
    compiled.set_initial_pressure(field)
    reference.run(120)
    compiled.run(120)
    _assert_identical(compiled, reference)
    assert compiled.absorbing_boundary is boundary


@pytest.mark.parametrize(
    "faces",
    [
        (),  # nothing lined: an empty shell, every wall impedance
        ((0, 1),),  # one wall
        ((2, 0),),  # one wall on another axis
        ((0, 0), (1, 1)),  # two walls meeting along one edge
        ((0, 0), (1, 0), (2, 1)),  # three walls meeting at one corner
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 1)),  # five walls: everything but the floor
    ],
)
def test_wall_subsets_with_impedance_on_the_rest(faces):
    """The unlined walls keep their admittance; lined ones and shell edge cells ignore it."""
    boundary = AbsorbingBoundary.open_field(THICKNESS, 1200.0, faces=faces)
    signal = _ricker(150)
    reference, compiled = _pair(walls=ASYMMETRIC, absorbing_boundary=boundary)
    for sim in (reference, compiled):
        sim.add_volume_source(CENTRE, signal)
        sim.add_pressure_receiver((0.5, 0.4, 0.3))
        sim.add_pressure_receiver((0.01, 0.01, 0.01))  # corner cell: in the shell or on 3 walls
        sim.add_pressure_receiver((0.65, 0.56, 0.5))  # the opposite corner cell
        sim.run(150)
    _assert_identical(compiled, reference)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)


def test_partial_skin_is_bit_identical():
    """kappa_max = 1, order 2, shallow: the other CPML parameter regime."""
    boundary = AbsorbingBoundary.partial_skin(0.8, thickness=3)
    reference, compiled = _pair(absorbing_boundary=boundary)
    signal = _ricker(150)
    for sim in (reference, compiled):
        sim.add_volume_source(CENTRE, signal)
        sim.run(150)
    _assert_identical(compiled, reference)


def test_thickness_one_and_source_inside_the_shell():
    """The thinnest shell, with a soft source in it (added after the CPML overwrite)."""
    boundary = AbsorbingBoundary(thickness=1, alpha_max=2000.0)
    reference, compiled = _pair(absorbing_boundary=boundary)
    signal = _ricker(120)
    for sim in (reference, compiled):
        sim.add_volume_source((0.01, 0.28, 0.25), signal)  # cell (0, 9, 8): in the x_min shell
        sim.add_volume_source((0.65, 0.56, 0.5), 0.5 * signal)  # cell (21, 18, 16): a corner
        sim.add_pressure_receiver((0.01, 0.28, 0.25))
        sim.run(120)
    _assert_identical(compiled, reference)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)


def test_everything_together():
    """CPML on z, impedance on x and y, rigid obstacles (one reaching into the shell), receivers."""
    boundary = AbsorbingBoundary.open_field(THICKNESS, 1000.0, faces=((2, 0), (2, 1)))
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[13:16, 4:8, 6:11] = True  # interior block
    rigid[3:6, 12:15, 0:8] = True  # a pillar standing in the floor's shell
    signal = _ricker(160)
    reference, compiled = _pair(walls=ASYMMETRIC, rigid_cells=rigid, absorbing_boundary=boundary)
    for sim in (reference, compiled):
        sim.add_volume_source((0.2, 0.2, 0.3), signal)
        sim.add_pressure_receiver((0.5, 0.15, 0.25))
        sim.add_pressure_receiver((0.3, 0.3, 0.05))  # in the floor's shell
        sim.add_velocity_receiver((0.4, 0.3, 0.06), 2)  # face (13, 10, 2): in the shell
        sim.run(160)
    _assert_identical(compiled, reference)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    np.testing.assert_array_equal(compiled.recorded_velocity, reference.recorded_velocity)
    assert np.all(compiled.p[rigid] == 0.0)


def test_chunked_runs_carry_the_shell_state():
    """Velocity and memory fields persist across run calls, not just the two pressure levels."""
    boundary = AbsorbingBoundary.open_field(THICKNESS, 1000.0)
    field = np.random.default_rng(1).standard_normal(GRID.shape)
    whole = c_backend.CScalarFDTD(GRID, absorbing_boundary=boundary)
    pieces = c_backend.CScalarFDTD(GRID, absorbing_boundary=boundary)
    whole.set_initial_pressure(field)
    pieces.set_initial_pressure(field)
    whole.run(41)
    for chunk in (5, 1, 17, 18):
        pieces.run(chunk)
    np.testing.assert_array_equal(pieces.p, whole.p)
    np.testing.assert_array_equal(pieces.previous, whole.previous)
    assert pieces.step_index == whole.step_index == 41


def test_thread_count_does_not_change_the_answer():
    boundary = AbsorbingBoundary.open_field(THICKNESS, 1000.0, faces=((0, 1), (1, 0), (2, 0)))
    signal = _ricker(100)
    results = []
    for threads in (1, 3, 8):
        compiled = c_backend.CScalarFDTD(GRID, walls=ASYMMETRIC, absorbing_boundary=boundary)
        compiled.set_threads(threads)
        compiled.add_volume_source(CENTRE, signal)
        compiled.run(100)
        results.append(compiled.p.copy())
    compiled.set_threads(0)
    for other in results[1:]:
        np.testing.assert_array_equal(results[0], other)


def test_single_precision_matches_the_single_precision_reference():
    boundary = AbsorbingBoundary.open_field(THICKNESS, 1000.0)
    signal = _ricker(150)
    reference, compiled = _pair(absorbing_boundary=boundary, dtype=np.float32)
    for sim in (reference, compiled):
        sim.add_volume_source(CENTRE, signal)
        sim.run(150)
    assert compiled.p.dtype == np.float32
    scale = float(np.max(np.abs(reference.p)))
    assert scale > 0.0
    assert float(np.max(np.abs(compiled.p - reference.p))) / scale < 1e-4


def test_the_shell_absorbs():
    """Not just equal to the reference: a lined room loses its energy, a closed one keeps it."""
    field = np.zeros(GRID.shape)
    field[9:13, 8:11, 7:10] = 1.0
    lined = c_backend.CScalarFDTD(GRID, absorbing_boundary=AbsorbingBoundary.open_field(4, 1000.0))
    closed = c_backend.CScalarFDTD(GRID)
    for sim in (lined, closed):
        sim.set_initial_pressure(field)
        sim.run(300)
    assert np.isfinite(lined.p).all()
    assert float(np.sum(lined.p**2)) < 1e-2 * float(np.sum(closed.p**2))


def test_shell_state_is_stored_in_the_shell_only():
    """The interior stays two arrays: no velocity or memory field the size of the grid."""
    grid = Grid(0.02, 96, 80, 72)
    compiled = c_backend.CScalarFDTD(
        grid, absorbing_boundary=AbsorbingBoundary.open_field(3, 1000.0)
    )
    buffers = {id(compiled.p), id(compiled.previous)}
    extra = []
    for value in vars(compiled).values():
        items = value.values() if isinstance(value, dict) else value
        items = items if isinstance(items, (list, tuple)) else [value]
        extra += [a for a in items if isinstance(a, np.ndarray) and id(a) not in buffers]
    assert not any(a.shape == grid.shape for a in extra)
    # The reference's layout (three velocities, three psi_v, three psi_p, all grid-sized) is
    # 9 * n_cells. Six slabs of depth 3 on 96 x 80 x 72 hold about 0.22 * n_cells cells, so nine
    # slab fields (corners stored once per overlapping slab) come to about 2 * n_cells.
    assert sum(a.size for a in extra) < 3 * grid.n_cells
    compiled.run(3)  # and it runs
    assert np.isfinite(compiled.p).all()


def test_a_shell_that_does_not_fit_is_refused():
    # 2 * 9 = 18 >= 17 = nz: refused, as the reference refuses it.
    with pytest.raises(ValueError):
        ScalarFDTD(GRID, absorbing_boundary=AbsorbingBoundary(thickness=9))
    with pytest.raises(ValueError):
        c_backend.CScalarFDTD(GRID, absorbing_boundary=AbsorbingBoundary(thickness=9))
    c_backend.CScalarFDTD(GRID, absorbing_boundary=AbsorbingBoundary(thickness=8))  # 16 < 17: fits


def test_no_boundary_is_unchanged():
    """``absorbing_boundary=None`` (the default) is the closed-room kernel, bit for bit."""
    field = np.random.default_rng(2).standard_normal(GRID.shape)
    explicit = c_backend.CScalarFDTD(GRID, walls=ASYMMETRIC, absorbing_boundary=None)
    default = c_backend.CScalarFDTD(GRID, walls=ASYMMETRIC)
    for sim in (explicit, default):
        sim.set_initial_pressure(field)
        sim.run(50)
    np.testing.assert_array_equal(explicit.p, default.p)
    assert explicit.absorbing_boundary is None

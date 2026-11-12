"""Prescribed (hard) sources on the compiled pressure-only 2D kernel: the reference, bit for bit.

Every case is checked against the NumPy :class:`~ac_fdtd.scalar2d.ScalarFDTD2D` with
``assert_array_equal`` in fp64, on both stored levels and on the recordings. A tolerance would hide
a driven face mapped to the wrong cell or axis, or a correction that skipped the wall relation at a
corner; neither is a round-off-sized effect, but some are small. This means the kernel's sweep, wall
post-pass and source passes must evaluate the reference's expressions in the reference's order.
"""

import numpy as np
import pytest

from ac_fdtd.boundaries2d import WallAdmittances2D
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scalar2d import ScalarFDTD2D

c_backend = pytest.importorskip("ac_fdtd.c_backend")

pytestmark = pytest.mark.skipif(not c_backend.is_available(), reason="no C compiler available")

# 31 x 25 cells: two different counts, so a swapped axis cannot pass by symmetry.
GRID = Grid2D.from_lengths((0.62, 0.5), 0.02)
DX = GRID.dx
STEPS = 200

INTERIOR = (15.5 * DX, 12.5 * DX)  # cell (15, 12)
X_WALL = (0.0, 12.5 * DX)  # lower x-face on the x_min wall line
Y_WALL = (15.5 * DX, 0.0)  # lower y-face on the y_min wall line
CORNER = (0.0, 0.0)

RECEIVERS = [(0.0, 0.0), (0.0, 0.3), (0.31, 0.49), (0.45, 0.2), INTERIOR]


def _drive(n: int = 60, scale: float = 1e-3) -> np.ndarray:
    """Smooth ramp to ``scale`` with a tone burst on top; ends at ``scale``, not at zero."""
    t = np.arange(n, dtype=float)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.minimum(t / 30.0, 1.0)))
    burst = 0.5 * np.sin(2.0 * np.pi * t / 13.0) * np.exp(-(((t - 25.0) / 10.0) ** 2))
    return scale * (ramp + burst)


def _pair(**kwargs):
    reference = ScalarFDTD2D(GRID, **kwargs)
    compiled = c_backend.CScalarFDTD2D(GRID, **kwargs)
    for point in RECEIVERS:
        reference.add_pressure_receiver(point)
        compiled.add_pressure_receiver(point)
    return reference, compiled


def _add(solvers, point, signal, start=0, kind="pressure", axis=None) -> None:
    for solver in solvers:
        solver.add_prescribed_source(point, signal, start=start, kind=kind, axis=axis)


def _assert_identical(reference, compiled, steps: int = STEPS) -> None:
    reference.run(steps)
    compiled.run(steps)
    assert np.max(np.abs(reference.p)) > 0.0
    np.testing.assert_array_equal(compiled.p, reference.p)
    np.testing.assert_array_equal(compiled.previous, reference.p_old)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)


def test_prescribed_pressure_sources_match():
    """Two held nodes; one window ends mid-run, so the release is checked too."""
    reference, compiled = _pair()
    _add((reference, compiled), INTERIOR, _drive(40, 1.0), start=5)
    _add((reference, compiled), (0.1, 0.4), -_drive(STEPS + 10, 0.5))
    _assert_identical(reference, compiled)


@pytest.mark.parametrize(
    ("point", "axis", "start"),
    [(INTERIOR, 0, 3), (INTERIOR, 1, 0), (X_WALL, 0, 5), (Y_WALL, 1, 2)],
    ids=["interior-x", "interior-y", "wall-x", "wall-y"],
)
def test_single_velocity_source_matches(point, axis, start):
    reference, compiled = _pair()
    _add((reference, compiled), point, _drive(), start=start, kind="velocity", axis=axis)
    _assert_identical(reference, compiled)


def test_rotated_pistons_in_an_absorbing_room_with_an_obstacle():
    """Everything at once: both axes on one interior cell and on the corner, impedance walls
    (asymmetric, so a wrong wall index shows), a face touching a solid cell (silenced), a soft
    volume source and a prescribed pressure node."""
    walls = WallAdmittances2D(x_min=0.3, x_max=0.7, y_min=0.1, y_max=1.0)
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[20:24, 10:14] = True
    reference, compiled = _pair(walls=walls, rigid_cells=rigid)
    solvers = (reference, compiled)
    angle = 0.6
    _add(solvers, INTERIOR, np.cos(angle) * _drive(), kind="velocity", axis=0)
    _add(solvers, INTERIOR, np.sin(angle) * _drive(), start=1, kind="velocity", axis=1)
    _add(solvers, CORNER, _drive(), start=3, kind="velocity", axis=0)
    _add(solvers, CORNER, -_drive(), start=3, kind="velocity", axis=1)
    _add(solvers, (24.5 * DX, 12.5 * DX), _drive(), kind="velocity", axis=0)  # touches the solid
    _add(solvers, (0.5, 0.45), _drive(30, 0.2), start=40)
    burst = np.sin(np.arange(80) / 5.0) * 1e-2
    reference.add_volume_source((0.15, 0.3), burst)
    compiled.add_volume_source((0.15, 0.3), burst)
    _assert_identical(reference, compiled, steps=300)
    assert np.all(compiled.p[rigid] == 0.0)


def test_several_sources_on_one_face_share_one_stored_value():
    """Driven, released, driven again, and an overlap where the later-added source wins."""
    reference, compiled = _pair()
    for signal, start in ((_drive(30), 0), (-_drive(30), 50), (0.5 * _drive(30), 70)):
        _add((reference, compiled), INTERIOR, signal, start=start, kind="velocity", axis=0)
    _assert_identical(reference, compiled)


def test_chunked_runs_carry_the_stored_faces():
    """The running face values live across ``run`` calls: windows straddle every chunk boundary."""
    reference, whole = _pair()
    _, pieces = _pair()
    for solver in (reference, whole, pieces):
        solver.add_prescribed_source(INTERIOR, _drive(), start=3, kind="velocity", axis=0)
        solver.add_prescribed_source(Y_WALL, _drive(), start=1, kind="velocity", axis=1)
        solver.add_prescribed_source((0.1, 0.4), _drive(25, 0.3), start=7)
    reference.run(120)
    whole.run(120)
    for chunk in (5, 1, 17, 40, 2, 55):
        pieces.run(chunk)
    assert pieces.step_index == whole.step_index == 120
    np.testing.assert_array_equal(pieces.p, whole.p)
    np.testing.assert_array_equal(pieces.previous, whole.previous)
    np.testing.assert_array_equal(pieces.recorded_pressure, whole.recorded_pressure)
    np.testing.assert_array_equal(whole.p, reference.p)


def test_thread_count_does_not_change_the_answer():
    results = []
    for threads in (1, 3, 8):
        compiled = c_backend.CScalarFDTD2D(GRID)
        compiled.set_threads(threads)
        compiled.add_prescribed_source(INTERIOR, _drive(), kind="velocity", axis=1)
        compiled.add_prescribed_source(X_WALL, _drive(), kind="velocity", axis=0)
        compiled.run(100)
        results.append(compiled.p.copy())
    compiled.set_threads(0)
    for other in results[1:]:
        np.testing.assert_array_equal(results[0], other)


def test_single_precision_tracks_the_single_precision_reference():
    reference, compiled = (
        ScalarFDTD2D(GRID, dtype=np.float32),
        c_backend.CScalarFDTD2D(GRID, dtype=np.float32),
    )
    for solver in (reference, compiled):
        solver.add_prescribed_source(INTERIOR, _drive(), kind="velocity", axis=0)
        solver.add_prescribed_source(Y_WALL, _drive(), kind="velocity", axis=1)
        solver.add_prescribed_source((0.1, 0.4), _drive(25, 0.3), start=7)
    reference.run(STEPS)
    compiled.run(STEPS)
    assert compiled.p.dtype == np.float32
    peak = float(np.max(np.abs(reference.p)))
    assert peak > 0.0
    assert float(np.max(np.abs(compiled.p - reference.p))) < 1e-5 * peak


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "velocity"},
        {"kind": "velocity", "axis": 2},
        {"kind": "pressure", "axis": 0},
        {"kind": "nonsense"},
        {"kind": "velocity", "axis": 0, "start": -1},
        {"kind": "pressure", "start": -1},
    ],
)
def test_contradictory_arguments_are_rejected(kwargs):
    compiled = c_backend.CScalarFDTD2D(GRID)
    with pytest.raises(ValueError):
        compiled.add_prescribed_source(INTERIOR, np.ones(4), **kwargs)

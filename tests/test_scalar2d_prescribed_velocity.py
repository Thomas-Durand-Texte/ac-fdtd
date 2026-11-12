"""Prescribed (hard) velocity sources in the pressure-only 2D scheme: still the first-order scheme.

A driven face is the one place the eliminated velocity is *not* the momentum step: the reference
overwrites it before the pressure update reads it. The pressure-only form keeps one running value
per driven face (as it does for a velocity probe) and folds the difference between the driven value
and the free momentum step into the two cells the face separates. Every case is checked against
:class:`AcousticFDTD2D` to 1e-10 of the peak, like ``tests/test_scalar2d.py``: the elimination is
exact, only the operation order differs.

The drive ends on a nonzero value on purpose: a driven *wall* face then keeps that value forever
(the piston that stopped but did not retract: steady mass inflow), while an interior face is
released to the momentum equation. Both regimes are therefore exercised by every equivalence run.
"""

import numpy as np
import pytest

from ac_fdtd.boundaries2d import WallAdmittances2D
from ac_fdtd.cpml2d import AbsorbingBoundary2D
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scalar2d import ScalarFDTD2D
from ac_fdtd.scheme2d import AcousticFDTD2D

# 31 x 25 cells: two different counts, so a swapped axis cannot pass by symmetry.
GRID = Grid2D.from_lengths((0.62, 0.5), 0.02)
DX = GRID.dx
STEPS = 200

# Cell centres, so the point -> cell mapping is never in doubt.
INTERIOR = (15.5 * DX, 12.5 * DX)  # cell (15, 12)
X_WALL = (0.0, 12.5 * DX)  # cell (0, 12): its lower x-face is the x_min wall line
Y_WALL = (15.5 * DX, 0.0)  # cell (15, 0): its lower y-face is the y_min wall line
CORNER = (0.0, 0.0)  # cell (0, 0): both lower faces are wall lines

RECEIVERS = [(0.0, 0.0), (0.0, 0.3), (0.31, 0.49), (0.45, 0.2), INTERIOR]


def _drive(n: int = 60, scale: float = 1e-3) -> np.ndarray:
    """Smooth ramp to ``scale`` with a tone burst on top; ends at ``scale``, not at zero."""
    t = np.arange(n, dtype=float)
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.minimum(t / 30.0, 1.0)))
    burst = 0.5 * np.sin(2.0 * np.pi * t / 13.0) * np.exp(-(((t - 25.0) / 10.0) ** 2))
    return scale * (ramp + burst)


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def _pair(**kwargs) -> tuple[AcousticFDTD2D, ScalarFDTD2D]:
    reference = AcousticFDTD2D(GRID, **kwargs)
    variant = ScalarFDTD2D(GRID, **kwargs)
    for point in RECEIVERS:
        reference.add_pressure_receiver(point)
        variant.add_pressure_receiver(point)
    return reference, variant


def _velocity(solvers, point, signal, axis, start=0) -> None:
    for solver in solvers:
        solver.add_prescribed_source(point, signal, start=start, kind="velocity", axis=axis)


def _assert_matches(reference: AcousticFDTD2D, variant: ScalarFDTD2D, steps: int = STEPS) -> None:
    reference.run(steps)
    variant.run(steps)
    assert np.max(np.abs(reference.p)) > 0.0  # the drive did something
    assert _max_rel(variant.p, reference.p) < 1e-10
    assert _max_rel(variant.recorded_pressure, reference.recorded_pressure) < 1e-10


# -- exact one-step oracle ------------------------------------------------------------------
@pytest.mark.parametrize(
    ("point", "axis"),
    [(INTERIOR, 0), (INTERIOR, 1), (X_WALL, 0), (Y_WALL, 1)],
    ids=["interior-x", "interior-y", "wall-x", "wall-y"],
)
def test_first_driven_step_from_rest_is_exact(point, axis):
    """From rest, ``p^{n+1} = -rho c^2 dt/dx * div v``: +K V in the cell above the face, -K V in
    the cell below it (absent on a wall line), zero everywhere else; nothing before ``start``."""
    value = 2.5e-3
    start = 4
    solver = ScalarFDTD2D(GRID)
    solver.add_prescribed_source(point, np.array([value]), start=start, kind="velocity", axis=axis)

    solver.run(start)
    assert np.all(solver.p == 0.0) and np.all(solver.p_old == 0.0)

    solver.step()
    medium = solver.medium
    k_value = medium.density * medium.sound_speed**2 * solver.dt / DX * value
    expected = np.zeros(GRID.shape)
    index = GRID.cell_index(point)
    expected[index] = k_value
    if index[axis] > 0:
        lower = list(index)
        lower[axis] -= 1
        expected[tuple(lower)] = -k_value
    np.testing.assert_allclose(solver.p, expected, rtol=1e-12, atol=1e-12 * abs(k_value))


# -- equivalence with the first-order scheme ------------------------------------------------
@pytest.mark.parametrize(
    ("point", "axis", "start"),
    [(INTERIOR, 0, 3), (INTERIOR, 1, 0), (X_WALL, 0, 5), (Y_WALL, 1, 2)],
    ids=["interior-x", "interior-y", "wall-x", "wall-y"],
)
def test_single_velocity_source_matches_the_first_order_scheme(point, axis, start):
    reference, variant = _pair()
    _velocity((reference, variant), point, _drive(), axis, start)
    _assert_matches(reference, variant)


@pytest.mark.parametrize("point", [INTERIOR, CORNER], ids=["interior", "corner"])
def test_both_axes_on_one_cell_matches(point):
    """The studio's rotated piston: one source per axis on the same cell, weighted cos / sin."""
    angle = 0.6
    reference, variant = _pair()
    _velocity((reference, variant), point, np.cos(angle) * _drive(), 0)
    _velocity((reference, variant), point, np.sin(angle) * _drive(), 1, start=1)
    _assert_matches(reference, variant)


def test_a_driven_wall_line_and_interior_sources_match():
    """A 2D loudspeaker: a line of driven x_min faces, plus independent interior sources of both
    axes with their own windows and a soft volume source."""
    reference, variant = _pair()
    for j in range(8, 17):
        _velocity((reference, variant), (0.0, (j + 0.5) * DX), _drive(), 0, start=2)
    _velocity((reference, variant), (0.4, 0.15), -_drive(40), 1, start=30)
    _velocity((reference, variant), (0.2, 0.4), _drive(50, 2e-3), 0, start=11)
    burst = np.sin(np.arange(80) / 5.0) * 1e-2
    reference.add_volume_source((0.5, 0.3), burst)
    variant.add_volume_source((0.5, 0.3), burst)
    _assert_matches(reference, variant)


def test_velocity_sources_against_impedance_walls_match():
    """Driven wall faces on absorbing walls (asymmetric admittances) and at a corner, where the
    correction goes through the three-level wall relation and the corner sums two coefficients."""
    walls = WallAdmittances2D(x_min=0.3, x_max=0.7, y_min=0.1, y_max=1.0)
    reference, variant = _pair(walls=walls)
    _velocity((reference, variant), X_WALL, _drive(), 0)
    _velocity((reference, variant), CORNER, _drive(), 0, start=3)
    _velocity((reference, variant), CORNER, -_drive(), 1, start=3)
    _velocity((reference, variant), ((30.5) * DX, 0.0), _drive(), 1)  # x_max corner, y_min wall
    _velocity((reference, variant), (1.5 * DX, 12.5 * DX), _drive(), 0, start=7)  # next to wall
    _assert_matches(reference, variant, steps=300)


def test_several_sources_on_one_face_share_one_stored_value():
    """One face, three windows: driven, released (free momentum evolution), driven again, and an
    overlap where the source added later wins (the reference overwrites in insertion order)."""
    reference, variant = _pair()
    _velocity((reference, variant), INTERIOR, _drive(30), 0, start=0)
    _velocity((reference, variant), INTERIOR, -_drive(30), 0, start=50)
    _velocity((reference, variant), INTERIOR, 0.5 * _drive(30), 0, start=70)
    _assert_matches(reference, variant)


def test_a_face_touching_a_rigid_cell_stays_closed():
    """The reference zeroes every face touching a solid after the prescribed overwrite, so a
    source there is silent; the other source still drives, and solid cells stay at exactly 0."""
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[20:24, 10:14] = True
    reference, variant = _pair(rigid_cells=rigid)
    _velocity((reference, variant), (24.5 * DX, 12.5 * DX), _drive(), 0)  # lower cell is solid
    _velocity((reference, variant), (21.5 * DX, 10.5 * DX), _drive(), 1)  # upper cell is solid
    _velocity((reference, variant), (10.5 * DX, 12.5 * DX), _drive(), 1)  # open face
    _assert_matches(reference, variant)
    assert np.all(variant.p[rigid] == 0.0)


def test_velocity_receiver_on_a_driven_face_reads_the_driven_value():
    """A velocity probe on a driven face must record the face's real velocity (driven, then free
    for an interior face; held for a wall face), as the reference's ``probe_velocity`` does."""
    reference, variant = _pair()
    _velocity((reference, variant), INTERIOR, _drive(), 0, start=4)
    _velocity((reference, variant), Y_WALL, _drive(), 1, start=4)
    probes = [(INTERIOR, 0), (Y_WALL, 1), ((16.5 * DX, 12.5 * DX), 0), (INTERIOR, 1)]
    channels = [variant.add_velocity_receiver(point, axis) for point, axis in probes]

    expected = [[] for _ in probes]
    for _ in range(150):
        for samples, (point, axis) in zip(expected, probes, strict=True):
            samples.append(reference.probe_velocity(point, axis))
        reference.step()
        variant.step()
    recorded = variant.recorded_velocity
    for channel, samples in zip(channels, expected, strict=True):
        assert _max_rel(recorded[channel], np.array(samples)) < 1e-10


# -- storage and the CPML shell -------------------------------------------------------------
def test_velocity_sources_add_no_field_sized_array():
    variant = ScalarFDTD2D(GRID)
    for j in range(GRID.ny):
        variant.add_prescribed_source((0.0, (j + 0.5) * DX), _drive(), kind="velocity", axis=0)
    variant.add_prescribed_source(INTERIOR, _drive(), kind="velocity", axis=1)
    variant.run(3)
    large = [
        name
        for name, value in vars(variant).items()
        if isinstance(value, np.ndarray) and value.size >= GRID.nx * GRID.ny // 2
    ]
    assert sorted(large) == ["p", "p_old"]


def test_interior_source_in_an_open_room_matches_the_closed_room_until_the_shell_hears_it():
    """With a CPML shell the interior runs the same update. The disturbance spreads one cell per
    step from the two cells beside each driven face; after 8 steps it reaches row 4 at most
    (3-cell shell on a 31 x 25 grid, the y-face sits between rows 11 and 12 of column 15), so no
    shell cell has moved and the two runs are identical bit for bit."""
    closed = ScalarFDTD2D(GRID)
    open_room = ScalarFDTD2D(GRID, absorbing_boundary=AbsorbingBoundary2D(thickness=3))
    signal = np.sin(np.arange(20) / 2.0) * 1e-3
    for solver in (closed, open_room):
        solver.add_prescribed_source(INTERIOR, signal, kind="velocity", axis=0)
        solver.add_prescribed_source(INTERIOR, signal[::-1], kind="velocity", axis=1)
    closed.run(8)
    open_room.run(8)
    assert np.any(closed.p != 0.0)
    np.testing.assert_array_equal(open_room.p, closed.p)


@pytest.mark.parametrize(
    ("point", "axis"),
    [((0.5 * DX, 12.5 * DX), 1), ((3.5 * DX, 12.5 * DX), 0)],
    ids=["cell-in-shell", "lower-cell-in-shell"],
)
def test_velocity_source_on_a_shell_face_is_rejected(point, axis):
    """Assumption (no reference has a CPML): a face bordering a shell cell is refused, not
    silently mis-driven."""
    solver = ScalarFDTD2D(GRID, absorbing_boundary=AbsorbingBoundary2D(thickness=3))
    with pytest.raises(ValueError):
        solver.add_prescribed_source(point, _drive(), kind="velocity", axis=axis)


# -- arguments ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "velocity"},
        {"kind": "velocity", "axis": 2},
        {"kind": "velocity", "axis": -1},
        {"kind": "pressure", "axis": 0},
        {"kind": "nonsense"},
        {"kind": "velocity", "axis": 0, "start": -1},
    ],
)
def test_contradictory_arguments_are_rejected(kwargs):
    solver = ScalarFDTD2D(GRID)
    with pytest.raises(ValueError):
        solver.add_prescribed_source(INTERIOR, np.ones(4), **kwargs)

"""The compiled pressure-only 3D kernel with locally-reacting walls must BE the reference scheme.

Every walled case is checked against the NumPy :class:`~ac_fdtd.scalar.ScalarFDTD` with
``assert_array_equal`` in fp64, on both stored levels: the wall relation
``(1+a) p^{n+1} = 2 p^n - (1-a) p^{n-1} + lam^2 L_N(p^n)`` is evaluated in the reference's operation
order, so there is no round-off to excuse. A tolerance would hide a face mapped to the wrong
array index, which at these sizes is a 1e-3 effect, not a 1e-14 one — but also a corner whose
coefficients were not summed, which can be far smaller.
"""

import numpy as np
import pytest

from ac_fdtd.boundaries import WallAdmittances
from ac_fdtd.grid import Grid
from ac_fdtd.scalar import ScalarFDTD

c_backend = pytest.importorskip("ac_fdtd.c_backend")

pytestmark = pytest.mark.skipif(not c_backend.is_available(), reason="no C compiler available")

# 20 x 17 x 15: three different counts, so a swapped axis cannot pass by symmetry.
GRID = Grid.from_lengths((0.6, 0.5, 0.45), 0.03)

# Six different admittances, one per face, so a face read from the wrong side or axis shows.
ASYMMETRIC = WallAdmittances(x_min=0.1, x_max=0.7, y_min=0.25, y_max=1.0, z_min=0.5, z_max=0.9)


def _field(seed: int, grid: Grid = GRID) -> np.ndarray:
    # White noise touches every cell, every face and every corner from the first step.
    return np.random.default_rng(seed).standard_normal(grid.shape)


def _pair(grid: Grid = GRID, **kwargs):
    reference = ScalarFDTD(grid, **kwargs)
    compiled = c_backend.CScalarFDTD(grid, **kwargs)
    return reference, compiled


def _assert_same_state(compiled, reference) -> None:
    np.testing.assert_array_equal(compiled.p, reference.p)
    np.testing.assert_array_equal(compiled.previous, reference.p_old)


@pytest.mark.parametrize(
    "walls",
    [
        WallAdmittances.matched(),
        WallAdmittances.from_absorption(0.3),
        ASYMMETRIC,
        # rigid and absorbing faces mixed: the edges where they meet carry one face's a only
        WallAdmittances(x_min=0.0, x_max=0.7, y_min=0.2, y_max=0.0, z_min=0.5, z_max=0.0),
        # the inner (contiguous, peeled) axis alone
        WallAdmittances(z_max=0.8),
        # the outer (threaded) axis alone
        WallAdmittances(x_min=0.8),
    ],
    ids=["matched", "alpha0.3", "asymmetric", "mixed-rigid", "z_max-only", "x_min-only"],
)
def test_walls_match_the_reference_bit_for_bit(walls):
    reference, compiled = _pair(walls=walls)
    field = _field(0)
    reference.set_initial_pressure(field)
    compiled.set_initial_pressure(field)
    reference.run(150)
    compiled.run(150)
    _assert_same_state(compiled, reference)
    assert compiled.step_index == reference.step_index == 150


def test_walls_actually_absorb():
    """Guards the guard: a kernel that ignored ``walls`` would be the rigid run, not the walled
    one. Matched walls drain a closed room, which a rigid one never does."""
    field = _field(1)
    rigid = c_backend.CScalarFDTD(GRID)
    matched = c_backend.CScalarFDTD(GRID, walls=WallAdmittances.matched())
    rigid.set_initial_pressure(field)
    matched.set_initial_pressure(field)
    rigid.run(300)
    matched.run(300)
    assert float(np.sum(matched.p**2)) < 0.05 * float(np.sum(rigid.p**2))


@pytest.mark.parametrize("steps", [1, 2, 7])
def test_moving_start_uses_the_previous_level_at_the_wall(steps):
    """``p^{-1} != p^0`` (a field already in motion): the ``(1-a) p^{n-1}`` term is live on the
    very first step, so a kernel that read the wrong level there differs at once."""
    reference, compiled = _pair(walls=ASYMMETRIC)
    now, before = _field(2), _field(3)
    reference.p[:] = now
    reference.p_old[:] = before
    compiled.p[:] = now
    compiled.previous[:] = before
    reference.run(steps)
    compiled.run(steps)
    _assert_same_state(compiled, reference)


@pytest.mark.parametrize("nz", [1, 2])
def test_thin_grid_where_opposite_walls_share_cells(nz):
    """With one or two cells along z, every cell is on both z walls (nz=1) or on one of them
    next to the other (nz=2): the summed corner coefficients and the z faces peeled out of the
    inner loop must both still be right."""
    grid = Grid(dx=0.03, nx=9, ny=7, nz=nz)
    reference, compiled = _pair(grid, walls=ASYMMETRIC)
    field = _field(4, grid)
    reference.set_initial_pressure(field)
    compiled.set_initial_pressure(field)
    reference.run(60)
    compiled.run(60)
    _assert_same_state(compiled, reference)


def test_below_the_courant_limit():
    """``a = (c dt / 2dx) xi`` depends on dt, which depends on courant."""
    reference, compiled = _pair(walls=ASYMMETRIC, courant=0.7)
    field = _field(5)
    reference.set_initial_pressure(field)
    compiled.set_initial_pressure(field)
    reference.run(100)
    compiled.run(100)
    _assert_same_state(compiled, reference)


def test_chunked_walled_runs_equal_one_run():
    """The wall needs ``p^{n-1}`` at the wall cells; a chunk boundary must not lose it, and the
    caller must still find the newest level after the C side swapped pointers an odd number of
    times."""
    field = _field(6)
    whole = c_backend.CScalarFDTD(GRID, walls=ASYMMETRIC)
    pieces = c_backend.CScalarFDTD(GRID, walls=ASYMMETRIC)
    whole.set_initial_pressure(field)
    pieces.set_initial_pressure(field)
    whole.run(21)
    for chunk in (5, 1, 7, 0, 8):
        pieces.run(chunk)
    np.testing.assert_array_equal(pieces.p, whole.p)
    np.testing.assert_array_equal(pieces.previous, whole.previous)
    assert pieces.step_index == whole.step_index == 21


def test_thread_count_does_not_change_the_answer():
    field = _field(7)
    results = []
    try:
        for threads in (1, 3, 8):
            compiled = c_backend.CScalarFDTD(GRID, walls=ASYMMETRIC)
            assert compiled.set_threads(threads) >= 1
            compiled.set_initial_pressure(field)
            compiled.run(80)
            results.append(compiled.p.copy())
    finally:
        c_backend.CScalarFDTD(GRID).set_threads(0)
    for other in results[1:]:
        np.testing.assert_array_equal(results[0], other)


def test_rigid_default_is_unchanged():
    """``walls=None`` and an all-zero ``WallAdmittances`` are the same rigid box, and it still
    reproduces the reference (to round-off: the rigid path predates the bit-exact contract)."""
    field = _field(8)
    default = c_backend.CScalarFDTD(GRID)
    explicit = c_backend.CScalarFDTD(GRID, walls=WallAdmittances())
    assert default.walls == explicit.walls == WallAdmittances()
    reference = ScalarFDTD(GRID)
    for sim in (default, explicit, reference):
        sim.set_initial_pressure(field)
        sim.run(100)
    np.testing.assert_array_equal(explicit.p, default.p)
    np.testing.assert_allclose(default.p, reference.p, rtol=1e-11, atol=1e-12)


def test_walls_keyword_matches_the_reference_signature():
    """Same constructor arguments as ScalarFDTD, in the same order: walls follows dtype."""
    compiled = c_backend.CScalarFDTD(GRID, ScalarFDTD(GRID).medium, 1.0, np.float64, ASYMMETRIC)
    assert compiled.walls == ASYMMETRIC


def test_single_precision_tracks_the_single_precision_reference():
    reference, compiled = _pair(walls=ASYMMETRIC, dtype=np.float32)
    field = _field(9).astype(np.float32)
    reference.set_initial_pressure(field)
    compiled.set_initial_pressure(field)
    reference.run(200)
    compiled.run(200)
    assert compiled.p.dtype == np.float32
    error = np.max(np.abs(compiled.p - reference.p)) / max(np.max(np.abs(reference.p)), 1e-30)
    assert error < 1e-4


def test_invalid_courant_is_rejected():
    with pytest.raises(ValueError):
        c_backend.CScalarFDTD(GRID, courant=1.5, walls=ASYMMETRIC)

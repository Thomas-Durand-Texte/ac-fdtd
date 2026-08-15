"""The pressure-only variant is the same scheme, not a similar one — so it must prove it."""

import numpy as np
import pytest

from ac_fdtd.grid import Grid
from ac_fdtd.scheme import AcousticFDTD

c_backend = pytest.importorskip("ac_fdtd.c_backend")

pytestmark = pytest.mark.skipif(not c_backend.is_available(), reason="no C compiler available")

GRID = Grid.from_lengths((0.6, 0.5, 0.45), 0.03)


def _initial() -> np.ndarray:
    return np.random.default_rng(0).standard_normal(GRID.shape)


@pytest.mark.parametrize("steps", [1, 2, 7, 200])
def test_it_reproduces_the_pressure_velocity_scheme(steps):
    """Eliminating velocity is algebra, not approximation, so the two agree to round-off.

    The correspondence is exact only for matching initial conditions: ``v^{-1/2} = 0`` in the
    first-order form is ``p^{-1} = p^0`` in the three-level one.
    """
    initial = _initial()
    reference = AcousticFDTD(GRID)
    reference.p[:] = initial
    variant = c_backend.CScalarFDTD(GRID)
    variant.set_initial_pressure(initial)

    reference.run(steps)
    variant.run(steps)
    np.testing.assert_allclose(variant.p, reference.p, rtol=1e-11, atol=1e-13)


def test_the_newest_level_is_tracked_across_an_odd_number_of_steps():
    """The C loop swaps its own pointers, so the caller has to count. Off by one and the
    returned field is a step stale — which looks like a slightly wrong wave speed."""
    initial = _initial()
    whole = c_backend.CScalarFDTD(GRID)
    whole.set_initial_pressure(initial)
    pieces = c_backend.CScalarFDTD(GRID)
    pieces.set_initial_pressure(initial)

    whole.run(21)
    for chunk in (5, 1, 7, 8):
        pieces.run(chunk)

    np.testing.assert_array_equal(pieces.p, whole.p)
    assert pieces.step_index == whole.step_index == 21


def test_at_rest_the_two_levels_agree():
    variant = c_backend.CScalarFDTD(GRID)
    variant.set_initial_pressure(_initial())
    np.testing.assert_array_equal(variant.p, variant.previous)


def test_the_thread_count_does_not_change_the_answer():
    initial = _initial()
    results = []
    for threads in (1, 3, 8):
        variant = c_backend.CScalarFDTD(GRID)
        variant._library.acfdtd_set_threads(threads)
        variant.set_initial_pressure(initial)
        variant.run(60)
        results.append(variant.p.copy())
    variant._library.acfdtd_set_threads(0)

    for other in results[1:]:
        np.testing.assert_array_equal(results[0], other)


def test_single_precision_tracks_double():
    initial = _initial()
    double = c_backend.CScalarFDTD(GRID, dtype=np.float64)
    single = c_backend.CScalarFDTD(GRID, dtype=np.float32)
    double.set_initial_pressure(initial)
    single.set_initial_pressure(initial)

    double.run(300)
    single.run(300)
    error = np.max(np.abs(double.p - single.p)) / np.max(np.abs(double.p))
    assert error < 1e-4


def test_a_rigid_box_keeps_its_energy_roughly_where_it_started():
    """No energy functional is defined for the three-level form here, so this is only a
    sanity bound: a lossless rigid box must not gain or lose the bulk of its amplitude."""
    variant = c_backend.CScalarFDTD(GRID)
    variant.set_initial_pressure(_initial())
    start = float(np.sum(variant.p**2))
    variant.run(2000)
    assert 0.2 * start < float(np.sum(variant.p**2)) < 5.0 * start

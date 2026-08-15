"""The compiled backend has to be the same computation as the reference, on any thread count."""

import numpy as np
import pytest

from ac_fdtd.air import AirAbsorption
from ac_fdtd.boundaries import AbsorbingLayer, WallAdmittances
from ac_fdtd.grid import Grid
from ac_fdtd.scheme import AcousticFDTD

c_backend = pytest.importorskip("ac_fdtd.c_backend")

pytestmark = pytest.mark.skipif(not c_backend.is_available(), reason="no C compiler available")

GRID = Grid.from_lengths((0.6, 0.5, 0.45), 0.03)
SOURCE = (0.2, 0.2, 0.2)
RECEIVER = (0.4, 0.3, 0.3)

CONFIGURATIONS = {
    "rigid": {},
    "absorbing walls": {"walls": WallAdmittances.uniform(0.3)},
    "absorbing layer": {"absorbing_layer": AbsorbingLayer(thickness=4)},
    "air absorption": {"air_absorption": AirAbsorption()},
    "everything at once": {
        "walls": WallAdmittances(x_min=0.2, z_max=0.7),
        "absorbing_layer": AbsorbingLayer(thickness=3),
        "air_absorption": AirAbsorption(temperature=15.0, relative_humidity=70.0),
    },
}


def _paired(dtype=np.float64, **configuration):
    reference = AcousticFDTD(GRID, dtype=dtype, **configuration)
    candidate = c_backend.CAcousticFDTD(GRID, dtype=dtype, **configuration)

    initial = np.random.default_rng(0).standard_normal(GRID.shape)
    signal = np.exp(-(((np.arange(200) - 30) / 8.0) ** 2))
    for solver in (reference, candidate):
        solver.p[:] = initial
        solver.add_volume_source(SOURCE, signal)
        solver.add_pressure_receiver(RECEIVER)
    return reference, candidate


def test_the_ctypes_mirror_matches_the_c_struct():
    """Constructing the solver checks this, so reaching it at all is the test passing."""
    solver = c_backend.CAcousticFDTD(GRID)
    assert solver.threads >= 1


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_matches_the_reference(name):
    reference, candidate = _paired(**CONFIGURATIONS[name])
    reference.run(150)
    candidate.run(150)

    np.testing.assert_allclose(candidate.p, reference.p, rtol=1e-11, atol=1e-13)
    for axis in range(3):
        np.testing.assert_allclose(
            candidate.velocity[axis], reference.velocity[axis], rtol=1e-11, atol=1e-13
        )
    np.testing.assert_allclose(
        candidate.recorded_pressure, reference.recorded_pressure, rtol=1e-11, atol=1e-14
    )
    assert candidate.energy() == pytest.approx(reference.energy(), rel=1e-11)


def test_the_thread_count_does_not_change_the_answer():
    """The one property a hand-written thread pool has to have, and the easiest to lose."""
    results = []
    for threads in (1, 3, 8):
        solver = c_backend.CAcousticFDTD(GRID, walls=WallAdmittances.uniform(0.2))
        solver.set_threads(threads)
        assert solver.threads == threads
        solver.p[:] = np.random.default_rng(1).standard_normal(GRID.shape)
        solver.add_volume_source(SOURCE, np.ones(100))
        solver.run(120)
        results.append(solver.p.copy())

    solver.set_threads(0)
    for other in results[1:]:
        np.testing.assert_array_equal(results[0], other)


def test_a_rebuilt_pool_does_not_reuse_the_previous_run(tmp_path):
    """Regression: new workers used to inherit a stale generation counter and a freed pointer.

    They would run one chunk against the previous run's state before waiting for work. It
    surfaced as a bus error a long way from the cause, and only when the thread count changed
    between runs — which is exactly what a benchmark does.
    """
    for threads in (4, 1, 6, 2):
        solver = c_backend.CAcousticFDTD(GRID)
        solver.set_threads(threads)
        solver.p[:] = np.random.default_rng(2).standard_normal(GRID.shape)
        solver.run(30)
        assert np.all(np.isfinite(solver.p))
    solver.set_threads(0)


def test_single_precision_stays_close_to_double():
    reference, _ = _paired()
    _, single = _paired(dtype=np.float32)
    reference.run(400)
    single.run(400)

    recorded = reference.recorded_pressure[0]
    error = np.max(np.abs(recorded - single.recorded_pressure[0])) / np.max(np.abs(recorded))
    assert error < 1e-4


def test_running_in_pieces_is_the_same_as_running_at_once():
    """The compiled loop takes a step offset, so sources have to line up across calls."""
    whole, pieces = _paired()
    whole.run(120)
    for _ in range(4):
        pieces.run(30)

    np.testing.assert_array_equal(pieces.p, whole.p)
    np.testing.assert_array_equal(pieces.recorded_pressure, whole.recorded_pressure)


def test_build_report_says_how_it_was_built():
    report = c_backend.build_report()
    assert report["threads"] >= 1
    assert "-O3" in report["flags"]

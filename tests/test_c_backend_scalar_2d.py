"""The compiled pressure-only 2D kernel must be the reference scheme, not a near miss.

Every case is checked against the NumPy ScalarFDTD2D (which is itself bit-identical to the
first-order AcousticFDTD2D), across rigid / impedance walls / obstacles / driven, single and double
precision, chunked runs and thread counts.
"""

import numpy as np
import pytest

from ac_fdtd.boundaries2d import WallAdmittances2D
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scalar2d import ScalarFDTD2D

c_backend = pytest.importorskip("ac_fdtd.c_backend")

pytestmark = pytest.mark.skipif(not c_backend.is_available(), reason="no C compiler available")

GRID = Grid2D.from_lengths((1.3, 1.1), 0.02)


def _ricker(n: int) -> np.ndarray:
    t = np.arange(n)
    return (1.0 - 0.5 * ((t - 60) / 20) ** 2) * np.exp(-(((t - 60) / 20) ** 2) / 2.0)


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def test_rigid_box_matches_the_reference():
    field = np.random.default_rng(0).standard_normal(GRID.shape)
    reference = ScalarFDTD2D(GRID)
    reference.set_initial_pressure(field)
    compiled = c_backend.CScalarFDTD2D(GRID)
    compiled.set_initial_pressure(field)
    reference.run(200)
    compiled.run(200)
    np.testing.assert_allclose(compiled.p, reference.p, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize(
    "walls",
    [WallAdmittances2D.matched(), WallAdmittances2D.from_absorption(0.3)],
)
def test_impedance_walls_and_receiver_match(walls):
    signal = _ricker(400)
    reference = ScalarFDTD2D(GRID, walls=walls)
    compiled = c_backend.CScalarFDTD2D(GRID, walls=walls)
    reference.add_volume_source((0.4, 0.5), signal)
    compiled.add_volume_source((0.4, 0.5), signal)
    reference.add_pressure_receiver((0.7, 0.6))
    compiled.add_pressure_receiver((0.7, 0.6))
    reference.run(400)
    compiled.run(400)
    assert _max_rel(compiled.p, reference.p) < 1e-10
    assert _max_rel(compiled.recorded_pressure, reference.recorded_pressure) < 1e-10


def test_rigid_obstacle_matches_and_stays_solid():
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[20:30, 15:18] = True
    field = np.random.default_rng(1).standard_normal(GRID.shape) * (~rigid)
    reference = ScalarFDTD2D(GRID, rigid_cells=rigid)
    reference.set_initial_pressure(field)
    compiled = c_backend.CScalarFDTD2D(GRID, rigid_cells=rigid)
    compiled.set_initial_pressure(field)
    reference.run(300)
    compiled.run(300)
    assert _max_rel(compiled.p, reference.p) < 1e-10
    assert np.all(compiled.p[rigid] == 0.0)


def test_obstacle_in_an_absorbing_room_driven():
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[40:44, 20:40] = True
    walls = WallAdmittances2D.from_absorption(0.4)
    signal = _ricker(400)
    reference = ScalarFDTD2D(GRID, walls=walls, rigid_cells=rigid)
    compiled = c_backend.CScalarFDTD2D(GRID, walls=walls, rigid_cells=rigid)
    reference.add_volume_source((0.3, 0.6), signal)
    compiled.add_volume_source((0.3, 0.6), signal)
    reference.run(400)
    compiled.run(400)
    assert _max_rel(compiled.p, reference.p) < 1e-10


def test_chunked_runs_track_the_pointer_swap():
    field = np.random.default_rng(2).standard_normal(GRID.shape)
    whole = c_backend.CScalarFDTD2D(GRID)
    whole.set_initial_pressure(field)
    pieces = c_backend.CScalarFDTD2D(GRID)
    pieces.set_initial_pressure(field)
    whole.run(21)
    for chunk in (5, 1, 7, 8):
        pieces.run(chunk)
    np.testing.assert_array_equal(pieces.p, whole.p)
    assert pieces.step_index == whole.step_index == 21


def test_single_precision_tracks_double():
    signal = _ricker(300)
    double = c_backend.CScalarFDTD2D(GRID, dtype=np.float64)
    single = c_backend.CScalarFDTD2D(GRID, dtype=np.float32)
    double.add_volume_source((0.4, 0.5), signal)
    single.add_volume_source((0.4, 0.5), signal)
    double.run(300)
    single.run(300)
    error = np.max(np.abs(double.p - single.p)) / max(np.max(np.abs(double.p)), 1e-30)
    assert error < 1e-4


def test_thread_count_does_not_change_the_answer():
    signal = _ricker(200)
    results = []
    for threads in (1, 3, 8):
        compiled = c_backend.CScalarFDTD2D(GRID)
        compiled.set_threads(threads)
        compiled.add_volume_source((0.4, 0.5), signal)
        compiled.run(200)
        results.append(compiled.p.copy())
    compiled.set_threads(0)
    for other in results[1:]:
        np.testing.assert_array_equal(results[0], other)

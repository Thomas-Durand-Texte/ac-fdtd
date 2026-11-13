"""Rigid obstacles on the compiled pressure-only 3D kernel must BE the reference's.

``ScalarFDTD`` closes every face that has a solid cell on either side (its Laplacian multiplies the
face's difference by an ``open`` factor) and zeroes the solid cells' new pressure at the end of
each step. Every case with a solid cell is compared to it with ``assert_array_equal`` in fp64, the
all-rigid room included: the fast rigid sweep has no mask, so a mask selects the exact path. The
masks put obstacles against walls, edges and a corner, and leave one-cell gaps, where a face
closed on the wrong side or a wall cell solved with the unmasked Laplacian shows at once.
"""

import inspect

import numpy as np
import pytest

from ac_fdtd.boundaries import WallAdmittances
from ac_fdtd.grid import Grid
from ac_fdtd.medium import AIR
from ac_fdtd.scalar import ScalarFDTD

c_backend = pytest.importorskip("ac_fdtd.c_backend")

pytestmark = pytest.mark.skipif(not c_backend.is_available(), reason="no C compiler available")

# 20 x 17 x 15: three different counts, so a swapped axis cannot pass by symmetry.
GRID = Grid.from_lengths((0.6, 0.5, 0.45), 0.03)

ASYMMETRIC = WallAdmittances(x_min=0.1, x_max=0.7, y_min=0.25, y_max=1.0, z_min=0.5, z_max=0.9)
WALLS = [None, ASYMMETRIC, WallAdmittances(x_min=0.8)]
WALL_IDS = ["rigid", "asymmetric", "x_min-only"]

INTERIOR = (0.31, 0.22, 0.17)  # cell (10, 7, 5): open in every mask below


def _block() -> np.ndarray:
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[5:9, 9:13, 6:10] = True
    return rigid


def _complex() -> np.ndarray:
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[0:3, 4:8, 5:9] = True  # against the x_min wall
    rigid[4:6, 4:8, 5:9] = True  # one-cell gap (x = 3) to the block above
    rigid[17:20, 0:2, 0:3] = True  # in the (x_max, y_min, z_min) corner
    rigid[8:11, 14:17, 12:15] = True  # along the (y_max, z_max) edge
    rigid[11:14, 1:3, 9:12] = True  # one-cell gap (y = 0) to the y_min wall
    rigid[14, :, :] = True  # a plate across the room, touching four walls...
    rigid[14, 8, 7] = False  # ...pierced by a one-cell hole
    rigid[7, 12, 3] = True  # an isolated cell
    return rigid


MASKS = {"block": _block, "complex": _complex}


def _signal(seed: int, n: int = 40) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def _start(seed: int = 0) -> np.ndarray:
    # Nonzero everywhere, solid cells included: set_initial_pressure does not zero them, and the
    # first step must make that harmless exactly as the reference does.
    return np.random.default_rng(seed).standard_normal(GRID.shape)


def _pair(**kwargs):
    return ScalarFDTD(GRID, **kwargs), c_backend.CScalarFDTD(GRID, **kwargs)


def _both(reference, compiled, method: str, *args):
    first = getattr(reference, method)(*args)
    second = getattr(compiled, method)(*args)
    return first, second


def _assert_same_state(compiled, reference) -> None:
    np.testing.assert_array_equal(compiled.p, reference.p)
    np.testing.assert_array_equal(compiled.previous, reference.p_old)


def test_rigid_cells_keyword_matches_the_reference_signature():
    reference = list(inspect.signature(ScalarFDTD).parameters)
    compiled = list(inspect.signature(c_backend.CScalarFDTD).parameters)
    assert compiled[:6] == reference[:6]
    assert compiled[5] == "rigid_cells"
    assert inspect.signature(c_backend.CScalarFDTD).parameters["rigid_cells"].default is None
    # Positional, as in the reference.
    compiled_instance = c_backend.CScalarFDTD(GRID, AIR, 1.0, np.float64, None, _block())
    reference_instance = ScalarFDTD(GRID, AIR, 1.0, np.float64, None, _block())
    compiled_instance.set_initial_pressure(_start(1))
    reference_instance.set_initial_pressure(_start(1))
    compiled_instance.run(5)
    reference_instance.run(5)
    _assert_same_state(compiled_instance, reference_instance)


def test_mask_is_cast_to_bool_and_checked():
    as_int = _complex().astype(np.int32)
    reference, compiled = _pair(walls=ASYMMETRIC, rigid_cells=as_int)
    _both(reference, compiled, "set_initial_pressure", _start(2))
    reference.run(20)
    compiled.run(20)
    _assert_same_state(compiled, reference)
    for shape in [(20, 17, 14), (15, 17, 20), (20, 17)]:
        with pytest.raises(ValueError, match="rigid_cells"):
            c_backend.CScalarFDTD(GRID, rigid_cells=np.zeros(shape, dtype=bool))


@pytest.mark.parametrize("mask", list(MASKS), ids=list(MASKS))
@pytest.mark.parametrize("walls", WALLS, ids=WALL_IDS)
def test_obstacles_match_the_reference_bit_for_bit(walls, mask):
    rigid = MASKS[mask]()
    reference, compiled = _pair(walls=walls, rigid_cells=rigid)
    _both(reference, compiled, "set_initial_pressure", _start(3))
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(4))
    _both(reference, compiled, "add_volume_source", (0.0, 0.0, 0.0), _signal(5))
    for point in [INTERIOR, (0.6, 0.5, 0.45), (0.1, 0.2, 0.2), (0.42, 0.25, 0.22)]:
        _both(reference, compiled, "add_pressure_receiver", point)
    reference.run(90)
    compiled.run(90)
    assert np.max(np.abs(reference.p)) > 0.0
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    _assert_same_state(compiled, reference)
    assert np.all(compiled.p[rigid] == 0.0)
    assert np.all(compiled.previous[rigid] == 0.0)


def test_a_source_inside_a_solid_cell_does_nothing():
    rigid = _complex()
    solid_point = (0.06, 0.15, 0.18)  # cell (2, 5, 6), inside the block on the x_min wall
    assert rigid[GRID.cell_index(solid_point)]
    with_it, without = (
        c_backend.CScalarFDTD(GRID, walls=ASYMMETRIC, rigid_cells=rigid) for _ in range(2)
    )
    reference = ScalarFDTD(GRID, walls=ASYMMETRIC, rigid_cells=rigid)
    for simulation in (with_it, without, reference):
        simulation.add_volume_source(INTERIOR, _signal(6))
    with_it.add_volume_source(solid_point, _signal(7))
    reference.add_volume_source(solid_point, _signal(7))
    for simulation in (with_it, without, reference):
        simulation.run(50)
    np.testing.assert_array_equal(with_it.p, without.p)
    _assert_same_state(with_it, reference)


def test_sources_and_receivers_around_obstacles():
    """A pressure receiver in a solid cell records zeros; a velocity probe with one solid cell is
    rebuilt from the two pressures like any other (not zero), one with two solid cells is zero."""
    rigid = _complex()
    reference, compiled = _pair(walls=ASYMMETRIC, rigid_cells=rigid)
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(8))
    gap = (0.09, 0.15, 0.18)  # (3, 5, 6): the one-cell gap between two blocks
    _both(reference, compiled, "add_volume_source", gap, _signal(9))
    solid = (0.06, 0.15, 0.18)  # (2, 5, 6)
    _both(reference, compiled, "add_pressure_receiver", solid)
    _both(reference, compiled, "add_pressure_receiver", (0.09, 0.15, 0.18))
    probes = [
        ((0.09, 0.15, 0.18), 0),  # faces (2|3): solid below, open here
        ((0.06, 0.15, 0.18), 0),  # faces (1|2): both solid
        ((0.42, 0.24, 0.21), 0),  # faces (13|14) at the hole (14, 8, 7): both open
        ((0.45, 0.15, 0.21), 0),  # faces (14|15): solid below, open here
        (INTERIOR, 1),
    ]
    for point, axis in probes:
        _both(reference, compiled, "add_velocity_receiver", point, axis)
    reference.run(80)
    compiled.run(80)
    pressure = compiled.recorded_pressure
    velocity = compiled.recorded_velocity
    np.testing.assert_array_equal(pressure, reference.recorded_pressure)
    np.testing.assert_array_equal(velocity, reference.recorded_velocity)
    _assert_same_state(compiled, reference)
    assert np.all(pressure[0] == 0.0)
    assert np.max(np.abs(pressure[1])) > 0.0
    assert np.max(np.abs(velocity[0])) > 0.0
    assert np.all(velocity[1] == 0.0)
    assert np.max(np.abs(velocity[2])) > 0.0
    assert np.max(np.abs(velocity[3])) > 0.0


def test_a_solid_partition_blocks_everything():
    rigid = np.zeros(GRID.shape, dtype=bool)
    rigid[10, :, :] = True
    reference, compiled = _pair(walls=ASYMMETRIC, rigid_cells=rigid)
    _both(reference, compiled, "add_volume_source", (0.1, 0.2, 0.2), _signal(10))
    _both(reference, compiled, "add_volume_source", (0.0, 0.5, 0.0), _signal(11))
    _both(reference, compiled, "add_pressure_receiver", (0.5, 0.25, 0.2))
    reference.run(100)
    compiled.run(100)
    _assert_same_state(compiled, reference)
    assert np.max(np.abs(compiled.p[:10])) > 0.0
    assert np.all(compiled.p[10:] == 0.0)
    assert np.all(compiled.previous[10:] == 0.0)
    assert np.all(compiled.recorded_pressure == 0.0)


def test_an_all_solid_room_falls_silent():
    rigid = np.ones(GRID.shape, dtype=bool)
    reference, compiled = _pair(rigid_cells=rigid)
    _both(reference, compiled, "set_initial_pressure", _start(12))
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(13))
    _both(reference, compiled, "add_pressure_receiver", INTERIOR)
    reference.run(1)
    compiled.run(1)
    _assert_same_state(compiled, reference)
    assert np.all(compiled.p == 0.0)
    reference.run(6)
    compiled.run(6)
    _assert_same_state(compiled, reference)
    assert np.all(compiled.p == 0.0) and np.all(compiled.previous == 0.0)
    assert np.all(compiled.recorded_pressure == 0.0)


@pytest.mark.parametrize("walls", [None, ASYMMETRIC], ids=["rigid", "asymmetric"])
def test_an_empty_mask_is_no_mask(walls):
    empty = np.zeros(GRID.shape, dtype=bool)
    reference, compiled = _pair(walls=walls, rigid_cells=empty)
    unmasked = c_backend.CScalarFDTD(GRID, walls=walls)
    for simulation in (reference, compiled, unmasked):
        simulation.set_initial_pressure(_start(14))
        simulation.add_volume_source(INTERIOR, _signal(15))
        simulation.run(60)
    if walls is None:
        # The fast rigid sweep may be kept: round-off only.
        scale = np.max(np.abs(reference.p))
        np.testing.assert_allclose(compiled.p, reference.p, rtol=0, atol=1e-11 * scale)
        np.testing.assert_allclose(compiled.p, unmasked.p, rtol=0, atol=1e-11 * scale)
    else:
        _assert_same_state(compiled, reference)
        np.testing.assert_array_equal(compiled.p, unmasked.p)


def test_chunked_runs_with_obstacles():
    """The masks are part of the state, not of a chunk: they must hold across chunk boundaries
    and across the buffer swap of odd chunks."""
    rigid = _complex()
    reference, compiled = _pair(walls=ASYMMETRIC, rigid_cells=rigid)
    _both(reference, compiled, "set_initial_pressure", _start(16))
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(17, 25))
    _both(reference, compiled, "add_pressure_receiver", (0.5, 0.1, 0.4))
    _both(reference, compiled, "add_velocity_receiver", (0.45, 0.15, 0.21), 0)
    reference.run(61)
    for chunk in (5, 1, 7, 0, 8, 40):
        compiled.run(chunk)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    np.testing.assert_array_equal(compiled.recorded_velocity, reference.recorded_velocity)
    _assert_same_state(compiled, reference)


def test_below_the_courant_limit():
    rigid = _complex()
    reference, compiled = _pair(walls=ASYMMETRIC, courant=0.7, rigid_cells=rigid)
    _both(reference, compiled, "add_volume_source", (0.0, 0.25, 0.45), _signal(18))
    _both(reference, compiled, "add_pressure_receiver", INTERIOR)
    reference.run(80)
    compiled.run(80)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    _assert_same_state(compiled, reference)


@pytest.mark.parametrize("walls", [None, ASYMMETRIC], ids=["rigid", "asymmetric"])
def test_thread_count_does_not_change_the_answer(walls):
    results = []
    try:
        for threads in (1, 3, 8):
            compiled = c_backend.CScalarFDTD(GRID, walls=walls, rigid_cells=_complex())
            assert compiled.set_threads(threads) >= 1
            compiled.set_initial_pressure(_start(19))
            compiled.add_volume_source(INTERIOR, _signal(20))
            compiled.run(50)
            results.append(compiled.p.copy())
    finally:
        c_backend.CScalarFDTD(GRID).set_threads(0)
    for field in results[1:]:
        np.testing.assert_array_equal(field, results[0])


def test_single_precision_tracks_the_single_precision_reference():
    rigid = _complex()
    reference, compiled = _pair(walls=ASYMMETRIC, dtype=np.float32, rigid_cells=rigid)
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(21))
    _both(reference, compiled, "add_volume_source", (0.0, 0.0, 0.0), _signal(22))
    _both(reference, compiled, "add_pressure_receiver", (0.5, 0.1, 0.4))
    reference.run(120)
    compiled.run(120)
    assert compiled.p.dtype == np.float32
    assert np.all(compiled.p[rigid] == 0.0)
    for ours, theirs in (
        (compiled.recorded_pressure, reference.recorded_pressure),
        (compiled.p, reference.p),
    ):
        theirs = np.asarray(theirs, dtype=np.float64)
        error = np.max(np.abs(ours - theirs)) / max(np.max(np.abs(theirs)), 1e-30)
        assert error < 1e-4

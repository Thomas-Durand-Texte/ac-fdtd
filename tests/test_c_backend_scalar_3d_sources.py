"""Volume sources and receivers on the compiled pressure-only 3D kernel must BE the reference's.

Every case with an absorbing wall is checked against the NumPy :class:`~ac_fdtd.scalar.ScalarFDTD`
with ``assert_array_equal`` in fp64: the fields, and every recorded sample. The soft source is the
reference's difference ``rho c^2 dt (S_n - S_{n-1})`` added after the wall solve, so a source on a
wall cell that was divided by ``1 + a``, or a sample read one step early, is caught at once. The
all-rigid room keeps its faster sweep, which agrees to round-off only (see T001), so there the
comparison is to 1e-11 of the peak.
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

ASYMMETRIC = WallAdmittances(x_min=0.1, x_max=0.7, y_min=0.25, y_max=1.0, z_min=0.5, z_max=0.9)

# Interior, the (0,0,0) corner, the far corner (clipped to the last cell), and the middle of a face
# on each of the inner (z) and outer (x) axes: every kind of wall cell plus none.
INTERIOR = (0.31, 0.22, 0.17)
POINTS = [
    INTERIOR,
    (0.0, 0.0, 0.0),
    (0.6, 0.5, 0.45),
    (0.3, 0.25, 0.0),
    (0.6, 0.25, 0.2),
]


def _signal(seed: int, n: int = 40) -> np.ndarray:
    # Noise, not a smooth pulse: every sample differs from the previous one, so an off-by-one in
    # which sample is read, or injecting S_n instead of S_n - S_{n-1}, changes every step.
    return np.random.default_rng(seed).standard_normal(n)


def _pair(grid: Grid = GRID, **kwargs):
    return ScalarFDTD(grid, **kwargs), c_backend.CScalarFDTD(grid, **kwargs)


def _both(reference, compiled, method: str, *args):
    first = getattr(reference, method)(*args)
    second = getattr(compiled, method)(*args)
    return first, second


def _assert_same_state(compiled, reference) -> None:
    np.testing.assert_array_equal(compiled.p, reference.p)
    np.testing.assert_array_equal(compiled.previous, reference.p_old)


@pytest.mark.parametrize(
    "walls",
    [WallAdmittances.matched(), ASYMMETRIC, WallAdmittances(x_min=0.8)],
    ids=["matched", "asymmetric", "x_min-only"],
)
def test_sources_and_pressure_receivers_match_bit_for_bit(walls):
    reference, compiled = _pair(walls=walls)
    for seed, point in enumerate(POINTS):
        _both(reference, compiled, "add_volume_source", point, _signal(seed))
    for point in POINTS + [(0.45, 0.4, 0.33)]:
        _both(reference, compiled, "add_pressure_receiver", point)
    reference.run(120)
    compiled.run(120)
    assert np.max(np.abs(reference.recorded_pressure)) > 0.0
    assert compiled.recorded_pressure.shape == (len(POINTS) + 1, 120)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    _assert_same_state(compiled, reference)


def test_velocity_receivers_match_bit_for_bit():
    """``v^{n-1/2}`` on the lower face along each axis, reconstructed from the two cells as the
    reference does. On a lower wall (index 0 along the axis) the face is the wall itself, and the
    reference records zero forever."""
    reference, compiled = _pair(walls=ASYMMETRIC)
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(10))
    _both(reference, compiled, "add_volume_source", (0.0, 0.0, 0.0), _signal(11))
    probes = [(INTERIOR, axis) for axis in range(3)]
    probes += [((0.03, 0.03, 0.03), axis) for axis in range(3)]  # face next to the corner cell
    probes += [((0.0, 0.2, 0.2), 0), ((0.3, 0.0, 0.2), 1), ((0.3, 0.2, 0.0), 2)]  # on the wall
    channels = [_both(reference, compiled, "add_velocity_receiver", p, a)[1] for p, a in probes]
    assert channels == list(range(len(probes)))
    reference.run(90)
    compiled.run(90)
    recorded = compiled.recorded_velocity
    assert recorded.shape == (len(probes), 90)
    np.testing.assert_array_equal(recorded, reference.recorded_velocity)
    assert np.all(recorded[:, 0] == 0.0)  # v^{-1/2} = 0: a start from rest
    assert np.all(recorded[6:] == 0.0)
    assert np.max(np.abs(recorded[:6])) > 0.0


def test_receivers_are_numbered_per_kind():
    _, compiled = _pair(walls=ASYMMETRIC)
    assert compiled.add_pressure_receiver(INTERIOR) == 0
    assert compiled.add_velocity_receiver(INTERIOR, 2) == 0
    assert compiled.add_pressure_receiver((0.0, 0.0, 0.0)) == 1
    assert compiled.add_velocity_receiver(INTERIOR, 0) == 1


def test_empty_recordings_have_the_reference_shape():
    reference, compiled = _pair(walls=ASYMMETRIC)
    assert compiled.recorded_pressure.shape == reference.recorded_pressure.shape == (0, 0)
    assert compiled.recorded_velocity.shape == reference.recorded_velocity.shape == (0, 0)
    compiled.run(5)  # running without receivers records nothing
    assert compiled.recorded_pressure.shape == (0, 0)
    assert compiled.recorded_velocity.shape == (0, 0)
    _both(reference, compiled, "add_pressure_receiver", INTERIOR)
    _both(reference, compiled, "add_velocity_receiver", INTERIOR, 1)
    other = c_backend.CScalarFDTD(GRID, walls=ASYMMETRIC)
    other.add_pressure_receiver(INTERIOR)
    other.add_velocity_receiver(INTERIOR, 1)
    assert other.recorded_pressure.shape == (1, 0)
    assert other.recorded_velocity.shape == (1, 0)


def test_sources_sharing_a_cell_add_in_insertion_order():
    """Two sources on one (wall) cell: the reference adds them one after the other, so the sum's
    rounding depends on the order; a list given as the signal is cast like an array."""
    reference, compiled = _pair(walls=ASYMMETRIC)
    point = (0.6, 0.0, 0.2)
    _both(reference, compiled, "add_volume_source", point, _signal(20))
    _both(reference, compiled, "add_volume_source", point, list(_signal(21, 25) * 1e-3))
    _both(reference, compiled, "add_pressure_receiver", point)
    reference.run(70)
    compiled.run(70)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    _assert_same_state(compiled, reference)


@pytest.mark.parametrize("length", [0, 1, 2])
def test_short_signals_end_with_their_own_negative(length):
    """After the last sample the source injects ``-S_last`` once and then nothing: a soft source
    leaves no net volume. An empty signal injects nothing at all."""
    reference, compiled = _pair(walls=ASYMMETRIC)
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(30, length))
    _both(reference, compiled, "add_pressure_receiver", INTERIOR)
    reference.run(12)
    compiled.run(12)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    _assert_same_state(compiled, reference)
    if length == 0:
        assert np.all(compiled.p == 0.0)


def test_chunked_runs_carry_the_source_and_the_recordings():
    """The signal is indexed by the global step, not the chunk's: a chunk boundary (here also one
    inside the signal and one at its end) must neither restart nor skip it, and the recordings of
    successive chunks are concatenated in time."""
    reference, compiled = _pair(walls=ASYMMETRIC)
    _both(reference, compiled, "add_volume_source", (0.03, 0.47, 0.2), _signal(40, 13))
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(41, 30))
    _both(reference, compiled, "add_pressure_receiver", (0.5, 0.1, 0.4))
    _both(reference, compiled, "add_velocity_receiver", (0.5, 0.1, 0.4), 2)
    reference.run(61)
    for chunk in (5, 1, 7, 0, 8, 40):
        compiled.run(chunk)
    assert compiled.step_index == 61
    assert compiled.recorded_pressure.shape == (1, 61)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    np.testing.assert_array_equal(compiled.recorded_velocity, reference.recorded_velocity)
    _assert_same_state(compiled, reference)


def test_source_added_after_a_run_starts_at_the_current_step():
    """As in the reference, a source added at step k reads its sample k (``S_k - S_{k-1}``), not
    sample 0: the signal is on the simulation's clock."""
    reference, compiled = _pair(walls=ASYMMETRIC)
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(50))
    _both(reference, compiled, "add_pressure_receiver", (0.2, 0.3, 0.1))
    reference.run(10)
    compiled.run(10)
    _both(reference, compiled, "add_volume_source", (0.0, 0.5, 0.45), _signal(51))
    reference.run(30)
    compiled.run(30)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    _assert_same_state(compiled, reference)


def test_below_the_courant_limit():
    """The source coefficient rho c^2 dt and the velocity one dt / (rho dx) both depend on dt."""
    reference, compiled = _pair(walls=ASYMMETRIC, courant=0.7)
    _both(reference, compiled, "add_volume_source", (0.0, 0.25, 0.45), _signal(60))
    _both(reference, compiled, "add_pressure_receiver", INTERIOR)
    _both(reference, compiled, "add_velocity_receiver", INTERIOR, 0)
    reference.run(80)
    compiled.run(80)
    np.testing.assert_array_equal(compiled.recorded_pressure, reference.recorded_pressure)
    np.testing.assert_array_equal(compiled.recorded_velocity, reference.recorded_velocity)


def test_rigid_room_matches_to_round_off():
    """walls=None keeps the fast rigid sweep (T001), so this is round-off, not bits; the recording
    is still the field itself, sampled after each step."""
    reference, compiled = _pair()
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(70))
    _both(reference, compiled, "add_volume_source", (0.0, 0.0, 0.0), _signal(71))
    _both(reference, compiled, "add_pressure_receiver", (0.6, 0.5, 0.45))
    _both(reference, compiled, "add_velocity_receiver", (0.4, 0.3, 0.2), 1)
    reference.run(100)
    compiled.run(100)
    pressure = reference.recorded_pressure
    velocity = reference.recorded_velocity
    np.testing.assert_allclose(
        compiled.recorded_pressure, pressure, rtol=0, atol=1e-11 * np.max(np.abs(pressure))
    )
    np.testing.assert_allclose(
        compiled.recorded_velocity, velocity, rtol=0, atol=1e-11 * np.max(np.abs(velocity))
    )
    assert compiled.recorded_pressure[0, -1] == compiled.p[-1, -1, -1]


def test_thread_count_does_not_change_the_answer():
    results = []
    try:
        for threads in (1, 3, 8):
            compiled = c_backend.CScalarFDTD(GRID, walls=ASYMMETRIC)
            assert compiled.set_threads(threads) >= 1
            compiled.add_volume_source(INTERIOR, _signal(80))
            compiled.add_volume_source((0.6, 0.0, 0.0), _signal(81))
            compiled.add_pressure_receiver((0.1, 0.4, 0.3))
            compiled.run(60)
            results.append((compiled.p.copy(), compiled.recorded_pressure))
    finally:
        c_backend.CScalarFDTD(GRID).set_threads(0)
    for field, recording in results[1:]:
        np.testing.assert_array_equal(field, results[0][0])
        np.testing.assert_array_equal(recording, results[0][1])


def test_single_precision_tracks_the_single_precision_reference():
    reference, compiled = _pair(walls=ASYMMETRIC, dtype=np.float32)
    _both(reference, compiled, "add_volume_source", INTERIOR, _signal(90))
    _both(reference, compiled, "add_volume_source", (0.0, 0.0, 0.0), _signal(91))
    _both(reference, compiled, "add_pressure_receiver", (0.5, 0.1, 0.4))
    _both(reference, compiled, "add_velocity_receiver", (0.5, 0.1, 0.4), 0)
    reference.run(150)
    compiled.run(150)
    assert compiled.p.dtype == np.float32
    assert compiled.recorded_pressure.dtype == np.float32
    for ours, theirs in (
        (compiled.recorded_pressure, reference.recorded_pressure),
        (compiled.recorded_velocity, reference.recorded_velocity),
    ):
        theirs = np.asarray(theirs, dtype=np.float64)
        error = np.max(np.abs(ours - theirs)) / max(np.max(np.abs(theirs)), 1e-30)
        assert error < 1e-4

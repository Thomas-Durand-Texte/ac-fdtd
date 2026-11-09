"""Correctness of the lossless scheme, measured against things that have exact answers."""

import numpy as np
import pytest

from ac_fdtd.analytic import discrete_mode_frequency, mode_initial_state, mode_pressure_field
from ac_fdtd.grid import Grid
from ac_fdtd.medium import AIR
from ac_fdtd.scheme import AcousticFDTD


def _box(dx=0.05, lengths=(0.8, 0.65, 0.55), courant=1.0):
    return AcousticFDTD(Grid.from_lengths(lengths, dx), medium=AIR, courant=courant)


def test_rigid_walls_are_never_written_to():
    solver = _box()
    solver.p[:] = np.random.default_rng(0).standard_normal(solver.grid.shape)
    solver.run(20)
    for axis in range(3):
        wall = solver.velocity[axis]
        lower = tuple(0 if a == axis else slice(None) for a in range(3))
        upper = tuple(-1 if a == axis else slice(None) for a in range(3))
        assert np.all(wall[lower] == 0.0)
        assert np.all(wall[upper] == 0.0)


@pytest.mark.parametrize("courant", [1.0, 0.9, 0.5])
def test_energy_is_conserved_to_round_off(courant):
    """The sharpest available check: drift here means a bug, not a modelling choice."""
    solver = _box(courant=courant)
    solver.p[:] = np.random.default_rng(1).standard_normal(solver.grid.shape)

    initial = solver.energy()
    assert initial > 0.0

    drift = []
    for _ in range(40):
        solver.run(50)
        drift.append(abs(solver.energy() - initial) / initial)

    assert max(drift) < 1e-11


def test_energy_needs_the_staggered_coupling_term():
    """Guard on the correction that the naive p^2 + v^2 form drops.

    The two forms differ by the mixed-time coupling, whose size relative to the total is of
    order ``omega*dt/2``: fractions of a percent for a well-resolved mode, order one for
    content near the grid limit. So the naive form oscillates instead of holding, and cannot
    be used as a correctness check — which is why the real one gets a guard of its own.
    """
    solver = _box()
    solver.p[:] = np.random.default_rng(2).standard_normal(solver.grid.shape)
    correct = [solver.energy()]

    def naive():
        medium, grid = solver.medium, solver.grid
        potential = np.sum(solver.p**2) / (2 * medium.density * medium.sound_speed**2)
        kinetic = sum(0.5 * medium.density * np.sum(v**2) for v in solver.velocity)
        return float(grid.dx**3 * (potential + kinetic))

    naive_values = [naive()]
    for _ in range(20):
        solver.run(7)
        correct.append(solver.energy())
        naive_values.append(naive())

    correct = np.array(correct)
    naive_values = np.array(naive_values)
    assert np.ptp(correct) / correct[0] < 1e-12
    assert np.ptp(naive_values) / naive_values[0] > 1e-6


@pytest.mark.parametrize("mode", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 1, 1), (3, 2, 1)])
def test_a_single_mode_stays_exactly_sinusoidal(mode):
    """A mode of the box is an eigenvector of the discrete operator, so the response is pure.

    This catches sign errors, slice off-by-ones and half-step mistakes in one shot: any of them
    detunes the oscillation or leaks energy into other modes, and both show up here at a
    tolerance no tuning can hide.
    """
    solver = _box(dx=0.04)
    frequency = discrete_mode_frequency(solver.grid, mode, solver.medium, solver.dt)
    shape = mode_pressure_field(solver.grid, mode)

    pressure, velocity = mode_initial_state(solver.grid, mode, solver.medium, solver.dt)
    solver.p[:] = pressure
    for axis in range(3):
        solver.velocity[axis][:] = velocity[axis]

    errors = []
    for _ in range(600):
        solver.step()
        expected = np.cos(2 * np.pi * frequency * solver.time) * shape
        errors.append(np.max(np.abs(solver.p - expected)))

    assert max(errors) < 1e-11


def test_discrete_frequency_approaches_the_analytic_one_at_second_order():
    from ac_fdtd.analytic import mode_frequency

    mode = (2, 1, 1)
    lengths = (1.6, 1.2, 1.0)
    errors = []
    for dx in (0.05, 0.025, 0.0125):
        grid = Grid.from_lengths(lengths, dx)
        dt = grid.dx / (AIR.sound_speed * np.sqrt(3))
        exact = mode_frequency(grid.lengths, mode, AIR.sound_speed)
        numerical = discrete_mode_frequency(grid, mode, AIR, dt)
        errors.append(abs(numerical - exact) / exact)

    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert all(3.5 < ratio < 4.5 for ratio in ratios), (errors, ratios)


def test_a_soft_source_conserves_nothing_but_stays_bounded():
    """Sanity check on the source path: it injects energy, and the run does not blow up."""
    solver = _box()
    n_steps = 400
    signal = np.zeros(n_steps)
    signal[:20] = np.hanning(20)
    solver.add_volume_source((0.3, 0.3, 0.3), signal)
    solver.run(n_steps)

    assert np.isfinite(solver.p).all()
    assert solver.energy() > 0.0
    assert np.max(np.abs(solver.p)) < 1e6


def test_a_prescribed_pressure_source_holds_its_node_then_frees_it():
    """Inside the window the node equals the signal exactly; after it, it is a free node again."""
    solver = _box()
    signal = np.random.default_rng(3).standard_normal(15)
    point = (0.4, 0.3, 0.25)
    solver.add_prescribed_source(point, signal, kind="pressure")
    index = solver.grid.cell_index(point)

    for k in range(signal.size):
        solver.step()
        assert solver.p[index] == pytest.approx(signal[k]), k

    # Released: the node is no longer pinned, so over a few further steps it moves off the last
    # held value rather than staying clamped to it.
    held = solver.p[index]
    values = []
    for _ in range(5):
        solver.step()
        values.append(solver.p[index])
    assert any(v != held for v in values)


def test_a_prescribed_velocity_source_drives_the_field_from_rest():
    """A velocity face driven from silence does work on the fluid and radiates pressure."""
    solver = _box()
    signal = np.full(10, 0.01)
    point = (0.4, 0.3, 0.25)
    axis = 0
    solver.add_prescribed_source(point, signal, kind="velocity", axis=axis)
    index = solver.grid.cell_index(point)

    assert solver.energy() == 0.0
    solver.step()
    # The driven face holds exactly the prescribed value this step...
    assert solver.velocity[axis][index] == pytest.approx(signal[0])
    # ...and it has already radiated: pressure is no longer uniformly zero.
    assert np.any(solver.p != 0.0)

    solver.run(20)
    assert np.isfinite(solver.p).all()
    assert solver.energy() > 0.0


def test_a_prescribed_source_is_inert_outside_its_window():
    """Nothing happens before ``start``; the first sample lands exactly on step ``start``."""
    solver = _box()
    signal = np.array([1.0, 1.0, 1.0])
    point = (0.4, 0.3, 0.25)
    solver.add_prescribed_source(point, signal, start=5, kind="pressure")
    index = solver.grid.cell_index(point)

    solver.run(5)  # steps 0..4, strictly before the window
    assert solver.energy() == 0.0
    assert solver.p[index] == 0.0

    solver.step()  # step 5: the window's first sample is imposed
    assert solver.p[index] == pytest.approx(1.0)


def test_prescribed_source_rejects_contradictory_arguments():
    solver = _box()
    signal = np.ones(4)
    point = (0.1, 0.1, 0.1)
    with pytest.raises(ValueError):
        solver.add_prescribed_source(point, signal, kind="velocity")  # axis missing
    with pytest.raises(ValueError):
        solver.add_prescribed_source(point, signal, kind="pressure", axis=0)  # axis with pressure
    with pytest.raises(ValueError):
        solver.add_prescribed_source(point, signal, kind="nonsense")
    with pytest.raises(ValueError):
        solver.add_prescribed_source(point, signal, start=-1)

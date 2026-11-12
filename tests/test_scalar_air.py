"""Air relaxation (ISO 9613-1) in the pressure-only references: T006.

The p-v schemes drive each relaxation state with ``div v^{n+1/2}``. The pressure-only schemes hold
no velocity, so they rebuild that divergence from the two pressure levels and the relaxation offset
of the same step. The elimination is exact in exact arithmetic but reorders the floating-point
operations, so the oracle here is the p-v scheme with air **to fp64 round-off** (1e-10 of the peak),
not bit for bit as in ``tests/test_scalar.py``. A prototype of the elimination measured ~1e-14.
"""

import numpy as np
import pytest

from ac_fdtd.air import AirAbsorption
from ac_fdtd.boundaries import WallAdmittances
from ac_fdtd.boundaries2d import WallAdmittances2D
from ac_fdtd.cpml import AbsorbingBoundary
from ac_fdtd.cpml2d import AbsorbingBoundary2D
from ac_fdtd.grid import Grid
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scalar import ScalarFDTD
from ac_fdtd.scalar2d import ScalarFDTD2D
from ac_fdtd.scheme import AcousticFDTD
from ac_fdtd.scheme2d import AcousticFDTD2D

AIR_ISO = AirAbsorption(temperature=20.0, relative_humidity=50.0)
GRID_3D = Grid.from_lengths((0.6, 0.5, 0.45), 0.03)
GRID_2D = Grid2D.from_lengths((1.3, 1.1), 0.02)
STEPS = 2000
#: Round-off budget relative to the peak recorded pressure. Not zero: see the module docstring.
TOLERANCE = 1e-10

# Receivers include corner and wall-adjacent cells, where the wall correction and the relaxation
# reconstruction meet, as well as an interior cell.
SOURCE_3D, RECEIVERS_3D = (
    (0.3, 0.25, 0.2),
    [(0.01, 0.01, 0.01), (0.59, 0.49, 0.44), (0.45, 0.3, 0.05)],
)
SOURCE_2D, RECEIVERS_2D = (0.4, 0.5), [(0.01, 0.01), (1.29, 1.09), (0.9, 0.02), (0.7, 0.6)]


def _ricker(n: int) -> np.ndarray:
    t = np.arange(n)
    return (1.0 - 0.5 * ((t - 60) / 20) ** 2) * np.exp(-(((t - 60) / 20) ** 2) / 2.0)


def _run_pair(reference, variant, source, receivers, steps=STEPS):
    signal = _ricker(300)
    for solver in (reference, variant):
        solver.add_volume_source(source, signal)
        for point in receivers:
            solver.add_pressure_receiver(point)
    reference.run(steps)
    variant.run(steps)
    return variant.recorded_pressure, reference.recorded_pressure


def _assert_round_off_close(variant: np.ndarray, reference: np.ndarray) -> None:
    peak = float(np.max(np.abs(reference)))
    assert peak > 0.0
    assert variant.shape == reference.shape
    error = float(np.max(np.abs(variant - reference)))
    assert error <= TOLERANCE * peak, f"max error {error / peak:.3e} of the peak"


# -- 1. equivalence with the p-v scheme with air -----------------------------------------------


@pytest.mark.parametrize(
    "walls",
    [
        None,
        WallAdmittances(x_min=0.0, x_max=0.7, y_min=0.2, y_max=0.0, z_min=0.5, z_max=0.9),
    ],
    ids=["rigid", "impedance"],
)
def test_3d_with_air_matches_the_first_order_scheme(walls):
    reference = AcousticFDTD(GRID_3D, walls=walls, air_absorption=AIR_ISO)
    variant = ScalarFDTD(GRID_3D, walls=walls, air_absorption=AIR_ISO)
    recorded, expected = _run_pair(reference, variant, SOURCE_3D, RECEIVERS_3D)
    _assert_round_off_close(recorded, expected)
    _assert_round_off_close(variant.p, reference.p)


@pytest.mark.parametrize(
    "walls",
    [None, WallAdmittances2D(x_min=0.0, x_max=0.8, y_min=0.2, y_max=1.0)],
    ids=["rigid", "impedance"],
)
def test_2d_with_air_matches_the_first_order_scheme(walls):
    reference = AcousticFDTD2D(GRID_2D, walls=walls, air_absorption=AIR_ISO)
    variant = ScalarFDTD2D(GRID_2D, walls=walls, air_absorption=AIR_ISO)
    recorded, expected = _run_pair(reference, variant, SOURCE_2D, RECEIVERS_2D)
    _assert_round_off_close(recorded, expected)
    _assert_round_off_close(variant.p, reference.p)


def test_2d_with_air_and_rigid_obstacle_matches_the_first_order_scheme():
    """A solid cell has every face closed: its divergence and relaxation states stay zero."""
    rigid = np.zeros(GRID_2D.shape, dtype=bool)
    rigid[30:40, 20:28] = True
    walls = WallAdmittances2D.from_absorption(0.3)
    reference = AcousticFDTD2D(GRID_2D, walls=walls, air_absorption=AIR_ISO, rigid_cells=rigid)
    variant = ScalarFDTD2D(GRID_2D, walls=walls, air_absorption=AIR_ISO, rigid_cells=rigid)
    recorded, expected = _run_pair(reference, variant, SOURCE_2D, RECEIVERS_2D)
    _assert_round_off_close(recorded, expected)
    assert np.all(variant.p[rigid] == 0.0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_free_decay_from_pressure_at_rest_matches(dimension):
    """``set_initial_pressure`` is the p-v start ``v = 0`` with every relaxation state at zero."""
    if dimension == 3:
        grid, reference_class, variant_class = GRID_3D, AcousticFDTD, ScalarFDTD
    else:
        grid, reference_class, variant_class = GRID_2D, AcousticFDTD2D, ScalarFDTD2D
    field = np.random.default_rng(3).standard_normal(grid.shape)
    reference = reference_class(grid, air_absorption=AIR_ISO)
    reference.p[:] = field
    variant = variant_class(grid, air_absorption=AIR_ISO)
    variant.set_initial_pressure(field)
    for steps in (1, 2, 299):  # 1, 3 then 302 steps in total
        reference.run(steps)
        variant.run(steps)
        np.testing.assert_allclose(variant.p, reference.p, rtol=0.0, atol=TOLERANCE * 4.0)


# -- 2. no regression without air -------------------------------------------------------------


@pytest.mark.parametrize("dimension", [2, 3])
def test_air_with_no_process_is_bit_identical_to_no_air(dimension):
    """An ``AirAbsorption`` with both terms off has zero processes: the scheme must be untouched."""
    silent = AirAbsorption(classical=False, relaxation=False)
    if dimension == 3:
        grid, solver_class, walls = GRID_3D, ScalarFDTD, WallAdmittances.from_absorption(0.3)
        source = SOURCE_3D
    else:
        grid, solver_class, walls = GRID_2D, ScalarFDTD2D, WallAdmittances2D.from_absorption(0.3)
        source = SOURCE_2D
    plain = solver_class(grid, walls=walls)
    disabled = solver_class(grid, walls=walls, air_absorption=silent)
    assert disabled.air_absorption is silent
    assert plain.air_absorption is None
    for solver in (plain, disabled):
        solver.add_volume_source(source, _ricker(300))
        solver.run(200)
    assert np.array_equal(plain.p, disabled.p)
    assert np.array_equal(plain.p_old, disabled.p_old)


# -- 3. physics: attenuation vs ISO 9613-1, as tests/test_air.py does for the p-v scheme ---------


@pytest.mark.parametrize("frequency", [2000.0, 4000.0])
def test_measured_attenuation_matches_iso(frequency):
    """``tests/test_air.py::test_measured_attenuation_matches_iso`` on ``ScalarFDTD``, same
    tolerance (5 % of the ISO value)."""
    dx = 0.003
    solver = ScalarFDTD(
        Grid(dx=dx, nx=3000, ny=1, nz=1),
        walls=WallAdmittances(x_min=1.0, x_max=1.0),
        air_absorption=AIR_ISO,
    )

    n_steps, width = 5200, 4.0
    index = np.arange(n_steps)
    solver.add_volume_source((0.5, dx / 2, dx / 2), np.exp(-(((index - 4 * width) / width) ** 2)))

    near, far = 1.5, 8.0
    solver.add_pressure_receiver((near, dx / 2, dx / 2))
    solver.add_pressure_receiver((far, dx / 2, dx / 2))
    solver.run(n_steps)
    recorded = solver.recorded_pressure

    spectra = np.abs(np.fft.rfft(recorded, axis=1))
    frequencies = np.fft.rfftfreq(n_steps, solver.dt)
    bin_index = int(np.argmin(np.abs(frequencies - frequency)))
    measured = -20.0 * np.log10(spectra[1, bin_index] / spectra[0, bin_index]) / (far - near)

    expected = float(AIR_ISO.attenuation(frequencies[bin_index]))
    assert measured == pytest.approx(expected, rel=0.05)


# -- 4. energy ---------------------------------------------------------------------------------


def test_absorbing_air_removes_energy_and_a_long_rigid_run_stays_bounded():
    """Rigid room, random field at rest, 4000 steps. ``ScalarFDTD`` has no ``energy()``, so the
    proxy is the mean square pressure over 500-step windows: lossless it stays flat (within a few
    per cent), with air it falls window after window and ends well below the lossless run."""
    grid = Grid.from_lengths((0.4, 0.35, 0.3), 0.02)
    field = np.random.default_rng(6).standard_normal(grid.shape)
    windows = {}
    for label, air in (("lossless", None), ("air", AIR_ISO)):
        solver = ScalarFDTD(grid, air_absorption=air)
        solver.set_initial_pressure(field)
        mean_square = []
        for _ in range(4000):
            solver.step()
            mean_square.append(float(np.mean(solver.p**2)))
        assert np.isfinite(solver.p).all()
        windows[label] = np.array(mean_square).reshape(8, 500).mean(axis=1)

    lossless, lossy = windows["lossless"], windows["air"]
    assert np.max(np.abs(lossless / lossless[0] - 1.0)) < 0.05
    assert np.all(np.diff(lossy) < 0.0)
    assert lossy[-1] < 0.6 * lossy[0]
    assert lossy[-1] < 0.6 * lossless[-1]


# -- 5. memory ---------------------------------------------------------------------------------


def _cell_array_elements(solver, n_cells: int) -> int:
    """Elements held in grid-sized (or larger) arrays reachable from the solver's attributes.

    Looks inside lists, tuples and dicts, so per-process states stored in a list (as the p-v scheme
    does) or stacked in one array are both counted. Each array object is counted once.
    """
    seen: set[int] = set()
    total = 0

    def visit(value) -> None:
        nonlocal total
        if isinstance(value, np.ndarray):
            if id(value) not in seen and value.size >= n_cells:
                seen.add(id(value))
                total += value.size
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for value in vars(solver).values():
        visit(value)
    return total


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize(
    "air", [AIR_ISO, AirAbsorption(relaxation=False)], ids=["iso", "classical-only"]
)
def test_closed_room_with_air_holds_two_levels_plus_one_array_per_process(dimension, air):
    """2 pressure levels + one array per relaxation process, plus at most one scratch array (the p-v
    scheme keeps scratch too). No velocity grid: that is the whole point of the scheme."""
    if dimension == 3:
        grid, solver = GRID_3D, ScalarFDTD(GRID_3D, air_absorption=air)
        assert not hasattr(solver, "_velocity")
    else:
        grid, solver = GRID_2D, ScalarFDTD2D(GRID_2D, air_absorption=air)
        assert not hasattr(solver, "vx") and not hasattr(solver, "vy")
    n_cells = int(np.prod(grid.shape))
    processes = len(air.processes(solver.medium.sound_speed, solver.dt))
    assert processes == (3 if air is AIR_ISO else 1)

    solver.add_volume_source(SOURCE_3D if dimension == 3 else SOURCE_2D, _ricker(50))
    solver.run(5)  # lazily allocated state counts too
    held = _cell_array_elements(solver, n_cells)
    assert (2 + processes) * n_cells <= held <= (3 + processes) * n_cells


# -- API edges ---------------------------------------------------------------------------------


def test_air_with_an_absorbing_boundary_is_rejected():
    """Assumption: the CPML shell is first-order p-v and has no relaxation states, so combining it
    with air would silently drop absorption in the shell. Until that is ported, refuse it."""
    with pytest.raises(ValueError):
        ScalarFDTD(
            Grid.from_lengths((0.6, 0.6, 0.6), 0.03),
            absorbing_boundary=AbsorbingBoundary.open_field(4, 1000.0),
            air_absorption=AIR_ISO,
        )
    with pytest.raises(ValueError):
        ScalarFDTD2D(
            Grid2D.from_lengths((0.6, 0.6), 0.03),
            absorbing_boundary=AbsorbingBoundary2D.open_field(4, 1000.0),
            air_absorption=AIR_ISO,
        )

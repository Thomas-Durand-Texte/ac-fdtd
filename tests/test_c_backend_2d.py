"""The compiled 2D backend has to be the same computation as the 2D reference scheme."""

import numpy as np
import pytest

from ac_fdtd.air import AirAbsorption
from ac_fdtd.boundaries2d import AbsorbingLayer2D, WallAdmittances2D
from ac_fdtd.grid2d import Grid2D
from ac_fdtd.scheme2d import AcousticFDTD2D

c_backend = pytest.importorskip("ac_fdtd.c_backend")

pytestmark = pytest.mark.skipif(not c_backend.is_available(), reason="no C compiler available")

GRID = Grid2D.from_lengths((0.6, 0.5), 0.03)
SOURCE = (0.2, 0.2)
RECEIVER = (0.4, 0.3)


def _obstacle_mask() -> np.ndarray:
    """A solid block off to one side — an interior rigid obstacle to exercise the masked sweeps."""
    nx, ny = GRID.shape
    mask = np.zeros(GRID.shape, dtype=bool)
    mask[nx // 2 : nx // 2 + 3, ny // 3 : 2 * ny // 3] = True
    return mask


CONFIGURATIONS = {
    "rigid": {},
    "absorbing walls": {"walls": WallAdmittances2D.uniform(0.3)},
    "absorbing layer": {"absorbing_layer": AbsorbingLayer2D(thickness=4)},
    "air absorption": {"air_absorption": AirAbsorption()},
    "rigid obstacle": {"rigid_cells": _obstacle_mask()},
    "obstacle in an absorbing room": {
        "rigid_cells": _obstacle_mask(),
        "walls": WallAdmittances2D.uniform(0.4),
    },
    "everything at once": {
        "walls": WallAdmittances2D(x_min=0.2, y_max=0.7),
        "absorbing_layer": AbsorbingLayer2D(thickness=3),
        "air_absorption": AirAbsorption(temperature=15.0, relative_humidity=70.0),
        "rigid_cells": _obstacle_mask(),
    },
}


def _paired(dtype=np.float64, **configuration):
    reference = AcousticFDTD2D(GRID, dtype=dtype, **configuration)
    candidate = c_backend.CAcousticFDTD2D(GRID, dtype=dtype, **configuration)

    initial = np.random.default_rng(0).standard_normal(GRID.shape)
    signal = np.exp(-(((np.arange(200) - 30) / 8.0) ** 2))
    for solver in (reference, candidate):
        solver.p[:] = initial
        solver.add_volume_source(SOURCE, signal)
        solver.add_pressure_receiver(RECEIVER)
    return reference, candidate


def test_the_ctypes_mirror_matches_the_2d_c_struct():
    """Constructing the solver checks the struct sizes, so reaching here is the test passing."""
    solver = c_backend.CAcousticFDTD2D(GRID)
    assert solver.threads >= 1


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_matches_the_reference(name):
    reference, candidate = _paired(**CONFIGURATIONS[name])
    reference.run(150)
    candidate.run(150)

    np.testing.assert_allclose(candidate.p, reference.p, rtol=1e-11, atol=1e-13)
    for axis in range(2):
        np.testing.assert_allclose(
            candidate.velocity[axis], reference.velocity[axis], rtol=1e-11, atol=1e-13
        )
    np.testing.assert_allclose(
        candidate.recorded_pressure, reference.recorded_pressure, rtol=1e-11, atol=1e-13
    )


def test_prescribed_sources_match_the_reference():
    """A driven velocity piston and a pressure node must step exactly like the reference scheme."""
    signal = np.exp(-(((np.arange(200) - 30) / 8.0) ** 2))

    def build(cls):
        solver = cls(GRID, walls=WallAdmittances2D.uniform(0.3))
        for j in (5, 6, 7):  # a short velocity piston (axis 0) and a pressure node
            solver.add_prescribed_source((0.2, j * 0.03), signal, kind="velocity", axis=0)
        solver.add_prescribed_source((0.4, 0.3), signal, kind="pressure")
        solver.add_pressure_receiver(RECEIVER)
        return solver

    reference, candidate = build(AcousticFDTD2D), build(c_backend.CAcousticFDTD2D)
    reference.run(150)
    candidate.run(150)
    np.testing.assert_allclose(candidate.p, reference.p, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(
        candidate.recorded_pressure, reference.recorded_pressure, rtol=1e-11, atol=1e-13
    )


def test_single_precision_tracks_the_reference_loosely():
    """fp32 cannot match to 1e-11, but it must still be the same computation to a few ulp/step."""
    reference, candidate = _paired(dtype=np.float32, walls=WallAdmittances2D.uniform(0.2))
    reference.run(120)
    candidate.run(120)
    np.testing.assert_allclose(candidate.p, reference.p, rtol=1e-3, atol=1e-4)

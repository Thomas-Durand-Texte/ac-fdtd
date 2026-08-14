"""The PyTorch backend has to be the same computation, not a similar one.

Every feature is checked against the NumPy reference in double precision, where the two should
agree to round-off — on this machine they agree bitwise. That is the only thing keeping two
implementations of the same scheme from drifting apart.
"""

import numpy as np
import pytest

from ac_fdtd.air import AirAbsorption
from ac_fdtd.boundaries import AbsorbingLayer, WallAdmittances
from ac_fdtd.grid import Grid
from ac_fdtd.scheme import AcousticFDTD

torch = pytest.importorskip("torch")

from ac_fdtd.torch_backend import (  # noqa: E402
    TorchAcousticFDTD,
    available_devices,
)

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


def _paired(dtype=torch.float64, device="cpu", **configuration):
    """The same problem set up on both backends, with identical initial conditions."""
    reference = AcousticFDTD(GRID, **configuration)
    candidate = TorchAcousticFDTD(GRID, device=device, dtype=dtype, **configuration)

    initial = np.random.default_rng(0).standard_normal(GRID.shape)
    reference.p[:] = initial
    candidate.p.copy_(torch.as_tensor(initial, dtype=dtype, device=device))

    signal = np.exp(-(((np.arange(200) - 30) / 8.0) ** 2))
    for solver in (reference, candidate):
        solver.add_volume_source(SOURCE, signal)
        solver.add_pressure_receiver(RECEIVER)
    return reference, candidate


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_double_precision_matches_the_reference(name):
    reference, candidate = _paired(**CONFIGURATIONS[name])
    reference.run(150)
    candidate.run(150)

    np.testing.assert_allclose(candidate.p.numpy(), reference.p, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        candidate.recorded_pressure, reference.recorded_pressure, rtol=1e-10, atol=1e-14
    )
    assert candidate.energy() == pytest.approx(reference.energy(), rel=1e-10)


def test_velocity_fields_match_too():
    """Pressure agreeing is not enough: it would hide a compensating error in the other half."""
    reference, candidate = _paired(walls=WallAdmittances.uniform(0.2))
    reference.run(80)
    candidate.run(80)
    for axis in range(3):
        np.testing.assert_allclose(
            candidate.velocity[axis].numpy(), reference.velocity[axis], rtol=1e-10, atol=1e-14
        )


def test_single_precision_noise_floor_is_far_below_the_signal():
    """fp32 is not free, but what it costs here is a noise floor rather than a wrong answer."""
    reference, candidate = _paired(dtype=torch.float32, walls=WallAdmittances.uniform(0.05))
    reference.run(2000)
    candidate.run(2000)

    recorded = reference.recorded_pressure[0]
    error = np.max(np.abs(recorded - candidate.recorded_pressure[0])) / np.max(np.abs(recorded))
    assert error < 1e-4


def test_receivers_record_the_same_samples_as_probing_by_hand():
    solver = TorchAcousticFDTD(GRID, dtype=torch.float64)
    solver.add_volume_source(SOURCE, np.ones(50))
    solver.add_pressure_receiver(RECEIVER)
    solver.run(20)
    solver.run(20)

    recorded = solver.recorded_pressure
    assert recorded.shape == (1, 40)
    index = GRID.cell_index(RECEIVER)
    assert recorded[0, -1] == pytest.approx(float(solver.p[index]))


def test_available_devices_always_includes_the_cpu():
    assert available_devices()[0] == "cpu"


@pytest.mark.skipif("mps" not in available_devices(), reason="no MPS device")
def test_mps_refuses_double_precision_with_an_explanation():
    with pytest.raises(ValueError, match="float64"):
        TorchAcousticFDTD(GRID, device="mps", dtype=torch.float64)


@pytest.mark.skipif("mps" not in available_devices(), reason="no MPS device")
def test_the_gpu_agrees_with_the_cpu_at_the_same_precision():
    _, on_cpu = _paired(dtype=torch.float32, device="cpu", walls=WallAdmittances.uniform(0.2))
    _, on_gpu = _paired(dtype=torch.float32, device="mps", walls=WallAdmittances.uniform(0.2))
    on_cpu.run(100)
    on_gpu.run(100)
    np.testing.assert_allclose(on_gpu.p.cpu().numpy(), on_cpu.p.numpy(), rtol=1e-4, atol=1e-6)

"""The same scheme in PyTorch, to be checked against the NumPy reference rather than trusted.

Why this should win, and by how much
------------------------------------
A 3D FDTD step is about ten passes over four arrays and roughly one arithmetic operation per
byte moved. That makes it **memory-bandwidth-bound**, and bandwidth is exactly what a GPU has
more of. Measured on this machine with a plain fused add over 16.8 M elements: 306 GB/s on the
CPU through PyTorch's threads, 727 GB/s on the MPS device. So the ceiling is a factor of about
2.4 for the GPU over the threaded CPU, and single precision doubles both by halving the traffic.

That is the opposite regime from the companion `plate-fdtd` project, where the same framework
was found to be *kernel-launch bound* — the arrays there were small enough that dispatching an
operation cost more than performing it. Here the crossover is a property of the grid: below a
few hundred thousand cells the launches dominate and NumPy can win outright; above a few
million the hardware's bandwidth is all that matters. The benchmark measures where that happens
instead of asserting it.

Precision
---------
**MPS does not support float64 at all.** The GPU path is therefore single precision, and single
precision is not free: this scheme accumulates round-off over 10^4 to 10^5 steps for a room
impulse response. What that costs is measured against the double-precision reference rather
than assumed, and the answer is in `PROGRESS.md`.

Sharing with the reference
--------------------------
Everything that decides *what* the scheme does — wall coefficients, layer profiles, relaxation
strengths, the time step — is computed by the same shared functions the NumPy backend uses.
Only the ten lines that move numbers are written twice. That is the part parity tests can
police; a second copy of the physics would not be.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .air import AirAbsorption
from .boundaries import (
    AbsorbingLayer,
    WallAdmittances,
    edge_slabs,
    layer_factors,
    wall_update_coefficients,
)
from .grid import Grid, max_stable_time_step
from .medium import AIR, Medium

__all__ = ["TorchAcousticFDTD", "available_devices", "synchronize"]

_AXES = (0, 1, 2)


def available_devices() -> tuple[str, ...]:
    """Devices this machine can actually run on, best last."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return tuple(devices)


def synchronize(device: str) -> None:
    """Wait for queued work to finish. Timing anything on a GPU without this measures nothing."""
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def _upper(axis: int) -> tuple[slice, ...]:
    return tuple(slice(1, None) if a == axis else slice(None) for a in _AXES)


def _lower(axis: int) -> tuple[slice, ...]:
    return tuple(slice(None, -1) if a == axis else slice(None) for a in _AXES)


def _interior_faces(axis: int) -> tuple[slice, ...]:
    return tuple(slice(1, -1) if a == axis else slice(None) for a in _AXES)


class TorchAcousticFDTD:
    """The staggered pressure–velocity scheme on a PyTorch device.

    Operation for operation the same as :class:`~ac_fdtd.scheme.AcousticFDTD`, including the
    ``(p^n, v^{n-1/2})`` state pairing and the rule that wall velocity planes are never written
    to. Differences from the reference are bugs, and the parity tests say so.

    Args:
        grid: Geometry and spacing.
        medium: Fluid properties.
        courant: Fraction of the stability limit.
        device: ``"cpu"``, ``"mps"`` or ``"cuda"``.
        dtype: ``torch.float32`` or ``torch.float64``. MPS supports only the former.
        walls: Wall admittances. Default rigid.
        absorbing_layer: Graded matched layer, for free field.
        air_absorption: Atmospheric conditions, or ``None`` for lossless air.
    """

    def __init__(
        self,
        grid: Grid,
        medium: Medium = AIR,
        courant: float = 1.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        walls: WallAdmittances | None = None,
        absorbing_layer: AbsorbingLayer | None = None,
        air_absorption: AirAbsorption | None = None,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")
        if device.startswith("mps") and dtype == torch.float64:
            raise ValueError(
                "MPS has no float64 support at all, so the GPU path is single precision. "
                "Use device='cpu' for double, and see PROGRESS.md for what fp32 costs here."
            )

        self.grid = grid
        self.medium = medium
        self.courant = courant
        self.walls = walls or WallAdmittances()
        self.absorbing_layer = absorbing_layer
        self.air_absorption = air_absorption
        self.device = device
        self.dtype = dtype
        self.dt = courant * max_stable_time_step(grid.dx, medium.sound_speed)
        self.step_index = 0

        options = {"device": device, "dtype": dtype}
        self.p = torch.zeros(grid.shape, **options)
        self.velocity = [torch.zeros(grid.velocity_shape(axis), **options) for axis in _AXES]

        self._velocity_coefficient = self.dt / (medium.density * grid.dx)
        self._pressure_coefficient = medium.density * medium.sound_speed**2 * self.dt / grid.dx
        self._source_coefficient = medium.density * medium.sound_speed**2 * self.dt

        self._divergence = torch.empty(grid.shape, **options)
        self._scratch = torch.empty(grid.shape, **options)

        self._sources: list[tuple[int, torch.Tensor]] = []
        self._receiver_indices: list[int] = []
        self._recording: torch.Tensor | None = None

        indices, coefficient = wall_update_coefficients(grid, medium, self.dt, self.walls)
        self._wall_indices = torch.as_tensor(indices, device=device, dtype=torch.long)
        self._wall_from_updated = torch.as_tensor(1.0 / (1.0 + coefficient), **options)
        self._wall_from_previous = torch.as_tensor(coefficient / (1.0 + coefficient), **options)

        self._relaxation: list[tuple[torch.Tensor, float, float, float]] = []
        self._relaxation_scratch: torch.Tensor | None = None
        if air_absorption is not None:
            processes = air_absorption.processes(medium.sound_speed, self.dt)
            relaxed_speed_squared = medium.sound_speed**2 / (
                1.0 + sum(process.strength for process in processes)
            )
            for process in processes:
                decay = math.exp(-self.dt / process.relaxation_time)
                mean_weight = (process.relaxation_time / self.dt) * (1.0 - decay)
                self._relaxation.append(
                    (
                        torch.zeros(grid.shape, **options),
                        decay,
                        self.dt * mean_weight,
                        medium.density * relaxed_speed_squared * process.strength / grid.dx,
                    )
                )
            self._relaxation_scratch = torch.empty(grid.shape, **options)
        self._relaxation_target_scale = self.dt * sum(gain for _, _, _, gain in self._relaxation)

        self._pressure_damping = []
        self._velocity_damping = [[] for _ in _AXES]
        if absorbing_layer is not None:
            pressure_factors, velocity_factors = layer_factors(
                grid, medium, self.dt, absorbing_layer
            )

            def to_tensor(slabs):
                return [(index, torch.as_tensor(factor, **options)) for index, factor in slabs]

            self._pressure_damping = [
                slab
                for axis in _AXES
                for slab in to_tensor(edge_slabs(pressure_factors[axis], axis))
            ]
            for component in _AXES:
                self._velocity_damping[component] = [
                    slab
                    for axis in _AXES
                    for slab in to_tensor(edge_slabs(velocity_factors[component][axis], axis))
                ]

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.dt

    @property
    def time(self) -> float:
        return self.step_index * self.dt

    def add_volume_source(self, point: tuple[float, float, float], signal: np.ndarray) -> None:
        """Inject a volume-velocity source, as in the reference backend.

        The signal is moved to the device once rather than sampled from the host each step: a
        scalar crossing the bus per step would serialise the whole run against it.
        """
        flat = int(np.ravel_multi_index(self.grid.cell_index(point), self.grid.shape))
        self._sources.append(
            (flat, torch.as_tensor(np.asarray(signal), device=self.device, dtype=self.dtype))
        )

    def add_pressure_receiver(self, point: tuple[float, float, float]) -> int:
        """Record pressure at ``point``. Samples stay on the device until :meth:`recorded_pressure`.

        Reading one value per step back to the host would force a synchronisation every step and
        throw away everything the device was gaining.
        """
        self._receiver_indices.append(
            int(np.ravel_multi_index(self.grid.cell_index(point), self.grid.shape))
        )
        return len(self._receiver_indices) - 1

    @property
    def recorded_pressure(self) -> np.ndarray:
        """Everything recorded so far, as ``(channel, sample)`` on the host."""
        if self._recording is None:
            return np.zeros((len(self._receiver_indices), 0))
        return self._recording.detach().cpu().numpy()

    def run(self, n_steps: int) -> None:
        """Advance ``n_steps``, growing the recording buffer once rather than per step."""
        if self._receiver_indices:
            addition = torch.zeros(
                (len(self._receiver_indices), n_steps), device=self.device, dtype=self.dtype
            )
            self._recording = (
                addition
                if self._recording is None
                else torch.cat((self._recording, addition), dim=1)
            )
            offset = self._recording.shape[1] - n_steps
            indices = torch.as_tensor(self._receiver_indices, device=self.device, dtype=torch.long)
        for local_step in range(n_steps):
            self.step()
            if self._receiver_indices:
                self._recording[:, offset + local_step] = self.p.view(-1)[indices]

    def step(self) -> None:
        """Advance one time step, preserving the ``(p^n, v^{n-1/2})`` pairing."""
        self._update_velocity()
        for axis in _AXES:
            self._damp(self.velocity[axis], self._velocity_damping[axis])
        self._update_pressure()
        self._damp(self.p, self._pressure_damping)
        self._inject_sources()
        self.step_index += 1

    @staticmethod
    def _damp(field: torch.Tensor, slabs) -> None:
        for index, factor in slabs:
            field[index] *= factor

    def _update_velocity(self) -> None:
        for axis in _AXES:
            self.velocity[axis][_interior_faces(axis)] -= self._velocity_coefficient * (
                self.p[_upper(axis)] - self.p[_lower(axis)]
            )

    def _update_pressure(self) -> None:
        flat = self.p.view(-1)
        previous_at_walls = flat[self._wall_indices].clone() if self._wall_indices.numel() else None

        divergence = self._divergence
        torch.sub(self.velocity[0][_upper(0)], self.velocity[0][_lower(0)], out=divergence)
        for axis in (1, 2):
            component = self.velocity[axis]
            torch.sub(component[_upper(axis)], component[_lower(axis)], out=self._scratch)
            divergence += self._scratch

        self._advance_relaxation(divergence)

        divergence *= self._pressure_coefficient
        self.p -= divergence

        if previous_at_walls is not None:
            flat[self._wall_indices] = (
                flat[self._wall_indices] * self._wall_from_updated
                - previous_at_walls * self._wall_from_previous
            )

    def _advance_relaxation(self, divergence: torch.Tensor) -> None:
        target = self._scratch
        offset = self._relaxation_scratch
        for state, decay, mean_weight, gain in self._relaxation:
            torch.mul(divergence, gain, out=target)
            torch.sub(state, target, out=offset)

            torch.mul(offset, decay, out=state)
            state += target

            torch.mul(offset, mean_weight, out=offset)
            self.p += offset

        if self._relaxation:
            torch.mul(divergence, self._relaxation_target_scale, out=target)
            self.p += target

    def _inject_sources(self) -> None:
        flat = self.p.view(-1)
        for index, signal in self._sources:
            if self.step_index < signal.shape[0]:
                flat[index] += self._source_coefficient * signal[self.step_index]

    def energy(self) -> float:
        """Total acoustic energy, in joules, for the stored ``(p^n, v^{n-1/2})`` pairing."""
        medium = self.medium
        cell_volume = self.grid.dx**3

        potential = float(torch.sum(self.p * self.p)) / (
            2.0 * medium.density * medium.sound_speed**2
        )

        kinetic = 0.0
        coupling = 0.0
        for axis in _AXES:
            component = self.velocity[axis]
            kinetic += 0.5 * medium.density * float(torch.sum(component * component))
            gradient = (self.p[_upper(axis)] - self.p[_lower(axis)]) / self.grid.dx
            coupling += float(torch.sum(component[_interior_faces(axis)] * gradient))

        return cell_volume * (potential + kinetic - 0.5 * self.dt * coupling)

    def synchronize(self) -> None:
        """Wait for the device to finish. Needed before timing, never needed for correctness."""
        synchronize(self.device)

"""Driving the compiled time loop, and building it on demand.

The library is compiled on first use rather than vendored, so there is no architecture-specific
binary in the repository and it is built by whatever compiler is present. Compilation takes a
few hundred milliseconds, the result is cached next to the source, and it is rebuilt whenever
the C file is newer — an edit cannot be silently ignored.

There is nothing to install first: threading is pthreads rather than OpenMP, for reasons the
C file explains, so the build needs only a C compiler.

**Precision is a build option.** The kernel is compiled twice, once with ``REAL=float`` and once
with ``REAL=double``. Single precision is not a formality here: it halves the memory traffic on
a problem where memory traffic is the cost.

``-ffp-contract=off`` is deliberate. At ``-O3`` clang fuses ``a*b+c`` into a single
fused multiply-add, which is *more* accurate than doing it in two steps but not *equal* to what
NumPy does — and a backend that differs from the reference by a few ulp is harder to dismiss
than one that differs by a lot, because it reads as a physics result.
"""

from __future__ import annotations

import ctypes
import math
import subprocess
import sysconfig
from pathlib import Path

import numpy as np

from .air import AirAbsorption
from .boundaries import AbsorbingLayer, WallAdmittances, layer_factors, wall_update_coefficients
from .boundaries2d import (
    AbsorbingLayer2D,
    WallAdmittances2D,
    layer_factors_2d,
    wall_update_coefficients_2d,
)
from .grid import Grid, max_stable_time_step
from .grid2d import Grid2D, max_stable_time_step_2d
from .medium import AIR, Medium

__all__ = [
    "CAcousticFDTD",
    "CAcousticFDTD2D",
    "CScalarFDTD",
    "CScalarFDTD2D",
    "build",
    "build_report",
    "is_available",
]

_SOURCE = Path(__file__).resolve().parent / "c" / "acfdtd.c"
_MAX_RELAXATION = 4

_BASE_FLAGS = ["-O3", "-fno-fast-math", "-ffp-contract=off", "-shared", "-fPIC", "-pthread"]

_libraries: dict[str, ctypes.CDLL] = {}


def _library_path(dtype: np.dtype) -> Path:
    suffix = sysconfig.get_config_var("SHLIB_SUFFIX") or ".so"
    return _SOURCE.with_name(f"acfdtd_{'f64' if dtype == np.float64 else 'f32'}{suffix}")


def _compiler() -> list[str]:
    return (sysconfig.get_config_var("CC") or "cc").split()


def _compile(target: Path, dtype: np.dtype) -> None:
    real = "double" if dtype == np.float64 else "float"
    command = [
        *_compiler(),
        *_BASE_FLAGS,
        f"-DREAL={real}",
        str(_SOURCE),
        "-o",
        str(target),
        "-lm",
    ]
    subprocess.run(command, check=True, capture_output=True)


def build(dtype: np.dtype = np.float64, force: bool = False) -> Path:
    """Compile the shared library for one precision, skipping work that is already done."""
    dtype = np.dtype(dtype)
    target = _library_path(dtype)
    if not force and target.exists() and target.stat().st_mtime >= _SOURCE.stat().st_mtime:
        return target
    try:
        _compile(target, dtype)
    except (subprocess.CalledProcessError, OSError) as error:
        raise RuntimeError(f"could not compile {_SOURCE}") from error
    return target


def _library(dtype: np.dtype) -> ctypes.CDLL:
    key = str(dtype)
    if key not in _libraries:
        library = ctypes.CDLL(str(build(dtype)))
        library.acfdtd_struct_size.restype = ctypes.c_size_t
        library.acfdtd_scalar_struct_size.restype = ctypes.c_size_t
        library.acfdtd_scalar_run.restype = None
        library.acfdtd_thread_count.restype = ctypes.c_int
        library.acfdtd_set_threads.restype = ctypes.c_int
        library.acfdtd_set_threads.argtypes = [ctypes.c_int]
        library.acfdtd_is_double.restype = ctypes.c_int
        library.acfdtd_run.restype = None
        library.acfdtd2d_struct_size.restype = ctypes.c_size_t
        library.acfdtd2d_run.restype = None
        library.acfdtd2d_scalar_struct_size.restype = ctypes.c_size_t
        library.acfdtd2d_scalar_run.restype = None
        _libraries[key] = library
    return _libraries[key]


def is_available() -> bool:
    """Whether a compiler that can build the kernel is present."""
    try:
        build()
        return True
    except (RuntimeError, OSError):
        return False


def build_report() -> dict[str, object]:
    """How the library was built and what it can use. Read this before quoting a timing."""
    build()
    library = _library(np.float64)
    return {
        "compiler": " ".join(_compiler()),
        "flags": list(_BASE_FLAGS),
        "threads": int(library.acfdtd_thread_count()),
    }


def _structure(dtype: np.dtype) -> type[ctypes.Structure]:
    real = ctypes.c_double if dtype == np.float64 else ctypes.c_float
    pointer = ctypes.POINTER(real)
    index_pointer = ctypes.POINTER(ctypes.c_int64)

    class State(ctypes.Structure):
        _fields_ = [
            ("nx", ctypes.c_int32),
            ("ny", ctypes.c_int32),
            ("nz", ctypes.c_int32),
            ("n_relax", ctypes.c_int32),
            ("n_wall", ctypes.c_int64),
            ("n_sources", ctypes.c_int64),
            ("n_receivers", ctypes.c_int64),
            ("n_steps", ctypes.c_int64),
            ("source_length", ctypes.c_int64),
            ("vel_coef", real),
            ("pres_coef", real),
            ("src_coef", real),
            ("relax_scale", real),
            ("p", pointer),
            ("vx", pointer),
            ("vy", pointer),
            ("vz", pointer),
            ("psi", pointer),
            ("relax_decay", real * _MAX_RELAXATION),
            ("relax_mean", real * _MAX_RELAXATION),
            ("relax_gain", real * _MAX_RELAXATION),
            ("cell", pointer * 3),
            ("face", pointer * 3),
            ("has_layer", ctypes.c_int32),
            ("wall_index", index_pointer),
            ("wall_from_updated", pointer),
            ("wall_from_previous", pointer),
            ("wall_scratch", pointer),
            ("source_index", index_pointer),
            ("source_signal", pointer),
            ("receiver_index", index_pointer),
            ("recording", pointer),
        ]

    return State


def _structure_2d(dtype: np.dtype) -> type[ctypes.Structure]:
    real = ctypes.c_double if dtype == np.float64 else ctypes.c_float
    pointer = ctypes.POINTER(real)
    index_pointer = ctypes.POINTER(ctypes.c_int64)

    class State2D(ctypes.Structure):
        _fields_ = [
            ("nx", ctypes.c_int32),
            ("ny", ctypes.c_int32),
            ("n_relax", ctypes.c_int32),
            ("n_wall", ctypes.c_int64),
            ("n_sources", ctypes.c_int64),
            ("n_receivers", ctypes.c_int64),
            ("n_steps", ctypes.c_int64),
            ("source_length", ctypes.c_int64),
            ("vel_coef", real),
            ("pres_coef", real),
            ("src_coef", real),
            ("relax_scale", real),
            ("p", pointer),
            ("vx", pointer),
            ("vy", pointer),
            ("psi", pointer),
            ("relax_decay", real * _MAX_RELAXATION),
            ("relax_mean", real * _MAX_RELAXATION),
            ("relax_gain", real * _MAX_RELAXATION),
            ("cell", pointer * 2),
            ("face", pointer * 2),
            ("has_layer", ctypes.c_int32),
            ("wall_index", index_pointer),
            ("wall_from_updated", pointer),
            ("wall_from_previous", pointer),
            ("wall_scratch", pointer),
            ("source_index", index_pointer),
            ("source_signal", pointer),
            ("receiver_index", index_pointer),
            ("recording", pointer),
            ("vx_open", pointer),
            ("vy_open", pointer),
            ("p_keep", pointer),
            ("has_rigid", ctypes.c_int32),
            ("n_pres_pressure", ctypes.c_int64),
            ("n_pres_velocity", ctypes.c_int64),
            ("pres_length", ctypes.c_int64),
            ("pp_index", index_pointer),
            ("pp_start", index_pointer),
            ("pp_signal", pointer),
            ("pv_axis", ctypes.POINTER(ctypes.c_int32)),
            ("pv_index", index_pointer),
            ("pv_start", index_pointer),
            ("pv_signal", pointer),
        ]

    return State2D


def _scalar_structure(dtype: np.dtype) -> type[ctypes.Structure]:
    real = ctypes.c_double if dtype == np.float64 else ctypes.c_float

    class ScalarState(ctypes.Structure):
        _fields_ = [
            ("nx", ctypes.c_int32),
            ("ny", ctypes.c_int32),
            ("nz", ctypes.c_int32),
            ("n_steps", ctypes.c_int64),
            ("lambda_squared", real),
            ("current", ctypes.POINTER(real)),
            ("previous", ctypes.POINTER(real)),
        ]

    return ScalarState


def _scalar_structure_2d(dtype: np.dtype) -> type[ctypes.Structure]:
    real = ctypes.c_double if dtype == np.float64 else ctypes.c_float
    pointer = ctypes.POINTER(real)
    index_pointer = ctypes.POINTER(ctypes.c_int64)

    class ScalarState2D(ctypes.Structure):
        _fields_ = [
            ("nx", ctypes.c_int32),
            ("ny", ctypes.c_int32),
            ("n_wall", ctypes.c_int64),
            ("n_sources", ctypes.c_int64),
            ("n_receivers", ctypes.c_int64),
            ("n_steps", ctypes.c_int64),
            ("source_length", ctypes.c_int64),
            ("lambda_squared", real),
            ("src_coef", real),
            ("current", pointer),
            ("previous", pointer),
            ("x_open", pointer),
            ("y_open", pointer),
            ("p_keep", pointer),
            ("has_rigid", ctypes.c_int32),
            ("wall_index", index_pointer),
            ("wall_a", pointer),
            ("wall_scratch", pointer),
            ("source_index", index_pointer),
            ("source_signal", pointer),
            ("receiver_index", index_pointer),
            ("recording", pointer),
        ]

    return ScalarState2D


class CAcousticFDTD:
    """The staggered scheme with its time loop compiled, fused into two sweeps per step.

    Same physics and same state pairing as :class:`~ac_fdtd.scheme.AcousticFDTD`; the fields are
    ordinary NumPy arrays that C writes into, so they can be inspected between runs at no cost.

    The one API difference is that :meth:`run` is the unit of work rather than ``step``: the
    whole loop happens inside C, which is the entire point. A single step is ``run(1)``.
    """

    def __init__(
        self,
        grid: Grid,
        medium: Medium = AIR,
        courant: float = 1.0,
        dtype: np.dtype = np.float64,
        walls: WallAdmittances | None = None,
        absorbing_layer: AbsorbingLayer | None = None,
        air_absorption: AirAbsorption | None = None,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")

        self.grid = grid
        self.medium = medium
        self.courant = courant
        self.dtype = np.dtype(dtype)
        self.walls = walls or WallAdmittances()
        self.absorbing_layer = absorbing_layer
        self.air_absorption = air_absorption
        self.dt = courant * max_stable_time_step(grid.dx, medium.sound_speed)
        self.step_index = 0

        self._library = _library(self.dtype)
        self._state_type = _structure(self.dtype)
        if ctypes.sizeof(self._state_type) != self._library.acfdtd_struct_size():
            raise RuntimeError(
                "the ctypes mirror of the C struct no longer matches the C definition: "
                f"{ctypes.sizeof(self._state_type)} bytes here, "
                f"{self._library.acfdtd_struct_size()} bytes there"
            )

        self.p = np.zeros(grid.shape, dtype=self.dtype)
        self.velocity = [np.zeros(grid.velocity_shape(axis), dtype=self.dtype) for axis in range(3)]

        processes = air_absorption.processes(medium.sound_speed, self.dt) if air_absorption else ()
        if len(processes) > _MAX_RELAXATION:
            raise ValueError(f"the kernel holds at most {_MAX_RELAXATION} relaxation processes")
        self._relaxation_states = np.zeros((len(processes), *grid.shape), dtype=self.dtype)

        relaxed_speed_squared = medium.sound_speed**2 / (
            1.0 + sum(process.strength for process in processes)
        )
        self._decay = np.zeros(_MAX_RELAXATION)
        self._mean = np.zeros(_MAX_RELAXATION)
        self._gain = np.zeros(_MAX_RELAXATION)
        for slot, process in enumerate(processes):
            decay = math.exp(-self.dt / process.relaxation_time)
            self._decay[slot] = decay
            self._mean[slot] = self.dt * (process.relaxation_time / self.dt) * (1.0 - decay)
            self._gain[slot] = medium.density * relaxed_speed_squared * process.strength / grid.dx
        self._relaxation_scale = self.dt * float(self._gain.sum())

        indices, coefficient = wall_update_coefficients(grid, medium, self.dt, self.walls)
        self._wall_index = np.ascontiguousarray(indices, dtype=np.int64)
        self._wall_from_updated = (1.0 / (1.0 + coefficient)).astype(self.dtype)
        self._wall_from_previous = (coefficient / (1.0 + coefficient)).astype(self.dtype)
        self._wall_scratch = np.zeros(self._wall_index.size, dtype=self.dtype)

        if absorbing_layer is None:
            self._cell = [np.ones(grid.shape[axis], dtype=self.dtype) for axis in range(3)]
            self._face = [np.ones(grid.shape[axis] + 1, dtype=self.dtype) for axis in range(3)]
        else:
            pressure_factors, velocity_factors = layer_factors(
                grid, medium, self.dt, absorbing_layer
            )
            self._cell = [pressure_factors[axis].astype(self.dtype) for axis in range(3)]
            self._face = [velocity_factors[axis][axis].astype(self.dtype) for axis in range(3)]

        self._source_index = np.zeros(0, dtype=np.int64)
        self._source_signal = np.zeros((0, 0), dtype=self.dtype)
        self._receiver_index = np.zeros(0, dtype=np.int64)
        self._recordings: list[np.ndarray] = []

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.dt

    @property
    def time(self) -> float:
        return self.step_index * self.dt

    @property
    def threads(self) -> int:
        """Threads the compiled library will use."""
        return int(self._library.acfdtd_thread_count())

    def set_threads(self, count: int) -> int:
        """Set the pool size; ``0`` means one per online processor. Returns what was set.

        The pool is shared by every instance using the same precision, because it belongs to
        the library rather than to a solver.
        """
        return int(self._library.acfdtd_set_threads(int(count)))

    def add_volume_source(self, point: tuple[float, float, float], signal: np.ndarray) -> None:
        """Add a volume-velocity source.

        Signals of different lengths are zero-padded to a common length, because the kernel
        indexes them as one rectangular array. Padding with silence is what a shorter signal
        means anyway.
        """
        flat = int(np.ravel_multi_index(self.grid.cell_index(point), self.grid.shape))
        signal = np.asarray(signal, dtype=self.dtype)
        length = max(self._source_signal.shape[1], signal.size)

        padded = np.zeros((self._source_signal.shape[0] + 1, length), dtype=self.dtype)
        padded[:-1, : self._source_signal.shape[1]] = self._source_signal
        padded[-1, : signal.size] = signal

        self._source_index = np.append(self._source_index, flat)
        self._source_signal = padded

    def add_pressure_receiver(self, point: tuple[float, float, float]) -> int:
        flat = int(np.ravel_multi_index(self.grid.cell_index(point), self.grid.shape))
        self._receiver_index = np.append(self._receiver_index, flat)
        return self._receiver_index.size - 1

    @property
    def recorded_pressure(self) -> np.ndarray:
        """Everything recorded so far, as ``(channel, sample)``."""
        if not self._recordings:
            return np.zeros((self._receiver_index.size, 0))
        return np.concatenate(self._recordings, axis=1)

    def _pointer(self, array: np.ndarray):
        real = ctypes.c_double if self.dtype == np.float64 else ctypes.c_float
        return array.ctypes.data_as(ctypes.POINTER(real))

    def _indices(self, array: np.ndarray):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

    def run(self, n_steps: int) -> None:
        """Advance ``n_steps`` inside the compiled loop."""
        if n_steps <= 0:
            return

        recording = np.zeros((self._receiver_index.size, n_steps), dtype=self.dtype)
        pointer_array = (
            ctypes.POINTER(ctypes.c_double if self.dtype == np.float64 else ctypes.c_float) * 3
        )

        state = self._state_type(
            nx=self.grid.nx,
            ny=self.grid.ny,
            nz=self.grid.nz,
            n_relax=self._relaxation_states.shape[0],
            n_wall=self._wall_index.size,
            n_sources=self._source_index.size,
            n_receivers=self._receiver_index.size,
            n_steps=n_steps,
            source_length=self._source_signal.shape[1],
            vel_coef=self.dt / (self.medium.density * self.grid.dx),
            pres_coef=self.medium.density * self.medium.sound_speed**2 * self.dt / self.grid.dx,
            src_coef=self.medium.density * self.medium.sound_speed**2 * self.dt,
            relax_scale=self._relaxation_scale,
            p=self._pointer(self.p),
            vx=self._pointer(self.velocity[0]),
            vy=self._pointer(self.velocity[1]),
            vz=self._pointer(self.velocity[2]),
            psi=self._pointer(self._relaxation_states),
            relax_decay=(ctypes.c_double * 4 if self.dtype == np.float64 else ctypes.c_float * 4)(
                *self._decay
            ),
            relax_mean=(ctypes.c_double * 4 if self.dtype == np.float64 else ctypes.c_float * 4)(
                *self._mean
            ),
            relax_gain=(ctypes.c_double * 4 if self.dtype == np.float64 else ctypes.c_float * 4)(
                *self._gain
            ),
            cell=pointer_array(*[self._pointer(profile) for profile in self._cell]),
            face=pointer_array(*[self._pointer(profile) for profile in self._face]),
            has_layer=int(self.absorbing_layer is not None),
            wall_index=self._indices(self._wall_index),
            wall_from_updated=self._pointer(self._wall_from_updated),
            wall_from_previous=self._pointer(self._wall_from_previous),
            wall_scratch=self._pointer(self._wall_scratch),
            source_index=self._indices(self._source_index),
            source_signal=self._pointer(self._source_signal),
            receiver_index=self._indices(self._receiver_index),
            recording=self._pointer(recording),
        )

        self._library.acfdtd_run(ctypes.byref(state), ctypes.c_int64(self.step_index))
        self.step_index += n_steps
        if self._receiver_index.size:
            self._recordings.append(recording)

    def step(self) -> None:
        self.run(1)

    def energy(self) -> float:
        """Total acoustic energy, in joules, for the stored ``(p^n, v^{n-1/2})`` pairing."""
        medium = self.medium
        cell_volume = self.grid.dx**3

        potential = float(np.dot(self.p.ravel(), self.p.ravel())) / (
            2.0 * medium.density * medium.sound_speed**2
        )

        kinetic = 0.0
        coupling = 0.0
        for axis in range(3):
            component = self.velocity[axis]
            flat = component.ravel()
            kinetic += 0.5 * medium.density * float(np.dot(flat, flat))
            upper = tuple(slice(1, None) if a == axis else slice(None) for a in range(3))
            lower = tuple(slice(None, -1) if a == axis else slice(None) for a in range(3))
            interior = tuple(slice(1, -1) if a == axis else slice(None) for a in range(3))
            gradient = (self.p[upper] - self.p[lower]) / self.grid.dx
            coupling += float(np.sum(component[interior] * gradient))

        return cell_volume * (potential + kinetic - 0.5 * self.dt * coupling)


class CAcousticFDTD2D:
    """The 2D staggered scheme with its time loop compiled (``ac_fdtd.scheme2d`` in C).

    Same physics and same ``(p^n, v^{n-1/2})`` pairing as
    :class:`~ac_fdtd.scheme2d.AcousticFDTD2D`; the fields are ordinary NumPy arrays that C writes
    into. As with the 3D kernel, :meth:`run` is the unit of work rather than ``step``, and the two
    share the one compiled library and its thread pool.

    ``rigid_cells`` carries interior rigid obstacles exactly as in the reference scheme: a boolean
    mask of solid cells whose touching velocity faces are held at ``v.n = 0``. It composes with
    absorbing walls and the layer (the resonator-in-an-absorbing-room case), and the compiled run
    is the same computation as :class:`~ac_fdtd.scheme2d.AcousticFDTD2D` with the same mask.
    """

    def __init__(
        self,
        grid: Grid2D,
        medium: Medium = AIR,
        courant: float = 1.0,
        dtype: np.dtype = np.float64,
        walls: WallAdmittances2D | None = None,
        absorbing_layer: AbsorbingLayer2D | None = None,
        air_absorption: AirAbsorption | None = None,
        rigid_cells: np.ndarray | None = None,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")

        self.grid = grid
        self.medium = medium
        self.courant = courant
        self.dtype = np.dtype(dtype)
        self.walls = walls or WallAdmittances2D()
        self.absorbing_layer = absorbing_layer
        self.air_absorption = air_absorption
        self.dt = courant * max_stable_time_step_2d(grid.dx, medium.sound_speed)
        self.step_index = 0

        self._library = _library(self.dtype)
        self._state_type = _structure_2d(self.dtype)
        if ctypes.sizeof(self._state_type) != self._library.acfdtd2d_struct_size():
            raise RuntimeError(
                "the ctypes mirror of the 2D C struct no longer matches the C definition: "
                f"{ctypes.sizeof(self._state_type)} bytes here, "
                f"{self._library.acfdtd2d_struct_size()} bytes there"
            )

        self.p = np.zeros(grid.shape, dtype=self.dtype)
        self.velocity = [np.zeros(grid.velocity_shape(axis), dtype=self.dtype) for axis in range(2)]

        processes = air_absorption.processes(medium.sound_speed, self.dt) if air_absorption else ()
        if len(processes) > _MAX_RELAXATION:
            raise ValueError(f"the kernel holds at most {_MAX_RELAXATION} relaxation processes")
        self._relaxation_states = np.zeros((len(processes), *grid.shape), dtype=self.dtype)

        relaxed_speed_squared = medium.sound_speed**2 / (
            1.0 + sum(process.strength for process in processes)
        )
        self._decay = np.zeros(_MAX_RELAXATION)
        self._mean = np.zeros(_MAX_RELAXATION)
        self._gain = np.zeros(_MAX_RELAXATION)
        for slot, process in enumerate(processes):
            decay = math.exp(-self.dt / process.relaxation_time)
            self._decay[slot] = decay
            self._mean[slot] = self.dt * (process.relaxation_time / self.dt) * (1.0 - decay)
            self._gain[slot] = medium.density * relaxed_speed_squared * process.strength / grid.dx
        self._relaxation_scale = self.dt * float(self._gain.sum())

        indices, coefficient = wall_update_coefficients_2d(grid, medium, self.dt, self.walls)
        self._wall_index = np.ascontiguousarray(indices, dtype=np.int64)
        self._wall_from_updated = (1.0 / (1.0 + coefficient)).astype(self.dtype)
        self._wall_from_previous = (coefficient / (1.0 + coefficient)).astype(self.dtype)
        self._wall_scratch = np.zeros(self._wall_index.size, dtype=self.dtype)

        if absorbing_layer is None:
            self._cell = [np.ones(grid.shape[axis], dtype=self.dtype) for axis in range(2)]
            self._face = [np.ones(grid.shape[axis] + 1, dtype=self.dtype) for axis in range(2)]
        else:
            pressure_factors, velocity_factors = layer_factors_2d(
                grid, medium, self.dt, absorbing_layer
            )
            self._cell = [pressure_factors[axis].astype(self.dtype) for axis in range(2)]
            self._face = [velocity_factors[axis][axis].astype(self.dtype) for axis in range(2)]

        self._source_index = np.zeros(0, dtype=np.int64)
        self._source_signal = np.zeros((0, 0), dtype=self.dtype)
        self._receiver_index = np.zeros(0, dtype=np.int64)
        self._recordings: list[np.ndarray] = []

        # Rigid obstacles as 0/1 factors the C sweeps multiply by (see the reference scheme for the
        # construction). When there is no mask the flag is off and the sweeps are skipped, so the
        # single-element placeholders below only exist to keep the struct pointers valid.
        self._has_rigid = rigid_cells is not None
        if rigid_cells is None:
            self._vx_open = np.ones(1, dtype=self.dtype)
            self._vy_open = np.ones(1, dtype=self.dtype)
            self._p_keep = np.ones(1, dtype=self.dtype)
        else:
            rigid = np.ascontiguousarray(rigid_cells, dtype=bool)
            if rigid.shape != grid.shape:
                raise ValueError(
                    f"rigid_cells shape {rigid.shape} does not match the grid {grid.shape}"
                )
            vx_open = np.ones(grid.velocity_shape(0), dtype=self.dtype)
            vx_open[1:-1, :] = ~(rigid[:-1, :] | rigid[1:, :])
            vy_open = np.ones(grid.velocity_shape(1), dtype=self.dtype)
            vy_open[:, 1:-1] = ~(rigid[:, :-1] | rigid[:, 1:])
            self._vx_open = np.ascontiguousarray(vx_open.reshape(-1))
            self._vy_open = np.ascontiguousarray(vy_open.reshape(-1))
            self._p_keep = np.ascontiguousarray((~rigid).astype(self.dtype).reshape(-1))

        # Prescribed (hard) sources, the counterpart of the reference scheme's prescribed sources.
        # (flat index, start, signal) for pressure, and (axis, flat index, start, signal) for
        # velocity. Flat indices are into p / vx / vy; they are packed into matrices in run().
        self._pressure_sources: list[tuple[int, int, np.ndarray]] = []
        self._velocity_sources: list[tuple[int, int, int, np.ndarray]] = []

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.dt

    @property
    def time(self) -> float:
        return self.step_index * self.dt

    @property
    def threads(self) -> int:
        return int(self._library.acfdtd_thread_count())

    def set_threads(self, count: int) -> int:
        return int(self._library.acfdtd_set_threads(int(count)))

    def add_volume_source(self, point: tuple[float, float], signal: np.ndarray) -> None:
        """Add a volume-velocity source; signals are zero-padded to a common length, as in 3D."""
        flat = int(np.ravel_multi_index(self.grid.cell_index(point), self.grid.shape))
        signal = np.asarray(signal, dtype=self.dtype)
        length = max(self._source_signal.shape[1], signal.size)

        padded = np.zeros((self._source_signal.shape[0] + 1, length), dtype=self.dtype)
        padded[:-1, : self._source_signal.shape[1]] = self._source_signal
        padded[-1, : signal.size] = signal

        self._source_index = np.append(self._source_index, flat)
        self._source_signal = padded

    def add_pressure_receiver(self, point: tuple[float, float]) -> int:
        flat = int(np.ravel_multi_index(self.grid.cell_index(point), self.grid.shape))
        self._receiver_index = np.append(self._receiver_index, flat)
        return self._receiver_index.size - 1

    def add_prescribed_source(
        self,
        point: tuple[float, float],
        signal: np.ndarray,
        start: int = 0,
        kind: str = "pressure",
        axis: int | None = None,
    ) -> None:
        """Hard-drive a node for a window of steps (see ``AcousticFDTD2D.add_prescribed_source``).

        The same contract as the reference scheme: a ``pressure`` node is set after the volume
        sources, a ``velocity`` face (``axis`` 0 or 1) before the pressure update reads it. The flat
        index is into the pressure field or the chosen velocity grid, computed here so the kernel
        just writes it.
        """
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        samples = np.asarray(signal, dtype=self.dtype)
        cell = self.grid.cell_index(point)
        if kind == "pressure":
            if axis is not None:
                raise ValueError("axis is meaningless for a pressure source; leave it None")
            flat = int(np.ravel_multi_index(cell, self.grid.shape))
            self._pressure_sources.append((flat, start, samples))
        elif kind == "velocity":
            if axis not in (0, 1):
                raise ValueError(f"velocity source needs axis in (0, 1), got {axis!r}")
            flat = int(np.ravel_multi_index(cell, self.grid.velocity_shape(axis)))
            self._velocity_sources.append((axis, flat, start, samples))
        else:
            raise ValueError(f"kind must be 'pressure' or 'velocity', got {kind!r}")

    @property
    def recorded_pressure(self) -> np.ndarray:
        if not self._recordings:
            return np.zeros((self._receiver_index.size, 0))
        return np.concatenate(self._recordings, axis=1)

    def _pointer(self, array: np.ndarray):
        real = ctypes.c_double if self.dtype == np.float64 else ctypes.c_float
        return array.ctypes.data_as(ctypes.POINTER(real))

    def _indices(self, array: np.ndarray):
        return array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))

    def run(self, n_steps: int) -> None:
        """Advance ``n_steps`` inside the compiled 2D loop."""
        if n_steps <= 0:
            return

        recording = np.zeros((self._receiver_index.size, n_steps), dtype=self.dtype)
        real = ctypes.c_double if self.dtype == np.float64 else ctypes.c_float
        pointer_array = ctypes.POINTER(real) * 2
        array4 = (ctypes.c_double * 4) if self.dtype == np.float64 else (ctypes.c_float * 4)

        # Pack the prescribed sources into matrices the kernel reads row by row (kept alive as
        # locals through the C call). pres_length is their common, zero-padded window length.
        pres_length = max(
            [0, *(sig.size for _, _, sig in self._pressure_sources)]
            + [sig.size for _, _, _, sig in self._velocity_sources]
        )
        pp_index = np.array([i for i, _, _ in self._pressure_sources], dtype=np.int64)
        pp_start = np.array([s for _, s, _ in self._pressure_sources], dtype=np.int64)
        pp_signal = np.zeros((len(self._pressure_sources), pres_length), dtype=self.dtype)
        for row, (_, _, sig) in enumerate(self._pressure_sources):
            pp_signal[row, : sig.size] = sig
        pv_axis = np.array([a for a, _, _, _ in self._velocity_sources], dtype=np.int32)
        pv_index = np.array([i for _, i, _, _ in self._velocity_sources], dtype=np.int64)
        pv_start = np.array([s for _, _, s, _ in self._velocity_sources], dtype=np.int64)
        pv_signal = np.zeros((len(self._velocity_sources), pres_length), dtype=self.dtype)
        for row, (_, _, _, sig) in enumerate(self._velocity_sources):
            pv_signal[row, : sig.size] = sig

        state = self._state_type(
            nx=self.grid.nx,
            ny=self.grid.ny,
            n_relax=self._relaxation_states.shape[0],
            n_wall=self._wall_index.size,
            n_sources=self._source_index.size,
            n_receivers=self._receiver_index.size,
            n_steps=n_steps,
            source_length=self._source_signal.shape[1],
            vel_coef=self.dt / (self.medium.density * self.grid.dx),
            pres_coef=self.medium.density * self.medium.sound_speed**2 * self.dt / self.grid.dx,
            src_coef=self.medium.density * self.medium.sound_speed**2 * self.dt,
            relax_scale=self._relaxation_scale,
            p=self._pointer(self.p),
            vx=self._pointer(self.velocity[0]),
            vy=self._pointer(self.velocity[1]),
            psi=self._pointer(self._relaxation_states),
            relax_decay=array4(*self._decay),
            relax_mean=array4(*self._mean),
            relax_gain=array4(*self._gain),
            cell=pointer_array(*[self._pointer(profile) for profile in self._cell]),
            face=pointer_array(*[self._pointer(profile) for profile in self._face]),
            has_layer=int(self.absorbing_layer is not None),
            wall_index=self._indices(self._wall_index),
            wall_from_updated=self._pointer(self._wall_from_updated),
            wall_from_previous=self._pointer(self._wall_from_previous),
            wall_scratch=self._pointer(self._wall_scratch),
            source_index=self._indices(self._source_index),
            source_signal=self._pointer(self._source_signal),
            receiver_index=self._indices(self._receiver_index),
            recording=self._pointer(recording),
            vx_open=self._pointer(self._vx_open),
            vy_open=self._pointer(self._vy_open),
            p_keep=self._pointer(self._p_keep),
            has_rigid=int(self._has_rigid),
            n_pres_pressure=len(self._pressure_sources),
            n_pres_velocity=len(self._velocity_sources),
            pres_length=pres_length,
            pp_index=self._indices(pp_index),
            pp_start=self._indices(pp_start),
            pp_signal=self._pointer(pp_signal),
            pv_axis=pv_axis.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            pv_index=self._indices(pv_index),
            pv_start=self._indices(pv_start),
            pv_signal=self._pointer(pv_signal),
        )

        self._library.acfdtd2d_run(ctypes.byref(state), ctypes.c_int64(self.step_index))
        self.step_index += n_steps
        if self._receiver_index.size:
            self._recordings.append(recording)

    def step(self) -> None:
        self.run(1)

    def energy(self) -> float:
        """Total acoustic energy per metre of depth, for the stored ``(p^n, v^{n-1/2})`` pairing."""
        medium = self.medium
        cell_area = self.grid.dx**2

        potential = float(np.dot(self.p.ravel(), self.p.ravel())) / (
            2.0 * medium.density * medium.sound_speed**2
        )

        kinetic = 0.0
        coupling = 0.0
        for axis in range(2):
            component = self.velocity[axis]
            flat = component.ravel()
            kinetic += 0.5 * medium.density * float(np.dot(flat, flat))
            upper = tuple(slice(1, None) if a == axis else slice(None) for a in range(2))
            lower = tuple(slice(None, -1) if a == axis else slice(None) for a in range(2))
            interior = tuple(slice(1, -1) if a == axis else slice(None) for a in range(2))
            gradient = (self.p[upper] - self.p[lower]) / self.grid.dx
            coupling += float(np.sum(component[interior] * gradient))

        return cell_area * (potential + kinetic - 0.5 * self.dt * coupling)


class CScalarFDTD:
    """The pressure-only variant: the same scheme with velocity eliminated.

    Two cell-sized arrays instead of four, and three array touches per cell per step instead of
    twelve. It is not an approximation — starting from ``p^{-1} = p^0`` it reproduces a
    pressure-velocity run started from ``v^{-1/2} = 0`` to the last bit, which is what
    ``tests/test_scalar_variant.py`` checks.

    **Rigid walls only.** The impedance wall, the absorbing layer and the relaxation states are
    all natural in a first-order system and would each need re-deriving as a three-level
    relation here. This exists to measure what that work would be worth before doing it.
    """

    def __init__(
        self,
        grid: Grid,
        medium: Medium = AIR,
        courant: float = 1.0,
        dtype: np.dtype = np.float64,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")

        self.grid = grid
        self.medium = medium
        self.courant = courant
        self.dtype = np.dtype(dtype)
        self.dt = courant * max_stable_time_step(grid.dx, medium.sound_speed)
        self.step_index = 0

        self._library = _library(self.dtype)
        self._state_type = _scalar_structure(self.dtype)
        if ctypes.sizeof(self._state_type) != self._library.acfdtd_scalar_struct_size():
            raise RuntimeError("the ctypes mirror of acfdtd_scalar no longer matches the C one")

        self._buffers = [
            np.zeros(grid.shape, dtype=self.dtype),
            np.zeros(grid.shape, dtype=self.dtype),
        ]
        #: Which buffer holds the newest level. The C loop swaps its own pointers once per
        #: step, so after an odd number of steps the roles have changed places.
        self._newest = 0

    @property
    def p(self) -> np.ndarray:
        """The newest pressure level."""
        return self._buffers[self._newest]

    @property
    def previous(self) -> np.ndarray:
        """The level before it. Equal to :attr:`p` at rest, which is the state ``v = 0``."""
        return self._buffers[1 - self._newest]

    def set_initial_pressure(self, field: np.ndarray) -> None:
        """Start from a given pressure at rest — both levels equal, so the velocity is zero."""
        for buffer in self._buffers:
            buffer[:] = field

    @property
    def threads(self) -> int:
        return int(self._library.acfdtd_thread_count())

    def run(self, n_steps: int) -> None:
        if n_steps <= 0:
            return
        real = ctypes.c_double if self.dtype == np.float64 else ctypes.c_float
        state = self._state_type(
            nx=self.grid.nx,
            ny=self.grid.ny,
            nz=self.grid.nz,
            n_steps=n_steps,
            lambda_squared=(self.medium.sound_speed * self.dt / self.grid.dx) ** 2,
            current=self.p.ctypes.data_as(ctypes.POINTER(real)),
            previous=self.previous.ctypes.data_as(ctypes.POINTER(real)),
        )
        self._library.acfdtd_scalar_run(ctypes.byref(state))
        self.step_index += n_steps
        self._newest ^= n_steps & 1


class CScalarFDTD2D:
    """The pressure-only 2D scheme with its time loop compiled (``ac_fdtd.scalar2d`` in C).

    Two cell arrays instead of three, one sweep instead of two: the fast, low-memory path for a
    closed 2D room. It is the same scheme as :class:`~ac_fdtd.scheme2d.AcousticFDTD2D`, not an
    approximation — bit-identical to :class:`~ac_fdtd.scalar2d.ScalarFDTD2D` — with rigid and
    locally-reacting walls, rigid obstacles, soft volume sources and pressure receivers. Open
    boundaries (the CFS-CPML shell) and the matched layer are not compiled here yet; the NumPy
    reference carries those.

    As with the other compiled kernels :meth:`run` is the unit of work, and the two pressure levels
    are ordinary NumPy arrays C writes into.
    """

    def __init__(
        self,
        grid: Grid2D,
        medium: Medium = AIR,
        courant: float = 1.0,
        dtype: np.dtype = np.float64,
        walls: WallAdmittances2D | None = None,
        rigid_cells: np.ndarray | None = None,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")

        self.grid = grid
        self.medium = medium
        self.courant = courant
        self.dtype = np.dtype(dtype)
        self.walls = walls or WallAdmittances2D()
        self.dt = courant * max_stable_time_step_2d(grid.dx, medium.sound_speed)
        self.step_index = 0

        self._library = _library(self.dtype)
        self._state_type = _scalar_structure_2d(self.dtype)
        if ctypes.sizeof(self._state_type) != self._library.acfdtd2d_scalar_struct_size():
            raise RuntimeError(
                "the ctypes mirror of the 2D scalar C struct no longer matches the C definition: "
                f"{ctypes.sizeof(self._state_type)} bytes here, "
                f"{self._library.acfdtd2d_scalar_struct_size()} bytes there"
            )

        self._lambda_squared = (medium.sound_speed * self.dt / grid.dx) ** 2
        self._src_coef = medium.density * medium.sound_speed**2 * self.dt

        self._buffers = [
            np.zeros(grid.shape, dtype=self.dtype),
            np.zeros(grid.shape, dtype=self.dtype),
        ]
        self._newest = 0

        indices, coefficient = wall_update_coefficients_2d(grid, medium, self.dt, self.walls)
        self._wall_index = np.ascontiguousarray(indices, dtype=np.int64)
        self._wall_a = coefficient.astype(self.dtype)
        self._wall_scratch = np.zeros(self._wall_index.size, dtype=self.dtype)

        self._has_rigid = rigid_cells is not None
        if rigid_cells is None:
            self._x_open = np.ones(1, dtype=self.dtype)
            self._y_open = np.ones(1, dtype=self.dtype)
            self._p_keep = np.ones(1, dtype=self.dtype)
        else:
            rigid = np.ascontiguousarray(rigid_cells, dtype=bool)
            if rigid.shape != grid.shape:
                raise ValueError(
                    f"rigid_cells shape {rigid.shape} does not match the grid {grid.shape}"
                )
            self._x_open = np.ascontiguousarray(
                (~(rigid[:-1, :] | rigid[1:, :])).astype(self.dtype).reshape(-1)
            )
            self._y_open = np.ascontiguousarray(
                (~(rigid[:, :-1] | rigid[:, 1:])).astype(self.dtype).reshape(-1)
            )
            self._p_keep = np.ascontiguousarray((~rigid).astype(self.dtype).reshape(-1))

        self._sources: list[tuple[tuple[int, int], np.ndarray]] = []
        self._receivers: list[tuple[int, int]] = []
        self._recordings: list[np.ndarray] = []

    @property
    def p(self) -> np.ndarray:
        """The newest pressure level."""
        return self._buffers[self._newest]

    @property
    def previous(self) -> np.ndarray:
        return self._buffers[1 - self._newest]

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.dt

    @property
    def time(self) -> float:
        return self.step_index * self.dt

    @property
    def threads(self) -> int:
        return int(self._library.acfdtd_thread_count())

    def set_threads(self, count: int) -> int:
        return int(self._library.acfdtd_set_threads(int(count)))

    def set_initial_pressure(self, field: np.ndarray) -> None:
        """Start from a given pressure at rest — both levels equal, so the velocity is zero."""
        for buffer in self._buffers:
            buffer[:] = field

    def add_volume_source(self, point: tuple[float, float], signal: np.ndarray) -> None:
        self._sources.append((self.grid.cell_index(point), np.asarray(signal, dtype=self.dtype)))

    def add_pressure_receiver(self, point: tuple[float, float]) -> int:
        index = self.grid.cell_index(point)
        flat = index[0] * self.grid.ny + index[1]
        self._receivers.append((len(self._receivers), flat))
        return len(self._receivers) - 1

    @property
    def recorded_pressure(self) -> np.ndarray:
        if not self._recordings:
            return np.zeros((len(self._receivers), 0))
        return np.concatenate(self._recordings, axis=1)

    def run(self, n_steps: int) -> None:
        if n_steps <= 0:
            return
        real = ctypes.c_double if self.dtype == np.float64 else ctypes.c_float

        if self._sources:
            length = max(len(signal) for _, signal in self._sources)
            source_index = np.array(
                [index[0] * self.grid.ny + index[1] for index, _ in self._sources], dtype=np.int64
            )
            source_signal = np.zeros((len(self._sources), length), dtype=self.dtype)
            for row, (_, signal) in enumerate(self._sources):
                source_signal[row, : len(signal)] = signal
        else:
            length = 0
            source_index = np.zeros(0, dtype=np.int64)
            source_signal = np.zeros((0, 0), dtype=self.dtype)

        receiver_index = np.array([flat for _, flat in self._receivers], dtype=np.int64)
        recording = np.zeros((len(self._receivers), n_steps), dtype=self.dtype)

        source_signal = np.ascontiguousarray(source_signal)
        state = self._state_type(
            nx=self.grid.nx,
            ny=self.grid.ny,
            n_wall=self._wall_index.size,
            n_sources=len(self._sources),
            n_receivers=len(self._receivers),
            n_steps=n_steps,
            source_length=length,
            lambda_squared=self.dtype.type(self._lambda_squared),
            src_coef=self.dtype.type(self._src_coef),
            current=self.p.ctypes.data_as(ctypes.POINTER(real)),
            previous=self.previous.ctypes.data_as(ctypes.POINTER(real)),
            x_open=self._x_open.ctypes.data_as(ctypes.POINTER(real)),
            y_open=self._y_open.ctypes.data_as(ctypes.POINTER(real)),
            p_keep=self._p_keep.ctypes.data_as(ctypes.POINTER(real)),
            has_rigid=int(self._has_rigid),
            wall_index=self._wall_index.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            wall_a=self._wall_a.ctypes.data_as(ctypes.POINTER(real)),
            wall_scratch=self._wall_scratch.ctypes.data_as(ctypes.POINTER(real)),
            source_index=source_index.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            source_signal=source_signal.ctypes.data_as(ctypes.POINTER(real)),
            receiver_index=receiver_index.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            recording=recording.ctypes.data_as(ctypes.POINTER(real)),
        )
        self._library.acfdtd2d_scalar_run(ctypes.byref(state), ctypes.c_int64(self.step_index))
        self.step_index += n_steps
        self._newest ^= n_steps & 1
        if self._receivers:
            self._recordings.append(recording)

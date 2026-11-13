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
from .cpml import AbsorbingBoundary, cpml_axis_profiles_3d
from .cpml2d import AbsorbingBoundary2D, cpml_axis_profiles
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
            ("n_wall", ctypes.c_int64),
            ("wall_index", ctypes.POINTER(ctypes.c_int64)),
            ("wall_a", ctypes.POINTER(real)),
            ("wall_scratch", ctypes.POINTER(real)),
            ("step_offset", ctypes.c_int64),
            ("src_coef", real),
            ("n_sources", ctypes.c_int64),
            ("source_index", ctypes.POINTER(ctypes.c_int64)),
            ("source_offset", ctypes.POINTER(ctypes.c_int64)),
            ("source_length", ctypes.POINTER(ctypes.c_int64)),
            ("source_pool", ctypes.POINTER(real)),
            ("n_receivers", ctypes.c_int64),
            ("receiver_index", ctypes.POINTER(ctypes.c_int64)),
            ("recording", ctypes.POINTER(real)),
            ("n_velocity", ctypes.c_int64),
            ("velocity_cell", ctypes.POINTER(ctypes.c_int64)),
            ("velocity_lower", ctypes.POINTER(ctypes.c_int64)),
            ("velocity_difference", ctypes.POINTER(real)),
            ("x_open", ctypes.POINTER(real)),
            ("y_open", ctypes.POINTER(real)),
            ("z_open", ctypes.POINTER(real)),
            ("p_keep", ctypes.POINTER(real)),
            ("has_rigid", ctypes.c_int32),
            ("n_cpml_regions", ctypes.c_int64),
            ("cpml_region", ctypes.POINTER(ctypes.c_int64)),
            ("cpml_velocity_units", ctypes.c_int32),
            ("cpml_pressure_units", ctypes.c_int32),
            ("inverse_dx", real),
            ("cpml_cv", real),
            ("cpml_cp", real),
            *((name, ctypes.POINTER(real)) for name in _CPML_PROFILE_NAMES_3D),
            *((name, ctypes.POINTER(real)) for name in _CPML_FIELD_NAMES_3D),
        ]

    return ScalarState


# The CPML arrays of the 3D scalar struct, in struct order. Profiles: (kappa, b, a) at cell centres
# then at faces, x, y, z. Fields: the per-box pools of the shell state.
_CPML_PROFILE_NAMES_3D = (
    "kx_p", "bx_p", "ax_p", "ky_p", "by_p", "ay_p", "kz_p", "bz_p", "az_p",
    "kx_v", "bx_v", "ax_v", "ky_v", "by_v", "ay_v", "kz_v", "bz_v", "az_v",
)  # fmt: skip
_CPML_FIELD_NAMES_3D = (
    "vx", "psi_vx", "vy", "psi_vy", "vz", "psi_vz", "psi_px", "psi_py", "psi_pz",
)  # fmt: skip
# Columns of one row of the 3D region table; must match the CPML3_* enum of the C file.
_CPML_COLUMNS_3D = 12


def _scalar_structure_2d(dtype: np.dtype) -> type[ctypes.Structure]:
    real = ctypes.c_double if dtype == np.float64 else ctypes.c_float
    pointer = ctypes.POINTER(real)
    index_pointer = ctypes.POINTER(ctypes.c_int64)
    int32_pointer = ctypes.POINTER(ctypes.c_int32)

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
            ("vel_coef", real),
            ("pres_coef", real),
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
            ("n_faces", ctypes.c_int64),
            ("face_cell", index_pointer),
            ("face_lower", index_pointer),
            ("face_a_cell", pointer),
            ("face_a_lower", pointer),
            ("face_closed", int32_pointer),
            ("face_value", pointer),
            ("face_drive", pointer),
            ("face_driven", int32_pointer),
            ("n_windows", ctypes.c_int64),
            ("window_kind", int32_pointer),
            ("window_target", index_pointer),
            ("window_start", index_pointer),
            ("window_length", index_pointer),
            ("window_offset", index_pointer),
            ("window_pool", pointer),
            ("n_cpml_regions", ctypes.c_int64),
            ("cpml_region", index_pointer),
            ("cpml_velocity_units", ctypes.c_int32),
            ("cpml_pressure_units", ctypes.c_int32),
            ("inverse_dx", real),
            ("cpml_cv", real),
            ("cpml_cp", real),
            *((name, pointer) for name in _CPML_PROFILE_NAMES),
            *((name, pointer) for name in _CPML_FIELD_NAMES),
        ]

    return ScalarState2D


# The CPML arrays of the 2D scalar struct, in struct order. Profiles: (kappa, b, a) at cell centres
# then at faces, x before y. Fields: the per-region pools of the shell state.
_CPML_PROFILE_NAMES = (
    "kx_p", "bx_p", "ax_p", "ky_p", "by_p", "ay_p",
    "kx_v", "bx_v", "ax_v", "ky_v", "by_v", "ay_v",
)  # fmt: skip
_CPML_FIELD_NAMES = ("vx", "psi_vx", "vy", "psi_vy", "psi_px", "psi_py")
# Columns of one row of the region table; must match the CPML_* enum of the C file.
_CPML_COLUMNS = 9


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

    Rigid and locally-reacting walls (``walls``, as in :class:`~ac_fdtd.scalar.ScalarFDTD`). With
    any absorbing wall the kernel evaluates the reference's arithmetic in the reference's order, so
    it is bit-identical to ``ScalarFDTD`` in fp64; the all-rigid room keeps a faster sweep that
    agrees to round-off. Rigid obstacles (``rigid_cells``) close every face touching a solid cell
    and zero the solid cells after each step, as the reference does; a mask with any solid cell
    selects the exact sweep, so it is bit-identical too, all-rigid room included.

    An open boundary (``absorbing_boundary``) is the reference's CFS-CPML shell, bit-identical in
    fp64: a p-v + CPML update on the shell cells overwriting the scalar update there, with the
    velocity and memory fields stored per lined wall in slabs of the shell only (see
    :meth:`_setup_cpml`), so the interior is still two arrays. A shell selects the exact sweep.
    The relaxation states (air absorption) are not compiled here.

    Soft volume sources and pressure and velocity receivers follow the reference too, with the same
    signatures. Their tables are handed to C once per :meth:`run` chunk and the recordings of
    successive chunks are concatenated, so a run can be split anywhere.
    """

    def __init__(
        self,
        grid: Grid,
        medium: Medium = AIR,
        courant: float = 1.0,
        dtype: np.dtype = np.float64,
        walls: WallAdmittances | None = None,
        rigid_cells: np.ndarray | None = None,
        absorbing_boundary: AbsorbingBoundary | None = None,
    ) -> None:
        if not 0.0 < courant <= 1.0:
            raise ValueError(f"courant must be in (0, 1], got {courant}")

        self.grid = grid
        self.medium = medium
        self.courant = courant
        self.walls = walls or WallAdmittances()
        self.absorbing_boundary = absorbing_boundary
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

        # The reference casts lam^2 and a to the working precision before using them; doing the
        # same here is part of being bit-identical (and of matching it in fp32).
        self._lambda_squared = self.dtype.type((medium.sound_speed * self.dt / grid.dx) ** 2)
        # Rigid faces contribute nothing, so an all-rigid room has no wall cells and the C side
        # takes its fast rigid sweep; any absorbing face selects the exact one.
        indices, coefficient = wall_update_coefficients(grid, medium, self.dt, self.walls)
        self._wall_index = np.ascontiguousarray(indices, dtype=np.int64)
        self._wall_a = coefficient.astype(self.dtype)

        self._cpml_region = np.zeros(0, dtype=np.int64)
        self._cpml_units = (0, 0)
        self._cpml_constants = (0.0, 0.0, 0.0)
        self._cpml_profiles: list[np.ndarray] = []
        self._cpml_fields: list[np.ndarray] = []
        if absorbing_boundary is not None:
            self._setup_cpml(absorbing_boundary)
        self._wall_scratch = np.zeros(self._wall_index.size, dtype=self.dtype)

        # Rigid obstacles as the reference builds them: per axis, a face is open iff neither of its
        # cells is solid, as 0/1 in the working dtype the C Laplacian multiplies by. A mask with no
        # solid cell is no mask (multiplying by 1 changes nothing), which keeps the fast rigid sweep
        # for it; the one-element placeholders only keep the struct pointers valid.
        placeholder = np.ones(1, dtype=self.dtype)
        self._open = [placeholder, placeholder, placeholder]
        self._p_keep = placeholder
        self._has_rigid = False
        if rigid_cells is not None:
            rigid = np.ascontiguousarray(rigid_cells, dtype=bool)
            if rigid.shape != grid.shape:
                raise ValueError(
                    f"rigid_cells shape {rigid.shape} does not match the grid {grid.shape}"
                )
            if rigid.any():
                self._has_rigid = True
                self._open = []
                for axis in range(3):
                    lower = tuple(slice(None, -1) if a == axis else slice(None) for a in range(3))
                    upper = tuple(slice(1, None) if a == axis else slice(None) for a in range(3))
                    self._open.append(
                        np.ascontiguousarray(~(rigid[lower] | rigid[upper]), dtype=self.dtype)
                    )
                self._p_keep = np.ascontiguousarray(~rigid, dtype=self.dtype)

        # Cast as ScalarFDTD casts it; the velocity coefficient stays a Python float, as there.
        self._src_coef = self.dtype.type(medium.density * medium.sound_speed**2 * self.dt)
        self._vel_coef = self.dt / (medium.density * grid.dx)

        self._sources: list[tuple[int, np.ndarray]] = []
        self._receivers: list[int] = []
        self._recordings: list[np.ndarray] = []
        # Velocity probes: (flat cell, flat cell below along the axis or -1 on the lower wall).
        # The running value v^{n-1/2} of each is carried across chunks in fp64, as the reference
        # carries it.
        self._velocity_probes: list[tuple[int, int]] = []
        self._velocity_running = np.zeros(0)
        self._velocity_recordings: list[np.ndarray] = []

    @property
    def p(self) -> np.ndarray:
        """The newest pressure level."""
        return self._buffers[self._newest]

    @property
    def sample_rate(self) -> float:
        return 1.0 / self.dt

    @property
    def time(self) -> float:
        return self.step_index * self.dt

    def _setup_cpml(self, boundary: AbsorbingBoundary) -> None:
        """The shell of ``ScalarFDTD._setup_cpml``, cut into one box per lined wall.

        The reference keeps the shell state on full-grid arrays and masks its updates; here each
        lined wall gets a box of the shell: the x slabs take the whole y-z cross-section, the y
        slabs fill the x range between them, the z slabs the x-y rectangle between both. The boxes
        are disjoint and cover exactly the shell; each stores its cells and every face bordering
        them, so a face on an edge or corner of the shell is stored by two boxes (the C struct
        explains why that is safe). Profiles and constants are cast as the reference casts them,
        so the kernel multiplies the same rounded numbers.
        """
        grid = self.grid
        shape = grid.shape
        thickness = boundary.thickness
        if 2 * thickness >= min(shape):
            raise ValueError(f"absorbing layer of {thickness} cells does not fit a {shape} grid")

        profiles = cpml_axis_profiles_3d(grid, self.medium, self.dt, boundary)
        self._cpml_profiles = [
            np.ascontiguousarray(c, dtype=self.dtype)
            for kind in ("p", "v")
            for axis in range(3)
            for c in profiles[(kind, axis)]
        ]
        cast = self.dtype.type
        self._cpml_constants = (
            cast(1.0 / grid.dx),
            cast(self.dt / self.medium.density),
            cast(self.medium.density * self.medium.sound_speed**2 * self.dt),
        )

        # Per axis, the range of cells left once that axis' lined slabs are taken off.
        lined = [boundary.has_axis(axis) for axis in range(3)]
        inner = [
            (thickness if low else 0, n - thickness if high else n)
            for n, (low, high) in zip(shape, lined, strict=True)
        ]
        boxes = []
        for axis in range(3):
            for side, is_lined in enumerate(lined[axis]):
                if not is_lined:
                    continue
                n = shape[axis]
                bounds = [(0, size) for size in shape]
                bounds[axis] = (0, thickness) if side == 0 else (n - thickness, n)
                for earlier in range(axis):  # the axes whose slabs already took the edges
                    bounds[earlier] = inner[earlier]
                boxes.append(bounds)

        # One row per box: bounds, pool offsets (cells, x, y, z faces), first units.
        rows = []
        cells = x_faces = y_faces = z_faces = velocity_units = pressure_units = 0
        for (i0, i1), (j0, j1), (k0, k1) in boxes:
            rows.append(
                (i0, i1, j0, j1, k0, k1, cells, x_faces, y_faces, z_faces)
                + (velocity_units, pressure_units)
            )
            ni, nj, nk = i1 - i0, j1 - j0, k1 - k0
            cells += ni * nj * nk
            x_faces += (ni + 1) * nj * nk
            y_faces += ni * (nj + 1) * nk
            z_faces += ni * nj * (nk + 1)
            velocity_units += ni + 1
            pressure_units += ni
        self._cpml_region = np.array(rows, dtype=np.int64).reshape(-1)
        self._cpml_units = (velocity_units, pressure_units)
        # vx, psi_vx, vy, psi_vy, vz, psi_vz, psi_px, psi_py, psi_pz: the state that persists
        # across run calls. ``faces=()`` leaves them empty, a valid (empty) shell as there.
        self._cpml_fields = [
            np.zeros(size, dtype=self.dtype)
            for size in (x_faces, x_faces, y_faces, y_faces, z_faces, z_faces, cells, cells, cells)
        ]

        # As in the reference, the impedance-wall correction applies to interior wall cells only.
        # Membership is tested on the coordinates: a grid-sized shell mask is what this avoids.
        if self._wall_index.size:
            coordinates = np.unravel_index(self._wall_index, shape)
            in_shell = np.zeros(self._wall_index.size, dtype=bool)
            for axis, (low, high) in enumerate(lined):
                if low:
                    in_shell |= coordinates[axis] < thickness
                if high:
                    in_shell |= coordinates[axis] >= shape[axis] - thickness
            self._wall_index = np.ascontiguousarray(self._wall_index[~in_shell])
            self._wall_a = self._wall_a[~in_shell]

    def _flat(self, cell: tuple[int, ...]) -> int:
        return (cell[0] * self.grid.ny + cell[1]) * self.grid.nz + cell[2]

    def add_volume_source(self, point: tuple[float, float, float], signal: np.ndarray) -> None:
        """A soft source on the simulation's clock: sample ``k`` belongs to global step ``k``."""
        cell = self.grid.cell_index(point)
        self._sources.append((self._flat(cell), np.asarray(signal, dtype=self.dtype)))

    def add_pressure_receiver(self, point: tuple[float, float, float]) -> int:
        self._receivers.append(self._flat(self.grid.cell_index(point)))
        return len(self._receivers) - 1

    def add_velocity_receiver(self, point: tuple[float, float, float], axis: int) -> int:
        """``v^{n-1/2}`` on the lower face of ``point`` along ``axis``, rebuilt from the two cells
        as :meth:`ac_fdtd.scalar.ScalarFDTD.add_velocity_receiver` does; zero on a lower wall."""
        cell = self.grid.cell_index(point)
        if cell[axis] > 0:
            below = list(cell)
            below[axis] -= 1
            lower = self._flat(tuple(below))
        else:
            lower = -1
        self._velocity_probes.append((self._flat(cell), lower))
        self._velocity_running = np.append(self._velocity_running, 0.0)
        return len(self._velocity_probes) - 1

    @property
    def recorded_pressure(self) -> np.ndarray:
        if not self._receivers:
            return np.zeros((0, 0))
        if not self._recordings:
            return np.zeros((len(self._receivers), 0), dtype=self.dtype)
        return np.concatenate(self._recordings, axis=1)

    @property
    def recorded_velocity(self) -> np.ndarray:
        if not self._velocity_probes:
            return np.zeros((0, 0))
        if not self._velocity_recordings:
            return np.zeros((len(self._velocity_probes), 0))
        return np.concatenate(self._velocity_recordings, axis=1)

    @property
    def previous(self) -> np.ndarray:
        """The level before it. Equal to :attr:`p` at rest, which is the state ``v = 0``."""
        return self._buffers[1 - self._newest]

    def set_threads(self, count: int) -> int:
        """Use ``count`` threads (``0``: the default for this machine); returns the count set."""
        return int(self._library.acfdtd_set_threads(int(count)))

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
        real_pointer = ctypes.POINTER(real)
        index_pointer = ctypes.POINTER(ctypes.c_int64)

        # Built once per chunk, never per step: the C loop does not call back into Python.
        # The signals go in whole, one after another; C indexes them by the global step.
        source_index = np.array([flat for flat, _ in self._sources], dtype=np.int64)
        source_length = np.array([signal.size for _, signal in self._sources], dtype=np.int64)
        source_offset = np.zeros(len(self._sources), dtype=np.int64)
        if self._sources:
            source_offset[1:] = np.cumsum(source_length)[:-1]
        # One spare element, so the pool has an address even when every signal is empty.
        source_pool = np.ascontiguousarray(
            np.concatenate([signal for _, signal in self._sources] + [np.zeros(1, self.dtype)]),
            dtype=self.dtype,
        )
        receiver_index = np.array(self._receivers, dtype=np.int64)
        recording = np.zeros((len(self._receivers), n_steps), dtype=self.dtype)
        velocity_cell = np.array([cell for cell, _ in self._velocity_probes], dtype=np.int64)
        velocity_lower = np.array([lower for _, lower in self._velocity_probes], dtype=np.int64)
        velocity_difference = np.zeros((len(self._velocity_probes), n_steps), dtype=self.dtype)

        state = self._state_type(
            nx=self.grid.nx,
            ny=self.grid.ny,
            nz=self.grid.nz,
            n_steps=n_steps,
            lambda_squared=self._lambda_squared,
            current=self.p.ctypes.data_as(ctypes.POINTER(real)),
            previous=self.previous.ctypes.data_as(ctypes.POINTER(real)),
            n_wall=self._wall_index.size,
            wall_index=self._wall_index.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            wall_a=self._wall_a.ctypes.data_as(ctypes.POINTER(real)),
            wall_scratch=self._wall_scratch.ctypes.data_as(ctypes.POINTER(real)),
            step_offset=self.step_index,
            src_coef=self._src_coef,
            n_sources=len(self._sources),
            source_index=source_index.ctypes.data_as(index_pointer),
            source_offset=source_offset.ctypes.data_as(index_pointer),
            source_length=source_length.ctypes.data_as(index_pointer),
            source_pool=source_pool.ctypes.data_as(real_pointer),
            n_receivers=len(self._receivers),
            receiver_index=receiver_index.ctypes.data_as(index_pointer),
            recording=recording.ctypes.data_as(real_pointer),
            n_velocity=len(self._velocity_probes),
            velocity_cell=velocity_cell.ctypes.data_as(index_pointer),
            velocity_lower=velocity_lower.ctypes.data_as(index_pointer),
            velocity_difference=velocity_difference.ctypes.data_as(real_pointer),
            x_open=self._open[0].ctypes.data_as(real_pointer),
            y_open=self._open[1].ctypes.data_as(real_pointer),
            z_open=self._open[2].ctypes.data_as(real_pointer),
            p_keep=self._p_keep.ctypes.data_as(real_pointer),
            has_rigid=int(self._has_rigid),
            n_cpml_regions=self._cpml_region.size // _CPML_COLUMNS_3D,
            cpml_region=self._cpml_region.ctypes.data_as(index_pointer),
            cpml_velocity_units=self._cpml_units[0],
            cpml_pressure_units=self._cpml_units[1],
            inverse_dx=self._cpml_constants[0],
            cpml_cv=self._cpml_constants[1],
            cpml_cp=self._cpml_constants[2],
        )
        # The shell arrays, set by name; left NULL without a shell (n_cpml_regions is then 0). The
        # field pools are written in place, so the shell state carries into the next run.
        for name, array in zip(_CPML_PROFILE_NAMES_3D, self._cpml_profiles, strict=False):
            setattr(state, name, array.ctypes.data_as(real_pointer))
        for name, array in zip(_CPML_FIELD_NAMES_3D, self._cpml_fields, strict=False):
            setattr(state, name, array.ctypes.data_as(real_pointer))
        self._library.acfdtd_scalar_run(ctypes.byref(state))
        self.step_index += n_steps
        self._newest ^= n_steps & 1
        if self._receivers:
            self._recordings.append(recording)
        if self._velocity_probes:
            self._integrate_velocity(velocity_difference)

    def _integrate_velocity(self, difference: np.ndarray) -> None:
        """The reference's running sum ``v <- v - (dt / (rho dx)) d``, a chunk at a time.

        Each sample is ``v`` before the step's increment, so the row is the cumulative sum of
        ``[v, -c d_0, -c d_1, ...]`` without its last element, which is the new running value.
        ``cumsum`` adds strictly left to right in fp64, and ``v - x`` equals ``v + (-x)`` exactly,
        so this is the reference's sequence of roundings. A probe on a lower wall has ``d = 0``
        and stays at zero.
        """
        n_steps = difference.shape[1]
        terms = np.empty((difference.shape[0], n_steps + 1))
        terms[:, 0] = self._velocity_running
        terms[:, 1:] = -(self._vel_coef * difference.astype(np.float64))
        sums = np.cumsum(terms, axis=1)
        self._velocity_recordings.append(sums[:, :n_steps])
        self._velocity_running = sums[:, n_steps].copy()


class CScalarFDTD2D:
    """The pressure-only 2D scheme with its time loop compiled (``ac_fdtd.scalar2d`` in C).

    Two cell arrays instead of three, one sweep instead of two: the fast, low-memory path for a
    closed 2D room. It is the same scheme as :class:`~ac_fdtd.scheme2d.AcousticFDTD2D`, not an
    approximation — bit-identical to :class:`~ac_fdtd.scalar2d.ScalarFDTD2D` — with rigid and
    locally-reacting walls, rigid obstacles, absorbing obstacles (``obstacle_admittance``: the
    normalised admittance of each solid cell, folded into the wall table so it combines with the
    walls, the rigid cells and the CPML shell), soft volume sources, prescribed pressure and
    velocity sources, pressure receivers, and the CFS-CPML open boundary (``absorbing_boundary``,
    the shell of :mod:`ac_fdtd.cpml2d`). The matched layer, air absorption and velocity receivers
    are not compiled here; the NumPy reference carries those.

    With an absorbing boundary the interior still holds two arrays: the shell's velocity and memory
    fields are stored per lined wall, as slabs of the shell's own depth, never as grid-sized fields.

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
        absorbing_boundary: AbsorbingBoundary2D | None = None,
        obstacle_admittance: np.ndarray | None = None,
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
        self.absorbing_boundary = absorbing_boundary

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
        # The two face constants, computed as ac_fdtd.scalar2d.ScalarFDTD2D computes them so the
        # kernel multiplies the same rounded numbers.
        self._pres_coef = medium.density * medium.sound_speed**2 * self.dt / grid.dx
        self._vel_coef = self.dt / (medium.density * grid.dx)

        self._buffers = [
            np.zeros(grid.shape, dtype=self.dtype),
            np.zeros(grid.shape, dtype=self.dtype),
        ]
        self._newest = 0

        indices, coefficient = wall_update_coefficients_2d(grid, medium, self.dt, self.walls)
        self._wall_index = np.ascontiguousarray(indices, dtype=np.int64)
        self._wall_a = coefficient.astype(self.dtype)

        self._has_rigid = rigid_cells is not None
        self._rigid: np.ndarray | None = None
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
            self._rigid = rigid
            self._x_open = np.ascontiguousarray(
                (~(rigid[:-1, :] | rigid[1:, :])).astype(self.dtype).reshape(-1)
            )
            self._y_open = np.ascontiguousarray(
                (~(rigid[:, :-1] | rigid[:, 1:])).astype(self.dtype).reshape(-1)
            )
            self._p_keep = np.ascontiguousarray((~rigid).astype(self.dtype).reshape(-1))

        if obstacle_admittance is not None:
            if rigid_cells is None:
                raise ValueError(
                    "obstacle_admittance needs rigid_cells: it is read only at solid cells"
                )
            self._merge_obstacle_admittance(obstacle_admittance)

        self._shell: np.ndarray | None = None
        self._cpml_region = np.zeros(0, dtype=np.int64)
        self._cpml_units = (0, 0)
        self._cpml_constants = (0.0, 0.0, 0.0)
        self._cpml_profiles: list[np.ndarray] = []
        self._cpml_fields: list[np.ndarray] = []
        if absorbing_boundary is not None:
            self._setup_cpml(absorbing_boundary)
        self._wall_scratch = np.zeros(self._wall_index.size, dtype=self.dtype)

        self._sources: list[tuple[tuple[int, int], np.ndarray]] = []
        self._receivers: list[tuple[int, int]] = []
        self._recordings: list[np.ndarray] = []

        # Prescribed sources, in insertion order: (kind, target, start, samples). kind 0 drives a
        # face (target = face index), kind 1 holds a pressure node (target = flat cell index).
        self._windows: list[tuple[int, int, int, np.ndarray]] = []
        # Driven faces, one entry per (axis, cell), in creation order. The stored velocity values
        # persist across run calls; the rest is fixed geometry, fixed when the face is created.
        self._face_index: dict[tuple[int, tuple[int, int]], int] = {}
        self._face_cells: list[int] = []
        self._face_lowers: list[int] = []
        self._face_a_cells: list[float] = []
        self._face_a_lowers: list[float] = []
        self._face_closed: list[int] = []
        self._face_value = np.zeros(0, dtype=self.dtype)

    def _merge_obstacle_admittance(self, obstacle_admittance: np.ndarray) -> None:
        """Fold the absorbing obstacle faces into the wall table, as ``ScalarFDTD2D`` does.

        Each fluid cell next to solid cells gets ``a = (c dt / 2 dx) * sum(xi)`` in float64, added
        to its outer-wall coefficient if it has one; the table is cast to ``dtype`` only at the
        end. Runs before ``_setup_cpml`` so the shell filter sees the merged table.
        """
        grid = self.grid
        rigid = self._rigid
        assert rigid is not None
        xi = np.asarray(obstacle_admittance, dtype=np.float64)
        if xi.shape != grid.shape:
            raise ValueError(
                f"obstacle_admittance shape {xi.shape} does not match the grid {grid.shape}"
            )
        solid_xi = xi[rigid]
        if not np.all(np.isfinite(solid_xi)) or np.any(solid_xi < 0.0):
            raise ValueError("obstacle_admittance must be finite and >= 0 at every solid cell")

        # Fluid-cell values are masked out first and solid-solid faces land on solid cells, which
        # are zeroed, so neither ever counts.
        weight = np.where(rigid, xi, 0.0)
        total = np.zeros(grid.shape)
        total[:-1, :] += weight[1:, :]
        total[1:, :] += weight[:-1, :]
        total[:, :-1] += weight[:, 1:]
        total[:, 1:] += weight[:, :-1]
        total[rigid] = 0.0
        flat = np.flatnonzero(total.reshape(-1) > 0.0)
        if not flat.size:
            return  # nothing absorbs: leave the wall arrays untouched

        scale = self.medium.sound_speed * self.dt / (2.0 * grid.dx)
        dense = np.zeros(total.size)
        indices, wall_a = wall_update_coefficients_2d(grid, self.medium, self.dt, self.walls)
        dense[indices] = wall_a
        dense[flat] += scale * total.reshape(-1)[flat]
        merged = np.union1d(indices, flat)
        self._wall_index = np.ascontiguousarray(merged, dtype=np.int64)
        self._wall_a = dense[merged].astype(self.dtype)

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

    def add_prescribed_source(
        self,
        point: tuple[float, float],
        signal: np.ndarray,
        start: int = 0,
        kind: str = "pressure",
        axis: int | None = None,
    ) -> None:
        """The contract of :meth:`ac_fdtd.scalar2d.ScalarFDTD2D.add_prescribed_source`, compiled.

        A pressure source holds the cell at ``signal``; a velocity source drives the lower face of
        the cell containing ``point`` along ``axis``. The stored face values and the windows are
        carried across :meth:`run` calls, so a run can be split anywhere.
        """
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}")
        samples = np.asarray(signal, dtype=self.dtype)
        cell = self.grid.cell_index(point)
        if kind == "pressure":
            if axis is not None:
                raise ValueError("axis is meaningless for a pressure source; leave it None")
            self._windows.append((1, cell[0] * self.grid.ny + cell[1], start, samples))
        elif kind == "velocity":
            if axis not in (0, 1):
                raise ValueError(f"velocity source needs axis in (0, 1), got {axis!r}")
            if self._shell is not None:
                below = list(cell)
                below[axis] -= 1
                if self._in_shell(cell) or (cell[axis] > 0 and self._in_shell(tuple(below))):
                    raise ValueError(
                        "a velocity source on a face bordering the absorbing shell is not "
                        "supported (no reference has a CPML plus a driven face)"
                    )
            self._windows.append((0, self._face(axis, cell), start, samples))
        else:
            raise ValueError(f"kind must be 'pressure' or 'velocity', got {kind!r}")

    def _face(self, axis: int, cell: tuple[int, int]) -> int:
        """The index of the driven face on the lower side of ``cell``, created at rest on first use.

        The geometry is decided here, once: the flat index of the cell and of the one below it, the
        wall coefficients of both (zero off the wall ring, read from the same table the walls use),
        and whether a rigid obstacle closes the face. Mirrors ``ScalarFDTD2D._velocity_face``.
        """
        key = (axis, cell)
        existing = self._face_index.get(key)
        if existing is not None:
            return existing
        if self.step_index > 0:
            # As in ScalarFDTD2D: a new face starts at v = 0, which is only true from rest.
            raise ValueError("a velocity source on a new face must be added before the first step")

        ny = self.grid.ny
        flat = cell[0] * ny + cell[1]
        if cell[axis] > 0:
            below = list(cell)
            below[axis] -= 1
            lower = below[0] * ny + below[1]
            a_lower = self._wall_coefficient(lower)
            closed = self._rigid is not None and bool(
                self._rigid[cell] or self._rigid[below[0], below[1]]
            )
        else:
            lower = -1
            a_lower = 0.0
            closed = False

        index = len(self._face_cells)
        self._face_index[key] = index
        self._face_cells.append(flat)
        self._face_lowers.append(lower)
        self._face_a_cells.append(float(self._wall_coefficient(flat)))
        self._face_a_lowers.append(float(a_lower))
        self._face_closed.append(int(closed))
        self._face_value = np.append(self._face_value, self.dtype.type(0.0))
        return index

    def _setup_cpml(self, boundary: AbsorbingBoundary2D) -> None:
        """The shell of ``ScalarFDTD2D._setup_cpml``, cut into one slab per lined wall.

        The reference keeps the shell state on full-grid arrays and masks its updates; here each
        lined wall gets a rectangle of the shell (the x slabs run the full height and take the
        corners, the y slabs fill between them), so the rectangles are disjoint and cover exactly
        the shell. Each stores its cells and every face bordering them; the C file explains why a
        face shared by two slabs is stored twice. Profiles and constants are cast as the reference
        casts them, so the kernel multiplies the same rounded numbers.
        """
        grid = self.grid
        nx, ny = grid.shape
        thickness = boundary.thickness
        if 2 * thickness >= min(nx, ny):
            raise ValueError(
                f"absorbing layer of {thickness} cells does not fit a {grid.shape} grid"
            )

        profiles = cpml_axis_profiles(grid, self.medium, self.dt, boundary)
        self._cpml_profiles = [
            np.ascontiguousarray(c, dtype=self.dtype)
            for key in (("p", 0), ("p", 1), ("v", 0), ("v", 1))
            for c in profiles[key]
        ]
        cast = self.dtype.type
        self._cpml_constants = (
            cast(1.0 / grid.dx),
            cast(self.dt / self.medium.density),
            cast(self.medium.density * self.medium.sound_speed**2 * self.dt),
        )

        low_x, high_x = boundary.has_axis(0)
        low_y, high_y = boundary.has_axis(1)
        x_start = thickness if low_x else 0
        x_end = nx - thickness if high_x else nx
        rectangles = []
        if low_x:
            rectangles.append((0, thickness, 0, ny))
        if high_x:
            rectangles.append((nx - thickness, nx, 0, ny))
        if low_y:
            rectangles.append((x_start, x_end, 0, thickness))
        if high_y:
            rectangles.append((x_start, x_end, ny - thickness, ny))

        # One row per region: bounds, pool offsets (cells, x faces, y faces), first units.
        rows = []
        cells = x_faces = y_faces = velocity_units = pressure_units = 0
        shell = []
        for i0, i1, j0, j1 in rectangles:
            rows.append((i0, i1, j0, j1, cells, x_faces, y_faces, velocity_units, pressure_units))
            ni, nj = i1 - i0, j1 - j0
            cells += ni * nj
            x_faces += (ni + 1) * nj
            y_faces += ni * (nj + 1)
            velocity_units += ni + 1
            pressure_units += ni
            shell.append((np.arange(i0, i1)[:, None] * ny + np.arange(j0, j1)[None, :]).ravel())
        self._cpml_region = np.array(rows, dtype=np.int64).reshape(-1)
        self._cpml_units = (velocity_units, pressure_units)
        # vx, psi_vx, vy, psi_vy, psi_px, psi_py: the state that persists across run calls.
        self._cpml_fields = [
            np.zeros(size, dtype=self.dtype)
            for size in (x_faces, x_faces, y_faces, y_faces, cells, cells)
        ]

        # The shell's flat cell indices, sorted: a grid-sized mask is what this class avoids.
        # ``faces=()`` is a valid (empty) shell, as in the reference.
        self._shell = np.sort(np.concatenate(shell)) if shell else np.zeros(0, dtype=np.int64)

        # As in the reference: an impedance wall correction only applies to interior wall cells,
        # and the driven-face wall coefficients are then read from the filtered table too.
        if self._wall_index.size:
            keep = ~np.isin(self._wall_index, self._shell)
            self._wall_index = np.ascontiguousarray(self._wall_index[keep])
            self._wall_a = self._wall_a[keep]

    def _in_shell(self, cell: tuple[int, int]) -> bool:
        flat = cell[0] * self.grid.ny + cell[1]
        index = int(np.searchsorted(self._shell, flat))
        return index < self._shell.size and int(self._shell[index]) == flat

    def _wall_coefficient(self, flat: int) -> float:
        hits = np.flatnonzero(self._wall_index == flat)
        return float(self._wall_a[hits[0]]) if hits.size else 0.0

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

        # Prescribed windows, flattened: their samples are concatenated into one pool, and each
        # window records where its own run starts in that pool.
        n_windows = len(self._windows)
        window_kind = np.array([w[0] for w in self._windows], dtype=np.int32)
        window_target = np.array([w[1] for w in self._windows], dtype=np.int64)
        window_start = np.array([w[2] for w in self._windows], dtype=np.int64)
        window_length = np.array([w[3].size for w in self._windows], dtype=np.int64)
        window_offset = np.zeros(n_windows, dtype=np.int64)
        if n_windows:
            window_offset[1:] = np.cumsum(window_length)[:-1]
        window_pool = np.ascontiguousarray(
            np.concatenate([w[3] for w in self._windows] or [np.zeros(0, dtype=self.dtype)]),
            dtype=self.dtype,
        )

        n_faces = len(self._face_cells)
        face_cell = np.array(self._face_cells, dtype=np.int64)
        face_lower = np.array(self._face_lowers, dtype=np.int64)
        face_a_cell = np.array(self._face_a_cells, dtype=self.dtype)
        face_a_lower = np.array(self._face_a_lowers, dtype=self.dtype)
        face_closed = np.array(self._face_closed, dtype=np.int32)
        face_value = np.ascontiguousarray(self._face_value, dtype=self.dtype)
        face_drive = np.zeros(n_faces, dtype=self.dtype)
        face_driven = np.zeros(n_faces, dtype=np.int32)

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
            vel_coef=self.dtype.type(self._vel_coef),
            pres_coef=self.dtype.type(self._pres_coef),
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
            n_faces=n_faces,
            face_cell=face_cell.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            face_lower=face_lower.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            face_a_cell=face_a_cell.ctypes.data_as(ctypes.POINTER(real)),
            face_a_lower=face_a_lower.ctypes.data_as(ctypes.POINTER(real)),
            face_closed=face_closed.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            face_value=face_value.ctypes.data_as(ctypes.POINTER(real)),
            face_drive=face_drive.ctypes.data_as(ctypes.POINTER(real)),
            face_driven=face_driven.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            n_windows=n_windows,
            window_kind=window_kind.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            window_target=window_target.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            window_start=window_start.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            window_length=window_length.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            window_offset=window_offset.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            window_pool=window_pool.ctypes.data_as(ctypes.POINTER(real)),
            n_cpml_regions=self._cpml_region.size // _CPML_COLUMNS,
            cpml_region=self._cpml_region.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            cpml_velocity_units=self._cpml_units[0],
            cpml_pressure_units=self._cpml_units[1],
            inverse_dx=self._cpml_constants[0],
            cpml_cv=self._cpml_constants[1],
            cpml_cp=self._cpml_constants[2],
        )
        # The shell arrays, set by name; left NULL without a shell (n_cpml_regions is then 0). The
        # field pools are written in place, so the shell state carries into the next run.
        for name, array in zip(_CPML_PROFILE_NAMES, self._cpml_profiles, strict=False):
            setattr(state, name, array.ctypes.data_as(ctypes.POINTER(real)))
        for name, array in zip(_CPML_FIELD_NAMES, self._cpml_fields, strict=False):
            setattr(state, name, array.ctypes.data_as(ctypes.POINTER(real)))
        self._library.acfdtd2d_scalar_run(ctypes.byref(state), ctypes.c_int64(self.step_index))
        self._face_value = face_value
        self.step_index += n_steps
        self._newest ^= n_steps & 1
        if self._receivers:
            self._recordings.append(recording)

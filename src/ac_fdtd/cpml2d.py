"""A CFS-CPML (complex-frequency-shifted convolutional PML) open boundary for the 2D scheme.

This is the pressure-only scheme's answer to free field, and it replaces the matched sponge layer of
:mod:`ac_fdtd.boundaries2d` with something strictly better: it absorbs at *all* angles, not just
near the normal, and corners need no special case. It is also the right tool for a *partial*
absorber — a locally-reacting wall saturates near a 13 dB return at oblique incidence however high
its absorption coefficient, whereas a shallow CPML skin tuned to a target reflection keeps improving
(see ``docs/kernel_study``).

The mechanism (Roden & Gedney). A spatial derivative ``d/dx`` inside the layer is replaced by

    (1/kappa_x) d/dx + psi,      psi^{new} = b_x psi^{old} + a_x (d/dx),
    b_x = exp(-(sigma_x/kappa_x + alpha_x) dt),
    a_x = sigma_x / (kappa_x (sigma_x + kappa_x alpha_x)) * (b_x - 1),

with ``sigma`` graded as ``depth**order`` from zero at the inner face to ``sigma_max`` at the outer
wall, ``kappa`` the real coordinate stretch (helps at modest thickness) and ``alpha`` the
complex-frequency shift (its job is grazing and evanescent incidence), graded the other way, from a
maximum at the inner face to zero outside. Those memory variables and the velocity they correct live
only in the boundary shell, so the interior stays pressure-only;
:class:`ac_fdtd.scalar2d.ScalarFDTD2D` runs the first-order update with these corrections in the
shell and the scalar update everywhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .boundaries2d import FACES
from .grid2d import Grid2D
from .medium import Medium

__all__ = ["AbsorbingBoundary2D", "cpml_axis_profiles"]


@dataclass(frozen=True)
class AbsorbingBoundary2D:
    """A CFS-CPML absorbing region lining a chosen subset of the four walls.

    Args:
        thickness: Depth of the layer in cells. Deeper is quieter: measured roughly −36 dB at 16
            cells, −44 at 24, −53 at 32 (kappa=5, alpha from f0). A few cells make a tunable partial
            absorber instead; see :meth:`partial_skin`.
        target_reflection: Round-trip amplitude reflection the grading is designed for; sets
            ``sigma_max``. For a full open boundary ``1e-6`` is typical; for a partial absorber of
            absorption ``alpha`` use ``1 - alpha`` (and a shallow ``thickness``).
        kappa_max: Peak real coordinate stretch at the outer face (``>= 1``). Helps a thin layer.
        alpha_max: Peak complex-frequency shift, in rad/s. The CFS term; a few times the band of
            interest (``2*pi*f`` for a characteristic ``f``) tames grazing incidence. Zero is a
            plain
            convolutional PML.
        order: Polynomial order of the ``sigma`` / ``kappa`` grading. Three is the usual choice.
        faces: Which walls to line, as ``(axis, side)`` pairs. All four by default.
    """

    thickness: int = 24
    target_reflection: float = 1e-6
    kappa_max: float = 5.0
    alpha_max: float = 0.0
    order: float = 3.0
    faces: tuple[tuple[int, int], ...] = FACES

    def __post_init__(self) -> None:
        if self.thickness < 1:
            raise ValueError(f"thickness must be at least one cell, got {self.thickness}")
        if not 0.0 < self.target_reflection < 1.0:
            raise ValueError(f"target_reflection must be in (0, 1), got {self.target_reflection}")
        if self.kappa_max < 1.0:
            raise ValueError(f"kappa_max must be >= 1, got {self.kappa_max}")
        if self.alpha_max < 0.0:
            raise ValueError(f"alpha_max must be non-negative, got {self.alpha_max}")

    @classmethod
    def open_field(
        cls,
        thickness: int = 24,
        characteristic_frequency: float = 1000.0,
        faces: tuple[tuple[int, int], ...] = FACES,
    ) -> AbsorbingBoundary2D:
        """A deep layer tuned for a near-anechoic open boundary at all angles."""
        return cls(
            thickness=thickness,
            target_reflection=1e-6,
            kappa_max=5.0,
            alpha_max=2.0 * math.pi * characteristic_frequency,
            faces=faces,
        )

    @classmethod
    def partial_skin(
        cls,
        absorption: float,
        thickness: int = 8,
        characteristic_frequency: float = 1000.0,
        faces: tuple[tuple[int, int], ...] = FACES,
    ) -> AbsorbingBoundary2D:
        """A shallow skin tuned to absorb a target fraction, angle-robustly.

        Unlike a locally-reacting wall at the same absorption, this does not collapse to near-total
        reflection at grazing incidence. ``absorption`` is the design figure; the realised value is
        measured, not promised, and depends on the angle distribution of the field.
        """
        if not 0.0 < absorption < 1.0:
            raise ValueError(f"absorption must be in (0, 1), got {absorption}")
        return cls(
            thickness=thickness,
            target_reflection=1.0 - absorption,
            kappa_max=1.0,
            alpha_max=2.0 * math.pi * characteristic_frequency,
            order=2.0,
            faces=faces,
        )

    def has_axis(self, axis: int) -> tuple[bool, bool]:
        """Whether this boundary lines the low and high side of ``axis``."""
        return ((axis, 0) in self.faces, (axis, 1) in self.faces)


def cpml_axis_profiles(
    grid: Grid2D, medium: Medium, dt: float, boundary: AbsorbingBoundary2D
) -> dict:
    """All per-axis CPML profiles the scheme needs, as a dict of ``(kappa, b, a)`` triples.

    Keys ``("p", axis)`` sample the pressure cell centres along ``axis`` (length ``n``); keys
    ``("v", axis)`` sample the velocity faces (length ``n + 1``). ``sigma_max`` comes from the
    reflection the standard way, ``sigma_max = -(order+1) c ln(R0) / (2 L)`` with ``L`` the layer
    depth in metres.
    """
    sound_speed = medium.sound_speed
    thickness = boundary.thickness
    layer_depth = thickness * grid.dx
    sigma_max = (
        -(boundary.order + 1.0)
        * sound_speed
        * math.log(boundary.target_reflection)
        / (2.0 * layer_depth)
    )

    def triple(n: int, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        low, high = boundary.has_axis(axis)
        position = np.arange(n)
        depth = np.zeros(n)
        if low:
            depth = np.maximum(depth, (thickness - position) / thickness)
        if high:
            depth = np.maximum(depth, (position - (n - 1 - thickness)) / thickness)
        depth = np.clip(depth, 0.0, 1.0)
        sigma = sigma_max * depth**boundary.order
        kappa = 1.0 + (boundary.kappa_max - 1.0) * depth**boundary.order
        alpha = boundary.alpha_max * (1.0 - depth)
        b = np.exp(-(sigma / kappa + alpha) * dt)
        denom = kappa * (sigma + kappa * alpha)
        # a is zero wherever there is no layer (denom -> 0); divide only where it is safe.
        a = np.zeros_like(denom)
        active = denom > 1e-30
        a[active] = sigma[active] / denom[active] * (b[active] - 1.0)
        return kappa, b, a

    profiles: dict = {}
    for axis in range(2):
        profiles[("p", axis)] = triple(grid.shape[axis], axis)
        profiles[("v", axis)] = triple(grid.shape[axis] + 1, axis)
    return profiles

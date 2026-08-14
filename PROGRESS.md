# Progress log

Newest last. Each entry records what was measured, under what conditions, and what it means.

---

## M0 — Scaffold

Repository laid out to match the companion `plate-fdtd` project: `src/` layout with uv,
ruff + pytest under GitHub Actions, `PLAN.md` / `PROGRESS.md`, figures as SVG in
`docs/figures/`.

`commit.sh` is deliberately untracked. It bypasses the unreadable global git config, sets the
identity per-repo, and stops at the local commit — pushing happens outside the sandbox. It also
refuses to commit if `tmp/` ever becomes tracked: that directory holds the 2D prototype this
work grew out of, the repository is public, and a guard is worth more than a comment.

---

## M1 — The lossless core

Staggered pressure–velocity leapfrog in 3D, rigid walls, NumPy, double precision.

### Design decisions

**The stored state is always `(p^n, v^{n-1/2})`.** Every diagnostic is written for that
pairing. This is the decision the whole module rests on: mixing the two fields without naming
an instant costs O(dt) in anything derived from both, and O(dt) errors do not announce
themselves.

**Rigid walls are structural, not a boundary routine.** The velocity planes lying on the walls
are never written to, so they hold the zero they were allocated with. Fastest, least fragile,
and it is what makes the energy identity exact rather than approximate.

**The time step comes from the Courant condition.** The 2D prototype used
`dt = 1e-2 / (1.2 * f_max)`, which is stable only because its `dx = 1e-2` was baked into the
numerator; changing the spacing there would have produced an unstable run with no warning.
Here `dt = C * dx / (c sqrt(3))` with `C = 1` by default — the stability limit, which is also
the point of minimum numerical dispersion.

**The simulated room is reported, not the requested one.** One uniform spacing cannot divide
three arbitrary lengths, so each axis rounds to whole cells. The half-cell difference shifts
the modal frequencies by about as much as the discretisation error being measured, so
`Grid.lengths` gives what is actually simulated and `Grid.length_error` gives the gap.

### Results

![M1 validation](docs/figures/m1_box_validation.svg)

**Energy conservation.** Rigid box 0.8 x 0.65 x 0.55 m, `dx = 50 mm`:

| initial condition | Courant | steps | relative energy drift |
| --- | --- | --- | --- |
| three exact modes | 1.0 | 4000 | `4.4e-16` |
| broadband noise | 1.0 | 2000 | `6.2e-16` |
| broadband noise | 0.9 | 2000 | `0.0` (exactly) |

This is round-off, i.e. the identity holds. The same run scored with the naive `p^2 + v^2`
energy — the one that ignores the half step — oscillates at `7.7e-2`. That form is not a
slightly worse diagnostic; it is not conserved at all, its error scaling as `omega dt / 2`,
which is a fraction of a percent for a well-resolved low mode and order one for content near
the grid limit. It cannot be used to detect bugs, which is what the correct one is for.

**A single mode stays a single mode.** The cosine mode shapes are exact eigenvectors of the
*discrete* rigid-wall operator, so with a consistently staggered initial velocity
(`v^{-1/2} = (dt / 2 rho) grad p^0`, not zero) the field must stay exactly
`cos(omega n dt) * Phi` forever. Over 3000 steps, mode (2,1,1), `dx = 40 mm`:

- max absolute error `1.3e-13` at unit amplitude.

This is the strongest single test here: a sign error, a slice off-by-one, a mistimed boundary
or a wrong half-step initialisation each detune the oscillation or leak energy into other
modes, and both are visible far above that tolerance.

**Convergence to the analytical frequencies.** Room 1.6 x 1.2 x 1.0 m, mode (2,1,1):

| dx [m] | points per wavelength | relative error |
| --- | --- | --- |
| 0.100 | 11 | `4.93e-4` |
| 0.050 | 22 | `1.23e-4` |
| 0.025 | 44 | `3.06e-5` |
| 0.0125 | 89 | `7.65e-6` |
| 0.00625 | 177 | `1.91e-6` |

Ratio 4.0 per halving — second order, as it should be. Note this is the *scheme's* frequency
from its dispersion relation against the continuous one; it is the discretisation error alone,
with no simulation noise in it.

**Broadband run.** Room 1.6 x 1.2 x 1.0 m, `dx = 25 mm` (64 x 48 x 40 cells), `dt = 42.05 us`,
0.6 s simulated, Gaussian pulse source, single pressure probe, both at asymmetric points:

- 25 analytical modes below 400 Hz, **24 matched within one FFT bin** (1.67 Hz)
- median offset between an analytical mode and the nearest interpolated peak: **0.080 Hz**

The one miss is not a discrepancy: that mode has a node at the source or the probe, so it is
not excited and its "nearest peak" belongs to a neighbour.

### Baseline throughput (NumPy, fp64)

Single-threaded NumPy on the M2 Ultra, for reference before the fast backends exist:

| grid | cells | cell-updates/s |
| --- | --- | --- |
| 32³ | 0.03 M | 173 M |
| 64³ | 0.26 M | 234 M |
| 96³ | 0.88 M | 214 M |

What that means in practice: one second of impulse response in a 5 x 4 x 3 m room at 1 kHz
(`dx = 34 mm`, 1.5 M cells, 17 000 steps) is about **2 minutes** at this rate, and the cost
scales as `f_max^4` — three spatial dimensions and the time step together. At 4 kHz the same
room is roughly 8 hours. That is the whole argument for M5–M7.

### Next

M2: boundaries. Rigid is the only wall condition so far, which also means the energy test
currently covers the easiest case there is — the passivity of the impedance and PML
formulations is the next thing that needs proving rather than assuming.

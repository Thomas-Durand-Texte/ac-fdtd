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

---

## M2 — Boundaries

Locally reacting absorbing walls, a graded absorbing layer for free field, and the free-field
Green's function as the check that ties propagation, source calibration and boundaries
together.

### Design decisions

**The wall condition is centred in time, and therefore implicit — but only within one cell.**
Imposing `v = Y p` directly means setting `v^{n+1/2}` from `p^n`, which is off by `dt/2`; the
sign of that error decides whether the wall quietly gains or loses energy, and it is the usual
reason FDTD boundaries go unstable. Centring it, `v^{n+1/2} = Y (p^{n+1} + p^n) / 2`, is
implicit in `p^{n+1}` — but the coupling is local, so it solves in closed form:

```
p^{n+1} = ((1 - a) p^n - rho c^2 dt div_interior) / (1 + a),     a = c dt xi / (2 dx)
```

Each absorbing face touching a cell adds its own `a`, so edges and corners need no special
case: the coefficients simply sum. This form can only remove energy, for any `xi >= 0`.

**The wall velocity planes stay at zero.** The flux through a wall enters through the
coefficient above and nowhere else. Storing a wall velocity as well would double-count it on
the next step — the kind of bug that halves an absorption coefficient and looks like a
modelling disagreement.

**The absorbing layer damps both equations by the same sigma.** That leaves the local impedance
at `rho c` everywhere, including where sigma varies. It is separable per axis, so it costs
three 1D profiles, no extra field, and is applied only to the slabs where it differs from one.

### Results

![M2 validation](docs/figures/m2_boundaries.svg)

**Absorbing walls, normal incidence.** Pulse down a one-cell duct, gated into incident and
reflected arrivals, mean over 60 Hz – 2 kHz:

| xi | measured \|R\| | theory `(1-xi)/(1+xi)` |
| --- | --- | --- |
| 0 (rigid) | 0.9998 | 1 |
| 0.25 | 0.6026 | 0.6 |
| 0.5 | 0.3384 | 0.3333 |
| 1 (matched) | 0.0477 | 0 |

The rigid row is the measurement's own accuracy — 0.02 % — so the two middle rows agree with
theory to within about half a percent of the incident amplitude.

The matched wall is the interesting one. It is exact at DC and degrades with frequency,
reaching `|R| = 0.09` at 2 kHz, which is 17 points per wavelength on this grid. Nothing is
wrong with the boundary: the *discrete* wave impedance is not exactly `rho c` once the grid
starts to disperse, so matching the continuous impedance stops being a match. That is a
property of first-order absorbing conditions in general, and it is the reason free-field runs
get a layer instead.

**The absorbing layer.** At normal incidence the layer is not approximately transparent but
exactly so: with the same sigma on both equations the two characteristics decouple, so a
right-going wave stays right-going whatever sigma does — grading included. Measured in the
duct, over the same band:

| layer depth | worst reflection in band |
| --- | --- |
| 8 cells | −86 dB |
| 16 cells | −108 dB |
| 32 cells | −111 dB |

Which is why the useful measurement is a 3D one. Running the same monopole problem twice — once
in a domain lined with the layer, once in a domain large enough that its own walls cannot answer
within the window — isolates exactly what the layer contributes:

| clear cells between receiver and layer | 16-cell layer | 32-cell layer |
| --- | --- | --- |
| 10 | −16.8 dB | −19.5 dB |
| 40 | −30.8 dB | −33.2 dB |

Doubling the layer buys 3 dB; moving the receiver 30 cells further from it buys 14. **The limit
is obliquity, not thickness** — a matched layer is matched at normal incidence only, and a
spherical wave hits it at every angle. If a job needs better than about −30 dB, the answer is a
true PML with coordinate stretching, not a thicker sponge. That is now a measured argument
rather than a guess, and −30 dB is the number it has to beat.

**Free field against the analytical Green's function.** Monopole at the centre, compared to
`p = rho Q'(t - r/c) / (4 pi r)` with `Q = q dx^3`:

| points per wavelength | r = 5 cells | r = 20 cells | r = 35 cells |
| --- | --- | --- | --- |
| 10.4 | 8.5 % | 30.9 % | 47.6 % |
| 28.6 | 2.1 % | 1.2 % | 2.4 % |
| 60.8 | 1.4 % | 0.21 % | 0.21 % |

At 61 points per wavelength the simulation and the closed-form solution agree to two parts in a
thousand, which also pins the source calibration: a mis-scaled volume source would show up as a
constant offset at every radius, and there is none to two decimal places. The residual at
5 cells is the near field of a source that is a cube rather than a point, and it does not
improve with resolution in the way the far field does.

The top row is the dispersion error growing with propagation distance, and it is the strongest
argument in this log for measuring the required resolution rather than assuming ten points per
wavelength: at 10 points, a pulse is 48 % wrong after 35 cells of travel.

### Two bugs this milestone found

**Cell indexing.** `1.2 / 0.01` is `119.99999999999999`, so a plain floor put a point on a cell
boundary one cell low — while `0.5 / 0.01` is exact and did not. Two grids of the same spacing
disagreed by one cell about where the same point was, which surfaced as a 7 % amplitude error
in the free-field comparison and looked like physics. `Grid.cell_index` now snaps within a
millionth of a cell.

**Comparing at a radius nothing was measured at.** A probe snaps to the cell containing it, so
a receiver asked for at 17.5 cells is measured at 17 — and comparing it against the analytical
solution at 17.5 costs a 7 % error that has nothing to do with the solver. Separations are now
read back from the indices rather than from the request.

Neither was caught by a test. Both were caught by comparing against a closed-form answer, which
is the argument for having one.

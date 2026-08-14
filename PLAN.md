# ac-fdtd — Plan

## Objective

Compute the **3D acoustic response of a room** with an explicit FDTD scheme: impulse responses
at receivers, from broadband (audible, several kHz) down to the modal region, with realistic
losses in the air and at the walls.

Secondary objective: find out **what this computation should actually run on**. 3D FDTD is
memory-bandwidth-bound, not compute-bound, which is the opposite of the regime measured in the
companion `plate-fdtd` repository — so the backend conclusion there does not transfer and has
to be re-measured here.

## Scope

**In scope**
- Lossless propagation on a staggered pressure–velocity grid (Yee arrangement), 2nd order in
  space and time; a 4th-order-in-space variant as an option.
- Air absorption: classical/rotational plus O2 and N2 relaxation, per ISO 9613-1, as a function
  of temperature, humidity and static pressure.
- Boundaries: rigid; locally reacting real admittance; frequency-dependent impedance via an IIR
  filter fitted to octave-band absorption data; PML and a first-order ABC for free field.
- Sources (soft volume source, band-limited pulses, sweeps) and receivers, with the half-step
  timing handled rather than left to the caller.
- Room acoustics metrics: Schroeder decay, T60, EDT, C50; wav/npz export.
- Backends: NumPy reference (fp64), PyTorch (CPU / MPS / CUDA), C with OpenMP.

**Out of scope**
- Non-rectangular geometry beyond staircased boundaries; no cut-cell or unstructured meshing.
- Moving sources or media, flow, temperature gradients.
- Scattering surfaces with non-local reaction; diffusers modelled explicitly rather than
  statistically.
- Auralisation itself (convolution, HRTF, binaural rendering) — this repo produces the IRs.

## Physics

First-order linear acoustics, staggered in time by half a step:

```
dv/dt = -(1/rho) grad p
dp/dt = -rho c^2 div v            (+ relaxation terms, see below)
```

Discretised with pressure at cell centres and normal velocity on faces. Stability requires the
Courant condition `c dt sqrt(3) / dx <= 1`, and the limit is also where numerical dispersion is
smallest — so the default is to run *at* it, not below.

**Why keep the p–v pair rather than a pressure-only scheme.** The second-order wave equation on
pressure alone carries two fields instead of four, which on a bandwidth-bound problem is close
to a factor two in wall time. That is a real advantage and it will be measured (M8). What the
first-order system buys is everything else in this plan: impedance boundaries, PML, and the
relaxation states of the air model all attach naturally to it.

**Air absorption.** Implemented as two auxiliary relaxation states (O2, N2) plus a classical
term, fitted to the ISO 9613-1 attenuation curve. Cost is +2 cell-sized fields and ~10 flops
per cell. To be clear about what this buys: at 20 °C and 50 % RH the attenuation is about
0.02 dB/m at 4 kHz, so over a 20 m path it is a fraction of a decibel. It is **not** a
stability mechanism — stability comes from the Courant condition and from energy-passive
boundary formulations. What it buys is a physically correct high-frequency decay slope, which
matters for T60 in large volumes, and a bound on accumulated high-frequency numerical energy.

## Validation ladder

0. **Energy conservation**, rigid box, lossless — must hold to round-off. *(M1, done: 4e-16.)*
1. **Analytical modes** of the rigid box, `f = (c/2) sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2)`,
   with the error shrinking as `dx^2`. *(M1, done.)*
2. **Free-field monopole** against the spherical Green's function: 1/r decay, arrival time,
   dispersion, and the reflection level of the absorbing boundary in dB.
3. **Air absorption** against the ISO 9613-1 curve, measured from spectral decay along a duct;
   target better than 5 % over 100 Hz – 8 kHz.
4. **T60** against Sabine and Eyring for uniform wall absorption — a statistical check, with
   the limits of statistical theory stated rather than glossed.
5. **Measured room IR**, if one is available from the `ir-estimation` work.

## Performance study (M7)

Sweep: backend {NumPy, Torch-CPU, Torch-MPS, C+OpenMP} x grid {64³ … 512³, 768³ as a stress
point} x max frequency {500 Hz, 1 kHz, 2 kHz, 4 kHz} x dtype {fp32, fp64}.

Reported as **cell-updates per second and achieved bandwidth against the machine's roofline**,
plus wall time for one second of impulse response — not raw wall clock, which hides the size
dependence. fp32 accuracy against fp64 is measured, not assumed: a one-second run at 4 kHz is
~10^5 steps of accumulated round-off, and MPS has no fp64 at all.

Memory ceiling: 64 GB unified, shared with the GPU working set. 512³ is ~2.1 GB of fp32 state
and comfortable; 768³ is ~7 GB; 1024³ is opt-in rather than part of the standard sweep.

Deliverable: a decision table — for this cell count and this bandwidth, use this backend.

A companion dispersion study measures phase-velocity error against propagation direction and
points per wavelength, and reports the resolution actually needed for a given error target.
That number decides the grid size, and an 8x reduction in cells from halving the required
points per wavelength is worth more than any backend choice.

## Milestones

| | | |
|---|---|---|
| M0 | Scaffold, CI, commit script | done |
| M1 | NumPy core, energy and modal validation | done |
| M2 | Boundaries: admittance, ABC, PML + Green's function test | |
| M3 | Air absorption + ISO 9613-1 validation | |
| M4 | Sources, receivers, IR export, room acoustics metrics | |
| M5 | PyTorch backend (CPU/MPS/CUDA) + parity tests | |
| M6 | C/OpenMP backend + parity tests | |
| M7 | Benchmark and dispersion studies, decision table | |
| M8 | 4th-order and pressure-only variants, examples and animations | |

## Conventions

Every result in `PROGRESS.md` comes with the numbers behind it and the conditions it was
measured under. A speed figure without the precision it was bought at, or an error without the
resolution it was measured at, is half a result.

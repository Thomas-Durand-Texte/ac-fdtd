# Status

The stable state file: where the project is, what is known to work, what is not implemented,
and what to do next. [`PROGRESS.md`](PROGRESS.md) is the append-only log of what was measured
and when; this file is the current picture and gets rewritten.

**Last updated:** after M8. All nine planned milestones are done, 100 tests pass, `ruff` clean.

---

## What works

| Area | State |
| --- | --- |
| Scheme | Staggered pressure–velocity, 2nd order in space and time, Courant 1 by default; a 2D peer `AcousticFDTD2D` (Courant 1/√2) with the same walls, layer, air and exact energy identity |
| Geometry | Rectangular rooms; a grid rounds each axis to whole cells and reports what it built |
| Walls | Rigid, locally reacting real admittance, impedance-matched; from an absorption coefficient or directly |
| Free field | Graded matched absorbing layer, any subset of the six faces |
| Air | ISO 9613-1: classical + O₂ + N₂ relaxation, from temperature / humidity / pressure |
| Sources | Band-limited impulse, Gaussian pulse, soft volume source, deconvolution; prescribed (hard) pressure/velocity sources with a time window (`add_prescribed_source`, reference backend) |
| Receivers | Pressure probes on all backends; velocity probes on the reference |
| Analysis | Schroeder decay, TR20/TR30, EDT, C50, octave bands, Sabine/Eyring, Schroeder frequency |
| Export | Polyphase resampling and 32-bit float wav |
| Backends | NumPy (fp64), PyTorch (CPU/MPS/CUDA, fp32/fp64), C (pthreads, fp32/fp64); 2D has NumPy + a C kernel (`CAcousticFDTD2D`), no Torch yet |
| Variant | Pressure-only C kernel, rigid walls only, ~5× the pressure–velocity form |

## What is deliberately not there

- **Non-rectangular geometry.** No cut cells, no unstructured meshes; staircasing only.
- **Frequency-dependent wall impedance.** Walls are a single real admittance. An IIR boundary
  fitted to octave-band absorption data is in `PLAN.md` and not implemented.
- **A true PML.** The absorbing layer is a matched sponge: exact at normal incidence, −17 to
  −31 dB in a real 3D field depending on receiver clearance. Beating that needs coordinate
  stretching.
- **Fourth order in space.** Analysed, not implemented — see below.
- **Moving sources or media, flow, temperature gradients.**
- **Auralisation.** This produces impulse responses; convolution and HRTF are someone else's job.

---

## What to do next, in value order

1. **Three-level boundary conditions for the pressure-only variant.** Measured at **5×** the
   pressure–velocity form, which is more than any other lever in the project — more than the
   GPU, more than the loop fusion. What it needs is the impedance wall, the absorbing layer and
   the relaxation states re-derived as three-level relations. The kernel and its equivalence
   test already exist; this is the work that makes it usable.

2. **Fourth order in space.** A further **6.8× fewer cell-updates** for the same 1 % phase
   error. The design is settled and written down: build the gradient with whatever stencils the
   boundary allows and *define* the divergence as its transpose, so the energy identity survives
   — a summation-by-parts construction, changing the pressure update in the three cells nearest
   each wall on each axis. Note what it does not buy: with a leapfrog time step the scheme stays
   second order overall; the constant improves by 2.8×, not the order.

3. **Frequency-dependent walls.** The single real admittance is the least realistic thing left
   in the physics. Octave-band absorption data fitted to an IIR boundary.

4. **A true PML**, if any application needs better than −30 dB of free field.

5. **Validation against a measured room impulse response.** Rung 5 of the ladder, still open.

---

## Working notes

**Environment.** `uv sync --extra dev` for everything except the GPU path, which is
`--extra torch`. The C backend compiles itself on first use and needs nothing but a compiler —
it uses pthreads rather than OpenMP because PyTorch ships its own libomp and a second OpenMP
runtime aborts the process.

**A gap to know about.** CI does not install PyTorch, so the cross-backend parity tests **skip**
on GitHub Actions and only run locally. Anything touching either backend wants
`uv sync --extra torch && uv run pytest` before pushing.

**Benchmarks.** Roughly 10 % run-to-run variance. Quote one run rather than mixing them, and
re-run `scripts/backend_benchmark.py` rather than editing a number in a table.

**Committing.** `./commit.sh "message"` — untracked, sets the identity per-repo, bypasses the
global git config, refuses to commit if `tmp/` ever becomes tracked, and stops at the local
commit. Pushing happens outside the sandbox.

**Figures.** Every script writes its own SVG into `docs/figures/`. They are committed, so a
figure and the text around it should be regenerated together.

**2D scheme.** A full-fidelity NumPy reference (`grid2d`/`boundaries2d`/`scheme2d`/`analytic2d`,
class `AcousticFDTD2D`) mirrors the 3D scheme: rigid + locally-reacting absorbing walls, matched
layer, air relaxation, soft and prescribed (hard) sources, pressure receivers, and the exact energy
identity — tested the same way (energy to round-off, a single mode staying sinusoidal, second-order
dispersion). It is the plane mode for the front end. A **compiled C kernel** (`CAcousticFDTD2D`, pthreads,
fp32/fp64, in the same `acfdtd.c`/`c_backend.py` as the 3D one) now matches the reference to
round-off — parity-tested across rigid / absorbing walls / matched layer / air — so 2D runs fast at
scale as well (`scripts/backend_benchmark_2d.py`: ~6 GCU/s fp32 on the M2 Ultra). Remaining 2D
follow-ups: a **WASM build** (for the in-browser demo) and a **Torch/MPS** path (large planes); the
prescribed-source hook on the C/Torch backends is still pending for both 2D and 3D. The reference
leads; the compiled kernels mirror it.

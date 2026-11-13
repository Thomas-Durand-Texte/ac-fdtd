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
| Geometry | Rectangular rooms; a grid rounds each axis to whole cells and reports what it built. 2D also takes a **rigid-cell mask** (`AcousticFDTD2D(rigid_cells=…)`): lossless interior obstacles of any staircased shape |
| Walls | Rigid, locally reacting real admittance, impedance-matched; from an absorption coefficient or directly |
| Free field | Graded matched absorbing layer, any subset of the six faces |
| Air | ISO 9613-1: classical + O₂ + N₂ relaxation, from temperature / humidity / pressure |
| Sources | Band-limited impulse, Gaussian pulse, soft volume source, deconvolution; prescribed (hard) pressure/velocity sources with a time window (`add_prescribed_source`, reference backend) |
| Receivers | Pressure probes on all backends; velocity probes on the reference |
| Analysis | Schroeder decay, TR20/TR30, EDT, C50, octave bands, Sabine/Eyring, Schroeder frequency |
| Export | Polyphase resampling and 32-bit float wav |
| Backends | NumPy (fp64), PyTorch (CPU/MPS/CUDA, fp32/fp64), C (pthreads, fp32/fp64); 2D has NumPy + a C kernel (`CAcousticFDTD2D`), no Torch yet |
| Variant | Pressure-only ("scalar", three-level) scheme: NumPy references **`ScalarFDTD2D`** / **`ScalarFDTD`** (3D) and a compiled **`CScalarFDTD2D`** — rigid + locally-reacting walls + rigid obstacles + volume sources + pressure & (reconstructed) velocity receivers, **bit-identical** to the p–v schemes, in **two cell arrays not 1+d** (measured −51% RAM in 3D, −33% in 2D). The 2D C kernel runs 1.8–2.5× the p–v C kernel; the 3D C kernel is the remaining step |
| Open boundary | Both scalar schemes take a **CFS-CPML** (`AbsorbingBoundary` / `AbsorbingBoundary2D`) on any subset of walls: a true PML, measured −45 dB+ absorbing a pulse into the corners (all angles), far past the sponge layer and the locally-reacting wall (which saturates near −13 dB obliquely). Velocity + the memory fields live only in the boundary shell, so a closed room is still two arrays. A shallow tuned skin (`partial_skin`) is the same mechanism as an angle-robust partial absorber |

## What is deliberately not there

- **Non-rectangular geometry.** No cut cells, no unstructured meshes; staircasing only.
- **Frequency-dependent wall impedance.** Walls are a single real admittance. An IIR boundary
  fitted to octave-band absorption data is in `PLAN.md` and not implemented.
- **A true PML.** The matched sponge (−17 to −31 dB in a real 3D field) is still all the p–v
  scheme and the 3D path have. The 2D *pressure-only* scheme now has a CFS-CPML
  (`AbsorbingBoundary2D`, −45 dB+ at all angles including corners); porting it to the p–v form and
  to 3D is open.
- **Fourth order in space.** Analysed, not implemented — see below.
- **Moving sources or media, flow, temperature gradients.**
- **Auralisation.** This produces impulse responses; convolution and HRTF are someone else's job.

---

## What to do next, in value order

1. **Pressure-only ("scalar") backend — in progress.** The three-level boundary derivations are
   done and validated (see `docs/kernel_study/`): rigid, locally-reacting impedance walls
   (`(1+a)p^{n+1} = 2p^n − (1−a)p^{n−1} + λ²L_N`), rigid obstacles, and the difference-injected
   soft source. The NumPy references **`ScalarFDTD2D`** and **`ScalarFDTD`** (3D) have
   landed, bit-identical to the p–v schemes in two arrays not 1+d (measured −51% RAM in 3D; the
   compiled speed-up was 2–3× in earlier benchmarks, the older "5×" optimistic), both carrying the
   **CFS-CPML open boundary** (−45 dB+ at all angles including corners, plus a tunable
   `partial_skin` that beats the locally-reacting wall). The **compiled 2D kernel `CScalarFDTD2D`**
   (three-level walls + obstacles + volume sources + prescribed pressure/velocity sources (T007) +
   receivers) has landed too — bit-identical to
   the reference, 1.8–2.5× the p–v C kernel. The 3D `CScalarFDTD` now has the locally-reacting
   walls (T001), volume sources + pressure/velocity receivers (T002) and rigid obstacles (T003,
   `rigid_cells`), all bit-identical to `ScalarFDTD`. The CPML shell is in both C kernels (T004 2D,
   T005 3D: `absorbing_boundary`, shell-only slabs, bit-identical). Still to do: adapter +
   auto-selection in the app. Air relaxation has landed in both NumPy references (T006: divergence rebuilt from the
   two pressure levels, +1 array/process, equal to the p–v scheme with air to ~1e-14; not yet with
   the CPML shell or prescribed pressure sources, nor in C); the matched sponge does not port — the
   CPML is its pressure-only replacement.

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
follow-ups: a **WASM build** (for the in-browser demo) and a **Torch/MPS** path (large planes). The
2D C kernel now has **prescribed (hard) sources** too — `CAcousticFDTD2D.add_prescribed_source`,
pressure and velocity, parity-tested against the reference — so a driven piston / pressure source
(a shaped loudspeaker) runs on C; the hook is still pending on the 3D C kernel and on Torch. The
reference leads; the compiled kernels mirror it.

**Rigid obstacles (2D).** `AcousticFDTD2D` takes `rigid_cells`, a boolean mask of solid cells. Each
velocity face touching a solid is held at zero (`v·n = 0`) before the pressure update reads it, so
an obstacle of any staircased shape is a lossless rigid reflector and the energy identity is
unchanged (`tests/test_obstacles2d.py`: energy conserved with an obstacle, a barrier seals the far
side, a shadow forms). The front end voxelises geometry (e.g. a Helmholtz resonator) into this mask.
The compiled 2D kernel takes the mask too (`CAcousticFDTD2D(rigid_cells=…)`): two extra 0/1-masking
sweeps per step, skipped when there is none, parity-tested against the reference including an
obstacle in an absorbing room (`tests/test_c_backend_2d.py`). Not yet: the mask on the 3D / Torch
kernels or a 3D mask. Absorbing masked surfaces exist in 2D scalar form (T008:
`ScalarFDTD2D(obstacle_admittance=…)`, ξ per solid cell, folded into the wall `a` terms; equal to an
outer wall of the same ξ to 1e-12; T009: `CScalarFDTD2D` takes the same keyword, merged in Python
into its wall table, bit-identical in fp64); `AcousticFDTD2D` does not take it yet.

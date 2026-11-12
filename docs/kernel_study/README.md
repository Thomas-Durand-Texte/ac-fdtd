# Kernel study — findings (living doc)

Goal (user's four points): drop velocity arrays (RAM, esp. small dx / 3D); re-derive from the
acoustics and find the best method/kernel (precision, speed, RAM) with tests of known results;
settle whether velocity must stay staggered (dt/2 shift) or can be collocated, and whether it can be
kept only for walls/probe/source; eventually velocity as arbitrary surfaces not bound to the grid.

Working dir: this scratchpad. Engine: `../ac-fdtd` (vendored into ac-fdtd-studio). Nothing in the
repos is edited yet — this is the research that precedes the code update.

## TL;DR so far
- The current scheme is the **staggered pressure–velocity leapfrog** (Yee), 2nd order, Courant 1
  (exact on the grid diagonal). It stores **1 + d** cell arrays: p + d velocity components
  (2D: 3 arrays, 3D: 4 arrays).
- The **pressure-only three-level** form (`p^{n+1}=2p^n-p^{n-1}+λ²Δp^n`, λ=cΔt/Δx) is the *exact*
  algebraic elimination of velocity — not an approximation. It stores **2** arrays regardless of
  dimension. RAM ratio 2/(1+d): **−33% in 2D, −50% in 3D.**
- The repo already ships a rigid-only pressure-only C kernel (`CScalarFDTD`) and flags "derive the
  boundaries as three-level relations" as the #1 lever (measured ~5× faster). This study does that
  derivation and proves it exact.

## Derivations (done, validated)

Eliminating v from the p-v leapfrog with a SOURCE gives, exactly:
    p^{n+1} = 2p^n − p^{n−1} + λ² L(p^n) + (S_n − S_{n−1})
- **Interior / rigid walls / rigid obstacles:** L is the standard Laplacian with Neumann
  (zero-gradient) at every rigid face — one-sided at domain walls, closed faces around obstacles,
  p=0 inside solids. VALIDATED to round-off (2e-13) vs `AcousticFDTD2D`.
- **Soft volume source:** inject the DIFFERENCE S_n−S_{n−1}, not S_n (the eliminated divergence
  already carries S_{n−1}). Injecting S_n is the classic pressure-only source bug. VALIDATED (2e-14).
- **Locally-reacting (impedance) wall, normalised admittance ξ:** a genuine three-level boundary
      (1+a) p^{n+1} = 2 p^n − (1−a) p^{n−1} + λ² L_N(p^n),   a = (cΔt/2Δx)·ξ   (summed at corners)
  Derived by carrying the previous step's own wall recurrence (the interior identity fails at the
  wall because p^n there was itself produced by the corrected update). VALIDATED: matches the p-v
  reference to **round-off (3.6e-13 over 400 steps)** for any IC consistent with the wall — i.e. any
  run driven from silence (case [B]: 5e-14). The damping sits on the (p^{n+1}−p^{n−1}) term, exactly
  a first-time-derivative radiation term, as physics demands.
  - Subtlety (not a bug): a *nonzero pressure at rest* against an absorbing wall is not a state the
    three-level wall recurrence is consistent with; it produces a one-time O(2a)≈6% transient that
    lives ONLY on the wall ring (interior stays 1.8e-15) and then propagates. Measure-zero start,
    irrelevant to driven runs. Confirmed: zeroing the 1-cell boundary ring of the IC restores
    round-off agreement.

## Speed + RAM, measured (C kernels, 3D, M2 Ultra, 16 threads)
RAM: p-v = (1+d) arrays = 4 cell-arrays in 3D; scalar = 2. **−50% RAM, exactly**, any precision.
Speed (GCU/s), scalar vs p-v, grows with grid size (both memory-bandwidth-bound; p-v also does the
extra velocity updates + divergence):

    grid            cells    fp32  p-v / scalar / ratio
    (100,80,70)     0.56M          1.62 / 4.13 / 2.5×
    (160,128,112)   2.29M          2.79 / 8.57 / 3.1×
    (250,200,175)   8.75M          6.42 / 20.1 / 3.1×
    fp64 (160^3-ish)               2.72 / 7.06 / 2.6×

Honest figure: **−50% RAM and ~2–3× speed**, the ratio rising toward ~3× at the large 3D sizes that
matter most (the user's "small dx / 3D" case). The "~5×" in STATUS was optimistic / an older run.

## Other methods tested (the "re-derive, find the best kernel" ask)

**Crank-Nicolson / ADI (implicit), re-derived and measured.** d2_t p = lam^2 d2_x (p^{n+1}+p^{n-1})/2,
unconditionally stable, dispersion sin^2(w dt/2)=q/(1+2q), q=lam^2 sin^2(k dx/2). Needs a tridiagonal
solve per axis per step (ADI in >1D) and cannot use the 2-array trick. Measured phase error for a 1%
target:

    explicit leapfrog @ CFL : ~2 ppw (1D, dispersionless) / ~10.5 ppw (worst 3D direction)
    Crank-Nicolson   C=1    : 31 ppw      C=2: 58 ppw      C=4: 115 ppw      C=8: 228 ppw

CN is 5-15x more resolution-hungry AND pays an implicit solve AND needs more memory. Its only lever -
dt beyond CFL - makes accuracy WORSE, and for audio dt is already pinned by sampling the signal in
time. CN loses on precision, speed and RAM. REJECTED. Explicit leapfrog at Courant 1 is essentially
optimal for uniform-grid acoustic propagation (exact on the grid diagonal).

**Fourth-order-in-space.** Already analysed in dispersion.py: -6.8x cell-updates for 1% error, but
still 2nd order overall with a leapfrog step, forces Courant 0.495 (losing diagonal exactness) and a
wider stencil (3 boundary-cell cases per wall per axis). It composes with the pressure-only form (the
stencil is on the Laplacian) and is an ORTHOGONAL lever to velocity elimination - they multiply.
Worthwhile but independent; do the RAM win first.

## Collocation vs the dt/2 stagger (user's explicit question) - MEASURED
Fourier symbol of the Laplacian (x dx^2): staggered is monotone to Nyquist (0->4.0); the COLLOCATED
operator (two centred first-differences) VANISHES at Nyquist -> the 2-cell checkerboard mode has zero
restoring force, a stationary spurious mode that accumulates and must be filtered/damped.
Conclusion: KEEP the half-cell space stagger and its companion dt/2 time stagger. They make the
difference compact, 2nd-order and null-space-free. Once velocity is eliminated the stagger COSTS
NOTHING - it is hidden inside the three-level pressure update. Collocation would cost accuracy, speed
and robustness for no benefit.

## Velocity only at walls/probe/source (user's goal) - MEASURED exact
In the pressure-only scheme the velocity GRID is gone. Where velocity is still needed it is O(1):
- probe: one running scalar per probe face, v^{n+1/2}=v^{n-1/2}-(dt/rho dx)(p^n_R-p^n_L).
  Reconstructed value matches the p-v stored velocity EXACTLY (0.0 error). Stays at the n-1/2 instant
  (documented half-step offset); average two to recentre on n.
- prescribed velocity source (piston / shaped loudspeaker line): hold that running value.
- rigid faces / obstacles (v.n=0): a closed face in the Laplacian - no stored velocity at all.
So velocity memory drops from O(N) (a full grid) to O(#walls + #probes + #source faces). Exactly the
user's intuition, and it falls out of the elimination for free.

## The two harder ports (so the recommendation is complete)
- Air absorption (ISO-9613 relaxation): portable. The unscaled divergence the relaxation states need
  is reconstructable from the two pressure levels: D = -(1/beta)[(p^n-p^{n-1})+lam^2 L(p^n)]. Keeps
  one auxiliary array per process (N2, O2), eroding - not erasing - the RAM win when air loss is on.
  Free for a dry/lossless room.
- Matched absorbing layer (free-field sponge): the awkward one. The matched layer damps p AND v with
  paired sigma profiles; dropping v breaks the match, so a naive damped scalar wave equation reflects.
  A reflectionless pressure-only layer needs an ADE/split-field PML with auxiliary fields - but only
  in the ~10-20-cell shell, not the whole grid. Options: (a) keep p-v for free-field runs, (b) port a
  thin-shell ADE-PML later. Do NOT block the room win on this. A CLOSED room never uses the layer, so
  the scalar kernel is already complete for it.

## Point 4: velocity as arbitrary off-grid surfaces (non-uniform)
"Non-uniform velocity on the surface" is already free - a prescribed-velocity source is a per-face
amplitude map. The open part is "any shape, not bound to the pressure grid." Three routes:
- (a) staircase onto Yee faces - current. O(dx) geometry error + facet scatter, but robust, cheap and
  IS exactly a boundary condition in the scalar Laplacian (closed/forced faces). Composes with the
  fast uniform stencil.
- (b) cut-cell / embedded boundary - sub-cell accurate, but the small-cell problem needs a special
  time step or flux redistribution: fragile, and it BREAKS the uniform stencil that gives the kernel
  its speed.
- (c) immersed-boundary spread forcing - distribute surface velocity onto nearby faces with a
  regularised kernel; smears over 1-2 cells, can hurt stability near resonances.
Recommendation: keep staircasing for the fast kernel (non-uniform amplitude free). A true off-grid
surface is a separate, stability-risky effort that does NOT compose cheaply with the pressure-only
form - prototype it only if a specific application needs sub-cell curved radiators.

## RECOMMENDATION
Add a pressure-only three-level backend as the fast kernel for closed rooms, 2D and 3D:
- -50% RAM (2 arrays vs 1+d), 2-3x faster, rising toward 3x at the large 3D sizes the user cares
  about ("small dx / 3D") - all PROVEN bit-exact vs the p-v reference for rigid + impedance walls +
  rigid obstacles + soft/prescribed sources + reconstructed probes.
- Keep the p-v scheme as the reference backend and for free-field (matched-layer) runs until a
  thin-shell ADE-PML is ported. Port air relaxation to three-level when wanted (cheap, 1 array/proc).
- Reject Crank-Nicolson/ADI. Keep the staggered dt/2 (eliminated, so free). Keep staircased velocity
  surfaces; treat cut-cell as a separate future study.
- Fourth-order-in-space is an orthogonal multiplier - a later, independent lever.

Scope of the code change (per the repos' golden rule): this lands in ac-fdtd UPSTREAM (extend the
existing CScalarFDTD with the three-level boundaries, add a 2D CScalarFDTD2D, adapter +
auto-selection), then re-vendor into ac-fdtd-studio. The derivations and validations here are the
spec for that work. NOT started - awaiting the go-ahead.

## Validation artefacts (this folder)
- scalar2d.py            - NumPy pressure-only prototype with the three-level boundaries (the spec).
- validate_scalar.py     - bit-equivalence vs AcousticFDTD2D (rigid/impedance/obstacle/driven).
- crank_nicolson.py      - the CN dispersion/cost comparison.
- collocation_and_probe.py - the Nyquist null-space + exact velocity-probe reconstruction.

================================================================================
# PART 2 — Absorption rework (user: go straight to a full PML; handle angles+corners;
#          also improve non-PML/partial-absorbent surfaces) and corner/angle handling
================================================================================

## Diagnosis: why alpha=0.999999 still over-reflects
Not a bug. The locally-reacting admittance wall (xi) is matched ONLY at normal incidence. Analytic
|R(theta)| for xi~1 (alpha=0.999999): R(0deg)=0.001 but R(60)=0.33, R(80)=0.70, R(85)=0.84. A 2D/3D
field hits a wall at all angles, so grazing energy reflects almost fully. This is correct physics for
a *material* wall; it is the wrong tool for *free-field truncation* (an external/open boundary).
=> two separate fixes: a real PML for open boundaries, and a better partial-absorber for materials.

## CFS-CPML validated (the open-boundary fix)
Roden-Gedney convolutional PML, complex-frequency-shifted. A stretched derivative d/dx becomes
(1/kappa)d/dx + psi, psi^new = b*psi + a*(d/dx), b=exp(-(sigma/kappa+alpha)dt),
a=sigma/(kappa(sigma+kappa alpha))(b-1). sigma graded ~depth^3 to sigma_max=-(m+1)c ln(R0)/(2L).
Measured reflection (robust: truncated domain minus a large free-field reference; MONITOR MUST SIT
BEFORE THE LAYER, not inside it -- that bug cost an hour):
    1D normal incidence: thickness 16 -> -35 dB, 32 -> -54 dB (monotone: formulation correct)
    2D (point source, oblique spread), kappa=5, alpha=2pi f0:
        thickness 16 -> -36 dB,  24 -> -44 dB,  32 -> -53 dB
The CFS alpha term adds 2-3 dB on the oblique field (its job: grazing/evanescent). Target -55 dB ->
~32-cell CPML. vs matched wall -13 dB and sponge layer ~ -17..-33 dB. PML wins decisively.

## Still to validate before coding the engine
- [ ] Corners: overlapping PMLs (both sigma_x and sigma_y active in the corner block). Standard CPML
      superposes per-axis; test a 4-side PML with a near-corner source.
- [ ] Hybrid: scalar (pressure-only) interior + p-v CPML shell, to KEEP the RAM win on open domains.
      Confirm the interior pressure matches a full p-v+CPML run and still reaches -50 dB.
- [ ] Partial-absorber (non-PML) improvement: a locally-reacting wall over-reflects obliquely. Test
      a better model (e.g. a thin lossy CPML "skin" tuned to a target absorption, or an
      angle-averaged admittance) so a wall set to alpha=0.6 behaves closer to alpha=0.6 at all angles.

## Corners + hybrid: VALIDATED
- Hybrid (scalar interior + p-v CFS-CPML shell) matches a full p-v+CPML run to round-off (1e-15) in
  the interior, INCLUDING an x_max+y_max corner. Velocity + 4 memory vars stored only on the shell
  (~20% of a 220^2 grid) -> the RAM win is kept on open domains. Interior stays 2 arrays.
- Corners handled by overlapping slabs (both sigma_x and sigma_y act in the corner block; standard).
- 4-side PML, central pulse: interior RMS decays -63 dB (all-angle + corner absorption confirmed).

Architecture decided: ONE scheme. Rigid = no shell; partial wall = shallow tuned lossy skin;
open/anechoic = deep CFS-CPML shell. Interior always pressure-only (2 arrays); velocity + CPML
memory live only in the boundary shell slabs. Corners = slab overlap.

## Partial absorber (non-PML walls): VALIDATED
Point-source (all-angle) reflection off x_max, dB re incident:
    target alpha   locally-reacting wall   8-cell skin   16-cell skin
        0.30             -1.9                 -3.4          -2.2
        0.60             -4.7                 -7.5          -6.2
        0.90            -11.8                -14.0         -13.8
        0.99            -12.8                -25.1         -23.7
The locally-reacting wall SATURATES at ~-13 dB however high alpha goes (grazing reflects fully) ==
the user's alpha=0.999999 complaint. A few-cell CFS-CPML skin breaks that ceiling (-25 dB at 0.99)
and a deeper skin -> fully anechoic. For WEAK absorption the single-admittance wall is fine & cheap.

## FINAL ABSORPTION DESIGN (one mechanism)
Boundary = a graded lossy shell (CFS-CPML), depth/strength sets the behaviour:
    rigid        : no shell (sigma=0)            -- v.n=0, exact, free
    locally-react: keep the 1-cell admittance    -- cheap, correct for genuinely locally-reacting
                   wall (three-level, bit-exact)     materials and weak absorption; angle-limited
    partial absorb: shallow tuned skin (few cells)-- angle-robust, breaks the grazing ceiling
    open/anechoic : deep CFS-CPML (~24-32 cells)  -- -55 dB+, all angles, corners via slab overlap
Interior stays pressure-only (2 arrays); velocity + 4 CPML memory vars live only in the shell slabs.
Corners: both sigma_x and sigma_y act in the overlap block (standard, validated to round-off).

## ENGINE BUILD PLAN (2D first, in ac-fdtd, then re-vendor)
v1 ScalarFDTD2D (NumPy ref): rigid + impedance walls + obstacles + volume/pressure sources +
   receivers + reconstructed velocity probe. Bit-exact vs AcousticFDTD2D. TESTS + commit.
v2 + CFS-CPML shell (open + partial skins) + prescribed velocity. TESTS + commit.
v3 CScalarFDTD2D (C mirror) + parity tests. v4 3D (ScalarFDTD + extend CScalarFDTD). v5 adapter +
   auto-selection in ac-fdtd-studio + re-vendor.

================================================================================
# IMPLEMENTATION LOG (engine: ac-fdtd)
================================================================================
v1 DONE: src/ac_fdtd/scalar2d.py ScalarFDTD2D (rigid+impedance+obstacles+volume/pressure
   sources+pressure&velocity receivers). tests/test_scalar2d.py (15 tests, bit-exact vs p-v).
   __init__ export, STATUS.md updated, docs/kernel_study/README.md (this doc). Commit script
   .claude/commits/07-pressure-only-2d-reference.sh (signed, user runs outside sandbox). Gates green.
v2 IN PROGRESS: add CFS-CPML open boundary + tuned partial skin to ScalarFDTD2D.
   Design: absorbing_boundary config (faces, thickness, target_reflection, kappa_max, alpha_max).
   Hybrid: scalar interior + p-v CPML shell. Reference stores full-grid velocity+memory when CPML
   present (C kernel will store shell-only); closed rooms still allocate NO velocity (2 arrays).
   Scope: CPML (deep or partial) on any face subset + rigid others + obstacles + sources. Impedance
   wall correction applied to INTERIOR wall cells only (shell corner cells stay pure p-v+CPML).
   Validated in tmp/kernel_study/hybrid_cpml.py (interior bit-exact vs full p-v+CPML incl. corner).

v2 DONE: CFS-CPML open boundary in ScalarFDTD2D.
   New module src/ac_fdtd/cpml2d.py: AbsorbingBoundary2D (open_field / partial_skin) + profiles.
   ScalarFDTD2D grows absorbing_boundary= : hybrid scalar-interior + p-v-CPML-shell, corners via
   slab overlap, impedance walls restricted to interior cells. Reference stores full-grid velocity+
   memory when CPML present; closed room stays 2 arrays (no vx). tests/test_scalar2d_cpml.py:
   central-pulse decay < -45 dB (all-angle+corner), reflection deepens with thickness (-25@16,
   -42@32), partial skin beats locally-reacting wall by >6 dB at alpha=0.99, one-axis-open stable,
   closed room 2 arrays. Full suite 166 green, ruff clean. Commit script 08 next.
NEXT: v3 CScalarFDTD2D (C kernel mirror: three-level walls + CPML shell). v4 3D. v5 adapter+vendor.

v3 DONE: 3D reference ScalarFDTD + 3D CPML (src/ac_fdtd/scalar.py, cpml.py). Bit-identical to
   AcousticFDTD (rigid/impedance/eigenmode <= 3e-13); 6-side CPML decays past -100 dB; -51% RAM
   (2 vs 4 arrays, measured); +rigid obstacle mask (3D p-v has none). tests/test_scalar.py (11).
   Full suite 178 green, ruff clean. Commit script renamed 07-pressure-only-scheme.sh covers 2D+3D.
NEXT: v4 compiled kernels CScalarFDTD2D / CScalarFDTD (three-level walls + CPML shell), checked vs
   these references. v5 adapter + auto-selection in ac-fdtd-studio + re-vendor (blocked on user push).

v4 DONE: compiled 2D kernel CScalarFDTD2D (src/ac_fdtd/c/acfdtd.c + c_backend.py).
   Three-level walls + rigid obstacles (masked-flux Laplacian) + volume sources (difference
   injection) + pressure receivers. One sweep, two arrays, pointer-swap per step. Bit-identical to
   ScalarFDTD2D (fp64 ~1e-14; fp32 tracks; thread counts agree; chunked runs exact). Measured
   1.8-2.5x the 2D p-v C kernel, -33% RAM. tests/test_c_backend_scalar_2d.py (8). Full suite 186
   green. Folded into commit 07 (references + CPML + 2D C kernel = one coherent "pressure-only
   scheme" commit, since STATUS/__init__ are shared and the user runs the script on the final tree).
REMAINING (clean follow-ups, de-risked):
   - 3D C kernel: extend CScalarFDTD (currently rigid-only 3D) with walls/obstacles/sources, OR add
     a full acfdtd3d_scalar like the 2D one. (Touches existing struct+tests -> do carefully.)
   - CPML shell in C (2D + 3D): the hybrid shell with velocity+memory slabs. Bigger C effort.
   - Air relaxation as three-level (reconstruct divergence from the two pressure levels).
   - ac-fdtd-studio: adapter auto-selection (prefer scalar for closed rooms) + re-vendor. BLOCKED on
     the user pushing the engine commit first (vendor script pulls the pushed repo; no push/gh in
     sandbox).

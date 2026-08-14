# ac-fdtd

3D acoustic FDTD: **room impulse responses** from a staggered pressure–velocity scheme, with
**air absorption** (ISO 9613-1) and boundaries that behave, validated against the cases where
an exact answer exists.

- Plan and scope: [`PLAN.md`](PLAN.md)
- Progress log, with the numbers: [`PROGRESS.md`](PROGRESS.md)

## Status

**M1 — the lossless core is in and validated.** Rigid rectangular rooms only, NumPy only.
Absorption, impedance boundaries and the fast backends are the milestones that follow.

![Validation of the lossless scheme in a rigid box](docs/figures/m1_box_validation.svg)

| Check | Result |
| --- | --- |
| Energy conservation, rigid box, 4000 steps | `4.4e-16` relative — round-off |
| Single mode stays sinusoidal, 3000 steps | `1.3e-13` max error, unit amplitude |
| Convergence to analytical modal frequencies | second order, `4.0x` per halving of `dx` |
| Broadband run, 25 modes below 400 Hz | 24 within one FFT bin, median offset `0.08 Hz` |

The one mode that misses is not excited: it has a node at the source or at the probe, so the
nearest peak in the spectrum belongs to a different mode.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/validate_box.py     # regenerates the figure above
```

```python
from ac_fdtd import AIR, AcousticFDTD, Grid

grid = Grid.from_max_frequency((5.0, 4.0, 3.0), max_frequency=1000.0,
                               sound_speed=AIR.sound_speed)
solver = AcousticFDTD(grid, medium=AIR)
solver.add_volume_source((1.0, 1.0, 1.2), signal)
solver.run(1000)
```

## Two things worth knowing before reading the code

**Pressure and velocity are half a time step apart, and the code says so out loud.** The
stored state is always the pair `(p^n, v^{n-1/2})`. Anything mixing the two — intensity,
energy, a velocity probe next to a pressure probe — has to name an instant, and getting it
wrong does not blow up, it just quietly biases the result. See the module docstring of
[`scheme.py`](src/ac_fdtd/scheme.py).

**The energy identity is the test that matters.** With rigid walls the discrete divergence and
gradient are exact negative adjoints, so the discrete energy is conserved *exactly*, not
approximately. Sign errors, off-by-one slices and mistimed boundary updates all show up there,
and most of them show up nowhere else.

## Licence

MIT.

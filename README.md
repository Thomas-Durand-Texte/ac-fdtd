# ac-fdtd

3D acoustic FDTD: **room impulse responses** from a staggered pressure–velocity scheme, with
**air absorption** (ISO 9613-1) and boundaries that behave, validated against the cases where
an exact answer exists.

- Plan and scope: [`PLAN.md`](PLAN.md)
- Progress log, with the numbers: [`PROGRESS.md`](PROGRESS.md)

## Status

**M2 — the core and its boundaries are in and validated.** Rectangular rooms, absorbing walls,
free field via an absorbing layer; NumPy only. Air absorption and the fast backends are the
milestones that follow.

![Validation of the lossless scheme in a rigid box](docs/figures/m1_box_validation.svg)

| Check | Result |
| --- | --- |
| Energy conservation, rigid box, 4000 steps | `4.4e-16` relative — round-off |
| Single mode stays sinusoidal, 3000 steps | `1.3e-13` max error, unit amplitude |
| Convergence to analytical modal frequencies | second order, `4.0x` per halving of `dx` |
| Broadband run, 25 modes below 400 Hz | 24 within one FFT bin, median offset `0.08 Hz` |

The one mode that misses is not excited: it has a node at the source or at the probe, so the
nearest peak in the spectrum belongs to a different mode.

![Validation of the boundaries](docs/figures/m2_boundaries.svg)

| Check | Result |
| --- | --- |
| Wall reflection vs `(1-xi)/(1+xi)` | within 0.005 of theory; rigid case measures 0.9998 |
| Impedance-matched wall | `\|R\| = 0.05` mean, rising to 0.09 at 17 points/wavelength |
| Absorbing layer, normal incidence | −108 dB at 16 cells — exact by construction |
| Absorbing layer in a 3D field | −17 to −31 dB, set by obliquity rather than thickness |
| Free field vs analytical Green's function | 0.21 % at 61 points/wavelength |

The layer result is the one to read carefully: doubling its thickness buys 3 dB, while moving
the receiver 30 cells further from it buys 14. A matched layer is matched at normal incidence
only. Beating −30 dB means a true PML, and −30 dB is the number it has to beat.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/validate_box.py         # regenerates the first figure
uv run python scripts/validate_boundaries.py  # the second (about two minutes)
```

```python
from ac_fdtd import AIR, AcousticFDTD, Grid, WallAdmittances

grid = Grid.from_max_frequency((5.0, 4.0, 3.0), max_frequency=1000.0, sound_speed=AIR.sound_speed)
solver = AcousticFDTD(grid, medium=AIR, walls=WallAdmittances.from_absorption(0.15))
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

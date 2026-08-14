# ac-fdtd

3D acoustic FDTD: **room impulse responses** from a staggered pressure–velocity scheme, with
**air absorption** (ISO 9613-1) and boundaries that behave, validated against the cases where
an exact answer exists.

- Plan and scope: [`PLAN.md`](PLAN.md)
- Progress log, with the numbers: [`PROGRESS.md`](PROGRESS.md)

## Status

**M4 — physics complete, performance next.** Rectangular rooms, absorbing walls, free field via
an absorbing layer, ISO 9613-1 air absorption, impulse responses out to a wav file with the
usual room-acoustics parameters. NumPy only — the fast backends are what remain.

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

![Validation of air absorption](docs/figures/m3_air_absorption.svg)

| Check | Result |
| --- | --- |
| Measured attenuation vs ISO 9613-1 | within 1.2 % above 20 points/wavelength, 3 % at 11 |
| Humidity dependence, 20–80 % RH | tracked across a factor of three at 8 kHz |
| Residual error vs humidity | independent of it — what is left is the grid, not the air |

The relaxation strengths come from the ISO coefficients by matching terms, so there is no
fitting step and no free parameter. Two extra fields, about ten flops per cell.

![Reverberation time against Sabine and Eyring](docs/figures/m4_reverberation.svg)

| octave band | simulated T20 | Sabine | Eyring |
| --- | --- | --- | --- |
| 125 Hz | 0.335 ± 0.019 s | 0.286 | 0.248 |
| 250 Hz | 0.266 ± 0.038 s | 0.285 | 0.248 |
| 500 Hz | 0.261 ± 0.035 s | 0.285 | 0.247 |
| 1000 Hz | 0.267 ± 0.026 s | 0.284 | 0.247 |

Above the Schroeder frequency (223 Hz for this room) the simulation lands between Eyring and
Sabine, within 6–8 % of Eyring. The 125 Hz band is 35 % away, and that is the right answer:
below the Schroeder frequency a room is a few separately decaying modes, not a diffuse field,
and Sabine's derivation does not apply there. Agreement in that band would mean the simulation
was reproducing the formula instead of the physics.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/validate_box.py         # regenerates the first figure
uv run python scripts/validate_boundaries.py  # the second (about two minutes)
uv run python scripts/validate_air_absorption.py
uv run python scripts/validate_reverberation.py  # also writes a wav you can listen to
```

A room impulse response, start to finish:

```python
from ac_fdtd import Room, simulate_impulse_response, write_wav

room = Room(dimensions=(5.0, 4.0, 3.0), absorption=0.15)
response = simulate_impulse_response(
    room,
    source=(1.0, 1.0, 1.2),
    receivers=[(3.5, 2.5, 1.4)],
    max_frequency=1000.0,
)
write_wav("room.wav", response.signals, response.sample_rate)
```

Or drive the scheme directly, which is what anything unusual should do:

```python
from ac_fdtd import AIR, AcousticFDTD, AirAbsorption, Grid, WallAdmittances

grid = Grid.from_max_frequency((5.0, 4.0, 3.0), max_frequency=1000.0, sound_speed=AIR.sound_speed)
solver = AcousticFDTD(
    grid,
    medium=AIR,
    walls=WallAdmittances.from_absorption(0.15),
    air_absorption=AirAbsorption(temperature=20.0, relative_humidity=50.0),
)
solver.add_volume_source((1.0, 1.0, 1.2), signal)
solver.add_pressure_receiver((3.5, 2.5, 1.4))
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

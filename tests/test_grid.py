import math

import pytest

from ac_fdtd.grid import Grid, max_stable_time_step
from ac_fdtd.medium import AIR


def test_lengths_report_the_room_actually_simulated():
    # 1.75 m does not divide by 0.1 m, so the simulated room is 1.8 m and the test says so.
    grid = Grid.from_lengths((1.75, 1.0, 1.0), dx=0.1)
    assert grid.shape == (18, 10, 10)
    assert grid.lengths[0] == pytest.approx(1.8)
    assert grid.length_error[0] == pytest.approx(0.05)
    assert grid.length_error[1] == pytest.approx(0.0)


def test_spacing_from_max_frequency():
    grid = Grid.from_max_frequency(
        (3.0, 3.0, 3.0), max_frequency=1000.0, sound_speed=343.0, points_per_wavelength=10.0
    )
    assert grid.dx == pytest.approx(0.0343)


def test_time_step_is_at_the_courant_limit():
    dt = max_stable_time_step(0.05, AIR.sound_speed)
    assert AIR.sound_speed * dt * math.sqrt(3) / 0.05 == pytest.approx(1.0)


def test_cell_index_clips_a_point_on_the_far_wall():
    grid = Grid.from_lengths((1.0, 1.0, 1.0), dx=0.1)
    assert grid.cell_index((0.0, 0.0, 0.0)) == (0, 0, 0)
    assert grid.cell_index((0.55, 0.05, 0.05)) == (5, 0, 0)
    assert grid.cell_index((1.0, 1.0, 1.0)) == (9, 9, 9)


def test_memory_estimate_counts_the_velocity_planes():
    grid = Grid.from_lengths((1.0, 1.0, 1.0), dx=0.1)
    velocity_cells = 3 * 11 * 10 * 10
    assert grid.memory_bytes(itemsize=4) == (1000 + velocity_cells) * 4
    assert grid.memory_bytes(itemsize=4, extra_fields=2) == (3000 + velocity_cells) * 4


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_rejects_nonsense_spacing(bad):
    with pytest.raises(ValueError):
        Grid(dx=bad, nx=1, ny=1, nz=1)

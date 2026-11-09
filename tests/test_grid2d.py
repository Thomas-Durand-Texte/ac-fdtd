"""The 2D grid: rounding, staggering, indexing, and the stability limit."""

import math

import numpy as np
import pytest

from ac_fdtd.grid2d import Grid2D, max_stable_time_step_2d


def test_from_lengths_rounds_each_axis_and_reports_the_error():
    grid = Grid2D.from_lengths((0.83, 0.61), 0.1)
    assert grid.shape == (8, 6)
    assert grid.lengths == pytest.approx((0.8, 0.6))
    assert grid.length_error == pytest.approx((0.8 - 0.83, 0.6 - 0.61))


def test_length_error_is_zero_without_a_recorded_request():
    assert Grid2D(dx=0.1, nx=8, ny=6).length_error == (0.0, 0.0)


def test_velocity_shape_adds_one_face_on_its_own_axis():
    grid = Grid2D(dx=0.1, nx=8, ny=6)
    assert grid.velocity_shape(0) == (9, 6)
    assert grid.velocity_shape(1) == (8, 7)


def test_cell_index_clips_to_the_grid_and_snaps_on_a_boundary():
    grid = Grid2D.from_lengths((1.0, 1.0), 0.01)
    # 1.2 / 0.01 is 119.9999... in binary; the tolerance must still land it in cell 120-equivalent,
    # and anything past the far wall clips to the last cell rather than one cell outside.
    assert grid.cell_index((0.5, 0.5)) == (50, 50)
    assert grid.cell_index((1.0, 1.0)) == (99, 99)
    assert grid.cell_index((-1.0, 2.0)) == (0, 99)


def test_max_stable_time_step_is_the_sqrt_two_courant_limit():
    dx, c = 0.01, 343.2
    assert max_stable_time_step_2d(dx, c) == pytest.approx(dx / (c * math.sqrt(2)))


def test_memory_bytes_counts_pressure_and_two_velocity_components():
    grid = Grid2D(dx=0.1, nx=8, ny=6)
    expected = (8 * 6 + 9 * 6 + 8 * 7) * 8
    assert grid.memory_bytes(itemsize=8) == expected
    assert grid.memory_bytes(itemsize=8, extra_fields=1) == expected + 8 * 6 * 8


def test_rejects_degenerate_geometry():
    with pytest.raises(ValueError):
        Grid2D(dx=0.0, nx=4, ny=4)
    with pytest.raises(ValueError):
        Grid2D(dx=0.1, nx=0, ny=4)
    with pytest.raises(ValueError):
        Grid2D.from_max_frequency((1.0, 1.0), max_frequency=-1.0, sound_speed=343.0)


def test_cell_centres_are_half_cell_offset():
    grid = Grid2D(dx=0.1, nx=3, ny=2)
    assert np.allclose(grid.cell_centres(0), [0.05, 0.15, 0.25])
    assert np.allclose(grid.cell_centres(1), [0.05, 0.15])

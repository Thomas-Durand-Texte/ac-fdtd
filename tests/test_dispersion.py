"""The dispersion relation, which is exact, and the design rule that comes out of it."""

import numpy as np
import pytest

from ac_fdtd.dispersion import (
    NAMED_DIRECTIONS,
    phase_velocity_ratio,
    required_points_per_wavelength,
    worst_direction_error,
)


def test_the_body_diagonal_is_exact_at_the_stability_limit():
    """Not approximately: the arcsine and its argument cancel identically at Courant 1."""
    resolutions = np.array([3.0, 5.0, 10.0, 40.0, 200.0])
    ratio = phase_velocity_ratio(resolutions, NAMED_DIRECTIONS["body diagonal (1,1,1)"])
    np.testing.assert_allclose(ratio, 1.0, atol=1e-14)


def test_below_the_limit_even_the_diagonal_disperses():
    """Which is why the default Courant number is 1 and not something with a margin."""
    ratio = phase_velocity_ratio(
        np.array([10.0]), NAMED_DIRECTIONS["body diagonal (1,1,1)"], courant=0.9
    )
    assert abs(ratio[0] - 1.0) > 1e-4


def test_the_axis_is_the_worst_direction():
    for resolution in (6.0, 10.0, 25.0):
        axis = abs(phase_velocity_ratio(np.array([resolution]), [1, 0, 0])[0] - 1.0)
        assert axis == pytest.approx(worst_direction_error(resolution), rel=1e-3)


def test_the_error_is_second_order_in_the_resolution():
    coarse = worst_direction_error(20.0)
    fine = worst_direction_error(40.0)
    assert coarse / fine == pytest.approx(4.0, rel=0.02)


def test_the_numerical_wave_is_never_faster_than_the_real_one():
    for resolution in (4.0, 8.0, 30.0):
        for direction in NAMED_DIRECTIONS.values():
            assert phase_velocity_ratio(np.array([resolution]), direction)[0] <= 1.0 + 1e-14


def test_ten_points_per_wavelength_is_about_one_percent():
    """The folk rule, with a number attached to it at last."""
    assert worst_direction_error(10.0) == pytest.approx(0.0111, abs=0.0005)
    assert required_points_per_wavelength(0.01) == pytest.approx(10.5, abs=0.2)


def test_a_finer_target_needs_more_points_and_a_lower_courant_needs_more_still():
    tighter = required_points_per_wavelength(0.002)
    looser = required_points_per_wavelength(0.02)
    assert tighter > looser
    assert required_points_per_wavelength(0.01, courant=0.5) > required_points_per_wavelength(
        0.01, courant=1.0
    )


def test_an_impossible_target_is_refused():
    with pytest.raises(ValueError):
        required_points_per_wavelength(0.0)

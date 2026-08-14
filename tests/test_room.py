"""Impulse responses end to end: excitation, recording, deconvolution, metrics, export."""

import numpy as np
import pytest
from scipy.io import wavfile

from ac_fdtd.boundaries import admittance_from_absorption, random_incidence_absorption
from ac_fdtd.io import resample, write_wav
from ac_fdtd.metrics import (
    clarity,
    early_decay_time,
    eyring_time,
    octave_band,
    reverberation_time,
    sabine_time,
    schroeder_decay,
    schroeder_frequency,
)
from ac_fdtd.room import Room, simulate_impulse_response
from ac_fdtd.sources import band_limited_impulse, deconvolve, gaussian_pulse

SAMPLE_RATE = 20000.0


def _decaying_noise(reverberation: float, duration: float = 2.0, seed: int = 0) -> np.ndarray:
    """White noise with an exactly known 60 dB decay time — a response with a right answer."""
    instants = np.arange(int(duration * SAMPLE_RATE)) / SAMPLE_RATE
    envelope = 10.0 ** (-3.0 * instants / reverberation)
    return np.random.default_rng(seed).standard_normal(instants.size) * envelope


@pytest.mark.parametrize("expected", [0.3, 0.8, 1.6])
def test_reverberation_time_recovers_a_known_decay(expected):
    response = _decaying_noise(expected)
    assert reverberation_time(response, SAMPLE_RATE, 20.0) == pytest.approx(expected, rel=0.02)
    assert reverberation_time(response, SAMPLE_RATE, 30.0) == pytest.approx(expected, rel=0.02)


def test_early_decay_time_matches_when_the_decay_is_a_single_slope():
    response = _decaying_noise(0.8)
    assert early_decay_time(response, SAMPLE_RATE) == pytest.approx(0.8, rel=0.05)


def test_schroeder_decay_starts_at_zero_and_is_monotonic():
    decay = schroeder_decay(_decaying_noise(0.5))
    assert decay[0] == pytest.approx(0.0)
    assert np.all(np.diff(decay) <= 1e-12)


def test_clarity_is_higher_for_a_drier_room():
    assert clarity(_decaying_noise(0.3), SAMPLE_RATE) > clarity(_decaying_noise(1.5), SAMPLE_RATE)


def test_octave_band_keeps_its_band_and_rejects_the_rest():
    instants = np.arange(int(0.5 * SAMPLE_RATE)) / SAMPLE_RATE
    in_band = np.sin(2 * np.pi * 500.0 * instants)
    out_of_band = np.sin(2 * np.pi * 4000.0 * instants)

    # Edges trimmed: a zero-phase filter has a startup transient at both ends, and including
    # it would measure the transient rather than the stopband.
    kept = octave_band(in_band, SAMPLE_RATE, 500.0)[1000:-1000]
    rejected = octave_band(out_of_band, SAMPLE_RATE, 500.0)[1000:-1000]
    assert np.std(kept) == pytest.approx(np.std(in_band), rel=0.05)
    assert np.std(rejected) < 1e-4 * np.std(out_of_band)


def test_eyring_predicts_a_shorter_decay_than_sabine():
    """And the gap widens with absorption, which is the whole reason Eyring exists."""
    for absorption in (0.1, 0.4, 0.8):
        sabine = sabine_time(100.0, 130.0, absorption)
        eyring = eyring_time(100.0, 130.0, absorption)
        assert eyring < sabine
    assert eyring_time(100.0, 130.0, 0.999) < 0.05


def test_random_incidence_absorption_exceeds_normal_incidence():
    """The factor that makes a correct simulation look wrong when it is forgotten."""
    for normal in (0.05, 0.15, 0.3):
        diffuse = random_incidence_absorption(admittance_from_absorption(normal))
        assert 1.3 < diffuse / normal < 2.0
    assert random_incidence_absorption(0.0) == pytest.approx(0.0)


def test_schroeder_frequency_is_where_it_should_be():
    # A small live room: statistical theory has nothing to say in the two lowest octaves.
    assert 200.0 < schroeder_frequency(20.0, 0.3) < 400.0


def test_band_limited_impulse_is_flat_in_band_and_empty_above():
    dt, max_frequency = 2e-5, 5000.0
    signal = band_limited_impulse(4096, dt, max_frequency)
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(signal.size, dt)

    in_band = spectrum[frequencies < 0.8 * max_frequency]
    above = spectrum[frequencies > 1.3 * max_frequency]
    assert np.ptp(in_band) / np.mean(in_band) < 0.05
    assert np.max(above) < 0.02 * np.mean(in_band)


def test_gaussian_pulse_is_down_by_forty_decibels_at_its_stated_limit():
    dt, max_frequency = 2e-5, 5000.0
    signal = gaussian_pulse(4096, dt, max_frequency)
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(signal.size, dt)
    at_limit = spectrum[int(np.argmin(np.abs(frequencies - max_frequency)))]
    assert 20 * np.log10(at_limit / spectrum[0]) == pytest.approx(-40.0, abs=1.0)


def test_deconvolution_recovers_a_known_response():
    dt, max_frequency, n = 2e-5, 5000.0, 8192
    excitation = band_limited_impulse(n, dt, max_frequency)

    truth = np.zeros(n)
    truth[[100, 340, 700]] = [1.0, -0.5, 0.25]
    recovered = deconvolve(np.convolve(truth, excitation)[:n], excitation, dt, max_frequency)

    # Only the band survives, so the arrivals come back as sinc-shaped rather than as spikes.
    # What must survive exactly is position, sign and relative size.
    peaks = [recovered[95:105].max(), recovered[335:345].min(), recovered[695:705].max()]
    assert peaks[0] > 0 and peaks[1] < 0 and peaks[2] > 0
    assert peaks[1] / peaks[0] == pytest.approx(-0.5, rel=0.1)
    assert peaks[2] / peaks[0] == pytest.approx(0.25, rel=0.1)


def test_resampling_preserves_a_tone():
    source_rate = 59400.0
    instants = np.arange(int(0.2 * source_rate)) / source_rate
    tone = np.sin(2 * np.pi * 1000.0 * instants)
    resampled = resample(tone, source_rate, 48000.0)

    assert resampled.size == pytest.approx(0.2 * 48000.0, rel=0.01)
    middle = resampled[1000:-1000]
    assert np.max(np.abs(middle)) == pytest.approx(1.0, rel=0.01)


def test_wav_export_round_trip(tmp_path):
    signals = np.random.default_rng(0).standard_normal((2, 6000))
    path = write_wav(tmp_path / "ir.wav", signals, 59400.0, target_rate=48000.0)
    rate, data = wavfile.read(path)
    assert rate == 48000
    assert data.shape[1] == 2
    assert np.max(np.abs(data)) == pytest.approx(1.0, rel=1e-6)


def test_a_small_room_decays_at_roughly_the_predicted_rate():
    """End to end. Loose bounds on purpose — the tight version is the M4 study, not a test."""
    room = Room(dimensions=(2.0, 1.7, 1.5), absorption=0.3, air=None)
    response = simulate_impulse_response(
        room,
        source=(0.6, 0.5, 0.4),
        receivers=[(1.4, 1.1, 1.0)],
        max_frequency=500.0,
        duration=4.0 * room.eyring_reverberation_time(),
    )

    assert np.all(np.isfinite(response.signals))
    measured = reverberation_time(response.recorded[0], response.sample_rate, 20.0)
    assert (
        0.6 * room.eyring_reverberation_time() < measured < 1.6 * room.sabine_reverberation_time()
    )

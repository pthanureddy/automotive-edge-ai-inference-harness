from __future__ import annotations

import numpy as np

from edge_ai.synthetic import CHANNEL_COUNT, WINDOW_SAMPLES

FEATURES_PER_CHANNEL = 6
FEATURE_COUNT = CHANNEL_COUNT * FEATURES_PER_CHANNEL
FEATURE_NAMES = tuple(
    f"axis_{axis}_{name}"
    for axis in range(CHANNEL_COUNT)
    for name in ("mean", "rms", "peak", "zero_crossing", "low_band_power", "high_band_power")
)


def extract_window_features(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64)
    if values.shape != (CHANNEL_COUNT, WINDOW_SAMPLES):
        raise ValueError(
            f"expected ({CHANNEL_COUNT}, {WINDOW_SAMPLES}) samples, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("sensor samples must be finite")

    feature_values: list[float] = []
    for axis_values in values:
        mean = float(np.mean(axis_values))
        centered = axis_values - mean
        rms = float(np.sqrt(np.mean(axis_values**2)))
        peak = float(np.max(np.abs(centered)))
        zero_crossing = float(
            np.count_nonzero((centered[:-1] < 0.0) != (centered[1:] < 0.0))
            / (WINDOW_SAMPLES - 1)
        )
        spectrum = np.fft.rfft(centered)
        power = (np.abs(spectrum) ** 2) / (WINDOW_SAMPLES**2)
        low_band_power = float(np.sum(power[1:4]))
        high_band_power = float(np.sum(power[4:13]))
        feature_values.extend(
            (mean, rms, peak, zero_crossing, low_band_power, high_band_power)
        )
    return np.asarray(feature_values, dtype=np.float32)


def extract_features(raw_windows: np.ndarray) -> np.ndarray:
    values = np.asarray(raw_windows, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (CHANNEL_COUNT, WINDOW_SAMPLES):
        raise ValueError(
            f"expected [windows, {CHANNEL_COUNT}, {WINDOW_SAMPLES}], got {values.shape}"
        )
    return np.stack([extract_window_features(window) for window in values])


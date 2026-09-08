from __future__ import annotations

import numpy as np
import pytest

from edge_ai.features import FEATURE_COUNT, extract_features, extract_window_features
from edge_ai.synthetic import CHANNEL_COUNT, WINDOW_SAMPLES, generate_windows, stack_windows


def test_feature_shape_and_finiteness() -> None:
    samples = generate_windows(cycles_per_class=5, windows_per_cycle=1)[0].samples
    features = extract_window_features(samples)
    assert features.shape == (FEATURE_COUNT,)
    assert np.isfinite(features).all()


def test_batch_feature_shape() -> None:
    raw, _, _ = stack_windows(generate_windows(cycles_per_class=5, windows_per_cycle=1))
    assert extract_features(raw).shape == (15, FEATURE_COUNT)


def test_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected"):
        extract_window_features(np.zeros((CHANNEL_COUNT, WINDOW_SAMPLES - 1)))


def test_rejects_nonfinite_samples() -> None:
    values = np.zeros((CHANNEL_COUNT, WINDOW_SAMPLES))
    values[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_window_features(values)

from __future__ import annotations

import numpy as np

from edge_ai.quantization import deployment_sizes, quantize_pipeline
from edge_ai.synthetic import CLASS_NAMES, generate_windows, split_by_cycle
from edge_ai.training import classification_metrics


def test_generation_is_deterministic() -> None:
    left = generate_windows(cycles_per_class=5, windows_per_cycle=1, seed=2)
    right = generate_windows(cycles_per_class=5, windows_per_cycle=1, seed=2)
    np.testing.assert_array_equal(left[0].samples, right[0].samples)


def test_cycle_split_has_no_leakage() -> None:
    split = split_by_cycle(generate_windows(cycles_per_class=10, windows_per_cycle=2))
    cycles = {name: {window.cycle_id for window in values} for name, values in split.items()}
    assert not cycles["train"] & cycles["validation"]
    assert not cycles["train"] & cycles["test"]
    assert not cycles["validation"] & cycles["test"]


def test_cycle_split_retains_all_classes() -> None:
    split = split_by_cycle(generate_windows(cycles_per_class=10, windows_per_cycle=2))
    expected = set(range(len(CLASS_NAMES)))
    assert all({window.label for window in values} == expected for values in split.values())


def test_float_model_learns_synthetic_task(fitted_bundle) -> None:
    predictions = fitted_bundle.pipeline.predict(fitted_bundle.split_features["test"])
    metrics = classification_metrics(fitted_bundle.split_labels["test"], predictions)
    assert metrics["macro_f1"] >= 0.80


def test_quantized_reference_tracks_float_model(fitted_bundle) -> None:
    quantized = quantize_pipeline(
        fitted_bundle.pipeline, fitted_bundle.split_features["train"]
    )
    test_features = fitted_bundle.split_features["test"]
    float_predictions = fitted_bundle.pipeline.predict(test_features)
    quantized_predictions = quantized.predict(test_features)
    assert np.mean(float_predictions == quantized_predictions) >= 0.90


def test_quantized_parameters_are_smaller(fitted_bundle) -> None:
    quantized = quantize_pipeline(
        fitted_bundle.pipeline, fitted_bundle.split_features["train"]
    )
    sizes = deployment_sizes(fitted_bundle.pipeline, quantized)
    assert sizes["parameter_byte_reduction_percent"] >= 50.0

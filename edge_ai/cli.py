from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from edge_ai.exporting import (
    export_onnx,
    predict_onnx,
    write_json,
    write_manifest,
    write_reference_vectors,
)
from edge_ai.features import extract_features
from edge_ai.quantization import deployment_sizes, quantize_pipeline, write_c_header
from edge_ai.synthetic import CLASS_NAMES, generate_windows, stack_windows
from edge_ai.training import classification_metrics, fit_bundle


def _split_summary(bundle: object) -> dict[str, object]:
    split_windows = bundle.split_windows
    cycle_sets = {
        name: {window.cycle_id for window in windows}
        for name, windows in split_windows.items()
    }
    return {
        "windows": {name: len(values) for name, values in split_windows.items()},
        "cycles": {name: len(values) for name, values in cycle_sets.items()},
        "cycle_overlap_count": sum(
            len(cycle_sets[left] & cycle_sets[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ),
    }


def build_artifacts(root: Path) -> dict[str, object]:
    artifact_dir = root / "artifacts"
    onnx_path = artifact_dir / "model.onnx"
    metrics_path = artifact_dir / "metrics.json"
    vectors_path = artifact_dir / "reference_vectors.csv"
    header_path = root / "firmware" / "generated" / "model_data.h"
    manifest_path = artifact_dir / "manifest.json"

    windows = generate_windows()
    bundle = fit_bundle(windows)
    pipeline = bundle.pipeline
    test_features = bundle.split_features["test"]
    test_labels = bundle.split_labels["test"]
    float_predictions = pipeline.predict(test_features)

    export_onnx(pipeline, onnx_path)
    onnx_predictions = predict_onnx(onnx_path, test_features)

    quantized = quantize_pipeline(pipeline, bundle.split_features["train"])
    quantized_predictions = quantized.predict(test_features)
    write_c_header(quantized, header_path)
    write_reference_vectors(bundle.split_windows["test"], quantized, vectors_path)

    robust_windows = generate_windows(
        cycles_per_class=12,
        windows_per_cycle=4,
        seed=20260909,
        noise_multiplier=1.75,
        cycle_prefix="noise",
    )
    robust_raw, robust_labels, _ = stack_windows(robust_windows)
    robust_features = extract_features(robust_raw)

    metrics: dict[str, object] = {
        "scope": {
            "data": "deterministic synthetic three-axis vibration windows",
            "execution": "host-based reference validation; no target MCU or vehicle",
            "sample_rate_hz": 400,
            "window_samples": 64,
            "classes": list(CLASS_NAMES),
        },
        "dataset": {
            "total_windows": len(windows),
            "total_cycles": len({window.cycle_id for window in windows}),
            "split": _split_summary(bundle),
        },
        "float_model": {
            "test": classification_metrics(test_labels, float_predictions),
            "validation": classification_metrics(
                bundle.split_labels["validation"],
                pipeline.predict(bundle.split_features["validation"]),
            ),
            "robust_noise": classification_metrics(
                robust_labels, pipeline.predict(robust_features)
            ),
            "test_confusion_matrix": confusion_matrix(
                test_labels, float_predictions, labels=np.arange(len(CLASS_NAMES))
            ).tolist(),
            "cross_validation_macro_f1": [
                round(value, 6) for value in bundle.cross_validation_macro_f1
            ],
            "cross_validation_macro_f1_mean": round(
                float(np.mean(bundle.cross_validation_macro_f1)), 6
            ),
        },
        "unsupervised_baseline": {
            "method": "IsolationForest fitted on normal training cycles",
            "test_roc_auc": round(bundle.isolation_forest_roc_auc, 6),
        },
        "onnx": {
            "opset": 17,
            "test_prediction_mismatches_vs_sklearn": int(
                np.count_nonzero(onnx_predictions != float_predictions)
            ),
            "model_bytes": onnx_path.stat().st_size,
        },
        "quantized_reference": {
            "kernel": "int8 weights/activations, int32 accumulation, float bias and rescaling",
            "test": classification_metrics(test_labels, quantized_predictions),
            "validation": classification_metrics(
                bundle.split_labels["validation"],
                quantized.predict(bundle.split_features["validation"]),
            ),
            "robust_noise": classification_metrics(
                robust_labels, quantized.predict(robust_features)
            ),
            "test_prediction_mismatches_vs_sklearn": int(
                np.count_nonzero(quantized_predictions != float_predictions)
            ),
            **deployment_sizes(pipeline, quantized),
        },
    }
    write_json(metrics_path, metrics)
    write_manifest([onnx_path, metrics_path, vectors_path, header_path], manifest_path, root)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current directory)",
    )
    args = parser.parse_args()
    metrics = build_artifacts(args.root.resolve())
    float_f1 = metrics["float_model"]["test"]["macro_f1"]
    quantized_f1 = metrics["quantized_reference"]["test"]["macro_f1"]
    print(f"artifacts generated: float macro-F1={float_f1}, quantized macro-F1={quantized_f1}")


if __name__ == "__main__":
    main()

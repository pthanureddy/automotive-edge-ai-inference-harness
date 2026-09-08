from __future__ import annotations

import json
from pathlib import Path


def verify_metrics(root: Path) -> list[str]:
    metrics_path = root / "artifacts" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checks = {
        "cycle split has no leakage": metrics["dataset"]["split"]["cycle_overlap_count"] == 0,
        "float test macro-F1 >= 0.90": metrics["float_model"]["test"]["macro_f1"] >= 0.90,
        "quantized test macro-F1 >= 0.88": metrics["quantized_reference"]["test"][
            "macro_f1"
        ]
        >= 0.88,
        "quantized degradation <= 0.04": (
            metrics["float_model"]["test"]["macro_f1"]
            - metrics["quantized_reference"]["test"]["macro_f1"]
            <= 0.04
        ),
        "ONNX labels match scikit-learn": metrics["onnx"][
            "test_prediction_mismatches_vs_sklearn"
        ]
        == 0,
        "parameter storage reduced >= 50%": metrics["quantized_reference"][
            "parameter_byte_reduction_percent"
        ]
        >= 50.0,
        "unsupervised ROC AUC >= 0.80": metrics["unsupervised_baseline"]["test_roc_auc"]
        >= 0.80,
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("verification failed: " + "; ".join(failures))
    return list(checks)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    checks = verify_metrics(args.root.resolve())
    print(f"{len(checks)} metric gates passed")


if __name__ == "__main__":
    main()

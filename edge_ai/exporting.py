from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

from edge_ai.features import FEATURE_COUNT, FEATURE_NAMES, extract_window_features
from edge_ai.quantization import QuantizedMLP
from edge_ai.synthetic import CHANNEL_COUNT, WINDOW_SAMPLES, SensorWindow


def export_onnx(pipeline: Pipeline, output_path: Path) -> None:
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=[("sensor_features", FloatTensorType([None, FEATURE_COUNT]))],
        target_opset=17,
        options={id(pipeline.named_steps["classifier"]): {"zipmap": False}},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())


def predict_onnx(model_path: Path, features: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.asarray(features, dtype=np.float32)})
    return np.asarray(outputs[0], dtype=np.int64).reshape(-1)


def write_reference_vectors(
    windows: list[SensorWindow], model: QuantizedMLP, output_path: Path, per_class: int = 5
) -> None:
    selected: list[SensorWindow] = []
    for label in sorted({window.label for window in windows}):
        selected.extend([window for window in windows if window.label == label][:per_class])

    fieldnames = ["cycle_id", "label", "prediction", "rpm", "load"]
    fieldnames.extend(
        f"sample_{axis}_{sample}"
        for axis in range(CHANNEL_COUNT)
        for sample in range(WINDOW_SAMPLES)
    )
    fieldnames.extend(FEATURE_NAMES)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for window in selected:
            features = extract_window_features(window.samples)
            prediction = int(model.predict(features)[0])
            row: dict[str, str | int | float] = {
                "cycle_id": window.cycle_id,
                "label": window.label,
                "prediction": prediction,
                "rpm": window.rpm,
                "load": window.load,
            }
            for axis in range(CHANNEL_COUNT):
                for sample in range(WINDOW_SAMPLES):
                    row[f"sample_{axis}_{sample}"] = f"{float(window.samples[axis, sample]):.9g}"
            for name, value in zip(FEATURE_NAMES, features, strict=True):
                row[name] = f"{float(value):.9g}"
            writer.writerow(row)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(paths: list[Path], output_path: Path, root: Path) -> None:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(paths)
    ]
    write_json(output_path, {"schema_version": 1, "artifacts": entries})


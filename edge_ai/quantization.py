from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from edge_ai.features import FEATURE_COUNT


@dataclass(frozen=True)
class QuantizedMLP:
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    input_scale: float
    weight1: np.ndarray
    weight1_scale: float
    bias1: np.ndarray
    hidden_scale: float
    weight2: np.ndarray
    weight2_scale: float
    bias2: np.ndarray
    classes: np.ndarray

    @property
    def hidden_count(self) -> int:
        return int(self.bias1.shape[0])

    def predict_logits(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] != FEATURE_COUNT:
            raise ValueError(f"expected {FEATURE_COUNT} features, got {values.shape[1]}")

        standardized = (values - self.scaler_mean) / self.scaler_scale
        input_q = np.clip(np.rint(standardized / self.input_scale), -127, 127).astype(
            np.int8
        )
        hidden_acc = input_q.astype(np.int32) @ self.weight1.astype(np.int32)
        hidden = np.maximum(
            0.0,
            hidden_acc.astype(np.float64) * self.input_scale * self.weight1_scale
            + self.bias1,
        )
        hidden_q = np.clip(np.rint(hidden / self.hidden_scale), 0, 127).astype(np.int8)
        output_acc = hidden_q.astype(np.int32) @ self.weight2.astype(np.int32)
        return (
            output_acc.astype(np.float64) * self.hidden_scale * self.weight2_scale
            + self.bias2
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        indices = np.argmax(self.predict_logits(features), axis=1)
        return self.classes[indices]


def _symmetric_quantize(values: np.ndarray) -> tuple[np.ndarray, float]:
    maximum = float(np.max(np.abs(values)))
    if maximum == 0.0:
        return np.zeros_like(values, dtype=np.int8), 1.0
    scale = maximum / 127.0
    quantized = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
    return quantized, scale


def quantize_pipeline(pipeline: Pipeline, calibration_features: np.ndarray) -> QuantizedMLP:
    scaler = pipeline.named_steps["standardize"]
    classifier = pipeline.named_steps["classifier"]
    if len(classifier.coefs_) != 2:
        raise ValueError("the firmware exporter supports one hidden layer")

    standardized = scaler.transform(calibration_features)
    input_abs_max = max(float(np.max(np.abs(standardized))), 1.0e-8)
    input_scale = input_abs_max / 127.0

    weight1, weight1_scale = _symmetric_quantize(classifier.coefs_[0])
    hidden_float = np.maximum(0.0, standardized @ classifier.coefs_[0] + classifier.intercepts_[0])
    hidden_abs_max = max(float(np.max(hidden_float)), 1.0e-8)
    hidden_scale = hidden_abs_max / 127.0
    weight2, weight2_scale = _symmetric_quantize(classifier.coefs_[1])

    return QuantizedMLP(
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float32),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float32),
        input_scale=input_scale,
        weight1=weight1,
        weight1_scale=weight1_scale,
        bias1=np.asarray(classifier.intercepts_[0], dtype=np.float32),
        hidden_scale=hidden_scale,
        weight2=weight2,
        weight2_scale=weight2_scale,
        bias2=np.asarray(classifier.intercepts_[1], dtype=np.float32),
        classes=np.asarray(classifier.classes_, dtype=np.int64),
    )


def deployment_sizes(pipeline: Pipeline, quantized: QuantizedMLP) -> dict[str, float | int]:
    classifier = pipeline.named_steps["classifier"]
    scaler = pipeline.named_steps["standardize"]
    fp32_bytes = int(
        4
        * (
            sum(values.size for values in classifier.coefs_)
            + sum(values.size for values in classifier.intercepts_)
            + scaler.mean_.size
            + scaler.scale_.size
        )
    )
    int8_bytes = int(
        quantized.weight1.nbytes
        + quantized.weight2.nbytes
        + 4
        * (
            quantized.bias1.size
            + quantized.bias2.size
            + quantized.scaler_mean.size
            + quantized.scaler_scale.size
            + 4
        )
    )
    return {
        "fp32_parameter_bytes": fp32_bytes,
        "quantized_parameter_bytes": int8_bytes,
        "parameter_byte_reduction_percent": round(
            100.0 * (fp32_bytes - int8_bytes) / fp32_bytes, 2
        ),
        "int8_weight_bytes": int(quantized.weight1.nbytes + quantized.weight2.nbytes),
    }


def _float_literal(value: float) -> str:
    rendered = f"{float(value):.9g}"
    if "e" not in rendered.lower() and "." not in rendered:
        rendered += ".0"
    return rendered + "f"


def _render_array(name: str, values: np.ndarray, c_type: str) -> str:
    flat = values.reshape(-1)
    if c_type == "float":
        rendered = ", ".join(_float_literal(float(value)) for value in flat)
    else:
        rendered = ", ".join(str(int(value)) for value in flat)
    return f"static const {c_type} {name}[{flat.size}] = {{{rendered}}};"


def write_c_header(model: QuantizedMLP, output_path: Path) -> None:
    lines = [
        "#ifndef EDGE_AI_GENERATED_MODEL_DATA_H",
        "#define EDGE_AI_GENERATED_MODEL_DATA_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define EDGE_AI_FEATURE_COUNT {FEATURE_COUNT}",
        f"#define EDGE_AI_HIDDEN_COUNT {model.hidden_count}",
        f"#define EDGE_AI_CLASS_COUNT {model.classes.size}",
        "",
        _render_array("EDGE_AI_SCALER_MEAN", model.scaler_mean, "float"),
        _render_array("EDGE_AI_SCALER_SCALE", model.scaler_scale, "float"),
        f"static const float EDGE_AI_INPUT_SCALE = {_float_literal(model.input_scale)};",
        _render_array("EDGE_AI_WEIGHT1", model.weight1, "int8_t"),
        f"static const float EDGE_AI_WEIGHT1_SCALE = {_float_literal(model.weight1_scale)};",
        _render_array("EDGE_AI_BIAS1", model.bias1, "float"),
        f"static const float EDGE_AI_HIDDEN_SCALE = {_float_literal(model.hidden_scale)};",
        _render_array("EDGE_AI_WEIGHT2", model.weight2, "int8_t"),
        f"static const float EDGE_AI_WEIGHT2_SCALE = {_float_literal(model.weight2_scale)};",
        _render_array("EDGE_AI_BIAS2", model.bias2, "float"),
        _render_array("EDGE_AI_CLASSES", model.classes.astype(np.int32), "int32_t"),
        "",
        "#endif",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")

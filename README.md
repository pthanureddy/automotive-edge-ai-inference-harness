# Automotive Edge AI Inference Harness

[![CI](https://github.com/pthanureddy/automotive-edge-ai-inference-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/pthanureddy/automotive-edge-ai-inference-harness/actions/workflows/ci.yml)

A reproducible, host-verified reference project for taking a small automotive sensor model from synthetic data generation through grouped validation, ONNX export, quantization, portable C inference, cross-language parity checks, and timing/resource measurement.

## What it demonstrates

- deterministic three-axis motor-vibration windows at 400 Hz with normal, imbalance, and bearing-fault labels;
- cycle-grouped train/validation/test splits that prevent windows from one simulated drive cycle crossing split boundaries;
- 18 time- and frequency-domain signal features implemented independently in NumPy and C;
- a scikit-learn MLP supervised classifier and a normal-only Isolation Forest anomaly baseline;
- ONNX opset 17 export with prediction-parity checking through ONNX Runtime;
- an int8 weight/activation reference kernel with int32 accumulation and floating-point bias/rescaling;
- fixed-size C arrays with no dynamic allocation in feature extraction or inference;
- C++ reference-vector tests and a host latency benchmark with an explicit 1 ms engineering budget;
- GitHub Actions verification on Python 3.11 and 3.12.

This is a compact engineering harness, not a claim that a synthetic classifier is useful for a real vehicle. The deliberately small model makes the entire deployment path inspectable.

## Architecture

```text
synthetic drive cycles -> NumPy features -> grouped ML validation
                                      |-> ONNX export + parity
                                      |-> int8 model export
raw reference windows -> portable C features -> C int8 inference -> C++ parity + timing
```

The quantized kernel is **not integer-only**: inputs, hidden activations, and weights are int8 and matrix accumulations are int32, while scaling, biases, ReLU, and feature extraction use floating point.

## Reproduce

Python 3.11+ is required. No external dataset is downloaded.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev,local-toolchain]"
python scripts/verify.py
```

The command regenerates all committed evidence, runs lint and Python tests, builds C/C++ with CMake/Ninja, runs cross-language tests, and writes `artifacts/host_benchmark.json`.

## Verification gates

- zero drive-cycle overlap across train, validation, and test;
- float test macro-F1 at least 0.90 on the deterministic synthetic dataset;
- quantized test macro-F1 at least 0.88 and no more than 0.04 below the float model;
- exact ONNX/scikit-learn test-label parity;
- at least 50% parameter-storage reduction versus the float32 parameter representation;
- Isolation Forest test ROC AUC at least 0.80;
- exact Python/C predictions for 15 held-out reference windows and bounded feature parity;
- p99 latency at or below a 1 ms **host-only** engineering budget, with all
  individual budget misses still recorded rather than hidden.

Generated evidence is committed under `artifacts/`, including the dataset/model metrics, ONNX model, SHA-256 manifest, reference vectors, and host timing. `firmware/generated/model_data.h` is the generated deployment header.

## Scope and limitations

Validated here:

- deterministic synthetic data and cycle-level validation;
- host execution on GitHub-hosted Linux runners and the development PC;
- portable C11 and C++20 builds;
- software-in-the-loop-style reference testing.

Not validated here:

- Infineon AURIX, Traveo, PSoC, or any other microcontroller;
- real vehicle, dyno, rig, or production sensor data;
- RTOS scheduling, interrupt behavior, target memory maps, hardware acceleration, or on-target timing;
- AUTOSAR integration, ISO 26262 work products, ASIL qualification, or safety certification;
- diagnostic accuracy, predictive-maintenance fitness, or production deployment.

Any port to an automotive MCU must revalidate numerical parity, memory, latency, toolchain behavior, compiler flags, and safety requirements on the actual target.

## Repository map

- `edge_ai/`: data generation, signal features, training, ONNX, quantization, and metric gates
- `firmware/`: portable C inference, C++ tests, benchmark, and generated model data
- `tests/`: Python unit and pipeline tests
- `artifacts/`: reproducible evidence and model outputs
- `.github/workflows/ci.yml`: clean-run automation

## License

MIT. Synthetic data, source, and generated artifacts in this repository are provided for educational and portfolio use.

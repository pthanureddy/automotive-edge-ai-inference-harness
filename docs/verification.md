# Verification record

Recorded on 2026-09-08 from a clean artifact regeneration and local host build.

## Data and model

| Check | Recorded result |
|---|---:|
| Synthetic windows / simulated cycles | 432 / 72 |
| Train / validation / test cycle overlap | 0 |
| Grouped 5-fold macro-F1, mean | 0.984118 |
| Float test macro-F1 | 1.000000 |
| Higher-noise float macro-F1 | 0.951129 |
| Isolation Forest test ROC AUC | 1.000000 |
| ONNX label mismatches vs scikit-learn | 0 / 72 |
| Quantized-reference test macro-F1 | 1.000000 |
| Parameter-storage reduction | 61.06% |

All accuracy figures are from deterministic synthetic data. They are reproducibility checks, not estimates of performance on a real vehicle population.

## Cross-language and host execution

| Check | Recorded result |
|---|---:|
| Python tests | 10 / 10 passed |
| Python/C held-out reference vectors | 15 / 15 passed |
| Host benchmark iterations | 20,000 |
| Mean / p95 / p99 | 90.68 / 93.90 / 156.20 microseconds |
| Samples above the 1 ms host budget | 1 |

The single 4.36 ms maximum sample is preserved in `artifacts/host_benchmark.json`. The non-real-time development host can be preempted, so the gate applies to p99 while all individual misses remain visible.

The hosted GitHub Actions run is the source of truth for clean Linux builds. Local figures above are machine-specific and must not be presented as MCU latency.

## Explicitly unverified

No test in this repository uses an Infineon AURIX, Traveo, or PSoC target; a vehicle, rig, or production dataset; an RTOS; AUTOSAR; or an ISO 26262 development process. Those remain future porting and validation work.

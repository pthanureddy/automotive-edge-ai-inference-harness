from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE_HZ = 400.0
WINDOW_SAMPLES = 64
CHANNEL_COUNT = 3
CLASS_NAMES = ("normal", "imbalance", "bearing_fault")


@dataclass(frozen=True)
class SensorWindow:
    cycle_id: str
    label: int
    rpm: float
    load: float
    samples: np.ndarray


def _impulse_train(time_s: np.ndarray, fundamental_hz: float, phase: float) -> np.ndarray:
    carrier = np.sin(2.0 * np.pi * 92.0 * time_s + phase)
    envelope = np.maximum(0.0, np.sin(2.0 * np.pi * fundamental_hz * 2.8 * time_s + phase))
    return carrier * envelope**5


def generate_windows(
    *,
    cycles_per_class: int = 24,
    windows_per_cycle: int = 6,
    seed: int = 20260908,
    noise_multiplier: float = 1.0,
    cycle_prefix: str = "base",
) -> list[SensorWindow]:
    """Generate deterministic three-axis vibration windows with cycle-level labels."""
    if cycles_per_class < 5 or windows_per_cycle < 1:
        raise ValueError("cycles_per_class must be >= 5 and windows_per_cycle must be >= 1")
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")

    rng = np.random.default_rng(seed)
    time_s = np.arange(WINDOW_SAMPLES, dtype=np.float64) / SAMPLE_RATE_HZ
    windows: list[SensorWindow] = []

    for label, class_name in enumerate(CLASS_NAMES):
        for cycle_index in range(cycles_per_class):
            cycle_id = f"{cycle_prefix}-{class_name}-{cycle_index:03d}"
            rpm = float((1100, 1800, 2600, 3400)[cycle_index % 4])
            load = float((0.25, 0.50, 0.75, 0.95)[(cycle_index // 2) % 4])
            fundamental_hz = rpm / 60.0
            cycle_phase = rng.uniform(0.0, 2.0 * np.pi)
            noise_sigma = noise_multiplier * (0.018 + 0.012 * load)

            for window_index in range(windows_per_cycle):
                phase = cycle_phase + window_index * 0.31
                base_amplitude = 0.10 + 0.08 * load
                axes = []
                for axis in range(CHANNEL_COUNT):
                    axis_phase = phase + axis * 0.67
                    signal = base_amplitude * np.sin(
                        2.0 * np.pi * fundamental_hz * time_s + axis_phase
                    )
                    signal += 0.035 * np.sin(
                        2.0 * np.pi * fundamental_hz * 2.0 * time_s + 0.4 * axis_phase
                    )

                    if label == 1:
                        radial_gain = (1.0, 2.8, 1.6)[axis]
                        signal += radial_gain * (0.14 + 0.10 * load) * np.sin(
                            2.0 * np.pi * fundamental_hz * time_s + axis_phase + 0.15
                        )
                    elif label == 2:
                        signal += (0.22 + 0.18 * load) * _impulse_train(
                            time_s, fundamental_hz, axis_phase
                        )
                        signal += 0.055 * np.sin(
                            2.0 * np.pi * 78.0 * time_s + 0.2 * axis_phase
                        )

                    signal += rng.normal(0.0, noise_sigma, WINDOW_SAMPLES)
                    axes.append(signal.astype(np.float32))

                windows.append(
                    SensorWindow(
                        cycle_id=cycle_id,
                        label=label,
                        rpm=rpm,
                        load=load,
                        samples=np.stack(axes),
                    )
                )

    return windows


def split_by_cycle(
    windows: list[SensorWindow], *, seed: int = 1729
) -> dict[str, list[SensorWindow]]:
    """Create class-balanced train/validation/test splits without cycle leakage."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[str]] = {index: [] for index in range(len(CLASS_NAMES))}
    for window in windows:
        if window.cycle_id not in by_label[window.label]:
            by_label[window.label].append(window.cycle_id)

    assignment: dict[str, str] = {}
    for label_cycles in by_label.values():
        ordered = np.asarray(sorted(label_cycles), dtype=object)
        rng.shuffle(ordered)
        train_end = max(1, int(round(len(ordered) * 0.70)))
        validation_end = max(train_end + 1, int(round(len(ordered) * 0.85)))
        validation_end = min(validation_end, len(ordered) - 1)
        for cycle_id in ordered[:train_end]:
            assignment[str(cycle_id)] = "train"
        for cycle_id in ordered[train_end:validation_end]:
            assignment[str(cycle_id)] = "validation"
        for cycle_id in ordered[validation_end:]:
            assignment[str(cycle_id)] = "test"

    result = {"train": [], "validation": [], "test": []}
    for window in windows:
        result[assignment[window.cycle_id]].append(window)
    return result


def stack_windows(windows: list[SensorWindow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.stack([window.samples for window in windows]).astype(np.float32)
    labels = np.asarray([window.label for window in windows], dtype=np.int64)
    groups = np.asarray([window.cycle_id for window in windows], dtype=object)
    return raw, labels, groups


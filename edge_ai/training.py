from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from edge_ai.features import extract_features
from edge_ai.synthetic import SensorWindow, split_by_cycle, stack_windows


@dataclass(frozen=True)
class TrainingBundle:
    pipeline: Pipeline
    split_windows: dict[str, list[SensorWindow]]
    split_features: dict[str, np.ndarray]
    split_labels: dict[str, np.ndarray]
    cross_validation_macro_f1: list[float]
    isolation_forest_roc_auc: float


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("standardize", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(12,),
                    activation="relu",
                    solver="lbfgs",
                    alpha=0.001,
                    max_iter=1200,
                    random_state=73,
                ),
            ),
        ]
    )


def fit_bundle(windows: list[SensorWindow]) -> TrainingBundle:
    split_windows = split_by_cycle(windows)
    split_features: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}
    for split_name, values in split_windows.items():
        raw, labels, _ = stack_windows(values)
        split_features[split_name] = extract_features(raw)
        split_labels[split_name] = labels

    pipeline = build_pipeline()
    pipeline.fit(split_features["train"], split_labels["train"])

    train_raw, train_labels, train_groups = stack_windows(split_windows["train"])
    train_features = extract_features(train_raw)
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1103)
    fold_scores = cross_val_score(
        build_pipeline(),
        train_features,
        train_labels,
        groups=train_groups,
        cv=folds,
        scoring="f1_macro",
    )

    standardizer = StandardScaler().fit(train_features)
    normal_mask = train_labels == 0
    detector = IsolationForest(n_estimators=160, contamination="auto", random_state=211)
    detector.fit(standardizer.transform(train_features[normal_mask]))
    test_scores = -detector.score_samples(standardizer.transform(split_features["test"]))
    binary_test_labels = (split_labels["test"] != 0).astype(np.int64)
    isolation_auc = roc_auc_score(binary_test_labels, test_scores)

    return TrainingBundle(
        pipeline=pipeline,
        split_windows=split_windows,
        split_features=split_features,
        split_labels=split_labels,
        cross_validation_macro_f1=[float(value) for value in fold_scores],
        isolation_forest_roc_auc=float(isolation_auc),
    )


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "macro_f1": round(float(f1_score(labels, predictions, average="macro")), 6),
    }


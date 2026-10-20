#evaluation.py
import json
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from src import config


def get_logistic_regression() -> LogisticRegression:
    return LogisticRegression(**config.LOGISTIC_REGRESSION_CONFIG)


def scale_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    print("Scaler fitting confirmation: fitted on train representations only.")
    x_test_scaled = scaler.transform(x_test)
    print("Scaler transformation confirmation: transformed train/test separately; no test fit.")
    return x_train_scaled, x_test_scaled, scaler


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> LogisticRegression:
    classifier = get_logistic_regression()
    print("Logistic Regression parameters:")
    print(json.dumps(config.LOGISTIC_REGRESSION_CONFIG, indent=2))
    classifier.fit(x_train, y_train)
    return classifier


def get_thresholds(scores: np.ndarray) -> np.ndarray:
    """Use unique scores and midpoints, matching the previous notebook."""
    scores = np.asarray(scores).ravel()
    unique_scores = np.sort(np.unique(scores))
    if len(unique_scores) == 1:
        return unique_scores
    mid_points = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    return np.unique(np.concatenate([unique_scores, mid_points]))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> dict[str, Any]:
    labels = labels or [0, 1]
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "precision_class_0": per_class_precision[0],
        "recall_class_0": per_class_recall[0],
        "f1_class_0": per_class_f1[0],
        "precision_class_1": per_class_precision[1],
        "recall_class_1": per_class_recall[1],
        "f1_class_1": per_class_f1[1],
    }


def evaluate_classifier(
    classifier: LogisticRegression,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    y_pred = classifier.predict(x_test)
    probabilities = classifier.predict_proba(x_test)[:, 1] if hasattr(classifier, "predict_proba") else None
    metrics = calculate_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, labels=[0, 1], zero_division=0, output_dict=True)
    report_text = classification_report(y_test, y_pred, labels=[0, 1], zero_division=0)

    print("Final metrics:")
    print(json.dumps(metrics, indent=2))
    print("Confusion matrix:")
    print(cm)
    print("Classification report:")
    print(report_text)

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "classification_report_text": report_text,
        "predictions": y_pred,
        "probabilities": probabilities,
    }


def evaluate_scores_with_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (scores >= threshold).astype(int)
    metrics = calculate_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], zero_division=0, output_dict=True)
    report_text = classification_report(y_true, y_pred, labels=[0, 1], zero_division=0, digits=4)
    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "classification_report": report,
        "classification_report_text": report_text,
        "predictions": y_pred,
        "probabilities": scores,
    }


def select_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    verbose: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Select threshold by macro F1, accuracy, macro recall, then class-1 recall."""
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores).ravel()
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must contain the same number of samples.")
    if len(y_true) == 0:
        raise ValueError("Threshold selection requires at least one sample.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Threshold selection requires binary labels encoded as 0 and 1.")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = y_true[order]
    unique_scores, first_indices = np.unique(sorted_scores, return_index=True)

    total_positive = int(np.sum(y_true == 1))
    total_negative = int(np.sum(y_true == 0))
    tp = total_positive
    fp = total_negative
    fn = 0
    tn = 0

    def metric_row(threshold: float) -> dict[str, Any]:
        precision_0 = tn / (tn + fn) if tn + fn else 0.0
        recall_0 = tn / (tn + fp) if tn + fp else 0.0
        f1_0 = (
            2 * precision_0 * recall_0 / (precision_0 + recall_0)
            if precision_0 + recall_0
            else 0.0
        )
        precision_1 = tp / (tp + fp) if tp + fp else 0.0
        recall_1 = tp / (tp + fn) if tp + fn else 0.0
        f1_1 = (
            2 * precision_1 * recall_1 / (precision_1 + recall_1)
            if precision_1 + recall_1
            else 0.0
        )
        present_class_0 = total_negative > 0 or tn + fn > 0
        present_class_1 = total_positive > 0 or tp + fp > 0
        present_count = int(present_class_0) + int(present_class_1)
        macro_precision = (
            precision_0 * present_class_0 + precision_1 * present_class_1
        ) / present_count
        macro_recall = (
            recall_0 * present_class_0 + recall_1 * present_class_1
        ) / present_count
        macro_f1 = (f1_0 * present_class_0 + f1_1 * present_class_1) / present_count
        return {
            "threshold": float(threshold),
            "accuracy": (tn + tp) / len(y_true),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "precision_class_0": precision_0,
            "recall_class_0": recall_0,
            "f1_class_0": f1_0,
            "precision_class_1": precision_1,
            "recall_class_1": recall_1,
            "f1_class_1": f1_1,
        }

    best = metric_row(unique_scores[0])
    best_key = (
        best["macro_f1"],
        best["accuracy"],
        best["macro_recall"],
        best["recall_class_1"],
    )

    for unique_index in range(len(unique_scores) - 1):
        start = first_indices[unique_index]
        end = first_indices[unique_index + 1]
        removed_labels = sorted_labels[start:end]
        removed_positive = int(np.sum(removed_labels == 1))
        removed_negative = int(np.sum(removed_labels == 0))
        tp -= removed_positive
        fn += removed_positive
        fp -= removed_negative
        tn += removed_negative

        threshold = (unique_scores[unique_index] + unique_scores[unique_index + 1]) / 2.0
        row = metric_row(threshold)
        row_key = (
            row["macro_f1"],
            row["accuracy"],
            row["macro_recall"],
            row["recall_class_1"],
        )
        if row_key > best_key:
            best = row
            best_key = row_key

    if verbose:
        print("Best validation threshold:")
        print(json.dumps(best, indent=2))
    return float(best["threshold"]), best


def run_lr_threshold_evaluation(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[LogisticRegression, StandardScaler, dict[str, Any], dict[str, Any]]:
    """Previous setup: pick threshold on val, then refit fixed LR on train+val."""
    x_train_scaled, x_val_scaled, validation_scaler = scale_train_test(x_train, x_val)
    validation_model = train_logistic_regression(x_train_scaled, y_train)
    val_scores = validation_model.predict_proba(x_val_scaled)[:, 1]
    best_threshold, validation_row = select_best_threshold(y_val, val_scores)

    x_trainval = np.vstack([x_train, x_val])
    y_trainval = np.concatenate([y_train, y_val])

    final_scaler = StandardScaler()
    x_trainval_scaled = final_scaler.fit_transform(x_trainval)
    print("Final scaler fitting confirmation: fitted on train+val only.")
    x_test_scaled = final_scaler.transform(x_test)
    print("Final scaler transformation confirmation: transformed test only; no test fit.")

    final_classifier = get_logistic_regression()
    print("Final Logistic Regression parameters:")
    print(json.dumps(config.LOGISTIC_REGRESSION_CONFIG, indent=2))
    final_classifier.fit(x_trainval_scaled, y_trainval)
    test_scores = final_classifier.predict_proba(x_test_scaled)[:, 1]
    evaluation = evaluate_scores_with_threshold(y_test, test_scores, best_threshold)

    print("Final metrics:")
    print(json.dumps(evaluation["metrics"], indent=2))
    print("Confusion matrix:")
    print(evaluation["confusion_matrix"])
    print("Classification report:")
    print(evaluation["classification_report_text"])

    threshold_info = {
        "best_threshold": best_threshold,
        "validation_selection_row": validation_row,
    }
    return final_classifier, final_scaler, evaluation, threshold_info


def run_lr_evaluation(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[LogisticRegression, StandardScaler, dict[str, Any]]:
    x_train_scaled, x_test_scaled, scaler = scale_train_test(x_train, x_test)
    classifier = train_logistic_regression(x_train_scaled, y_train)
    evaluation = evaluate_classifier(classifier, x_test_scaled, y_test)
    return classifier, scaler, evaluation

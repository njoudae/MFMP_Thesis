# result_manager.py
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config


def ensure_output_directories() -> None:
    config.ensure_project_directories()


def experiment_slug(method_name: str) -> str:
    return method_name.strip().lower().replace(" ", "_")


def result_filename(
    method_name: str,
    descriptor: str,
    extension: str,
    result_suffix: str | None = None,
) -> str:
    parts = [experiment_slug(method_name), descriptor]
    if result_suffix:
        parts.append(experiment_slug(result_suffix))
    return f"{'_'.join(part for part in parts if part)}.{extension}"


def save_metrics(
    method_name: str,
    metrics: dict[str, Any],
    confusion_matrix: np.ndarray,
    classification_report: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> Path:
    ensure_output_directories()
    slug = experiment_slug(method_name)
    row = {"method": method_name, **metrics}
    if metadata:
        row.update(metadata)
    metrics_path = config.METRICS_DIR / f"{slug}_metrics.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)

    json_path = config.METRICS_DIR / f"{slug}_details.json"
    details = {
        "method": method_name,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix.tolist(),
        "classification_report": classification_report,
        "metadata": metadata or {},
    }
    json_path.write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved metric details: {json_path}")
    append_master_result(row)
    return metrics_path


def append_master_result(row: dict[str, Any]) -> Path:
    master_path = config.METRICS_DIR / "master_results.csv"
    new_row = pd.DataFrame([row])
    if master_path.exists():
        existing = pd.read_csv(master_path)
        existing = existing[existing["method"] != row["method"]]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    if "macro_f1" in combined.columns:
        combined = combined.sort_values("macro_f1", ascending=False)
    combined.to_csv(master_path, index=False)
    print(f"Updated master result table: {master_path}")
    return master_path


def save_predictions(
    method_name: str,
    texts,
    y_true,
    y_pred,
    probabilities=None,
    result_suffix: str | None = None,
) -> Path:
    ensure_output_directories()
    data = {
        "original_text": list(texts),
        "true_label": list(y_true),
        "predicted_label": list(y_pred),
    }
    if probabilities is not None:
        data["probability_class_1"] = list(probabilities)
    path = config.PREDICTIONS_DIR / result_filename(
        method_name,
        "predictions",
        "csv",
        result_suffix,
    )
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved predictions: {path}")
    return path


def save_confusion_matrix_plot(
    method_name: str,
    confusion_matrix: np.ndarray,
    result_suffix: str | None = None,
) -> Path:
    ensure_output_directories()
    path = config.PLOTS_DIR / result_filename(
        method_name,
        "confusion_matrix",
        "png",
        result_suffix,
    )
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(confusion_matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_title(f"Confusion Matrix: {method_name}")
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix plot: {path}")
    return path


def save_execution_log(
    method_name: str,
    log_data: dict[str, Any],
    result_suffix: str | None = None,
) -> Path:
    ensure_output_directories()
    path = config.LOGS_DIR / result_filename(
        method_name,
        "execution_log",
        "json",
        result_suffix,
    )
    payload = {"saved_at": datetime.now().isoformat(timespec="seconds"), **log_data}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved execution log: {path}")
    return path


def save_experiment_metadata(
    method_name: str,
    metadata: dict[str, Any],
    result_suffix: str | None = None,
) -> Path:
    ensure_output_directories()
    path = config.LOGS_DIR / result_filename(
        method_name,
        "metadata",
        "json",
        result_suffix,
    )
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Saved experiment metadata: {path}")
    return path


def save_classifier_search_results(
    method_name: str,
    fold_results: pd.DataFrame,
    summary: pd.DataFrame,
    best_per_classifier: pd.DataFrame,
    test_results: pd.DataFrame,
    result_suffix: str | None = None,
) -> dict[str, Path]:
    """Save every CV fold, ranked parameter sets, and final held-out results."""
    ensure_output_directories()
    paths = {
        "fold_results": config.METRICS_DIR
        / result_filename(method_name, "cv_folds", "csv", result_suffix),
        "summary": config.METRICS_DIR
        / result_filename(method_name, "cv_summary", "csv", result_suffix),
        "best_per_classifier": config.METRICS_DIR
        / result_filename(method_name, "best_parameters", "csv", result_suffix),
        "test_results": config.METRICS_DIR
        / result_filename(method_name, "test_results", "csv", result_suffix),
        "overall_best": config.METRICS_DIR
        / result_filename(method_name, "overall_best", "csv", result_suffix),
        "excel": config.METRICS_DIR
        / result_filename(method_name, "classifier_search", "xlsx", result_suffix),
    }
    fold_results.to_csv(paths["fold_results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    best_per_classifier.to_csv(paths["best_per_classifier"], index=False)
    test_results.to_csv(paths["test_results"], index=False)
    test_results.head(1).to_csv(paths["overall_best"], index=False)

    with pd.ExcelWriter(paths["excel"]) as writer:
        fold_results.to_excel(writer, sheet_name="CV_Folds", index=False)
        summary.to_excel(writer, sheet_name="CV_Summary", index=False)
        best_per_classifier.to_excel(writer, sheet_name="Best_Parameters", index=False)
        test_results.to_excel(writer, sheet_name="Test_Results", index=False)
        test_results.head(1).to_excel(writer, sheet_name="Overall_Best", index=False)

    for test_row in test_results.to_dict(orient="records"):
        master_row = dict(test_row)
        master_parts = [method_name, str(test_row["classifier"])]
        if result_suffix:
            master_parts.append(result_suffix)
        master_row["method"] = "::".join(master_parts)
        append_master_result(master_row)

    print(f"Saved all CV folds: {paths['fold_results']}")
    print(f"Saved ranked CV summary: {paths['summary']}")
    print(f"Saved best parameters: {paths['best_per_classifier']}")
    print(f"Saved held-out test results: {paths['test_results']}")
    print(f"Saved combined Excel workbook: {paths['excel']}")
    return paths


def load_all_metrics() -> pd.DataFrame:
    metric_files = sorted(config.METRICS_DIR.glob("*_metrics.csv"))
    frames = [pd.read_csv(path) for path in metric_files if path.name != "master_results.csv"]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "macro_f1" in combined.columns:
        combined = combined.sort_values("macro_f1", ascending=False)
    return combined

#classifier_search.py

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from src import config
from src.evaluation import calculate_metrics, select_best_threshold


def _build_estimator(classifier_name: str) -> tuple[BaseEstimator, str]:
    if classifier_name == "LR":
        classifier = LogisticRegression(random_state=config.RANDOM_SEED)
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)]), "classifier__"
    if classifier_name == "SVM":
        classifier = LinearSVC(
            dual=False,
            random_state=config.RANDOM_SEED,
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)]), "classifier__"
    if classifier_name == "NB":
        return Pipeline([("scaler", StandardScaler()), ("classifier", GaussianNB())]), "classifier__"
    if classifier_name == "KNN":
        classifier = KNeighborsClassifier(n_jobs=-1)
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)]), "classifier__"
    if classifier_name == "DT":
        classifier = DecisionTreeClassifier(random_state=config.RANDOM_SEED)
        return classifier, ""
    if classifier_name in {"RF", "RFT"}:
        classifier = RandomForestClassifier(
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        return classifier, ""
    raise KeyError(f"Unknown classifier: {classifier_name}")


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return {
        "tn": int(np.sum((true == 0) & (pred == 0))),
        "fp": int(np.sum((true == 0) & (pred == 1))),
        "fn": int(np.sum((true == 1) & (pred == 0))),
        "tp": int(np.sum((true == 1) & (pred == 1))),
        "correct": int(np.sum(true == pred)),
        "incorrect": int(np.sum(true != pred)),
    }


def _parameter_columns(parameters: dict[str, Any]) -> dict[str, Any]:
    return {f"param_{name}": value for name, value in parameters.items()}


def _json_parameters(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, ensure_ascii=False)


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return {
        **calculate_metrics(y_true, y_pred),
        **_confusion_counts(y_true, y_pred),
    }


def _positive_class_scores(
    estimator: BaseEstimator,
    x_values: np.ndarray,
) -> tuple[np.ndarray, str]:
    if hasattr(estimator, "predict_proba"):
        classes = np.asarray(estimator.classes_)
        positive_indices = np.flatnonzero(classes == 1)
        if len(positive_indices) != 1:
            raise ValueError(f"Could not identify positive class 1 in classes {classes.tolist()}.")
        scores = estimator.predict_proba(x_values)[:, int(positive_indices[0])]
        return np.asarray(scores).ravel(), "probability_class_1"
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(x_values)).ravel(), "decision_score"
    raise TypeError(f"{type(estimator).__name__} does not expose probability or decision scores.")


def _summarize_folds(fold_results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = list(config.REQUIRED_METRIC_COLUMNS)
    count_columns = ["tn", "fp", "fn", "tp", "correct", "incorrect"]
    parameter_columns = [column for column in fold_results if column.startswith("param_")]
    group_columns = ["method", "classifier", "parameter_set", "parameters"]
    rows: list[dict[str, Any]] = []

    for group_values, group in fold_results.groupby(group_columns, sort=False, dropna=False):
        row = dict(zip(group_columns, group_values))
        row["fold_count"] = int(len(group))
        for column in parameter_columns:
            non_null = group[column].dropna()
            row[column] = non_null.iloc[0] if not non_null.empty else None
        row["score_type"] = group["score_type"].iloc[0]
        row["fold_threshold_mean"] = float(group["best_threshold"].mean())
        row["fold_threshold_std"] = float(group["best_threshold"].std(ddof=1))
        row["oof_best_threshold"] = float(group["oof_best_threshold"].iloc[0])
        for column in metric_columns:
            row[f"oof_{column}"] = float(group[f"oof_{column}"].iloc[0])
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1))
        for column in count_columns:
            row[f"{column}_sum"] = int(group[column].sum())
            row[f"oof_{column}"] = int(group[f"oof_{column}"].iloc[0])
        rows.append(row)

    summary = pd.DataFrame(rows)
    selection_column = f"{config.CV_SELECTION_METRIC}_mean"
    return summary.sort_values(
        [selection_column, "accuracy_mean", "macro_recall_mean"],
        ascending=False,
    ).reset_index(drop=True)


def _select_best_per_classifier(summary: pd.DataFrame) -> pd.DataFrame:
    best = (
        summary.sort_values(
            [f"{config.CV_SELECTION_METRIC}_mean", "accuracy_mean", "macro_recall_mean"],
            ascending=False,
        )
        .groupby(["method", "classifier"], sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.insert(0, "rank", np.arange(1, len(best) + 1))
    return best


def _fit_with_parameters(classifier_name: str, parameters: dict[str, Any]) -> BaseEstimator:
    if classifier_name == "SVM" and parameters.get("kernel") == "rbf":
        svm_parameters = dict(parameters)
        svm_parameters.pop("kernel")
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        probability=False,
                        cache_size=2048,
                        random_state=config.RANDOM_SEED,
                    ),
                ),
            ]
        )
        estimator.set_params(
            **{f"classifier__{name}": value for name, value in svm_parameters.items()}
        )
        return estimator

    estimator, prefix = _build_estimator(classifier_name)
    estimator.set_params(**{f"{prefix}{name}": value for name, value in parameters.items()})
    return estimator


def run_classifier_cv_search(
    method_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Search all configured classifiers with shared 10-fold stratified splits."""
    labels, counts = np.unique(y_train, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f"Binary labels are required; found labels {labels.tolist()}.")
    if int(counts.min()) < config.CV_N_SPLITS:
        raise ValueError(
            f"The smallest class has {int(counts.min())} samples, fewer than "
            f"CV_N_SPLITS={config.CV_N_SPLITS}."
        )

    splitter = StratifiedKFold(
        n_splits=config.CV_N_SPLITS,
        shuffle=config.CV_SHUFFLE,
        random_state=config.RANDOM_SEED if config.CV_SHUFFLE else None,
    )
    folds = list(splitter.split(x_train, y_train))
    fold_rows: list[dict[str, Any]] = []
    total_fits = sum(
        len(list(ParameterGrid(grid))) * config.CV_N_SPLITS
        for grid in config.CLASSIFIER_PARAM_GRIDS.values()
    )
    completed_fits = 0
    search_start = time.perf_counter()

    for classifier_name in config.CLASSIFIER_PARAM_GRIDS:
        grid = config.CLASSIFIER_PARAM_GRIDS[classifier_name]
        parameter_sets = list(ParameterGrid(grid))
        print(f"\n{classifier_name}: {len(parameter_sets)} parameter sets")
        for parameter_set, parameters in enumerate(parameter_sets, start=1):
            print(
                f"{classifier_name} parameter set {parameter_set}/{len(parameter_sets)}: "
                f"{_json_parameters(parameters)}"
            )
            base_estimator = _fit_with_parameters(classifier_name, parameters)
            oof_scores = np.empty(len(y_train), dtype=float)
            parameter_fold_rows: list[dict[str, Any]] = []
            for fold_number, (train_index, validation_index) in enumerate(folds, start=1):
                fold_start = time.perf_counter()
                estimator = clone(base_estimator)
                estimator.fit(x_train[train_index], y_train[train_index])
                validation_scores, score_type = _positive_class_scores(
                    estimator,
                    x_train[validation_index],
                )
                oof_scores[validation_index] = validation_scores
                best_threshold, _ = select_best_threshold(
                    y_train[validation_index],
                    validation_scores,
                    verbose=False,
                )
                predictions = (validation_scores >= best_threshold).astype(int)
                scores = _score_predictions(y_train[validation_index], predictions)
                fold_seconds = time.perf_counter() - fold_start
                completed_fits += 1
                elapsed_seconds = time.perf_counter() - search_start
                average_seconds = elapsed_seconds / completed_fits
                remaining_seconds = average_seconds * (total_fits - completed_fits)
                parameter_fold_rows.append(
                    {
                        "method": method_name,
                        "classifier": classifier_name,
                        "parameter_set": parameter_set,
                        "fold": fold_number,
                        "train_size": int(len(train_index)),
                        "validation_size": int(len(validation_index)),
                        "parameters": _json_parameters(parameters),
                        "score_type": score_type,
                        "best_threshold": best_threshold,
                        "fit_seconds": fold_seconds,
                        **_parameter_columns(parameters),
                        **scores,
                    }
                )
                print(
                    f"  fold {fold_number}/{config.CV_N_SPLITS}: "
                    f"macro_f1={scores['macro_f1']:.6f}, "
                    f"accuracy={scores['accuracy']:.6f}, "
                    f"threshold={best_threshold:.6f}, "
                    f"time={fold_seconds / 60:.2f} min, "
                    f"progress={completed_fits}/{total_fits}, "
                    f"ETA={remaining_seconds / 3600:.2f} h"
                )

            oof_best_threshold, oof_threshold_row = select_best_threshold(
                y_train,
                oof_scores,
                verbose=False,
            )
            oof_predictions = (oof_scores >= oof_best_threshold).astype(int)
            oof_scores_summary = _score_predictions(y_train, oof_predictions)
            for row in parameter_fold_rows:
                row["oof_best_threshold"] = oof_best_threshold
                row.update(
                    {
                        f"oof_{name}": value
                        for name, value in oof_scores_summary.items()
                    }
                )
            fold_rows.extend(parameter_fold_rows)
            print(
                f"  OOF threshold={oof_best_threshold:.6f}, "
                f"OOF macro_f1={oof_threshold_row['macro_f1']:.6f}"
            )

    fold_results = pd.DataFrame(fold_rows)
    summary = _summarize_folds(fold_results)
    best_per_classifier = _select_best_per_classifier(summary)

    test_rows: list[dict[str, Any]] = []
    prediction_results: dict[str, dict[str, Any]] = {}
    for _, best_row in best_per_classifier.iterrows():
        classifier_name = str(best_row["classifier"])
        parameters = json.loads(best_row["parameters"])
        estimator = _fit_with_parameters(classifier_name, parameters)
        estimator.fit(x_train, y_train)
        test_scores, score_type = _positive_class_scores(estimator, x_test)
        best_threshold = float(best_row["oof_best_threshold"])
        predictions = (test_scores >= best_threshold).astype(int)
        scores = _score_predictions(y_test, predictions)
        test_row = {
            "method": method_name,
            "classifier": classifier_name,
            "parameters": best_row["parameters"],
            "cv_macro_f1_mean": best_row["macro_f1_mean"],
            "cv_macro_f1_std": best_row["macro_f1_std"],
            "score_type": score_type,
            "best_threshold": best_threshold,
            "threshold_source": "pooled_out_of_fold_train_scores",
            "oof_macro_f1": best_row["oof_macro_f1"],
            **_parameter_columns(parameters),
            **scores,
        }
        test_rows.append(test_row)
        prediction_results[classifier_name] = {
            "estimator": estimator,
            "predictions": predictions,
            "scores": test_scores,
            "best_threshold": best_threshold,
            "confusion_matrix": np.array(
                [[scores["tn"], scores["fp"]], [scores["fn"], scores["tp"]]],
                dtype=int,
            ),
        }
        print(
            f"TEST {classifier_name}: threshold={best_threshold:.6f}, "
            f"macro_f1={scores['macro_f1']:.6f}, accuracy={scores['accuracy']:.6f}, "
            f"TN={scores['tn']}, FP={scores['fp']}, FN={scores['fn']}, TP={scores['tp']}"
        )

    test_results = pd.DataFrame(test_rows).sort_values(
        ["cv_macro_f1_mean", "cv_macro_f1_std"],
        ascending=[False, True],
    ).reset_index(drop=True)
    test_results.insert(0, "cv_rank", np.arange(1, len(test_results) + 1))
    test_results["test_rank"] = (
        test_results[config.CV_SELECTION_METRIC]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    overall_best = test_results.iloc[0].to_dict()
    print(
        "CV-selected overall result: "
        f"classifier={overall_best['classifier']}, "
        f"threshold={overall_best['best_threshold']:.6f}, "
        f"test_macro_f1={overall_best['macro_f1']:.6f}"
    )

    return {
        "fold_results": fold_results,
        "summary": summary,
        "best_per_classifier": best_per_classifier,
        "test_results": test_results,
        "overall_best": overall_best,
        "prediction_results": prediction_results,
    }

import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from src import config
from src.classifier_search import run_classifier_cv_search
from src.data_utils import load_prepared_train_test, set_global_seed
from src.embedding_utils import clear_memory, load_marbertv2
from src.pooling_methods import (
    dimension_mean_plus_std_zone_pooling_with_stats,
    dimension_mean_std_zone_pooling_with_stats,
    dimension_mean_zone_pooling_with_stats,
    dimension_std_zone_pooling_with_stats,
    mean_pooling,
)
from src.result_manager import (
    save_classifier_search_results,
    save_confusion_matrix_plot,
    save_execution_log,
    save_experiment_metadata,
    save_predictions,
)


DIM_ZONE_K_VALUES = [0.5, 1, 1.5, 2, 2.5, 3]
DIM_ZONE_METHOD_PREFIXES = {
    "dim_mean_zone_": "mean",
    "dim_std_zone_": "std",
    "dim_mean_std_zone_": "mean_std",
    "dim_mean_plus_std_zone_": "mean_plus_std",
}


def _k_suffix(k_value: float) -> str:
    if float(k_value).is_integer():
        return str(int(k_value))
    return str(k_value).replace(".", "_")


DIM_ZONE_METHODS = [
    "mean_baseline",
    *[f"dim_mean_zone_{_k_suffix(k)}std" for k in DIM_ZONE_K_VALUES],
    *[f"dim_std_zone_{_k_suffix(k)}std" for k in DIM_ZONE_K_VALUES],
    *[f"dim_mean_std_zone_{_k_suffix(k)}std" for k in DIM_ZONE_K_VALUES],
]
DIM_MEAN_PLUS_STD_ZONE_METHODS = [f"dim_mean_plus_std_zone_{_k_suffix(k)}std" for k in DIM_ZONE_K_VALUES]
DIM_ZONE_AVAILABLE_METHODS = [*DIM_ZONE_METHODS, *DIM_MEAN_PLUS_STD_ZONE_METHODS]


def _empty_stats() -> dict[str, float]:
    return {
        "total_values": 0,
        "kept_values": 0,
        "removed_values": 0,
        "sentence_dimension_count": 0,
        "fallback_count": 0,
    }


def _combine_stats(target: dict[str, float], update: dict[str, float]) -> None:
    for key in target:
        target[key] += update.get(key, 0)


def _finalize_stats(stats: dict[str, float]) -> dict[str, float]:
    total_values = max(stats["total_values"], 1)
    sentence_dimension_count = max(stats["sentence_dimension_count"], 1)
    return {
        **stats,
        "average_values_kept_per_dimension": stats["kept_values"] / sentence_dimension_count,
        "average_percentage_removed_values": (stats["removed_values"] / total_values) * 100,
    }


def _method_info(method_name: str) -> tuple[str, float | None]:
    if method_name == "mean_baseline":
        return "mean", None
    suffix = "std"
    for prefix, pooling_mode in DIM_ZONE_METHOD_PREFIXES.items():
        if method_name.startswith(prefix) and method_name.endswith(suffix):
            k_text = method_name.removeprefix(prefix).removesuffix(suffix).replace("_", ".")
            return pooling_mode, float(k_text)
    raise ValueError(f"Unknown dimension-zone method: {method_name}")


def _method_k(method_name: str) -> float | None:
    return _method_info(method_name)[1]


def _baseline_mean_stats(attention_mask: torch.Tensor, hidden_size: int) -> dict[str, float]:
    valid_tokens = int(attention_mask.bool().sum().item())
    total_values = valid_tokens * hidden_size
    sentence_dimension_count = int(attention_mask.shape[0] * hidden_size)
    return {
        "total_values": total_values,
        "kept_values": total_values,
        "removed_values": 0,
        "sentence_dimension_count": sentence_dimension_count,
        "fallback_count": 0,
    }


def extract_dimension_zone_representations(
    texts,
    tokenizer,
    model,
    device: torch.device,
    method_name: str,
    max_len: int = config.MAX_LEN,
    batch_size: int = config.BATCH_SIZE,
) -> tuple[np.ndarray, dict[str, float]]:
    """Extract final sentence vectors only; token embeddings are never saved."""
    text_list = [str(text) for text in texts]
    pooling_mode, k_value = _method_info(method_name)
    stats = _empty_stats()
    all_vectors: list[np.ndarray] = []

    print(f"Representation method name: {method_name}")
    print(f"Pooling mode: {pooling_mode}")
    print(f"K value: {k_value}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_len}")

    with torch.no_grad():
        for start in range(0, len(text_list), batch_size):
            batch_texts = text_list[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden_states = outputs.last_hidden_state

            if method_name == "mean_baseline":
                pooled = mean_pooling(hidden_states, encoded["attention_mask"])
                batch_stats = _baseline_mean_stats(encoded["attention_mask"], hidden_states.shape[-1])
            else:
                pooling_functions = {
                    "mean": dimension_mean_zone_pooling_with_stats,
                    "std": dimension_std_zone_pooling_with_stats,
                    "mean_std": dimension_mean_std_zone_pooling_with_stats,
                    "mean_plus_std": dimension_mean_plus_std_zone_pooling_with_stats,
                }
                pooled, batch_stats = pooling_functions[pooling_mode](
                    hidden_states,
                    encoded["attention_mask"],
                    k_value,
                )

            all_vectors.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
            _combine_stats(stats, batch_stats)

            del encoded, outputs, hidden_states, pooled
            clear_memory()

    representations = np.vstack(all_vectors).astype(np.float32, copy=False)
    finalized_stats = _finalize_stats(stats)
    print(f"Representation shape: {representations.shape}")
    print(f"Estimated memory size: {representations.nbytes / (1024**2):.2f} MB")
    print(f"Average values kept per dimension: {finalized_stats['average_values_kept_per_dimension']:.4f}")
    print(f"Average percentage of removed values: {finalized_stats['average_percentage_removed_values']:.4f}%")
    print(f"Fallback count: {int(finalized_stats['fallback_count'])}")
    return representations, finalized_stats


def run_dimension_zone_experiment(method_name: str) -> dict[str, Any]:
    start_time = time.time()
    config.ensure_project_directories()
    set_global_seed(config.RANDOM_SEED)

    if method_name not in DIM_ZONE_AVAILABLE_METHODS:
        raise ValueError(f"Unknown method {method_name!r}. Available: {DIM_ZONE_AVAILABLE_METHODS}")

    print("=" * 70)
    print(f"Experiment: {method_name}")
    print(f"Pooling mode: {_method_info(method_name)[0]}")
    print(f"K value: {_method_k(method_name)}")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"MAX_LEN: {config.MAX_LEN}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Cross-validation folds: {config.CV_N_SPLITS}")
    print(f"Classifiers: {list(config.CLASSIFIER_PARAM_GRIDS)}")
    print(f"Cleaning version: {config.CLEANING_VERSION}")
    print(f"Model name: {config.MARBERTV2_MODEL_NAME}")
    print(f"Local model path: {config.MARBERTV2_LOCAL_DIR}")
    print("=" * 70)

    train_df, test_df, text_column, label_column = load_prepared_train_test()
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Official train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print("Train labels:")
    print(train_df[label_column].value_counts().sort_index())
    print("Test labels:")
    print(test_df[label_column].value_counts().sort_index())

    tokenizer, model, device, model_source = load_marbertv2()

    x_train, train_filter_stats = extract_dimension_zone_representations(
        train_df[text_column].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        method_name=method_name,
    )
    x_test, test_filter_stats = extract_dimension_zone_representations(
        test_df[text_column].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        method_name=method_name,
    )

    y_train = train_df[label_column].to_numpy()
    y_test = test_df[label_column].to_numpy()

    search_results = run_classifier_cv_search(
        method_name=method_name,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    elapsed_seconds = time.time() - start_time
    print(f"Execution time: {elapsed_seconds:.2f} seconds")

    metadata = {
        "experiment_family": "dimension_zone",
        "zone_pooling_mode": _method_info(method_name)[0],
        "k_value": _method_k(method_name),
        "text_column": text_column,
        "label_column": label_column,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "representation_train_shape": x_train.shape,
        "representation_test_shape": x_test.shape,
        "representation_train_memory_mb": x_train.nbytes / (1024**2),
        "representation_test_memory_mb": x_test.nbytes / (1024**2),
        "train_filter_stats": train_filter_stats,
        "test_filter_stats": test_filter_stats,
        "batch_size": config.BATCH_SIZE,
        "max_len": config.MAX_LEN,
        "cv_n_splits": config.CV_N_SPLITS,
        "cv_selection_metric": config.CV_SELECTION_METRIC,
        "classifier_param_grids": config.CLASSIFIER_PARAM_GRIDS,
        "model_source": model_source,
        "model_name": config.MARBERTV2_MODEL_NAME,
        "cleaning_version": config.CLEANING_VERSION,
        "execution_time_seconds": elapsed_seconds,
    }

    original_text_column = config.TEXT_COLUMN if config.TEXT_COLUMN in test_df.columns else text_column
    save_classifier_search_results(
        method_name=method_name,
        fold_results=search_results["fold_results"],
        summary=search_results["summary"],
        best_per_classifier=search_results["best_per_classifier"],
        test_results=search_results["test_results"],
    )
    for classifier_name, prediction_result in search_results["prediction_results"].items():
        result_name = f"{method_name}_{classifier_name}"
        save_predictions(
            method_name=result_name,
            texts=test_df[original_text_column].tolist(),
            y_true=y_test,
            y_pred=prediction_result["predictions"],
            probabilities=prediction_result["scores"],
        )
        save_confusion_matrix_plot(result_name, prediction_result["confusion_matrix"])

    save_experiment_metadata(method_name, metadata)
    save_execution_log(
        method_name,
        {
            "method": method_name,
            "overall_best": search_results["overall_best"],
            "test_results": search_results["test_results"].to_dict(orient="records"),
            "metadata": metadata,
        },
    )

    result_row = {**search_results["overall_best"], **metadata}
    print("Summary-ready result row:")
    print(result_row)
    return result_row


def run_dimension_zone_experiment_family() -> pd.DataFrame:
    rows = []
    for method_name in DIM_ZONE_METHODS:
        rows.append(run_dimension_zone_experiment(method_name))

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary_path = config.METRICS_DIR / "dimension_zone_family_summary.csv"
    results.to_csv(summary_path, index=False)
    print(f"Saved dimension-zone family summary: {summary_path}")

    fig_path = config.PLOTS_DIR / "dimension_zone_macro_f1.png"
    ax = results.plot.bar(x="method", y="macro_f1", legend=False, figsize=(11, 4.5), rot=35)
    ax.set_title("Dimension-Zone Macro F1")
    ax.set_ylabel("macro_f1")
    ax.set_ylim(0, 1)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    print(f"Saved dimension-zone macro F1 plot: {fig_path}")
    return results


def run_dimension_mean_plus_std_zone_experiment_family() -> pd.DataFrame:
    rows = []
    for method_name in DIM_MEAN_PLUS_STD_ZONE_METHODS:
        rows.append(run_dimension_zone_experiment(method_name))

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary_path = config.METRICS_DIR / "dimension_mean_plus_std_zone_family_summary.csv"
    results.to_csv(summary_path, index=False)
    print(f"Saved dimension mean_plus_std-zone family summary: {summary_path}")

    fig_path = config.PLOTS_DIR / "dimension_mean_plus_std_zone_macro_f1.png"
    ax = results.plot.bar(x="method", y="macro_f1", legend=False, figsize=(9, 4.5), rot=35)
    ax.set_title("Dimension Mean Plus STD-Zone Macro F1")
    ax.set_ylabel("macro_f1")
    ax.set_ylim(0, 1)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    print(f"Saved dimension mean_plus_std-zone macro F1 plot: {fig_path}")
    return results

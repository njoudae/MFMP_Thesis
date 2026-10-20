import time
from typing import Any

import pandas as pd

from src import config
from src.classifier_search import run_classifier_cv_search
from src.data_utils import load_prepared_train_test, set_global_seed
from src.embedding_utils import extract_sentence_representations, load_marbertv2
from src.pooling_methods import get_pooling_method
from src.result_manager import (
    save_classifier_search_results,
    save_confusion_matrix_plot,
    save_execution_log,
    save_experiment_metadata,
    save_predictions,
)


def print_experiment_settings(method_name: str) -> None:
    print("=" * 70)
    print(f"Experiment: {method_name}")
    print(f"Random seed: {config.RANDOM_SEED}")
    print(f"MAX_LEN: {config.MAX_LEN}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Cross-validation folds: {config.CV_N_SPLITS}")
    print(f"Classifiers: {list(config.CLASSIFIER_PARAM_GRIDS)}")
    print(f"Cleaning version: {config.CLEANING_VERSION}")
    print(f"CACHE_REPRESENTATIONS: {config.CACHE_REPRESENTATIONS}")
    print(f"Model name: {config.MARBERTV2_MODEL_NAME}")
    print(f"Local model path: {config.MARBERTV2_LOCAL_DIR}")
    print("=" * 70)


def run_pooling_experiment(
    method_name: str,
    result_suffix: str | None = None,
) -> dict[str, Any]:
    """Run one MARBERTv2 pooling experiment with classifier grid search."""
    start_time = time.time()
    config.ensure_project_directories()
    set_global_seed(config.RANDOM_SEED)
    print_experiment_settings(method_name)

    train_df, test_df, text_column, label_column = load_prepared_train_test()
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Official train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print("Train labels:")
    print(train_df[label_column].value_counts().sort_index())
    print("Test labels:")
    print(test_df[label_column].value_counts().sort_index())
    pooling_fn = get_pooling_method(method_name)

    tokenizer, model, device, model_source = load_marbertv2()

    x_train = extract_sentence_representations(
        train_df[text_column].tolist(),
        tokenizer=tokenizer,
        model=model,
        pooling_fn=pooling_fn,
        method_name=method_name,
        split_name="train",
        max_len=config.MAX_LEN,
        batch_size=config.BATCH_SIZE,
        device=device,
        use_cache=config.CACHE_REPRESENTATIONS,
    )
    x_test = extract_sentence_representations(
        test_df[text_column].tolist(),
        tokenizer=tokenizer,
        model=model,
        pooling_fn=pooling_fn,
        method_name=method_name,
        split_name="test",
        max_len=config.MAX_LEN,
        batch_size=config.BATCH_SIZE,
        device=device,
        use_cache=config.CACHE_REPRESENTATIONS,
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
        "text_column": text_column,
        "label_column": label_column,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "representation_train_shape": x_train.shape,
        "representation_test_shape": x_test.shape,
        "representation_train_memory_mb": x_train.nbytes / (1024**2),
        "representation_test_memory_mb": x_test.nbytes / (1024**2),
        "batch_size": config.BATCH_SIZE,
        "max_len": config.MAX_LEN,
        "cv_n_splits": config.CV_N_SPLITS,
        "cv_selection_metric": config.CV_SELECTION_METRIC,
        "classifier_param_grids": config.CLASSIFIER_PARAM_GRIDS,
        "model_source": model_source,
        "model_name": config.MARBERTV2_MODEL_NAME,
        "cleaning_version": config.CLEANING_VERSION,
        "cache_representations": config.CACHE_REPRESENTATIONS,
        "execution_time_seconds": elapsed_seconds,
    }

    original_text_column = config.TEXT_COLUMN if config.TEXT_COLUMN in test_df.columns else text_column
    save_classifier_search_results(
        method_name=method_name,
        fold_results=search_results["fold_results"],
        summary=search_results["summary"],
        best_per_classifier=search_results["best_per_classifier"],
        test_results=search_results["test_results"],
        result_suffix=result_suffix,
    )
    single_classifier = len(search_results["prediction_results"]) == 1
    for classifier_name, prediction_result in search_results["prediction_results"].items():
        result_name = method_name if single_classifier else f"{method_name}_{classifier_name}"
        save_predictions(
            method_name=result_name,
            texts=test_df[original_text_column].tolist(),
            y_true=y_test,
            y_pred=prediction_result["predictions"],
            probabilities=prediction_result["scores"],
            result_suffix=result_suffix,
        )
        save_confusion_matrix_plot(
            result_name,
            prediction_result["confusion_matrix"],
            result_suffix=result_suffix,
        )

    save_experiment_metadata(method_name, metadata, result_suffix=result_suffix)
    save_execution_log(
        method_name,
        {
            "method": method_name,
            "overall_best": search_results["overall_best"],
            "test_results": search_results["test_results"].to_dict(orient="records"),
            "metadata": metadata,
        },
        result_suffix=result_suffix,
    )

    result_row = {**search_results["overall_best"], **metadata}
    print("Summary-ready result row:")
    print(result_row)
    return result_row


def run_pooling_experiment_suite(
    method_names: list[str],
    suite_name: str,
    plot_title: str | None = None,
    result_suffix: str | None = None,
) -> pd.DataFrame:
    """Run pooling methods with the shared classifier CV-search workflow."""
    rows = []
    for method_name in method_names:
        rows.append(run_pooling_experiment(method_name, result_suffix=result_suffix))

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    summary_suffix = f"_{result_suffix}" if result_suffix else ""
    summary_path = config.METRICS_DIR / f"{suite_name}_summary{summary_suffix}.csv"
    results.to_csv(summary_path, index=False)
    print(f"Saved suite summary: {summary_path}")

    fig_path = config.PLOTS_DIR / f"{suite_name}_macro_f1{summary_suffix}.png"
    ax = results.plot.bar(x="method", y="macro_f1", legend=False, figsize=(9, 4.5), rot=35)
    ax.set_title(plot_title or f"{suite_name} Macro F1")
    ax.set_ylabel("macro_f1")
    ax.set_ylim(0, 1)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    print(f"Saved suite macro F1 plot: {fig_path}")
    return results

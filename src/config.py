#config.py
from pathlib import Path


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = BASE_DIR / "models"
MARBERTV2_LOCAL_DIR = MODELS_DIR / "MARBERTv2"
MARBERTV2_MODEL_NAME = "UBC-NLP/MARBERTv2"

RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots"
LOGS_DIR = RESULTS_DIR / "logs"
CACHED_REPRESENTATIONS_DIR = RESULTS_DIR / "cached_representations"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"


# ---------------------------------------------------------------------------
# Data configuration
# ---------------------------------------------------------------------------
# Set these explicitly once your dataset columns are known. If left as None,
# data_utils.detect_text_label_columns will choose likely columns and print them.
DATASET_PATH = None
TEXT_COLUMN = "tweet"
LABEL_COLUMN = "sarcasm"

# ArSarcasm-v2 provides an official split. Experiments should reuse it when
# these files are present to avoid split drift between pooling methods.
USE_OFFICIAL_ARSARCASM_SPLIT = True
ARSARCASM_TRAIN_PATH = RAW_DATA_DIR / "ArSarcasm-v2" / "training_data.csv"
ARSARCASM_TEST_PATH = RAW_DATA_DIR / "ArSarcasm-v2" / "testing_data.csv"

# Cleaning versions from the supplied pipeline:
#   keep   -> keep Arabic + emoji + ! and ؟
#   remove -> keep Arabic only
CLEANING_VERSION = "keep"
CLEAN_KEEP_COLUMN = "clean_keep"
CLEAN_REMOVE_COLUMN = "clean_noemoji_nopunct"

PREFERRED_TEXT_COLUMNS = [
    "text",
    "tweet",
    "content",
    "sentence",
    "review",
    "comment",
    "arabic_text",
]
PREFERRED_LABEL_COLUMNS = [
    "sarcasm",
    "label",
    "target",
    "class",
    "sentiment",
    "category",
    "y",
]

TEST_SIZE = 0.20
VAL_SIZE = 0.15
SPLIT_TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
SPLIT_TEST_PATH = PROCESSED_DATA_DIR / "test.csv"
ROW_REMOVAL_LOG_PATH = LOGS_DIR / "row_removal_log.csv"

# Match the previous notebook setup: keep empty cleaned texts as "" instead of
# removing rows after cleaning.
DROP_EMPTY_TEXT_AFTER_CLEANING = False


# ---------------------------------------------------------------------------
# Reproducibility and model settings
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
MAX_LEN = 320
BATCH_SIZE = 32

# Default is intentionally off. Enable only when final sentence representations
# are small enough and useful to reuse.
CACHE_REPRESENTATIONS = False
MAX_CACHE_SIZE_MB = 512


# ---------------------------------------------------------------------------
# Classifier search and cross-validation
# ---------------------------------------------------------------------------
CV_N_SPLITS = 10
CV_SHUFFLE = True
CV_SELECTION_METRIC = "macro_f1"
C_VALUES = [0.001]

# Grid values are lists because sklearn.model_selection.ParameterGrid expands
# each dictionary into the parameter combinations evaluated on all folds.
CLASSIFIER_PARAM_GRIDS = {
    # Temporarily disabled classifiers. Uncomment any block to enable it.
    # "LR": {
    #     "C": C_VALUES,
    #     "solver": ["liblinear"],
    #     "class_weight": ["balanced"],
    #     "max_iter": [3000],
    # },
    # "SVM": [
    #     {
    #         "C": C_VALUES,
    #         "class_weight": ["balanced"],
    #         "max_iter": [10000],
    #     },
        # RBF is intentionally disabled because 6 additional parameter sets
        # make the 10-fold experiment prohibitively slow with sklearn SVC.
        # {
        #     "C": C_VALUES,
        #     "kernel": ["rbf"],
        #     "gamma": ["scale", "auto"],
        #     "class_weight": ["balanced"],
        # },
    # ],
    # "NB": {
    #     "var_smoothing": [1e-11, 1e-9, 1e-7],
    # },
    # "KNN": {
    #     "n_neighbors": [3, 5, 7],
    #     "weights": ["uniform", "distance"],
    #     "p": [1, 2],
    # },
    "RF": {
        "n_estimators": [400],
        "max_depth": [None],
        "min_samples_split": [2],
        "min_samples_leaf": [2],
        "max_features": ["sqrt"],
        "class_weight": ["balanced_subsample"],
    },
}

# Kept for backward compatibility with older notebooks that directly call the
# validation-threshold Logistic Regression helpers.
LOGISTIC_REGRESSION_CONFIG = {
    "C": 0.01,
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 3000,
    "random_state": RANDOM_SEED,
}


REQUIRED_METRIC_COLUMNS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "precision_class_0",
    "recall_class_0",
    "f1_class_0",
    "precision_class_1",
    "recall_class_1",
    "f1_class_1",
]


def ensure_project_directories() -> None:
    """Create all expected output directories."""
    for path in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MARBERTV2_LOCAL_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        PLOTS_DIR,
        LOGS_DIR,
        CACHED_REPRESENTATIONS_DIR,
        NOTEBOOKS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

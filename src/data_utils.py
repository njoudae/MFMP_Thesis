#data_utils.py
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".json", ".jsonl"}

QUESTION_MARKS = r"[؟?❓❔？︖﹖]"
EXCLAMATION_MARKS = r"[!！❗❕‼︕﹗]"
HEARTS = r"(?:❤️|❤|♥️|♥|❣️)"

ARABIC_UNIFY_MAP = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ة": "ه",
    "ى": "ي",
    "ؤ": "و",
    "ئ": "ي",
    "پ": "ب",
    "چ": "ج",
    "ڤ": "ف",
    "گ": "ك",
    "٪": "",
    "٫": "",
    "٬": "",
    "،": "",
    "؛": "",
    "…": "",
}

ALLOWED_SYMBOLS_KEEP = {"!", "؟"}


def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    """Set seeds for Python, NumPy, and PyTorch if it is installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _emoji_module():
    try:
        import emoji

        return emoji
    except ImportError as exc:
        raise ImportError(
            "The cleaning pipeline requires the 'emoji' package. Install project requirements before running notebooks."
        ) from exc


def unify_hearts(text):
    return re.sub(HEARTS, "♥", str(text))


def unify_question_marks(text):
    return re.sub(QUESTION_MARKS, "؟", str(text))


def unify_exclamation_marks(text):
    return re.sub(EXCLAMATION_MARKS, "!", str(text))


def get_all_emoji(series):
    emoji = _emoji_module()
    return [item["emoji"] for text in series for item in emoji.emoji_list(str(text))]


def count_q(series):
    return sum(str(text).count("؟") for text in series)


def count_ex(series):
    return sum(str(text).count("!") for text in series)


def diff(prev, new):
    return list(set(prev) - set(new))


def removed_chars(old_series, new_series):
    removed = []
    for old, new in zip(old_series, new_series):
        old = str(old)
        new = str(new)
        for ch in old:
            if ch not in new:
                removed.append(ch)
    return removed


def print_diff(step, prev_emoji, prev_ex, prev_q, old_series, new_series):
    new_emoji = get_all_emoji(new_series)
    new_ex = count_ex(new_series)
    new_q = count_q(new_series)

    removed_emoji = diff(prev_emoji, new_emoji)
    removed_ex = prev_ex - new_ex
    removed_q = prev_q - new_q
    removed_chars_step = removed_chars(old_series, new_series)

    print(f"\n==================== {step} ====================")
    print(f"Emoji before: {len(prev_emoji)} -> after: {len(new_emoji)}")
    print(f"Removed emoji count: {len(removed_emoji)}")
    if removed_emoji:
        print("Removed emoji list:", removed_emoji[:50], "..." if len(removed_emoji) > 50 else "")
    print(f"!: {prev_ex} -> {new_ex}, removed: {removed_ex}")
    print(f"؟: {prev_q} -> {new_q}, removed: {removed_q}")
    print(f"Removed characters in this step: {len(removed_chars_step)}")
    if removed_chars_step:
        print("Removed chars sample:", removed_chars_step[:50], "...")
    print("=====================================================")
    return new_emoji, new_ex, new_q


def clean_arabic_extra(ch):
    return ARABIC_UNIFY_MAP.get(ch, ch)


def remove_skin_tones(text):
    return re.sub(r"[🏻🏼🏽🏾🏿]", "", str(text))


def separate_emojis(text):
    emoji = _emoji_module()
    text = str(text)
    out = ""
    for ch in text:
        if ch in emoji.EMOJI_DATA:
            out += f" {ch} "
        else:
            out += ch
    return " ".join(out.split())


def detach_arabic_from_symbols(text):
    text = str(text)
    if not text:
        return text
    text = re.sub(r"([ء-ي])([^\sء-ي])", r"\1 \2", text)
    text = re.sub(r"([^\sء-ي])([ء-ي])", r"\1 \2", text)
    text = text.replace("\u200d", " \u200d ")
    text = text.replace("\ufe0f", " \ufe0f ")
    return " ".join(text.split())


def process_tags(text):
    text = str(text)
    text = re.sub(r"(@)(\S)", r"\1 \2", text)
    text = re.sub(r"(#)(\S)", r"\1 \2", text)
    text = re.sub(r"@\s*\S+", " ", text)
    text = re.sub(r"#\s*", "", text)
    text = text.replace("_", " ")
    return " ".join(text.split())


def remove_tashkeel(text):
    return re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", str(text))


def remove_kashida(text):
    return str(text).replace("ـ", "")


def replace_laugh_variants(text):
    return re.sub(r"(?<![ء-ي])(?:ه(?:[هخحعلا])*){3,}(?![ء-ي])", "هه", str(text))


def normalize_arabic(text):
    text = str(text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"گ", "ك", text)
    return text


def remove_digits_english(text):
    text = str(text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[A-Za-z]+", " ", text)
    return text


def limit_letter_repetition(text):
    return re.sub(r"([ء-ي])\1{2,}", r"\1\1", str(text))


def separate_symbols(text):
    text = str(text).replace("؟", " ؟ ").replace("!", " ! ")
    return " ".join(text.split())


def clean_spaces(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def filter_chars_keep(text):
    """Keep Arabic letters, emojis, !, ؟, and spaces."""
    emoji = _emoji_module()
    cleaned = ""
    for ch in str(text):
        ch = clean_arabic_extra(ch)
        if ch in emoji.EMOJI_DATA:
            cleaned += ch
        elif re.match(r"[ء-ي]", ch):
            cleaned += ch
        elif ch in ALLOWED_SYMBOLS_KEEP:
            cleaned += ch
        elif ch.isspace():
            cleaned += ch
        elif re.match(r"[\U0001F1E6-\U0001F1FF]", ch):
            continue
        else:
            continue
    return cleaned


def filter_chars_remove_emoji_punct(text):
    """Keep Arabic letters and spaces only."""
    emoji = _emoji_module()
    cleaned = ""
    for ch in str(text):
        ch = clean_arabic_extra(ch)
        if ch in emoji.EMOJI_DATA:
            continue
        if re.match(r"[ء-ي]", ch):
            cleaned += ch
        elif ch.isspace():
            cleaned += ch
    return cleaned


def build_cleaning_steps(version: str = "keep"):
    """Build the supplied Arabic cleaning pipeline."""
    steps = [
        ("NFKC", lambda x: unicodedata.normalize("NFKC", str(x))),
        ("UNIFY HEARTS", unify_hearts),
        ("UNIFY ?", unify_question_marks),
        ("UNIFY !", unify_exclamation_marks),
        ("DETACH ARABIC FROM SYMBOLS", detach_arabic_from_symbols),
        ("REMOVE SKIN TONE", remove_skin_tones),
        ("SEPARATE EMOJI", separate_emojis),
        ("PROCESS TAGS", process_tags),
        ("REMOVE TASHKEEL", remove_tashkeel),
        ("REMOVE KASHIDA", remove_kashida),
        ("NORMALIZE ARABIC", normalize_arabic),
        ("REMOVE DIGITS/ENGLISH", remove_digits_english),
    ]
    if version == "keep":
        steps += [
            ("FILTER KEEP (AR + EMOJI + !؟)", filter_chars_keep),
            ("UNIFY LAUGH", replace_laugh_variants),
            ("LIMIT REPETITION", limit_letter_repetition),
            ("SEPARATE SYMBOLS", separate_symbols),
            ("CLEAN SPACES", clean_spaces),
        ]
    elif version == "remove":
        steps += [
            ("FILTER REMOVE (AR ONLY)", filter_chars_remove_emoji_punct),
            ("UNIFY LAUGH", replace_laugh_variants),
            ("LIMIT REPETITION", limit_letter_repetition),
            ("CLEAN SPACES", clean_spaces),
        ]
    else:
        raise ValueError("version must be 'keep' or 'remove'")
    return steps


def apply_pipeline(series: pd.Series, version: str = "keep", debug: bool = False, debug_name: str = "DATA") -> pd.Series:
    """Apply the supplied cleaning pipeline to a pandas Series."""
    current = series.astype(str).copy()
    steps = build_cleaning_steps(version=version)
    if debug:
        prev_emoji = get_all_emoji(current)
        prev_ex = count_ex(current)
        prev_q = count_q(current)
        print_diff(f"{debug_name} | STEP 0 RAW", [], 0, 0, current, current)

    for name, func in steps:
        old = current
        current = current.apply(func)
        if debug:
            prev_emoji, prev_ex, prev_q = print_diff(
                f"{debug_name} | {name}", prev_emoji, prev_ex, prev_q, old, current
            )
    return current


def add_cleaning_columns(
    df: pd.DataFrame,
    raw_text_column: str,
    debug: bool = False,
    debug_name: str = "DATA",
) -> pd.DataFrame:
    """Add both requested cleaning versions as separate columns."""
    output = df.copy()
    output[config.CLEAN_KEEP_COLUMN] = apply_pipeline(
        output[raw_text_column],
        version="keep",
        debug=debug,
        debug_name=debug_name,
    )
    output[config.CLEAN_REMOVE_COLUMN] = apply_pipeline(
        output[raw_text_column],
        version="remove",
        debug=debug,
        debug_name=debug_name,
    )
    print(f"Done. New columns added: {config.CLEAN_KEEP_COLUMN}, {config.CLEAN_REMOVE_COLUMN}")
    return output


def active_clean_text_column(cleaning_version: str = config.CLEANING_VERSION) -> str:
    if cleaning_version == "keep":
        return config.CLEAN_KEEP_COLUMN
    if cleaning_version == "remove":
        return config.CLEAN_REMOVE_COLUMN
    raise ValueError("CLEANING_VERSION must be 'keep' or 'remove'")


def find_dataset_file(dataset_path: str | Path | None = None) -> Path:
    """Find a single likely dataset file without scanning generated outputs."""
    if dataset_path is not None:
        path = Path(dataset_path)
        if not path.is_absolute():
            path = config.BASE_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Configured dataset path does not exist: {path}")
        return path

    if config.DATASET_PATH:
        return find_dataset_file(config.DATASET_PATH)

    search_roots = [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.BASE_DIR]
    ignored_parts = {"results", "models", "notebooks", "src", ".git"}
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            relative_parts = set(path.relative_to(config.BASE_DIR).parts)
            if relative_parts & ignored_parts:
                continue
            if path.name in {config.SPLIT_TRAIN_PATH.name, config.SPLIT_TEST_PATH.name}:
                continue
            candidates.append(path)

    unique_candidates = sorted(set(candidates))
    if not unique_candidates:
        raise FileNotFoundError(
            "No dataset file found. Put your dataset in data/raw/ or set DATASET_PATH in src/config.py."
        )
    if len(unique_candidates) > 1:
        print("Multiple dataset candidates found. Using the first sorted path:")
        for candidate in unique_candidates:
            print(f"  - {candidate}")
    return unique_candidates[0]


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    """Load the dataset from CSV/TSV/Excel/Parquet/JSON."""
    path = find_dataset_file(dataset_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported dataset extension: {path.suffix}")

    print(f"Dataset path: {path}")
    print(f"Dataset shape: {df.shape}")
    return df


def official_arsarcasm_split_exists() -> bool:
    return config.ARSARCASM_TRAIN_PATH.exists() and config.ARSARCASM_TEST_PATH.exists()


def load_official_arsarcasm_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ArSarcasm-v2 official train/test CSV files when available."""
    train_df = pd.read_csv(config.ARSARCASM_TRAIN_PATH)
    test_df = pd.read_csv(config.ARSARCASM_TEST_PATH)
    print(f"Official train path: {config.ARSARCASM_TRAIN_PATH}")
    print(f"Official test path: {config.ARSARCASM_TEST_PATH}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    return train_df, test_df


def detect_text_label_columns(
    df: pd.DataFrame,
    text_column: str | None = config.TEXT_COLUMN,
    label_column: str | None = config.LABEL_COLUMN,
) -> tuple[str, str]:
    """Detect likely text and binary label columns, unless explicitly configured."""
    if text_column:
        if text_column not in df.columns:
            raise KeyError(f"TEXT_COLUMN={text_column!r} not found in dataset columns.")
        chosen_text = text_column
    else:
        lower_to_original = {str(col).lower(): col for col in df.columns}
        chosen_text = None
        for candidate in config.PREFERRED_TEXT_COLUMNS:
            if candidate.lower() in lower_to_original:
                chosen_text = lower_to_original[candidate.lower()]
                break
        if chosen_text is None:
            string_columns = [
                col
                for col in df.columns
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
            ]
            if not string_columns:
                raise ValueError("Could not detect a text column. Set TEXT_COLUMN in src/config.py.")
            chosen_text = max(
                string_columns,
                key=lambda col: df[col].dropna().astype(str).str.len().mean()
                if not df[col].dropna().empty
                else 0,
            )

    if label_column:
        if label_column not in df.columns:
            raise KeyError(f"LABEL_COLUMN={label_column!r} not found in dataset columns.")
        chosen_label = label_column
    else:
        lower_to_original = {str(col).lower(): col for col in df.columns if col != chosen_text}
        chosen_label = None
        for candidate in config.PREFERRED_LABEL_COLUMNS:
            if candidate.lower() in lower_to_original:
                chosen_label = lower_to_original[candidate.lower()]
                break
        if chosen_label is None:
            label_candidates = []
            for col in df.columns:
                if col == chosen_text:
                    continue
                nunique = df[col].dropna().nunique()
                if 2 <= nunique <= 20:
                    label_candidates.append((col, nunique))
            binary_candidates = [item for item in label_candidates if item[1] == 2]
            if binary_candidates:
                chosen_label = binary_candidates[0][0]
            elif label_candidates:
                chosen_label = sorted(label_candidates, key=lambda item: item[1])[0][0]
            else:
                raise ValueError("Could not detect a label column. Set LABEL_COLUMN in src/config.py.")

    print(f"Text column: {chosen_text}")
    print(f"Label column: {chosen_label}")
    return str(chosen_text), str(chosen_label)


def report_missing_duplicates_and_distribution(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
) -> None:
    print("Missing values:")
    print(df[[text_column, label_column]].isna().sum())
    print(f"Duplicate text rows: {df.duplicated(subset=[text_column]).sum()}")
    print("Label distribution:")
    print(df[label_column].value_counts(dropna=False).sort_index())


def apply_cleaning(
    df: pd.DataFrame,
    text_column: str,
    cleaning_fn: Callable[[str], str] | None = None,
) -> pd.DataFrame:
    """Apply a shared text cleaning function. Defaults to conservative whitespace cleanup."""
    cleaned = df.copy()
    cleaned[text_column] = cleaned[text_column].astype(str)
    if cleaning_fn is None:
        cleaned[text_column] = cleaned[text_column].str.replace(r"\s+", " ", regex=True).str.strip()
    else:
        cleaned[text_column] = cleaned[text_column].map(cleaning_fn)
    return cleaned


def prepare_binary_dataframe(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    cleaning_fn: Callable[[str], str] | None = None,
    drop_duplicate_texts: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Clean text, remove invalid rows with logging, and map labels to 0/1."""
    removal_log: list[dict] = []
    work = apply_cleaning(df, text_column, cleaning_fn=cleaning_fn)
    original_count = len(work)

    work[text_column] = work[text_column].fillna("").astype(str)
    missing_mask = work[label_column].isna()
    if config.DROP_EMPTY_TEXT_AFTER_CLEANING:
        missing_mask = missing_mask | (work[text_column].astype(str).str.len() == 0)
    missing_count = int(missing_mask.sum())
    if missing_count:
        reason = (
            "missing_or_empty_text_or_label"
            if config.DROP_EMPTY_TEXT_AFTER_CLEANING
            else "missing_label"
        )
        removal_log.append({"reason": reason, "count": missing_count})
        work = work.loc[~missing_mask].copy()

    unique_labels = list(pd.Series(work[label_column].dropna().unique()).sort_values())
    if len(unique_labels) != 2:
        raise ValueError(
            f"Binary classification requires exactly 2 labels after cleaning; found {len(unique_labels)}: {unique_labels}"
        )

    if all(isinstance(label, (bool, np.bool_)) for label in unique_labels):
        label_map = {False: 0, True: 1}
    elif set(unique_labels) == {0, 1}:
        label_map = {0: 0, 1: 1}
    else:
        normalized = pd.Series(unique_labels).astype(str).str.strip().str.lower().tolist()
        if set(normalized) == {"false", "true"}:
            label_map = {unique_labels[normalized.index("false")]: 0, unique_labels[normalized.index("true")]: 1}
        elif set(normalized) == {"0", "1"}:
            label_map = {unique_labels[normalized.index("0")]: 0, unique_labels[normalized.index("1")]: 1}
        else:
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}

    mapped_labels = work[label_column].map(label_map)
    unmapped_count = int(mapped_labels.isna().sum())
    if unmapped_count:
        raise ValueError(
            f"Label mapping produced {unmapped_count} unmapped labels. "
            f"Unique labels: {unique_labels}; label_map: {label_map}"
        )
    work[label_column] = mapped_labels.astype(int)
    print(f"Label mapping used: {label_map}")

    if drop_duplicate_texts:
        duplicate_mask = work.duplicated(subset=[text_column], keep="first")
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count:
            removal_log.append({"reason": "duplicate_text", "count": duplicate_count})
            work = work.loc[~duplicate_mask].copy()

    removed_total = original_count - len(work)
    if removed_total:
        print("Rows removed:")
        for item in removal_log:
            print(f"  {item['reason']}: {item['count']}")
        save_row_removal_log(removal_log)
    else:
        print("Rows removed: 0")

    return work.reset_index(drop=True), removal_log


def prepare_official_split_dataframes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str,
    label_column: str,
    cleaning_version: str = config.CLEANING_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Add cleaning columns and remove invalid rows from official train/test splits."""
    train_cleaned = add_cleaning_columns(train_df, text_column, debug=False, debug_name="TRAIN")
    test_cleaned = add_cleaning_columns(test_df, text_column, debug=False, debug_name="TEST")
    active_text = active_clean_text_column(cleaning_version)

    prepared_train, train_log = prepare_binary_dataframe(
        train_cleaned,
        active_text,
        label_column,
        cleaning_fn=None,
        drop_duplicate_texts=False,
    )
    prepared_test, test_log = prepare_binary_dataframe(
        test_cleaned,
        active_text,
        label_column,
        cleaning_fn=None,
        drop_duplicate_texts=False,
    )

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    prepared_train.to_csv(config.SPLIT_TRAIN_PATH, index=False, encoding="utf-8-sig")
    prepared_test.to_csv(config.SPLIT_TEST_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved processed train split: {config.SPLIT_TRAIN_PATH}")
    print(f"Saved processed test split: {config.SPLIT_TEST_PATH}")
    if train_log or test_log:
        combined_log = [
            {"split": "train", **item} for item in train_log
        ] + [
            {"split": "test", **item} for item in test_log
        ]
        save_row_removal_log(combined_log)

    print(f"Active experiment text column: {active_text}")
    print("Train label distribution:")
    print(prepared_train[label_column].value_counts().sort_index())
    print("Test label distribution:")
    print(prepared_test[label_column].value_counts().sort_index())
    return prepared_train, prepared_test, active_text


def save_row_removal_log(removal_log: list[dict]) -> None:
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(removal_log).to_csv(config.ROW_REMOVAL_LOG_PATH, index=False)
    print(f"Row removal log saved to: {config.ROW_REMOVAL_LOG_PATH}")


def get_or_create_train_test_split(
    df: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float = config.TEST_SIZE,
    random_seed: int = config.RANDOM_SEED,
    force_resplit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load an existing split if present; otherwise create and save a stratified split."""
    if (
        not force_resplit
        and config.SPLIT_TRAIN_PATH.exists()
        and config.SPLIT_TEST_PATH.exists()
    ):
        train_df = pd.read_csv(config.SPLIT_TRAIN_PATH)
        test_df = pd.read_csv(config.SPLIT_TEST_PATH)
        print("Loaded existing train/test split from data/processed/.")
    else:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=df[label_column],
        )
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(config.SPLIT_TRAIN_PATH, index=False)
        test_df.to_csv(config.SPLIT_TEST_PATH, index=False)
        print("Created and saved stratified train/test split.")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print("Train label distribution:")
    print(train_df[label_column].value_counts().sort_index())
    print("Test label distribution:")
    print(test_df[label_column].value_counts().sort_index())
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_prepared_train_test(
    dataset_path: str | Path | None = None,
    cleaning_fn: Callable[[str], str] | None = None,
    force_resplit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Full reproducible loading path used by every experiment notebook."""
    set_global_seed(config.RANDOM_SEED)

    if config.USE_OFFICIAL_ARSARCASM_SPLIT and official_arsarcasm_split_exists() and not force_resplit:
        train_raw, test_raw = load_official_arsarcasm_split()
        combined_for_detection = pd.concat([train_raw, test_raw], ignore_index=True)
        text_column, label_column = detect_text_label_columns(combined_for_detection)
        print("Official split detected; no random train/test split will be created.")
        print("Raw train checks:")
        report_missing_duplicates_and_distribution(train_raw, text_column, label_column)
        print("Raw test checks:")
        report_missing_duplicates_and_distribution(test_raw, text_column, label_column)
        train_df, test_df, active_text = prepare_official_split_dataframes(
            train_raw,
            test_raw,
            text_column=text_column,
            label_column=label_column,
            cleaning_version=config.CLEANING_VERSION,
        )
        return train_df, test_df, active_text, label_column

    df = load_dataset(dataset_path)
    text_column, label_column = detect_text_label_columns(df)
    report_missing_duplicates_and_distribution(df, text_column, label_column)
    if cleaning_fn is None and config.CLEANING_VERSION in {"keep", "remove"}:
        df = add_cleaning_columns(df, text_column, debug=False, debug_name="DATA")
        text_column = active_clean_text_column(config.CLEANING_VERSION)
    prepared_df, _ = prepare_binary_dataframe(df, text_column, label_column, cleaning_fn=cleaning_fn)
    train_df, test_df = get_or_create_train_test_split(
        prepared_df,
        text_column=text_column,
        label_column=label_column,
        force_resplit=force_resplit,
    )
    return train_df, test_df, text_column, label_column

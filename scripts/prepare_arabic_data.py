"""Recreate the processed ArSarcasm-v2 official train/test CSV files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_prepared_train_test


if __name__ == "__main__":
    train, test, text_column, label_column = load_prepared_train_test()
    print(f"Prepared {len(train)} train and {len(test)} test rows")
    print(f"Experiment columns: text={text_column!r}, label={label_column!r}")

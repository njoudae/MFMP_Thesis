"""Run lightweight consistency checks over the committed result artifacts."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    arabic = pd.read_csv(ROOT / "results/metrics/arabic/final_test_results.csv")
    selected = pd.read_csv(ROOT / "results/metrics/arabic/selected_configurations.csv")
    english = pd.read_csv(ROOT / "results/tables/english/selected_baselines_and_winners.csv")

    assert len(arabic) == 9, "Arabic final table must contain 3 methods x 3 classifiers"
    assert set(arabic["Approach"]) == {"CLS", "Mean", "MFMP"}
    assert set(arabic["Classifier"]) == {"LR", "SVM", "RF"}
    assert len(selected) == 9 and selected["configuration_selected"].all()

    cv_winners = english[english["selection"] == "best_cv_macro_f1"]
    assert set(cv_winners["algorithm"]) == {"LR", "SVM", "RF"}
    assert english["test_macro_f1"].between(0, 1).all()
    assert english["cv_macro_f1_mean"].between(0, 1).all()
    print("Artifact checks passed: Arabic 9 final rows; English 3 CV-selected winners.")


if __name__ == "__main__":
    main()

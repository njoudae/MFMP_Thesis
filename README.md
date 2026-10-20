# MFMP: Magnitude-based Binary Mask Filtering of Contextual Token Representations for Sarcasm Detection

MFMP is a sentence-representation method for sarcasm detection. It uses the
relative magnitudes of contextual token embeddings to construct a binary mask,
then averages the retained original token vectors before fitting a conventional
classifier. This repository preserves the completed Arabic and English
experiments, their training-only model-selection procedures, and the saved
result artifacts used to audit the findings.

## Contribution and method

MFMP asks whether tokens with unusually small or large contextual-vector
magnitudes should contribute equally to a sentence representation. For each
sentence, the implementation computes every attended token's L2 magnitude,
converts magnitudes to sentence-local population z-scores, retains tokens
inside configured lower/upper bounds, and mean-pools their original 768-D
contextual vectors. If the mask is empty, it falls back to the unfiltered mean.
CLS and ordinary mean pooling are the final baselines.

Important terminology: the completed implementation computes **L2 norms as
magnitudes**; it does not normalize token vectors to unit length. See
[`docs/methodology.md`](docs/methodology.md) for the exact mask rules, the
Arabic/English boundary difference, classifiers, and threshold selection.

## Repository layout

```text
configs/                  Saved Arabic environment/config plus English run config
docs/                     Method, data, experiments, audit, reproducibility
experiments/arabic/       Final self-contained MARBERTv2 notebook
experiments/english/      Final BERT LR, SVM, and RF notebooks
results/metrics/          Raw final searches and held-out metrics
results/tables/           Compact audit and token-distribution tables
results/figures/          Selected Arabic diagnostic figures
scripts/                  Data/model preparation and artifact verification
src/                      Preserved thesis modules plus tested MFMP core
tests/                    Lightweight MFMP invariants
```

Datasets, model weights, embedding caches, checkpoints, and row-level
predictions are deliberately absent.

## Installation

Python 3.10 is recommended. The saved Arabic run used PyTorch 2.10.0+cu126,
Transformers 5.0.0, scikit-learn 1.7.2, NumPy 2.2.6, and pandas 2.3.3.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset and encoder preparation

Follow [`docs/datasets.md`](docs/datasets.md). In short, place the official
ArSarcasm-v2 CSV files under `data/raw/ArSarcasm-v2/` and the Kaggle English CSV
under `data/raw/english/`, then run:

```bash
python scripts/prepare_arabic_data.py
python scripts/download_encoders.py
```

The second command requires network access and downloads sizeable model files.

## Running the experiments

Arabic baselines and MFMP are evaluated together by the final notebook:

```bash
jupyter nbconvert --to notebook --execute experiments/arabic/mfmp_final_arabic.ipynb --output mfmp_final_arabic.executed.ipynb --ExecutePreprocessor.timeout=-1
```

Run the second-language experiments with:

```bash
jupyter nbconvert --to notebook --execute experiments/english/english_all_tokens_lr.ipynb --output english_all_tokens_lr.executed.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute experiments/english/english_all_tokens_svm.ipynb --output english_all_tokens_svm.executed.ipynb --ExecutePreprocessor.timeout=-1
jupyter nbconvert --to notebook --execute experiments/english/english_all_tokens_rf.ipynb --output english_all_tokens_rf.executed.ipynb --ExecutePreprocessor.timeout=-1
```

These notebooks perform representation extraction, baseline/MFMP grids,
10-fold selection, pooled out-of-fold threshold selection, final classifier
fitting, and held-out evaluation. Full runs require substantial GPU time and
storage. Exact protocol details are in
[`docs/reproducibility.md`](docs/reproducibility.md).

## Evaluation

The primary selection metric is mean validation Macro-F1. Candidate ties are
resolved using sarcastic-class F1, sarcastic recall, and accuracy in the Arabic
notebook. Final tables also contain accuracy, macro precision/recall,
class-specific precision/recall/F1, balanced accuracy, MCC, Cohen's kappa,
ROC-AUC, average precision, PR-AUC, and confusion counts. Final operating
thresholds are chosen from pooled out-of-fold training scores, never from final
test labels. The English candidate CV values have the documented non-nested
limitation that fold thresholds are tuned and evaluated on the same validation
fold; see the methodology document.

## Verified results

Arabic official test results:

| Classifier | CLS Macro-F1 | Mean Macro-F1 | MFMP Macro-F1 | Selected MFMP bounds |
|---|---:|---:|---:|---|
| LR | 0.714846 | 0.712621 | 0.714537 | [0.0001, 1.5] |
| SVM | 0.714467 | 0.721864 | 0.718520 | [0.0001, 4.0] |
| RF | 0.690763 | 0.697984 | 0.700954 | [0.01, 1.5] |

English CV-selected MFMP variants:

| Classifier | CV-selected method | CV Macro-F1 | Test Macro-F1 | Test sarcastic F1 |
|---|---|---:|---:|---:|
| LR | `mean_gt_0.001` | 0.895855 | 0.855499 | 0.775309 |
| SVM | `mean_gt_0.0001` | 0.894553 | 0.851518 | 0.769608 |
| RF | `mean_gt_0.001_lt_1` | 0.849425 | 0.817389 | 0.716049 |

Values are transcribed from the committed CSV artifacts and checked by
`scripts/verify_artifacts.py`. Test-ranked English variants are retained in the
audit CSV but explicitly marked post hoc and are not presented as selected
results.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/verify_artifacts.py
```

Packaging did not re-run full transformer extraction or all CV fits. See
[`docs/repository_audit.md`](docs/repository_audit.md) for included/excluded
materials and unresolved publication items.

## Citation, license, and acknowledgments

Use `CITATION.cff` as a starting point and replace the Git identity with the
paper's complete author metadata before release. Cite ArSarcasm-v2 as requested
by its maintainers and cite the Kaggle dataset according to its current source
page.

The software in this repository is released under the MIT License; see
[`LICENSE`](LICENSE). Dataset and pretrained-model licenses remain governed by
their respective providers. This work acknowledges the ArSarcasm-v2/WANLP 2021 maintainers,
the English Sarcasm dataset provider, UBC-NLP for MARBERTv2, and the BERT and
Hugging Face ecosystems.

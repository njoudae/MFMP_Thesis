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

## Reported results

**Performance comparison of MFMP with CLS and mean pooling on ArSarcasm-v2.**

| Model | Pooling | Macro-F1 | F1-Sar |
|---|---|---:|---:|
| LR | CLS | 71% | 59% |
| LR | Mean | 71% | 58% |
| LR | **MFMP** | **72%** | **61%** |
| SVM | CLS | 71% | 60% |
| SVM | Mean | 72.2% | 61.1% |
| SVM | **MFMP** | **72.3%** | **61.3%** |
| RF | CLS | 68% | 54% |
| RF | Mean | 70% | 57% |
| RF | **MFMP** | **71%** | **60%** |

**Performance comparison of MFMP with CLS and mean pooling on the English
Sarcasm dataset.**

| Model | Pooling | Macro-F1 | F1-Sar |
|---|---|---:|---:|
| LR | CLS | 84% | 75% |
| LR | Mean | 86% | 78% |
| LR | **MFMP** | **87%** | **79%** |
| SVM | CLS | 85% | 75% |
| SVM | Mean | 85% | 77% |
| SVM | **MFMP** | **86%** | **78%** |
| RF | CLS | 77% | 66% |
| RF | Mean | 83% | 73% |
| RF | **MFMP** | **83%** | **75%** |

**Selected MFMP magnitude thresholds associated with the best performance of
each classifier.**

| Model | Dataset | K<sub>l</sub> | K<sub>u</sub> |
|---|---|---:|---:|
| LR | ArSarcasm-v2 | 0.001 | 0.5 |
| LR | English Sarcasm | 0.01 | 3.0 |
| SVM | ArSarcasm-v2 | — | 5.5 |
| SVM | English Sarcasm | 0.01 | 2.5 |
| RF | ArSarcasm-v2 | 0.001 | 2.5 |
| RF | English Sarcasm | — | 1.5 |

These are the manuscript-reported values supplied by the author. The committed
CSV artifacts remain available for detailed auditing, and
`scripts/verify_artifacts.py` checks their internal structure.

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

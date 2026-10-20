# Final experiments and result provenance

## Arabic

The final Arabic experiment is `experiments/arabic/mfmp_final_arabic.ipynb`.
Its saved environment is Python 3.10.19, NumPy 2.2.6, pandas 2.3.3, PyTorch
2.10.0+cu126, Transformers 5.0.0, and scikit-learn 1.7.2 on an NVIDIA RTX 4060
Laptop GPU. Seed 42, maximum length 320, batch size 32, and 10 stratified folds
are fixed.

The MFMP grid contains `K_lower` in `{0.0001, 0.001, 0.01}` and `K_upper` in
`{0.5, 1.0, ..., 5.0}`. LR and SVM search `C` in `{0.001, 0.0001}`. RF searches
400 trees, depth `None` or 30, minimum split 2, minimum leaf 2, `sqrt` features,
and balanced-subsample class weights.

Selected Arabic MFMP configurations and official-test Macro-F1:

| Classifier | K lower | K upper | Selected parameter | OOF-selection Macro-F1 | Test Macro-F1 |
|---|---:|---:|---|---:|---:|
| LR | 0.0001 | 1.5 | C=0.001 | 0.709612 | 0.714537 |
| SVM | 0.0001 | 4.0 | C=0.001 | 0.730371 | 0.718520 |
| RF | 0.01 | 1.5 | depth=30, 400 trees | 0.630280 | 0.700954 |

The complete nine-row baseline/MFMP test table is
`results/metrics/arabic/final_test_results.csv`. Candidate searches and selected
OOF thresholds are retained alongside it.

## English

The three final notebooks are under `experiments/english/`. They use
`bert-base-uncased`, seed 42, maximum length 320, batch size 32, a stratified
80/20 split, and 10 shuffled stratified folds. LR searches C values
`[0.1, 0.01, 0.001, 1]` with `liblinear`, balanced classes, and 3,000 maximum
iterations. Linear SVM searches the same C grid with balanced classes and
10,000 maximum iterations. RF uses 400 trees, no depth limit, minimum split 2,
minimum leaf 2, `sqrt` features, and balanced-subsample class weights.

The notebooks explicitly record a limitation: candidate CV thresholds are
optimized and evaluated on the same validation fold. Those CV values reproduce
the implemented selection but are not nested/unbiased performance estimates.
The final test threshold is separately derived from pooled out-of-fold training
scores.

The scientifically valid primary selections are those marked
`best_cv_macro_f1`; the rows explicitly marked `*_posthoc` were found by
ranking the held-out test set and must not be reported as prospectively
selected results.

| Classifier | CV-selected method | Bounds | CV Macro-F1 | Test Macro-F1 | Test sarcastic F1 |
|---|---|---|---:|---:|---:|
| LR | `mean_gt_0.001` | lower 0.001 | 0.895855 | 0.855499 | 0.775309 |
| SVM | `mean_gt_0.0001` | lower 0.0001 | 0.894553 | 0.851518 | 0.769608 |
| RF | `mean_gt_0.001_lt_1` | (0.001, 1.0) | 0.849425 | 0.817389 | 0.716049 |

All 61 methods per classifier, fold records, thresholds, dataset hashes, and
run signatures remain in `results/metrics/english/`. The compact audit table is
`results/tables/english/selected_baselines_and_winners.csv`.

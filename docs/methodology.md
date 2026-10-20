# MFMP methodology

This document describes the implementation that produced the committed final
artifacts. The provenance-bearing programs are the notebooks in
`experiments/`; `src/mfmp/core.py` is a tested extraction of their core tensor
operations.

## Contextual representations

Arabic experiments use the final hidden layer of the frozen
`UBC-NLP/MARBERTv2` encoder (768 dimensions). English experiments use the final
hidden layer of frozen `bert-base-uncased` (768 dimensions). Inputs are padded
and truncated to 320 tokens. All attended tokens are candidates, including
`[CLS]` and the final separator token; padding is always excluded.

For each attended token vector \(h_i\), MFMP computes the scalar Euclidean
magnitude \(m_i=\lVert h_i\rVert_2\). The final code does **not** divide token
vectors by their L2 norm and therefore does not perform unit-vector L2
normalization. “L2” in the experiment artifacts refers to the L2 magnitude.
LR and SVM sentence representations are subsequently standardized by a
fold-local `StandardScaler`; RF is not scaled.

## Sentence-local standardization and mask

Within each sentence, the mean and population standard deviation (`ddof=0`) of
the attended-token magnitudes are calculated. The absolute z-score is

\[
|z_i| = \left|\frac{m_i-\mu_m}{\max(\sigma_m,10^{-12})}\right|.
\]

The Arabic final experiment retains a whole token vector when
`K_lower <= abs(z_i) <= K_upper` (inclusive). The English notebooks use strict
bounds: `K_lower < abs(z_i) < K_upper`, with one-sided variants where only a
lower or upper bound is active. This difference is preserved because it is
present in the completed implementations and saved run contexts.

The sentence representation is the arithmetic mean of the **original**
contextual vectors selected by the scalar mask. If no token survives, the
unfiltered mean over all attended tokens is used. CLS and unfiltered mean
pooling are the two baselines in the final comparison.

## Classifiers and selection

Arabic uses Logistic Regression (`liblinear`, `class_weight="balanced"`,
`max_iter=3000`), linear SVM (`class_weight="balanced"`, `max_iter=10000`), and
Random Forest. LR and SVM use a fold-local `StandardScaler`. All configuration
selection uses the same seeded, shuffled, stratified 10 folds on training data.
Candidate configurations are ranked by mean validation Macro-F1, then mean
sarcastic-class F1, sarcastic-class recall, and accuracy. Only after selecting a
configuration is a single operating threshold optimized from pooled
out-of-fold training scores. The official Arabic test labels are used once.

English uses a seeded stratified 80/20 split because the source does not
provide the ArSarcasm-style official split. During English candidate selection,
each fold's threshold is optimized and scored on that same validation fold.
Consequently, the candidate CV scores are training-only but are not nested or
unbiased estimates. After method/hyperparameter selection, the saved final
operating threshold comes from pooled out-of-fold training scores and is then
applied to the held-out test set. Detailed grids and selected values are in
`docs/experiments.md` and the committed CSV files.

## Integrity note

If a manuscript describes unit-vector “L2 normalization,” it does not match the
committed final code. The code computes L2 magnitudes and sentence-local
z-scores. That wording should be reconciled before journal publication.

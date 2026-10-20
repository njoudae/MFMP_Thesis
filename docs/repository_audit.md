# Repository audit

## Included

- The self-contained final Arabic notebook and its exact config, search, test,
  distribution, and summary CSV artifacts.
- The three completed English classifier notebooks, their 61-method result
  CSVs, run signatures, dataset hash, and compact audited selections.
- Shared preprocessing, extraction, pooling, classifier-search, evaluation,
  and result-management modules used by the thesis project.
- A tested extraction of the final MFMP tensor operations.

## Excluded

- `data/`: third-party Arabic and English text is not redistributed.
- `models/`: encoder weights are hundreds of MB and are downloadable from the
  named upstream model repositories.
- `experiments/MFMP_Final_Experiment/cache/`: generated NumPy representation
  caches are large and reproducible from the notebooks.
- Prediction CSVs: they include row-level labels/scores and can expose source
  dataset content or facilitate unintended redistribution.
- Exploratory notebooks and scripts: the thesis root contains many abandoned,
  diagnostic, layer-sweep, token-cloud, and alternative-pooling experiments
  that are not the final methodology.
- Virtual environments, caches, temporary files, website prototypes, and
  unrelated outputs.

## Open publication items

- No software license was found or selected, so no `LICENSE` file was created.
- `CITATION.cff` uses the configured Git identity `njood`; the author should
  replace it with the paper's full author list, affiliations, DOI, and ORCIDs.
- The destination GitHub repository was verified through the connected GitHub
  API as public, empty, and using `main` as its default branch. The local GitHub
  CLI credential is invalid, but the connected account reports push/admin
  permission. Publication is intentionally paused until a license is selected.

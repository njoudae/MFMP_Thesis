# Datasets

No third-party dataset is committed to this repository. Download each dataset
from its original source and review its current terms before use or
redistribution.

## Arabic: ArSarcasm-v2

Source: <https://github.com/iabufarha/ArSarcasm-v2>

The source repository describes ArSarcasm-v2 as the WANLP 2021 shared-task
dataset and provides the official 12,548-row training and 3,000-row testing
files. It requests citation of:

> Ibrahim Abu Farha, Wajdi Zaghouani, and Walid Magdy. 2021. “Overview of the
> WANLP 2021 Shared Task on Sarcasm and Sentiment Detection in Arabic.”
> Proceedings of the Sixth Arabic Natural Language Processing Workshop.

Place the files as:

```text
data/raw/ArSarcasm-v2/training_data.csv
data/raw/ArSarcasm-v2/testing_data.csv
```

Then run `python scripts/prepare_arabic_data.py`. The preserved pipeline keeps
the official split, detects `tweet` and `sarcasm`, creates `clean_keep` and
`clean_noemoji_nopunct`, and uses `clean_keep` in the final experiment. It
normalizes Arabic character variants, laughter, tags, spacing, emoji/symbol
boundaries, and retains Arabic letters, emoji, `!`, and `؟`. Duplicates are
reported but not removed in the official Arabic final run. Empty cleaned text
is retained as an empty string, matching the original pipeline.

Expected processed fingerprints are recorded in `configs/dataset_metadata.json`.
They are pandas dataframe fingerprints, not raw-file SHA-256 hashes.

## English: English Sarcasm

Source: <https://www.kaggle.com/datasets/faridanusrat42/english-sarcasm>

Download `English Sarcasm.csv` through Kaggle and place it at:

```text
data/raw/english/English Sarcasm.csv
```

Alternatively place the original Kaggle archive at
`data/raw/english/archive.zip`, containing that filename. The notebooks also
accept `MFMP_ENGLISH_CSV` or `MFMP_ENGLISH_ZIP` environment variables.

The expected columns are `Tweet` and `Label`. Cleaning HTML-unescapes and NFKC
normalizes text, replaces URLs with `URL`, replaces mentions with `USER`,
collapses whitespace, restricts labels to 0/1, and removes duplicate
`(clean_text, Label)` pairs. The saved run had 4,553 raw rows, 4,546 cleaned and
deduplicated rows (3,546 class 0; 1,000 class 1), then a seed-42 stratified
80/20 split (3,636 train; 910 test). The cleaned-data SHA-256 recorded by all
three runs is
`c75769c63d704cdb21684777e2323431f62207d7d82c5618e92bc21464710d39`.

The Kaggle license metadata was not available in the inspected local project.
For that reason the data is excluded, and users must verify the current Kaggle
license and citation requirements at the source page.

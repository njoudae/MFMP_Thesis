# Reproducibility protocol

Run commands from the repository root.

1. Create a Python 3.10 environment and install `requirements.txt`. For exact
   GPU reproduction, use a PyTorch 2.10.0 CUDA 12.6 build compatible with your
   platform; CPU execution is supported but much slower.
2. Download the datasets exactly as described in `docs/datasets.md`.
3. Run `python scripts/prepare_arabic_data.py`.
4. Cache encoders with `python scripts/download_encoders.py`. The Arabic
   notebook requires `models/MARBERTv2`; the English notebooks use the normal
   Hugging Face cache for `bert-base-uncased`.
5. Execute the Arabic notebook with an unlimited notebook timeout:

   ```bash
   jupyter nbconvert --to notebook --execute experiments/arabic/mfmp_final_arabic.ipynb --output mfmp_final_arabic.executed.ipynb --ExecutePreprocessor.timeout=-1
   ```

6. Execute English LR, SVM, and RF separately:

   ```bash
   jupyter nbconvert --to notebook --execute experiments/english/english_all_tokens_lr.ipynb --output english_all_tokens_lr.executed.ipynb --ExecutePreprocessor.timeout=-1
   jupyter nbconvert --to notebook --execute experiments/english/english_all_tokens_svm.ipynb --output english_all_tokens_svm.executed.ipynb --ExecutePreprocessor.timeout=-1
   jupyter nbconvert --to notebook --execute experiments/english/english_all_tokens_rf.ipynb --output english_all_tokens_rf.executed.ipynb --ExecutePreprocessor.timeout=-1
   ```

7. Run lightweight checks with:

   ```bash
   python -m unittest discover -s tests -v
   python scripts/verify_artifacts.py
   ```

Full extraction and cross-validation were not re-run while packaging this
repository. The committed metrics are copied from completed experiment
artifacts and are validated structurally by the lightweight checks. Arabic
representation caches, English embeddings, model weights, predictions that
contain third-party text, and datasets are intentionally excluded.

`configs/environment.json` records the original Arabic environment. Exact
reproduction can still vary across GPU architectures, CUDA/cuDNN kernels, and
library builds despite deterministic seeds. The code requests deterministic
PyTorch algorithms where supported.

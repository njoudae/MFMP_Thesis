#embedding_utils.py
import gc
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src import config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_exists_locally(local_dir: Path = config.MARBERTV2_LOCAL_DIR) -> bool:
    required_files = ["config.json"]
    return local_dir.exists() and all((local_dir / filename).exists() for filename in required_files)


def load_marbertv2(
    local_dir: Path = config.MARBERTV2_LOCAL_DIR,
    model_name: str = config.MARBERTV2_MODEL_NAME,
    device: torch.device | None = None,
):
    """Load MARBERTv2 once, preferring the local project cache."""
    device = device or get_device()
    if model_exists_locally(local_dir):
        model_source = "local"
        print(f"Model loading source: local ({local_dir})")
        tokenizer = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
        model = AutoModel.from_pretrained(local_dir, local_files_only=True)
    else:
        model_source = "downloaded"
        print(f"Model loading source: downloaded ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        local_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        print(f"Saved MARBERTv2 locally to: {local_dir}")

    model.to(device)
    model.eval()
    return tokenizer, model, device, model_source


def estimate_numpy_size_mb(shape: tuple[int, ...], dtype=np.float32) -> float:
    return float(np.prod(shape) * np.dtype(dtype).itemsize / (1024**2))


def representation_cache_path(method_name: str, split_name: str, texts: list[str]) -> Path:
    joined = "\n".join(map(str, texts))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    filename = f"{method_name}_{split_name}_maxlen{config.MAX_LEN}_{digest}.npz"
    return config.CACHED_REPRESENTATIONS_DIR / filename


def maybe_load_cached_representations(
    method_name: str,
    split_name: str,
    texts: list[str],
    use_cache: bool = config.CACHE_REPRESENTATIONS,
) -> np.ndarray | None:
    if not use_cache:
        return None
    path = representation_cache_path(method_name, split_name, texts)
    if path.exists():
        print(f"Loaded cached representations: {path}")
        return np.load(path)["representations"]
    return None


def maybe_save_cached_representations(
    representations: np.ndarray,
    method_name: str,
    split_name: str,
    texts: list[str],
    use_cache: bool = config.CACHE_REPRESENTATIONS,
    max_cache_size_mb: float = config.MAX_CACHE_SIZE_MB,
) -> None:
    if not use_cache:
        print("CACHE_REPRESENTATIONS=False; skipping representation cache save.")
        return

    size_mb = representations.nbytes / (1024**2)
    print(f"Representation shape before caching: {representations.shape}")
    print(f"Estimated representation cache size: {size_mb:.2f} MB")
    if size_mb > max_cache_size_mb:
        print(f"Cache size exceeds {max_cache_size_mb:.2f} MB; skipping cache save automatically.")
        return

    config.CACHED_REPRESENTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = representation_cache_path(method_name, split_name, texts)
    np.savez_compressed(path, representations=representations.astype(np.float32, copy=False))
    print(f"Saved compressed representation cache: {path}")


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_sentence_representations(
    texts,
    tokenizer,
    model,
    pooling_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    method_name: str,
    split_name: str,
    max_len: int = config.MAX_LEN,
    batch_size: int = config.BATCH_SIZE,
    device: torch.device | None = None,
    use_cache: bool = config.CACHE_REPRESENTATIONS,
) -> np.ndarray:
    """Extract pooled sentence representations without storing token-level tensors."""
    device = device or get_device()
    text_list = [str(text) for text in texts]

    cached = maybe_load_cached_representations(method_name, split_name, text_list, use_cache=use_cache)
    if cached is not None:
        print(f"Representation method name: {method_name}")
        print(f"Representation shape: {cached.shape}")
        print(f"Estimated memory size: {cached.nbytes / (1024**2):.2f} MB")
        return cached

    print(f"Representation method name: {method_name}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_len}")

    all_vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(text_list), batch_size):
            batch_texts = text_list[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden_states = outputs.last_hidden_state
            pooled = pooling_fn(hidden_states, encoded["attention_mask"])
            all_vectors.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))

            del encoded, outputs, hidden_states, pooled
            clear_memory()

    representations = np.vstack(all_vectors).astype(np.float32, copy=False)
    print(f"Representation shape: {representations.shape}")
    print(f"Estimated memory size: {representations.nbytes / (1024**2):.2f} MB")
    maybe_save_cached_representations(
        representations,
        method_name=method_name,
        split_name=split_name,
        texts=text_list,
        use_cache=use_cache,
    )
    return representations

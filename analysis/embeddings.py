"""Sentence-transformer embeddings with a tiny on-disk cache.

The cache is keyed on (model_name, text) so re-running the pipeline after new
species are added only embeds the new strings.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_PATH = Path(__file__).resolve().parent / ".embedding_cache.npz"
MODEL_NAME = "all-MiniLM-L6-v2"


def _key(model_name: str, text: str) -> str:
    h = hashlib.sha1(f"{model_name}\x00{text}".encode()).hexdigest()
    return h


def _load_cache() -> dict[str, np.ndarray]:
    if not CACHE_PATH.exists():
        return {}
    with np.load(CACHE_PATH, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    np.savez_compressed(CACHE_PATH, **cache)


def embed(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    cache = _load_cache()
    keys = [_key(model_name, t) for t in texts]
    missing_idx = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        vecs = model.encode(
            [texts[i] for i in missing_idx],
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        for i, v in zip(missing_idx, vecs):
            cache[keys[i]] = v.astype(np.float32)
        _save_cache(cache)
    return np.stack([cache[k] for k in keys])

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .datasets import CanonicalDataset

Modality = Literal["semantic", "acoustic"]
SimilarityMethod = Literal["tfidf_cosine", "keyword_jaccard", "precomputed"]
Transform = Literal["none", "zscore"]


@dataclass(frozen=True)
class SimilaritySpec:
    modality: Modality
    method: SimilarityMethod = "tfidf_cosine"
    transform: Transform = "none"
    precomputed: pd.DataFrame | None = None


def _cosine_from_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=float)
    matrix = TfidfVectorizer(norm="l2").fit_transform(texts)
    out = (matrix @ matrix.T).toarray().astype(float)
    np.fill_diagonal(out, 1.0)
    return out


def _jaccard_from_keyword_lists(keyword_lists: list[list[str]]) -> np.ndarray:
    sets = [set(values) for values in keyword_lists]
    n = len(sets)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            union = sets[i] | sets[j]
            value = 0.0 if not union else len(sets[i] & sets[j]) / len(union)
            out[i, j] = value
            out[j, i] = value
    np.fill_diagonal(out, 1.0)
    return out


def _matrix_from_precomputed(dataset: CanonicalDataset, frame: pd.DataFrame) -> np.ndarray:
    required = {"call_id_1", "call_id_2", "similarity"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Precomputed similarities missing columns: {sorted(missing)}")
    ids = dataset.calls["call_id"].tolist()
    pos = {call_id: i for i, call_id in enumerate(ids)}
    out = np.eye(len(ids), dtype=float)
    for row in frame.itertuples(index=False):
        a = getattr(row, "call_id_1")
        b = getattr(row, "call_id_2")
        if a not in pos or b not in pos:
            continue
        value = float(getattr(row, "similarity"))
        out[pos[a], pos[b]] = value
        out[pos[b], pos[a]] = value
    return out


def _zscore_upper_triangle(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    idx = np.triu_indices_from(matrix, k=1)
    vals = matrix[idx].astype(float)
    if vals.size == 0:
        return matrix.copy()
    sd = float(vals.std())
    if sd == 0:
        return np.zeros_like(matrix, dtype=float)
    out = (matrix - float(vals.mean())) / sd
    np.fill_diagonal(out, 0.0)
    return out


def compute_similarity_matrix(dataset: CanonicalDataset, spec: SimilaritySpec) -> np.ndarray:
    calls = dataset.calls
    if spec.method == "precomputed":
        if spec.precomputed is None:
            raise ValueError("precomputed similarity requires a DataFrame")
        matrix = _matrix_from_precomputed(dataset, spec.precomputed)
    elif spec.method == "keyword_jaccard":
        col = "semantic_keywords" if spec.modality == "semantic" else "acoustic_keywords"
        matrix = _jaccard_from_keyword_lists(calls[col].tolist())
    elif spec.method == "tfidf_cosine":
        col = "semantic_description" if spec.modality == "semantic" else "acoustic_description"
        matrix = _cosine_from_texts(calls[col].fillna("").astype(str).tolist())
    else:
        raise ValueError(f"Unsupported similarity method: {spec.method}")

    if spec.transform == "zscore":
        return _zscore_upper_triangle(matrix)
    if spec.transform != "none":
        raise ValueError(f"Unsupported transform: {spec.transform}")
    return matrix

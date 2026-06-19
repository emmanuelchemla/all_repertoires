"""Reusable data and analysis primitives for repertoire explorer apps."""

from .datasets import CanonicalDataset, load_all_repertoires_json, load_apes_csv
from .paths import DATABASE_PATH, HUMAN_DATABASE_PATH
from .similarity import SimilaritySpec, compute_similarity_matrix
from .summaries import summarize_dataset, summarize_similarity

__all__ = [
    "CanonicalDataset",
    "DATABASE_PATH",
    "EMBEDDING_CACHE_PATH",
    "HUMAN_DATABASE_PATH",
    "SimilaritySpec",
    "compute_similarity_matrix",
    "load_all_repertoires_json",
    "load_apes_csv",
    "summarize_dataset",
    "summarize_similarity",
]

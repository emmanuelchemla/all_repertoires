"""Reusable data and analysis primitives for repertoire explorer apps."""

from .animallex_analysis import (
    AnalysisConfig,
    compute_acoustic_semantic_prediction,
    compute_cross_species_coverage,
    compute_cross_species_motifs,
    compute_description_embeddings,
    compute_form_meaning_alignment,
    compute_keyword_pmi,
    compute_overview,
    compute_pairwise_r,
    compute_species_pair_correlations,
    mantel,
    similarity_matrix,
)
from .bundle import AnimalLexBundle, load_bundle, source_hash, validate_bundle, write_bundle
from .datasets import (
    CanonicalDataset,
    load_all_repertoires_json,
    load_apes_csv,
    load_repertoire_yaml_directory,
)
from .paths import DATABASE_PATH, EMBEDDING_CACHE_PATH, HUMAN_DATABASE_PATH
from .similarity import SimilaritySpec, compute_similarity_matrix
from .summaries import summarize_dataset, summarize_similarity

__all__ = [
    "CanonicalDataset",
    "AnalysisConfig",
    "AnimalLexBundle",
    "DATABASE_PATH",
    "EMBEDDING_CACHE_PATH",
    "HUMAN_DATABASE_PATH",
    "SimilaritySpec",
    "compute_similarity_matrix",
    "compute_acoustic_semantic_prediction",
    "compute_description_embeddings",
    "compute_cross_species_coverage",
    "compute_cross_species_motifs",
    "compute_form_meaning_alignment",
    "compute_keyword_pmi",
    "compute_overview",
    "compute_pairwise_r",
    "compute_species_pair_correlations",
    "load_all_repertoires_json",
    "load_apes_csv",
    "load_bundle",
    "load_repertoire_yaml_directory",
    "mantel",
    "similarity_matrix",
    "source_hash",
    "summarize_dataset",
    "summarize_similarity",
    "validate_bundle",
    "write_bundle",
]

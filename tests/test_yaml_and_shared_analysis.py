from __future__ import annotations

from pathlib import Path

import numpy as np

from repertoire_explorer import (
    AnalysisConfig,
    compute_form_meaning_alignment,
    compute_keyword_pmi,
    compute_overview,
    load_repertoire_yaml_directory,
    similarity_matrix,
)
from repertoire_explorer.datasets import CANONICAL_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repertoires" / "llm_knowledge+search" / "species"


def test_yaml_loader_uses_existing_canonical_schema() -> None:
    dataset = load_repertoire_yaml_directory(SOURCE)

    assert dataset.calls.columns.tolist() == CANONICAL_COLUMNS
    assert len(dataset.calls) == 1507
    assert dataset.calls["species"].nunique() == 128
    assert dataset.calls["call_id"].is_unique
    assert dataset.calls["species"].str.contains(" ").all()
    assert dataset.species_metadata["Pan paniscus"]["common_name"] == "Bonobo"


def test_overview_and_pmi_use_explicit_keywords() -> None:
    dataset = load_repertoire_yaml_directory(SOURCE)

    overview = compute_overview(dataset)
    pmi = compute_keyword_pmi(dataset, minimum_calls=4)

    assert overview["n_calls"] == 1507
    assert overview["semantic_keywords"][0]["keyword"] == "attention"
    assert overview["semantic_keywords"][0]["percent_calls"] == (
        100 * overview["semantic_keywords"][0]["count"] / overview["n_calls"]
    )
    assert sum(row["n_species"] for row in overview["call_count_distribution"]) == 128
    bonobo = next(row for row in overview["species_counts"] if row["species"] == "Pan paniscus")
    assert bonobo["common_name"] == "Bonobo"
    assert bonobo["taxonomic_group"] == "Apes and gibbons"
    assert "high_frequency" in pmi["acoustic_keywords"]
    assert "alarm" in pmi["semantic_keywords"]
    assert np.asarray(pmi["p_values"]).shape == np.asarray(pmi["matrix"]).shape
    assert np.asarray(pmi["q_values"]).shape == np.asarray(pmi["matrix"]).shape
    assert np.array_equal(
        np.asarray(pmi["significant"]), np.asarray(pmi["q_values"]) < 0.05
    )
    assert np.all(np.asarray(pmi["q_values"]) >= np.asarray(pmi["p_values"]) - 1e-12)


def test_form_meaning_sampling_is_deterministic() -> None:
    dataset = load_repertoire_yaml_directory(SOURCE)
    dataset = type(dataset)(dataset.name, dataset.calls.iloc[:12].copy(), dataset.source_path)
    embeddings = np.arange(48, dtype=float).reshape(12, 4) + np.eye(12, 4)
    config = AnalysisConfig(n_permutations=3, scatter_sample_per_group=4)

    first = compute_form_meaning_alignment(dataset, embeddings, embeddings, config)
    second = compute_form_meaning_alignment(dataset, embeddings, embeddings, config)

    assert first == second
    np.testing.assert_allclose(similarity_matrix(embeddings), similarity_matrix(embeddings).T)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from repertoire_explorer import (
    AnalysisConfig,
    compute_acoustic_semantic_prediction,
    compute_cross_species_coverage,
    compute_form_meaning_alignment,
    compute_keyword_pmi,
    compute_overview,
    load_repertoire_yaml_directory,
    similarity_matrix,
)
from repertoire_explorer.datasets import CANONICAL_COLUMNS
from repertoire_explorer.datasets import CanonicalDataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repertoires" / "llm_knowledge+search" / "species"


def test_yaml_loader_uses_existing_canonical_schema() -> None:
    dataset = load_repertoire_yaml_directory(SOURCE)

    assert dataset.calls.columns.tolist() == CANONICAL_COLUMNS
    assert len(dataset.calls) == 1507
    assert dataset.calls["species"].nunique() == 128
    assert dataset.calls["call_id"].is_unique
    assert set(dataset.calls["confidence"]) == {"low", "medium", "high"}
    assert dataset.calls["species"].str.contains(" ").all()
    assert dataset.species_metadata["Pan paniscus"]["common_name"] == "Bonobo"


def test_overview_and_pmi_use_explicit_keywords() -> None:
    dataset = load_repertoire_yaml_directory(SOURCE)

    overview = compute_overview(dataset)
    pmi = compute_keyword_pmi(dataset, minimum_calls=4, n_permutations=19)

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


def test_pmi_uses_global_marginals_and_global_permutations() -> None:
    calls = pd.DataFrame(
        {
            "call_id": [f"call-{index}" for index in range(8)],
            "species": ["Species A"] * 4 + ["Species B"] * 4,
            "acoustic_keywords": [["loud"]] * 4 + [[] for _ in range(4)],
            "semantic_keywords": [["alarm"]] * 4 + [[] for _ in range(4)],
        }
    )
    dataset = CanonicalDataset("test", calls, Path("test"))

    result = compute_keyword_pmi(
        dataset,
        minimum_calls=1,
        n_permutations=31,
        random_seed=7,
    )

    assert result["joint_counts"] == [[4]]
    assert result["expected_counts"] == [[2.0]]
    assert result["matrix"] == [[1.0]]
    assert result["global_pmi_matrix"] == [[1.0]]
    assert result["p_values"] == [[0.125]]
    assert result["significance_test"] == "two-sided global permutation test"
    assert result["permutation_unit"] == (
        "complete acoustic keyword sets shuffled across all calls"
    )
    assert "among all calls" in result["null_model"]


def test_cross_species_coverage_uses_percentage_thresholds() -> None:
    calls = pd.DataFrame(
        {
            "call_id": [f"call-{index}" for index in range(4)],
            "species": ["Species A", "Species B", "Species C", "Species D"],
            "class": ["Mammalia"] * 4,
            "order": ["Test order"] * 4,
            "family": ["Test family"] * 4,
        }
    )
    dataset = CanonicalDataset("test", calls, Path("test"))
    semantic = np.eye(4)
    semantic[0, 1] = semantic[1, 0] = 0.9
    acoustic = np.eye(4)

    result = compute_cross_species_coverage(
        dataset,
        acoustic,
        semantic,
        thresholds=[0.8],
        default_threshold=0.8,
    )

    group = result["groups"]["all"]
    assert group["n_species"] == 4
    assert result["species_percentages"] == [25, 50, 75, 100]
    assert result["source_species_included"] is False
    assert group["n_target_species"] == 3
    assert group["semantic"]["percent_calls"][0] == [50.0, 0.0, 0.0, 0.0]
    assert group["acoustic"]["percent_calls"][0] == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]


def test_form_meaning_sampling_is_deterministic() -> None:
    dataset = load_repertoire_yaml_directory(SOURCE)
    dataset = type(dataset)(dataset.name, dataset.calls.iloc[:12].copy(), dataset.source_path)
    embeddings = np.arange(48, dtype=float).reshape(12, 4) + np.eye(12, 4)
    config = AnalysisConfig(n_permutations=3, scatter_sample_per_group=4)

    first = compute_form_meaning_alignment(dataset, embeddings, embeddings, config)
    second = compute_form_meaning_alignment(dataset, embeddings, embeddings, config)

    assert first == second
    np.testing.assert_allclose(similarity_matrix(embeddings), similarity_matrix(embeddings).T)


def test_prediction_uses_three_deterministic_holdout_schemes() -> None:
    rng = np.random.default_rng(11)
    embeddings = rng.normal(size=(30, 8))
    semantic = embeddings + rng.normal(scale=0.05, size=embeddings.shape)
    calls = pd.DataFrame(
        {
            "call_id": [f"call-{index}" for index in range(30)],
            "species": [f"Species {index // 5}" for index in range(30)],
            "family": [f"Family {index // 10}" for index in range(30)],
        }
    )
    dataset = CanonicalDataset("test", calls, Path("test"))
    config = AnalysisConfig(
        prediction_folds=3,
        prediction_ridge_alpha=1.0,
        prediction_bootstrap_samples=100,
    )

    first = compute_acoustic_semantic_prediction(dataset, embeddings, semantic, config)
    second = compute_acoustic_semantic_prediction(dataset, embeddings, semantic, config)

    assert first == second
    assert [condition["key"] for condition in first["conditions"]] == [
        "held_out_calls",
        "held_out_species",
        "held_out_families",
    ]
    assert all(condition["n_folds"] == 3 for condition in first["conditions"])
    assert first["models"] == {
        "random": "Random pairing",
        "retrieval": "Acoustic nearest neighbor",
        "ridge": "Linear ridge",
    }
    for condition in first["conditions"]:
        result = first["results"][condition["key"]]["ridge"]["cosine"]
        assert len(result["fold_values"]) == 3
        assert result["ci_low"] <= result["ci_high"]

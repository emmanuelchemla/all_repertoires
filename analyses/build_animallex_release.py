from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import numpy as np
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repertoire_explorer import (
    AnalysisConfig,
    compute_acoustic_semantic_prediction,
    compute_cross_species_coverage,
    compute_cross_species_motifs,
    compute_description_embeddings,
    compute_form_meaning_alignment,
    compute_keyword_form_meaning_alignment,
    compute_keyword_pmi,
    compute_overview,
    compute_species_pair_correlations,
    load_repertoire_yaml_directory,
    similarity_matrix,
    source_hash,
    validate_bundle,
    write_bundle,
)


DEFAULT_SOURCE = ROOT / "repertoires" / "llm_knowledge+search" / "species"
DEFAULT_OUTPUT = ROOT / "artifacts" / "animallex" / "latest"
DEFAULT_CACHE = ROOT / "cache" / "animallex_embedding_cache.json"
FULL_PERMUTATIONS = 9999
CONFIDENCE_FILTERS = {
    "medium_plus": {"medium", "high"},
    "high": {"high"},
}
SPECIES_FIT_FILTERS = {
    "include": {"include"},
    "include_caution": {"include", "caution"},
}


def compute_analysis_view(
    dataset,
    acoustic_embeddings,
    semantic_embeddings,
    acoustic_similarity,
    semantic_similarity,
    config: AnalysisConfig,
):
    thresholds = [
        round(index * config.coverage_threshold_step, 2)
        for index in range(int(round(1 / config.coverage_threshold_step)) + 1)
    ]
    return {
        "overview": compute_overview(dataset),
        "coverage": compute_cross_species_coverage(
            dataset,
            acoustic_similarity,
            semantic_similarity,
            thresholds=thresholds,
            default_threshold=config.coverage_default_threshold,
        ),
        "motifs": compute_cross_species_motifs(
            dataset, acoustic_similarity, semantic_similarity, config
        ),
        "form_meaning": compute_form_meaning_alignment(
            dataset, acoustic_embeddings, semantic_embeddings, config
        ),
        "form_meaning_keywords": compute_keyword_form_meaning_alignment(
            dataset, config
        ),
        "prediction": compute_acoustic_semantic_prediction(
            dataset, acoustic_embeddings, semantic_embeddings, config
        ),
        "species_matrix": compute_species_pair_correlations(
            dataset,
            acoustic_similarity,
            semantic_similarity,
            minimum_pairs=config.species_pair_minimum,
        ),
        "pmi": compute_keyword_pmi(
            dataset,
            minimum_calls=config.pmi_minimum,
            alpha=config.pmi_alpha,
            n_permutations=config.n_permutations,
            random_seed=config.random_seed,
        ),
    }


def compute_filtered_analysis(
    dataset,
    acoustic_embeddings,
    semantic_embeddings,
    acoustic_similarity,
    semantic_similarity,
    config: AnalysisConfig,
):
    analysis = compute_analysis_view(
        dataset,
        acoustic_embeddings,
        semantic_embeddings,
        acoustic_similarity,
        semantic_similarity,
        config,
    )
    confidence_views = {}
    for key, accepted_values in CONFIDENCE_FILTERS.items():
        selected = dataset.calls["confidence"].isin(accepted_values).to_numpy()
        selected_indices = selected.nonzero()[0]
        filtered_dataset = type(dataset)(
            dataset.name,
            dataset.calls.loc[selected].reset_index(drop=True),
            dataset.source_path,
            {
                name: metadata
                for name, metadata in dataset.species_metadata.items()
                if name in set(dataset.calls.loc[selected, "species"])
            },
        )
        confidence_views[key] = compute_analysis_view(
            filtered_dataset,
            acoustic_embeddings[selected],
            semantic_embeddings[selected],
            acoustic_similarity[np.ix_(selected_indices, selected_indices)],
            semantic_similarity[np.ix_(selected_indices, selected_indices)],
            config,
        )
    analysis["confidence_views"] = confidence_views
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the shared AnimalLex analysis bundle.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    permutation_group = parser.add_mutually_exclusive_group()
    permutation_group.add_argument(
        "--permutations",
        type=int,
        help="Override the iteration-mode permutation count (default: 999).",
    )
    permutation_group.add_argument(
        "--full",
        action="store_true",
        help="Build publication results with 9,999 permutations.",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    args = parse_args()
    n_permutations = (
        FULL_PERMUTATIONS
        if args.full
        else (
            args.permutations
            if args.permutations is not None
            else AnalysisConfig().n_permutations
        )
    )
    config = AnalysisConfig(n_permutations=n_permutations)
    if args.validate_only:
        validate_bundle(args.output, args.source, expected_config=config.to_dict())
        print(f"Bundle is current: {args.output}")
        return

    dataset = load_repertoire_yaml_directory(args.source)
    eligible = dataset.calls[
        dataset.calls["acoustic_description"].str.strip().ne("")
        & dataset.calls["semantic_description"].str.strip().ne("")
    ].copy()
    excluded = len(dataset.calls) - len(eligible)
    dataset = type(dataset)(
        dataset.name,
        eligible.reset_index(drop=True),
        dataset.source_path,
        dataset.species_metadata,
    )
    acoustic_embeddings = compute_description_embeddings(
        dataset.calls["acoustic_description"].tolist(),
        model_name=config.embedding_model,
        cache_path=args.cache,
    )
    semantic_embeddings = compute_description_embeddings(
        dataset.calls["semantic_description"].tolist(),
        model_name=config.embedding_model,
        cache_path=args.cache,
    )
    acoustic_similarity = similarity_matrix(acoustic_embeddings)
    semantic_similarity = similarity_matrix(semantic_embeddings)
    analysis = compute_filtered_analysis(
        dataset,
        acoustic_embeddings,
        semantic_embeddings,
        acoustic_similarity,
        semantic_similarity,
        config,
    )
    species_fit_views = {}
    for key, accepted_values in SPECIES_FIT_FILTERS.items():
        selected = dataset.calls["species_fit"].isin(accepted_values).to_numpy()
        selected_indices = selected.nonzero()[0]
        filtered_dataset = type(dataset)(
            dataset.name,
            dataset.calls.loc[selected].reset_index(drop=True),
            dataset.source_path,
            {
                name: metadata
                for name, metadata in dataset.species_metadata.items()
                if name in set(dataset.calls.loc[selected, "species"])
            },
        )
        species_fit_views[key] = compute_filtered_analysis(
            filtered_dataset,
            acoustic_embeddings[selected],
            semantic_embeddings[selected],
            acoustic_similarity[np.ix_(selected_indices, selected_indices)],
            semantic_similarity[np.ix_(selected_indices, selected_indices)],
            config,
        )
    analysis["species_fit_views"] = species_fit_views
    manifest = {
        "dataset": dataset.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.relative_to(ROOT)),
        "source_hash": source_hash(args.source),
        "source_commit": git_commit(),
        "n_calls": int(len(dataset.calls)),
        "n_species": int(dataset.calls["species"].nunique()),
        "confidence_counts": {
            value: int((dataset.calls["confidence"] == value).sum())
            for value in ("high", "medium", "low")
        },
        "species_fit_counts": {
            value: int((dataset.calls["species_fit"] == value).sum())
            for value in ("include", "caution", "exclude")
        },
        "excluded_calls": int(excluded),
        "build_mode": "full" if args.full else "iteration",
        "config": config.to_dict(),
    }
    write_bundle(
        args.output,
        dataset,
        analysis,
        manifest,
        acoustic_embeddings=acoustic_embeddings,
        semantic_embeddings=semantic_embeddings,
        acoustic_similarity=acoustic_similarity,
        semantic_similarity=semantic_similarity,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

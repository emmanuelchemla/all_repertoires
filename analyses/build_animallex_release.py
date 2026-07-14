from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repertoire_explorer import (
    AnalysisConfig,
    compute_cross_species_coverage,
    compute_description_embeddings,
    compute_form_meaning_alignment,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the shared AnimalLex analysis bundle.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--permutations", type=int, default=9999)
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
    config = AnalysisConfig(n_permutations=args.permutations)
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
    analysis = {
        "overview": compute_overview(dataset),
        "coverage": compute_cross_species_coverage(
            dataset,
            acoustic_similarity,
            semantic_similarity,
            thresholds=[
                round(index * config.coverage_threshold_step, 2)
                for index in range(
                    int(round(1 / config.coverage_threshold_step)) + 1
                )
            ],
            default_threshold=config.coverage_default_threshold,
        ),
        "form_meaning": compute_form_meaning_alignment(
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
    manifest = {
        "dataset": dataset.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.relative_to(ROOT)),
        "source_hash": source_hash(args.source),
        "source_commit": git_commit(),
        "n_calls": int(len(dataset.calls)),
        "n_species": int(dataset.calls["species"].nunique()),
        "excluded_calls": int(excluded),
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

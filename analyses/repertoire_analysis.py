from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repertoire_explorer import (
    DATABASE_PATH,
    SimilaritySpec,
    compute_similarity_matrix,
    load_all_repertoires_json,
    load_apes_csv,
    summarize_dataset,
    summarize_similarity,
)


DEFAULT_APES = ROOT.parent / "apes_comparison" / "paper" / "code" / "all_species_calls_analysis.csv"
DEFAULT_ALL_REPERTOIRES = DATABASE_PATH


def _load_precomputed(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    frame_path = Path(path)
    if frame_path.suffix.lower() == ".json":
        return pd.read_json(frame_path)
    return pd.read_csv(frame_path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run generic repertoire analyses over apes_comparison or all_repertoires data."
    )
    parser.add_argument(
        "--dataset",
        choices=["apes", "all_repertoires"],
        default="apes",
        help="Input database type.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to input CSV/JSON. Defaults to the known repo-local database.",
    )
    parser.add_argument(
        "--family",
        default=None,
        help="Optional taxonomic family filter, useful for all_repertoires.",
    )
    parser.add_argument(
        "--semantic-method",
        choices=["tfidf_cosine", "keyword_jaccard", "precomputed"],
        default="tfidf_cosine",
    )
    parser.add_argument(
        "--acoustic-method",
        choices=["tfidf_cosine", "keyword_jaccard", "precomputed"],
        default="tfidf_cosine",
    )
    parser.add_argument("--semantic-precomputed", default=None)
    parser.add_argument("--acoustic-precomputed", default=None)
    parser.add_argument(
        "--transform",
        choices=["none", "zscore"],
        default="none",
        help="Similarity transform applied independently to each matrix.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path. If omitted, prints to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    if input_path is None:
        input_path = DEFAULT_APES if args.dataset == "apes" else DEFAULT_ALL_REPERTOIRES

    if args.dataset == "apes":
        dataset = load_apes_csv(input_path)
    else:
        dataset = load_all_repertoires_json(input_path)
    if args.family:
        dataset = dataset.subset(family=args.family)

    sem = compute_similarity_matrix(
        dataset,
        SimilaritySpec(
            modality="semantic",
            method=args.semantic_method,
            transform=args.transform,
            precomputed=_load_precomputed(args.semantic_precomputed),
        ),
    )
    ac = compute_similarity_matrix(
        dataset,
        SimilaritySpec(
            modality="acoustic",
            method=args.acoustic_method,
            transform=args.transform,
            precomputed=_load_precomputed(args.acoustic_precomputed),
        ),
    )
    payload = {
        "dataset": summarize_dataset(dataset),
        "semantic_similarity": summarize_similarity(dataset, sem),
        "acoustic_similarity": summarize_similarity(dataset, ac),
        "config": {
            "dataset": args.dataset,
            "input": str(input_path),
            "family": args.family,
            "semantic_method": args.semantic_method,
            "acoustic_method": args.acoustic_method,
            "transform": args.transform,
        },
    }
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

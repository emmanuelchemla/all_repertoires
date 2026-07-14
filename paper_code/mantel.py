"""Compatibility access to the shared AnimalLex Mantel analysis."""

from __future__ import annotations

from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repertoire_explorer.animallex_analysis import (
    mantel as shared_mantel,
    similarity_matrix as shared_similarity_matrix,
)
from repertoire_explorer import load_bundle


N_PERM = 9999


def mantel(a_vec, s_vec, n_perm=N_PERM, seed=42):
    return shared_mantel(a_vec, s_vec, n_perm=n_perm, seed=seed)


def similarity_matrix(embeddings):
    return shared_similarity_matrix(embeddings)


def main() -> None:
    bundle = load_bundle(ROOT / "artifacts" / "animallex" / "latest")
    print(json.dumps(bundle.analysis["form_meaning"]["groups"], indent=2))


if __name__ == "__main__":
    main()

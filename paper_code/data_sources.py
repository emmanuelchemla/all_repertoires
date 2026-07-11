"""Shared data-source helpers for paper figure scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
PAPER_CODE = ROOT / "paper_code"

DATA_SOURCES = ("old", "new")


def _list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return [str(value)]


def _old_calls() -> list[dict[str, Any]]:
    with open(ROOT / "database.json", encoding="utf-8") as f:
        db = json.load(f)

    calls: list[dict[str, Any]] = []
    for species_entry in db.get("species", []):
        taxonomy = species_entry.get("taxonomy", {}) or {}
        tax_kingdom = taxonomy.get("kingdom", species_entry.get("kingdom", ""))
        tax_phylum = taxonomy.get("phylum", species_entry.get("phylum", ""))
        tax_class = taxonomy.get("class", species_entry.get("class", ""))
        tax_order = taxonomy.get("order", species_entry.get("order", ""))
        tax_family = taxonomy.get("family", species_entry.get("family", ""))
        tax_genus = taxonomy.get("genus", species_entry.get("genus", ""))
        species_name = species_entry.get("species_name", "unknown")

        for call in species_entry.get("calls", []):
            semantic_keywords = _list(call.get("ontology_keywords"))
            calls.append(
                {
                    **call,
                    "species": species_name,
                    "call_name": call.get("call_name", call.get("name", "unknown")),
                    "acoustic_description": call.get("acoustic_description", ""),
                    "semantic_description": call.get("semantic_description", ""),
                    "acoustic_keywords": [],
                    "semantic_keywords": semantic_keywords,
                    "ontology_keywords": semantic_keywords,
                    "taxonomy": taxonomy,
                    "kingdom": tax_kingdom,
                    "phylum": tax_phylum,
                    "class": tax_class,
                    "order": tax_order,
                    "family": tax_family,
                    "genus": tax_genus,
                }
            )
    return calls


def _new_calls() -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for path in sorted((ROOT / "repertoires" / "species").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        taxonomy = data.get("taxonomy", {}) or {}
        scientific_name = data.get("scientific_name", path.stem.replace("-", " "))
        common_name = data.get("common_name", scientific_name)
        species_name = f"{common_name} ({scientific_name})"

        for call in data.get("calls") or []:
            semantic_keywords = _list(call.get("semantic_keywords"))
            acoustic_keywords = _list(call.get("acoustic_keywords"))
            calls.append(
                {
                    **call,
                    "species": species_name,
                    "call_name": call.get("name", call.get("call_name", "unknown")),
                    "acoustic_description": call.get("acoustic_description", ""),
                    "semantic_description": call.get("semantic_description", ""),
                    "acoustic_keywords": acoustic_keywords,
                    "semantic_keywords": semantic_keywords,
                    "ontology_keywords": semantic_keywords,
                    "taxonomy": taxonomy,
                    "kingdom": taxonomy.get("kingdom", ""),
                    "phylum": taxonomy.get("phylum", ""),
                    "class": taxonomy.get("class", ""),
                    "order": taxonomy.get("order", ""),
                    "family": taxonomy.get("family", ""),
                    "genus": taxonomy.get("genus", ""),
                }
            )
    return calls


def load_calls(data_source: str = "old") -> list[dict[str, Any]]:
    """Load and flatten calls from either supported source."""
    if data_source == "old":
        calls = _old_calls()
    elif data_source == "new":
        calls = _new_calls()
    else:
        raise ValueError(f"Unknown data source {data_source!r}; expected one of {DATA_SOURCES}")

    if not calls:
        raise ValueError(f"No calls loaded for data source {data_source!r}")
    return calls


def artifact_path(kind: str, data_source: str, suffix: str) -> Path:
    """Return a dataset-specific paper_code artifact path."""
    return PAPER_CODE / f"{kind}_{data_source}.{suffix}"


def load_embedding_artifacts(data_source: str, n_calls: int) -> tuple[np.ndarray, np.ndarray]:
    ac_path = artifact_path("ac_emb", data_source, "npy")
    se_path = artifact_path("se_emb", data_source, "npy")
    missing = [str(p) for p in (ac_path, se_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing dataset-specific embedding artifact(s): "
            + ", ".join(missing)
            + ". Run `python paper_code/mantel.py --data-source "
            + data_source
            + "` first."
        )

    ac_emb = np.load(ac_path)
    se_emb = np.load(se_path)
    if len(ac_emb) != n_calls or len(se_emb) != n_calls:
        raise ValueError(
            f"Embedding/call count mismatch for {data_source!r}: "
            f"{len(ac_emb)} acoustic rows, {len(se_emb)} semantic rows, {n_calls} calls"
        )
    return ac_emb, se_emb


def load_mantel_results(data_source: str) -> dict[str, Any]:
    path = artifact_path("mantel_results", data_source, "json")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset-specific Mantel results: {path}. "
            f"Run `python paper_code/mantel.py --data-source {data_source}` first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)

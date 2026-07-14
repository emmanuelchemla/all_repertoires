from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


CANONICAL_COLUMNS = [
    "dataset",
    "call_id",
    "species",
    "call_name",
    "semantic_description",
    "acoustic_description",
    "semantic_keywords",
    "acoustic_keywords",
    "ontology_keywords",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "source",
]


@dataclass(frozen=True)
class CanonicalDataset:
    """Canonical call table plus enough metadata to drive reusable analyses."""

    name: str
    calls: pd.DataFrame
    source_path: Path
    species_metadata: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def species(self) -> list[str]:
        return sorted(self.calls["species"].dropna().astype(str).unique().tolist())

    @property
    def families(self) -> list[str]:
        if "family" not in self.calls:
            return []
        values = self.calls["family"].fillna("").astype(str).str.strip()
        return sorted(v for v in values.unique().tolist() if v)

    def subset(self, *, family: str | None = None, species: Iterable[str] | None = None) -> "CanonicalDataset":
        df = self.calls
        label_parts = [self.name]
        if family:
            df = df[df["family"].fillna("").astype(str) == family].copy()
            label_parts.append(f"family={family}")
        if species:
            species_set = set(species)
            df = df[df["species"].isin(species_set)].copy()
            label_parts.append(f"species={len(species_set)}")
        return CanonicalDataset(
            name=" | ".join(label_parts),
            calls=_finalize_calls(df),
            source_path=self.source_path,
            species_metadata={
                name: metadata
                for name, metadata in self.species_metadata.items()
                if name in set(df["species"])
            },
        )


def _split_keywords(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        raw = value
    else:
        if value is None or pd.isna(value):
            return []
        raw = str(value).replace(",", "|").split("|")
    out = []
    for item in raw:
        text = str(item).strip()
        if text:
            out.append(text)
    return sorted(set(out))


def _finalize_calls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = "" if not col.endswith("keywords") else [[] for _ in range(len(df))]
    for col in ["semantic_keywords", "acoustic_keywords", "ontology_keywords"]:
        df[col] = df[col].apply(_split_keywords)
    df["species"] = df["species"].fillna("").astype(str)
    df["call_name"] = df["call_name"].fillna("").astype(str)
    df["semantic_description"] = df["semantic_description"].fillna("").astype(str)
    df["acoustic_description"] = df["acoustic_description"].fillna("").astype(str)
    if "call_id" not in df or (df["call_id"].fillna("").astype(str) == "").any():
        df["call_id"] = [
            f"{row.species}|||{row.call_name}|||{i}"
            for i, row in enumerate(df.itertuples(index=False))
        ]
    return df[CANONICAL_COLUMNS].reset_index(drop=True)


def load_apes_csv(path: Path | str, *, name: str = "apes_comparison") -> CanonicalDataset:
    path = Path(path)
    raw = pd.read_csv(path)
    df = pd.DataFrame(
        {
            "dataset": name,
            "species": raw["species"],
            "call_name": raw["call_name"],
            "semantic_description": raw["context_description"],
            "acoustic_description": raw["acoustic_description"],
            "semantic_keywords": raw.get("semantic_keywords", ""),
            "acoustic_keywords": raw.get("acoustic_keywords", ""),
            "ontology_keywords": raw.get("semantic_keywords", ""),
            "family": "Hominidae",
            "source": raw.get("source_file", path.name),
        }
    )
    return CanonicalDataset(name=name, calls=_finalize_calls(df), source_path=path)


def load_all_repertoires_json(path: Path | str, *, name: str = "all_repertoires") -> CanonicalDataset:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for species_entry in payload.get("species", []):
        taxonomy = species_entry.get("taxonomy", {}) or {}
        species_name = species_entry.get("species_name", "unknown")
        tax = {
            "kingdom": taxonomy.get("kingdom", species_entry.get("kingdom", "")),
            "phylum": taxonomy.get("phylum", species_entry.get("phylum", "")),
            "class": taxonomy.get("class", species_entry.get("class", "")),
            "order": taxonomy.get("order", species_entry.get("order", "")),
            "family": taxonomy.get("family", species_entry.get("family", "")),
            "genus": taxonomy.get("genus", species_entry.get("genus", "")),
        }
        for call in species_entry.get("calls", []):
            rows.append(
                {
                    "dataset": name,
                    "species": species_name,
                    "call_name": call.get("call_name", "unknown"),
                    "semantic_description": call.get("semantic_description", ""),
                    "acoustic_description": call.get("acoustic_description", ""),
                    "semantic_keywords": call.get("ontology_keywords", []),
                    "acoustic_keywords": [],
                    "ontology_keywords": call.get("ontology_keywords", []),
                    "source": "; ".join(call.get("scientific_references", [])),
                    **tax,
                }
            )
    return CanonicalDataset(name=name, calls=_finalize_calls(pd.DataFrame(rows)), source_path=path)


def load_repertoire_yaml_directory(
    path: Path | str,
    *,
    name: str = "llm_knowledge+search",
) -> CanonicalDataset:
    """Load species repertoire YAML files into the existing canonical table."""
    path = Path(path)
    rows: list[dict[str, object]] = []
    species_metadata: dict[str, dict[str, str]] = {}
    for yaml_path in sorted(path.glob("*.yaml")):
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        taxonomy = payload.get("taxonomy", {}) or {}
        scientific_name = str(payload.get("scientific_name") or yaml_path.stem)
        species_metadata[scientific_name] = {
            "common_name": str(payload.get("common_name") or scientific_name),
        }
        tax = {
            "kingdom": taxonomy.get("kingdom", ""),
            "phylum": taxonomy.get("phylum", ""),
            "class": taxonomy.get("class", ""),
            "order": taxonomy.get("order", ""),
            "family": taxonomy.get("family", ""),
            "genus": taxonomy.get("genus", ""),
        }
        for call_index, call in enumerate(payload.get("calls", []) or []):
            call_name = str(call.get("name") or "unknown")
            references = call.get("references", []) or []
            rows.append(
                {
                    "dataset": name,
                    "call_id": f"{yaml_path.stem}|||{call_name}|||{call_index}",
                    "species": scientific_name,
                    "call_name": call_name,
                    "semantic_description": call.get("semantic_description", ""),
                    "acoustic_description": call.get("acoustic_description", ""),
                    "semantic_keywords": call.get("semantic_keywords", []),
                    "acoustic_keywords": call.get("acoustic_keywords", []),
                    "ontology_keywords": call.get("semantic_keywords", []),
                    "source": "; ".join(str(ref) for ref in references),
                    **tax,
                }
            )
    return CanonicalDataset(
        name=name,
        calls=_finalize_calls(pd.DataFrame(rows)),
        source_path=path,
        species_metadata=species_metadata,
    )

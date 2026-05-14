"""Load curated repertoires from repertoires/species/*.yaml into a flat call table.

Single source of truth for the main-paper analysis. Reads YAML files only — no
dependency on the legacy database.json or any cached embeddings.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPECIES_DIR = ROOT / "repertoires" / "species"


@dataclass(frozen=True)
class Call:
    species: str              # scientific name
    common_name: str
    class_: str               # taxonomy class
    order: str
    family: str
    genus: str
    name: str                 # call name
    acoustic_keywords: tuple[str, ...]
    semantic_keywords: tuple[str, ...]
    acoustic_description: str
    semantic_description: str


def load_calls() -> list[Call]:
    paths = sorted(glob.glob(str(SPECIES_DIR / "*.yaml")))
    if not paths:
        raise FileNotFoundError(f"no YAML repertoires under {SPECIES_DIR}")
    out: list[Call] = []
    for p in paths:
        with open(p) as f:
            doc = yaml.safe_load(f)
        tx = doc["taxonomy"]
        for c in doc["calls"]:
            out.append(Call(
                species=doc["scientific_name"],
                common_name=doc["common_name"],
                class_=tx["class"],
                order=tx["order"],
                family=tx["family"],
                genus=tx["genus"],
                name=c["name"],
                acoustic_keywords=tuple(c["acoustic_keywords"]),
                semantic_keywords=tuple(c["semantic_keywords"]),
                acoustic_description=c["acoustic_description"],
                semantic_description=c["semantic_description"],
            ))
    return out


def species_table(calls: list[Call]) -> list[dict]:
    """One row per species with class/family/order/n_calls."""
    by_sp: dict[str, dict] = {}
    for c in calls:
        d = by_sp.setdefault(c.species, {
            "species": c.species, "common_name": c.common_name,
            "class": c.class_, "order": c.order, "family": c.family,
            "genus": c.genus, "n_calls": 0,
        })
        d["n_calls"] += 1
    return sorted(by_sp.values(), key=lambda d: (d["class"], d["family"], d["species"]))

"""
Build database.json (legacy schema used by the paper figures) from the
curated YAML repertoires under ``repertoires/species/``.

Run this whenever the YAML repertoire database is updated; then re-run
the figure scripts in this directory.

    python paper_code/build_database.py

Output: ``database.json`` at the repository root.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
SPECIES_DIR = ROOT / "repertoires" / "species"
OUT_PATH = ROOT / "database.json"

# Map curated agreement labels (call_type_existence_agreement) to the
# coarse "subjective_reliability" tier used by legacy scripts.
RELIABILITY_MAP = {"strong": "high", "medium": "medium", "weak": "low"}


def _format_species_name(scientific: str, common: str) -> str:
    """Legacy display name: "Common name (Scientific name)"."""
    sci = scientific.strip()
    com = (common or sci).strip()
    return f"{com} ({sci})"


def _flatten_ref(ref: dict) -> str:
    return ref.get("url") or ref.get("id", "")


def convert_call(call: dict) -> dict:
    ac_kw = call.get("acoustic_keywords", []) or []
    se_kw = call.get("semantic_keywords", []) or []

    # Ensure acoustic keywords appear in the textual description so that
    # PMI scripts which grep the description for words like "tonal",
    # "broadband", "frequency-modulated" still find them.
    ac_desc = (call.get("acoustic_description") or "").strip()
    extra_terms = [k.replace("_", " ") for k in ac_kw]
    if extra_terms:
        suffix = "Acoustic features: " + ", ".join(extra_terms) + "."
        if suffix.lower() not in ac_desc.lower():
            ac_desc = (ac_desc + " " + suffix).strip()

    reliability = RELIABILITY_MAP.get(
        call.get("call_type_existence_agreement", ""), "medium"
    )

    scope = call.get("scope", {}) or {}
    sexes = scope.get("sexes") or []
    life_stages = scope.get("life_stages") or []
    users_parts = []
    if sexes and set(sexes) != {"female", "male", "unknown"} - {"unknown"}:
        users_parts.append("/".join(sexes))
    if life_stages and life_stages != ["adult"]:
        users_parts.append("/".join(life_stages))
    users = ", ".join(users_parts) or "Both sexes"

    return {
        "call_name": call.get("name", ""),
        "acoustic_description": ac_desc,
        "semantic_description": call.get("semantic_description", ""),
        # Legacy "ontology_keywords" was used by scripts as a semantic
        # tag set; the new schema separates them, so we forward the
        # semantic keywords here.
        "ontology_keywords": list(se_kw),
        "acoustic_keywords": list(ac_kw),
        "semantic_keywords": list(se_kw),
        "scientific_references": [
            _flatten_ref(r)
            for r in (call.get("acoustic_references", []) + call.get("semantic_references", []))
        ],
        "subjective_reliability": reliability,
        "comments": call.get("call_type_existence_explanation", ""),
        "users": users,
    }


def convert_species(doc: dict) -> dict:
    tax = doc.get("taxonomy", {}) or {}
    name = _format_species_name(doc["scientific_name"], doc.get("common_name", ""))
    return {
        "species_name": name,
        "class": tax.get("class", ""),
        "order": tax.get("order", ""),
        "family": tax.get("family", ""),
        "genus": tax.get("genus", ""),
        "taxonomy_basis": {
            "summary": (doc.get("primary_inventory") or {}).get("rationale", ""),
            "scientific_references": [],
        },
        "calls": [convert_call(c) for c in doc.get("calls", [])],
        "completeness_warning": "",
    }


def build() -> dict:
    files = sorted(SPECIES_DIR.glob("*.yaml"))
    if not files:
        raise SystemExit(f"No species YAMLs found in {SPECIES_DIR}")
    species = []
    for path in files:
        with path.open() as f:
            doc = yaml.safe_load(f)
        species.append(convert_species(doc))
    species.sort(key=lambda s: s["species_name"].lower())
    return {
        "scope_notes": {
            "source": "Generated from repertoires/species/*.yaml. Do not edit by hand.",
            "n_species": len(species),
            "n_calls": sum(len(s["calls"]) for s in species),
        },
        "species": species,
    }


def _refresh_embeddings(db: dict) -> None:
    """Recompute acoustic/semantic sentence embeddings cached as .npy.

    Several figure scripts (fig_embeddings, fig_landmark_*, fig_human_bursts)
    consume pre-computed ``paper_code/ac_emb.npy`` and ``se_emb.npy`` that
    must align row-for-row with the call order in ``database.json``. We
    regenerate them here so the entire pipeline stays consistent.
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - optional
        print(f"  (skipping embedding refresh: {exc})")
        return

    calls = [c for s in db["species"] for c in s["calls"]]
    ac_texts = [c.get("acoustic_description", "") for c in calls]
    se_texts = [c.get("semantic_description", "") for c in calls]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ac = model.encode(ac_texts, show_progress_bar=False, normalize_embeddings=True)
    se = model.encode(se_texts, show_progress_bar=False, normalize_embeddings=True)
    out_dir = Path(__file__).parent
    np.save(out_dir / "ac_emb.npy", ac.astype(np.float32))
    np.save(out_dir / "se_emb.npy", se.astype(np.float32))
    print(f"  Wrote ac_emb.npy / se_emb.npy ({ac.shape[0]} × {ac.shape[1]})")


def main() -> None:
    db = build()
    OUT_PATH.write_text(json.dumps(db, indent=2, ensure_ascii=False))
    n_sp = len(db["species"])
    n_calls = sum(len(s["calls"]) for s in db["species"])
    by_class: dict[str, int] = {}
    for s in db["species"]:
        by_class[s["class"]] = by_class.get(s["class"], 0) + 1
    print(f"Wrote {OUT_PATH} — {n_sp} species, {n_calls} calls")
    for cls, n in sorted(by_class.items()):
        print(f"  {cls or '(unknown)'}: {n}")
    _refresh_embeddings(db)


if __name__ == "__main__":
    main()

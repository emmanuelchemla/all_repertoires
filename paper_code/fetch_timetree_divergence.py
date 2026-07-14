"""
Fetch pairwise divergence times from the TimeTree REST API.

The script resolves each database species/taxon to a TimeTree/NCBI taxon ID,
then fetches pairwise divergence estimates for all represented pairs.

Outputs:
  plots/timetree_taxon_resolution.csv
  plots/timetree_divergence_pairs.csv
  plots/timetree_cache/*.json

TimeTree API documentation is described in:
  Kumar et al. 2022, TimeTree 5: An Expanded Resource for Species Divergence
  Times, Molecular Biology and Evolution.
"""

from __future__ import annotations

import itertools
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent
try:
    from paper_code.paths import DATABASE_PATH
except ModuleNotFoundError:
    from paths import DATABASE_PATH
DB_PATH = DATABASE_PATH
OUT_DIR = ROOT / "plots"
CACHE_DIR = OUT_DIR / "timetree_cache"
BASE_URL = "http://timetree.temple.edu/api"
REQUEST_SLEEP_SECONDS = 0.05


# Composite database entries need a representative TimeTree query. These are not
# paper-grade choices; they make the exploratory analysis use API-derived node
# ages wherever possible while preserving traceability in the resolution CSV.
MANUAL_QUERY_OVERRIDES = {
    "Poison dart frogs (Dendrobatidae; various genera)": "Dendrobatidae",
    "Red colobus (Piliocolobus spp.)": "Piliocolobus",
    "Langurs (Semnopithecus spp. / Trachypithecus spp.)": "Semnopithecus",
    "Gibbons (Hylobates spp., Nomascus spp., Hoolock spp.)": "Hylobates",
    "Brown lemurs (Eulemur spp.)": "Eulemur",
    "Mouse lemurs (Microcebus spp.)": "Microcebus",
    "Howler monkey (Alouatta spp.)": "Alouatta",
}


def load_species() -> list[dict[str, str]]:
    with open(DB_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    rows = []
    seen = set()
    for entry in db.get("species", []):
        species = str(entry.get("species_name", "unknown"))
        if species in seen:
            continue
        seen.add(species)
        rows.append(
            {
                "species": species,
                "class": str(entry.get("class", "") or ""),
                "order": str(entry.get("order", "") or ""),
                "family": str(entry.get("family", "") or ""),
                "genus": str(entry.get("genus", "") or ""),
            }
        )
    return rows


def infer_query(species_name: str) -> tuple[str, str]:
    if species_name in MANUAL_QUERY_OVERRIDES:
        return MANUAL_QUERY_OVERRIDES[species_name], "manual_composite_override"

    match = re.search(r"\(([^)]+)\)", species_name)
    if not match:
        return species_name, "display_name"

    scientific = match.group(1).strip()
    scientific = scientific.replace("’", "'")
    if ";" in scientific:
        scientific = scientific.split(";", 1)[0].strip()
    if "/" in scientific:
        scientific = scientific.split("/", 1)[0].strip()
    scientific = re.sub(r"\bspp?\.$", "", scientific).strip()
    return scientific, "parenthetical_scientific_name"


def cache_path(kind: str, key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")
    return CACHE_DIR / f"{kind}_{safe}.json"


def strip_php_notice(payload: str) -> str:
    start = payload.find("{")
    end = payload.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return payload
    return payload[start : end + 1]


def fetch_json(url: str, path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read().decode("utf-8", errors="replace")
    data = json.loads(strip_php_notice(payload))
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    time.sleep(REQUEST_SLEEP_SECONDS)
    return data


def resolve_taxon(species: str) -> dict[str, object]:
    query, query_source = infer_query(species)
    encoded = urllib.parse.quote_plus(query)
    url = f"{BASE_URL}/taxon/{encoded}"
    path = cache_path("taxon", query)

    try:
        data = fetch_json(url, path)
        return {
            "species": species,
            "timetree_query": query,
            "query_source": query_source,
            "resolved": True,
            "taxon_id": data.get("taxon_id") or data.get("ncbi_id"),
            "ncbi_id": data.get("ncbi_id"),
            "scientific_name": data.get("scientific_name"),
            "taxonomic_rank": data.get("taxonomic_rank"),
            "raw_error": "",
        }
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        return {
            "species": species,
            "timetree_query": query,
            "query_source": query_source,
            "resolved": False,
            "taxon_id": None,
            "ncbi_id": None,
            "scientific_name": "",
            "taxonomic_rank": "",
            "raw_error": repr(exc),
        }


def fetch_pairwise(row_a: pd.Series, row_b: pd.Series) -> dict[str, object]:
    species_a = row_a["species"]
    species_b = row_b["species"]
    id_a = int(row_a["taxon_id"])
    id_b = int(row_b["taxon_id"])
    url = f"{BASE_URL}/pairwise/{id_a}/{id_b}/summaryjson"
    path = cache_path("pairwise", f"{id_a}_{id_b}")

    base = {
        "species_a": species_a,
        "species_b": species_b,
        "timetree_taxon_id_a": id_a,
        "timetree_taxon_id_b": id_b,
        "timetree_query_a": row_a["timetree_query"],
        "timetree_query_b": row_b["timetree_query"],
        "timetree_name_a": row_a["scientific_name"],
        "timetree_name_b": row_b["scientific_name"],
    }

    try:
        data = fetch_json(url, path)
        age = data.get("precomputed_age")
        if age in ("", None):
            age = data.get("adjusted_age")
        return {
            **base,
            "timetree_mya": float(age) if age not in ("", None) else None,
            "timetree_ci_low": data.get("precomputed_ci_low"),
            "timetree_ci_high": data.get("precomputed_ci_high"),
            "timetree_study_count": data.get("all_total"),
            "timetree_source": "TimeTree API pairwise summaryjson",
            "timetree_error": "",
        }
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, ValueError) as exc:
        return {
            **base,
            "timetree_mya": None,
            "timetree_ci_low": None,
            "timetree_ci_high": None,
            "timetree_study_count": None,
            "timetree_source": "TimeTree API pairwise summaryjson",
            "timetree_error": repr(exc),
        }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    species_rows = load_species()
    resolution = pd.DataFrame(resolve_taxon(row["species"]) | row for row in species_rows)
    resolution_path = OUT_DIR / "timetree_taxon_resolution.csv"
    resolution.to_csv(resolution_path, index=False)

    resolved = resolution[resolution["resolved"] & resolution["taxon_id"].notna()].copy()
    pair_rows = []
    for _, row_a in resolved.iterrows():
        for _, row_b in resolved.loc[resolved.index > row_a.name].iterrows():
            pair_rows.append(fetch_pairwise(row_a, row_b))

    pairs = pd.DataFrame(pair_rows)
    pair_path = OUT_DIR / "timetree_divergence_pairs.csv"
    pairs.to_csv(pair_path, index=False)

    n_ok = int(pairs["timetree_mya"].notna().sum()) if len(pairs) else 0
    print(f"Resolved taxa: {len(resolved)}/{len(resolution)}")
    print(f"Fetched pairwise times: {n_ok}/{len(pairs)}")
    print(f"Saved: {resolution_path}")
    print(f"Saved: {pair_path}")

    unresolved = resolution[~resolution["resolved"]]
    if len(unresolved):
        print("\nUnresolved taxa:")
        print(unresolved[["species", "timetree_query", "raw_error"]].to_string(index=False))

    failed = pairs[pairs["timetree_mya"].isna()] if len(pairs) else pairs
    if len(failed):
        print("\nPairs without TimeTree age:")
        print(
            failed[
                ["species_a", "species_b", "timetree_query_a", "timetree_query_b", "timetree_error"]
            ].head(30).to_string(index=False)
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .datasets import CanonicalDataset


def summarize_dataset(dataset: CanonicalDataset) -> dict[str, Any]:
    calls = dataset.calls
    species_counts = (
        calls.groupby("species", sort=True)
        .size()
        .rename("n_calls")
        .reset_index()
        .to_dict(orient="records")
    )
    families = []
    if "family" in calls:
        families = (
            calls.groupby("family", sort=True)
            .agg(n_species=("species", "nunique"), n_calls=("call_id", "count"))
            .reset_index()
            .query("family != ''")
            .to_dict(orient="records")
        )
    return {
        "dataset": dataset.name,
        "source_path": str(dataset.source_path),
        "n_calls": int(len(calls)),
        "n_species": int(calls["species"].nunique()),
        "n_families": int(calls["family"].replace("", np.nan).nunique()) if "family" in calls else 0,
        "species_counts": species_counts,
        "family_counts": families,
        "columns": list(calls.columns),
    }


def _pair_values(dataset: CanonicalDataset, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    species = dataset.calls["species"].astype(str).to_numpy()
    within = []
    across = []
    for i, j in combinations(range(len(species)), 2):
        if species[i] == species[j]:
            within.append(float(matrix[i, j]))
        else:
            across.append(float(matrix[i, j]))
    return np.asarray(within, dtype=float), np.asarray(across, dtype=float)


def _safe_summary(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_similarity(dataset: CanonicalDataset, matrix: np.ndarray) -> dict[str, Any]:
    within, across = _pair_values(dataset, matrix)
    by_species_pair = []
    calls = dataset.calls
    species = dataset.species
    for i, source in enumerate(species):
        idx_i = calls.index[calls["species"] == source].tolist()
        for target in species[i:]:
            idx_j = calls.index[calls["species"] == target].tolist()
            values = []
            if source == target:
                for a, b in combinations(idx_i, 2):
                    values.append(float(matrix[a, b]))
            else:
                for a in idx_i:
                    for b in idx_j:
                        values.append(float(matrix[a, b]))
            by_species_pair.append(
                {
                    "species_pair": f"{source}~{target}",
                    **_safe_summary(np.asarray(values, dtype=float)),
                }
            )
    return {
        "within_species": _safe_summary(within),
        "across_species": _safe_summary(across),
        "species_pairs": by_species_pair,
    }

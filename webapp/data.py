from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repertoire_explorer import AnimalLexBundle, load_bundle


BUNDLE_PATH = ROOT / "artifacts" / "animallex" / "latest"
CONFIDENCE_VALUES = {
    "all": {"low", "medium", "high"},
    "medium_plus": {"medium", "high"},
    "high": {"high"},
}
SPECIES_FIT_VALUES = {
    "include": {"include"},
    "include_caution": {"include", "caution"},
    "all": {"include", "caution", "exclude"},
}


def load_animallex_bundle(path: Path | str = BUNDLE_PATH) -> AnimalLexBundle:
    return load_bundle(path)


def bundle_for_confidence(
    bundle: AnimalLexBundle, confidence_filter: str | None
) -> AnimalLexBundle:
    return bundle_for_filters(bundle, "all", confidence_filter)


def bundle_for_species_fit(
    bundle: AnimalLexBundle, species_fit_filter: str | None
) -> AnimalLexBundle:
    return bundle_for_filters(bundle, species_fit_filter, "all")


def bundle_for_filters(
    bundle: AnimalLexBundle,
    species_fit_filter: str | None,
    confidence_filter: str | None,
) -> AnimalLexBundle:
    fit_key = species_fit_filter if species_fit_filter in SPECIES_FIT_VALUES else "include"
    key = confidence_filter if confidence_filter in CONFIDENCE_VALUES else "all"
    base_analysis = (
        bundle.analysis
        if fit_key == "all"
        else bundle.analysis["species_fit_views"][fit_key]
    )
    if key == "all":
        analysis = base_analysis
    else:
        analysis = base_analysis["confidence_views"][key]
    accepted = CONFIDENCE_VALUES[key]
    accepted_fit = SPECIES_FIT_VALUES[fit_key]
    selected_indices = [
        index
        for index, call in enumerate(bundle.calls)
        if call.get("confidence") in accepted
        and call.get("species_fit", "include") in accepted_fit
    ]
    calls = [bundle.calls[index] for index in selected_indices]
    return replace(
        bundle,
        calls=calls,
        analysis=analysis,
        acoustic_embeddings=bundle.acoustic_embeddings[selected_indices],
        semantic_embeddings=bundle.semantic_embeddings[selected_indices],
        acoustic_similarity=bundle.acoustic_similarity[
            np.ix_(selected_indices, selected_indices)
        ],
        semantic_similarity=bundle.semantic_similarity[
            np.ix_(selected_indices, selected_indices)
        ],
    )

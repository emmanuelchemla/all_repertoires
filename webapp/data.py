from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys


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


def load_animallex_bundle(path: Path | str = BUNDLE_PATH) -> AnimalLexBundle:
    return load_bundle(path)


def bundle_for_confidence(
    bundle: AnimalLexBundle, confidence_filter: str | None
) -> AnimalLexBundle:
    key = confidence_filter if confidence_filter in CONFIDENCE_VALUES else "all"
    if key == "all":
        analysis = bundle.analysis
    else:
        analysis = bundle.analysis["confidence_views"][key]
    accepted = CONFIDENCE_VALUES[key]
    calls = [call for call in bundle.calls if call.get("confidence") in accepted]
    return replace(bundle, calls=calls, analysis=analysis)

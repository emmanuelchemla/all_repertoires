from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repertoire_explorer import AnimalLexBundle, load_bundle


DEFAULT_BUNDLE = ROOT / "artifacts" / "animallex" / "latest"


def load_analysis_bundle(path: Path | str = DEFAULT_BUNDLE) -> AnimalLexBundle:
    return load_bundle(path)

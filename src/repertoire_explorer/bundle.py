from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hashlib
import json
import numpy as np

from .datasets import CanonicalDataset


@dataclass(frozen=True)
class AnimalLexBundle:
    calls: list[dict[str, Any]]
    analysis: dict[str, Any]
    manifest: dict[str, Any]
    acoustic_embeddings: np.ndarray
    semantic_embeddings: np.ndarray
    acoustic_similarity: np.ndarray
    semantic_similarity: np.ndarray


def source_hash(source_dir: Path | str) -> str:
    source_dir = Path(source_dir)
    digest = hashlib.sha256()
    for path in sorted(source_dir.glob("*.yaml")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_bundle(
    output_dir: Path | str,
    dataset: CanonicalDataset,
    analysis: dict[str, Any],
    manifest: dict[str, Any],
    *,
    acoustic_embeddings: np.ndarray,
    semantic_embeddings: np.ndarray,
    acoustic_similarity: np.ndarray,
    semantic_similarity: np.ndarray,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calls = dataset.calls.to_dict(orient="records")
    (output_dir / "calls.json").write_text(
        json.dumps(calls, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "analysis.json").write_text(
        json.dumps(_json_safe(analysis), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "arrays.npz",
        acoustic_embeddings=acoustic_embeddings,
        semantic_embeddings=semantic_embeddings,
        acoustic_similarity=acoustic_similarity,
        semantic_similarity=semantic_similarity,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def load_bundle(output_dir: Path | str) -> AnimalLexBundle:
    output_dir = Path(output_dir)
    arrays = np.load(output_dir / "arrays.npz")
    return AnimalLexBundle(
        calls=json.loads((output_dir / "calls.json").read_text(encoding="utf-8")),
        analysis=json.loads((output_dir / "analysis.json").read_text(encoding="utf-8")),
        manifest=json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")),
        acoustic_embeddings=arrays["acoustic_embeddings"],
        semantic_embeddings=arrays["semantic_embeddings"],
        acoustic_similarity=arrays["acoustic_similarity"],
        semantic_similarity=arrays["semantic_similarity"],
    )


def validate_bundle(
    output_dir: Path | str,
    source_dir: Path | str,
    *,
    expected_config: dict[str, Any] | None = None,
) -> None:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    current_hash = source_hash(source_dir)
    if manifest.get("source_hash") != current_hash:
        raise ValueError("The AnimalLex analysis bundle is stale. Rebuild it before use.")
    if expected_config is not None and manifest.get("config") != expected_config:
        raise ValueError("The AnimalLex analysis settings changed. Rebuild the bundle before use.")

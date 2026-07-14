from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPERTOIRES_DIR = ROOT / "repertoires"
COMPILED_REPERTOIRES_DIR = REPERTOIRES_DIR / "compiled"
CACHE_DIR = ROOT / "cache"

DATABASE_PATH = COMPILED_REPERTOIRES_DIR / "database.json"
HUMAN_DATABASE_PATH = COMPILED_REPERTOIRES_DIR / "database_humans.json"
EMBEDDING_CACHE_PATH = CACHE_DIR / "embedding_cache.json"


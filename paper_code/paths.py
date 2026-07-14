from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"
DATABASE_PATH = ROOT / "repertoires" / "compiled" / "database.json"
HUMAN_DATABASE_PATH = ROOT / "repertoires" / "compiled" / "database_humans.json"
EMBEDDING_CACHE_PATH = CACHE_DIR / "embedding_cache.json"


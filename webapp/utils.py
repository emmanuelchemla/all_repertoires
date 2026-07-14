"""Utility functions for data loading, embeddings, and basic plots."""

from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from repertoire_explorer.datasets import load_all_repertoires_json


def species_common_name(name: str) -> str:
    """Return display-friendly common name by stripping trailing parentheses."""
    s = str(name)
    return re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()


def load_calls(json_path: Path) -> List[Dict[str, object]]:
    """Flatten species -> calls into a list of call dicts."""
    dataset = load_all_repertoires_json(json_path)
    calls = dataset.calls.to_dict(orient="records")
    for call in calls:
        call["taxonomy"] = {
            "kingdom": call.get("kingdom", ""),
            "phylum": call.get("phylum", ""),
            "class": call.get("class", ""),
            "order": call.get("order", ""),
            "family": call.get("family", ""),
            "genus": call.get("genus", ""),
        }
    return calls


def filter_calls(
    calls: List[Dict[str, object]],
    species_allowlist: Sequence[str] | None = None,
    taxon_rank: str | None = None,
    taxon_values: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    """Filter calls by species and/or a taxonomy rank."""
    out: List[Dict[str, object]] = []
    allow = set(species_allowlist) if species_allowlist else None
    tax_vals = set(taxon_values) if taxon_values else None

    for c in calls:
        sp = str(c.get("species", ""))
        if allow is not None and sp not in allow:
            continue

        if taxon_rank and tax_vals is not None:
            v = str(c.get(taxon_rank, ""))
            if v not in tax_vals:
                continue

        out.append(c)
    return out


def available_taxa(calls: List[Dict[str, object]], rank: str) -> List[str]:
    """Return sorted unique values for a given taxonomy rank."""
    vals = {str(c.get(rank, "")).strip() for c in calls}
    vals.discard("")
    return sorted(vals)


def build_taxonomy_sunburst(
    calls: List[Dict[str, object]],
    title: str = "Taxonomy overview",
    species_color_map: Dict[str, str] | None = None,
):
    """Build a simple Plotly sunburst chart from the optional taxonomy fields."""

    seen = {}
    for c in calls:
        sp = str(c.get("species", "unknown"))
        if sp not in seen:
            seen[sp] = c

    ranks = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    labels: List[str] = []
    ids: List[str] = []
    parents: List[str] = []
    colors: List[str] = []

    internal_color = "#E6E6E6"

    def node_id(rank: str, value: str) -> str:
        return f"{rank}:{value}" if rank != "species" else f"species:{value}"

    added = set()
    root = "root:Life"
    labels.append("Life")
    ids.append(root)
    parents.append("")
    colors.append(internal_color)
    added.add(root)

    for sp, c in seen.items():
        path = {
            "kingdom": str(c.get("kingdom", "")).strip(),
            "phylum": str(c.get("phylum", "")).strip(),
            "class": str(c.get("class", "")).strip(),
            "order": str(c.get("order", "")).strip(),
            "family": str(c.get("family", "")).strip(),
            "genus": str(c.get("genus", "")).strip(),
            "species": sp,
        }

        parent = root
        for rank in ranks:
            value = path.get(rank, "")
            if rank != "species" and not value:
                continue

            nid = node_id(rank, value)
            if nid not in added:
                labels.append(value)
                ids.append(nid)
                parents.append(parent)

                if rank == "species" and species_color_map is not None:
                    colors.append(species_color_map.get(value, internal_color))
                else:
                    colors.append(internal_color)

                added.add(nid)

            parent = nid

    fig = go.Figure(
        go.Sunburst(
            labels=labels,
            ids=ids,
            parents=parents,
            branchvalues="total",
            maxdepth=4,
            marker=dict(colors=colors),
        )
    )
    fig.update_layout(title=title, margin=dict(t=10, l=10, r=10, b=10))
    return fig


def load_cache(cache_path: Path, model: str) -> Dict[str, List[float]]:
    if not cache_path.exists():
        return {}
    with open(cache_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("model") != model:
        return {}
    return payload.get("embeddings", {})


def save_cache(cache_path: Path, model: str, cache: Dict[str, List[float]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"model": model, "embeddings": cache}), encoding="utf-8"
    )


def batch_iter(seq: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def embed_texts(
    texts: List[str],
    model_name: str,
    cache_path: Path,
    encoder: SentenceTransformer,
    batch_size: int = 64,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """Embed texts with SentenceTransformer, caching by exact text string."""
    cache = load_cache(cache_path, model_name)
    vectors: List[List[float]] = []
    missing: List[str] = [t for t in texts if t not in cache]

    for chunk in batch_iter(missing, batch_size):
        if not chunk:
            continue
        embeds = encoder.encode(
            chunk,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for text, emb in zip(chunk, embeds):
            cache[text] = emb.tolist()

    for text in texts:
        vectors.append(cache[text])

    save_cache(cache_path, model_name, cache)
    return np.array(vectors, dtype=np.float32), cache


def reduce_umap(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    reducer = UMAP(random_state=42, n_components=n_components)
    return reducer.fit_transform(embeddings)


def plot_umap(embeddings: np.ndarray, species: List[str], out_path: Path) -> None:
    coords = reduce_umap(embeddings, n_components=2)

    species_set = sorted(set(species))
    palette = plt.cm.get_cmap("tab20", len(species_set))
    color_map = {sp: palette(i) for i, sp in enumerate(species_set)}

    plt.figure(figsize=(10, 7))
    for sp in species_set:
        mask = [s == sp for s in species]
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=color_map[sp],
            label=species_common_name(sp),
            s=30,
            alpha=0.8,
        )

    plt.title("UMAP of Acoustic Descriptions")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(fontsize="small", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_umap_3d(embeddings: np.ndarray, species: List[str], out_path: Path) -> None:
    coords = reduce_umap(embeddings, n_components=3)
    species_set = sorted(set(species))
    palette = plt.cm.get_cmap("tab20", len(species_set))
    color_map = {sp: palette(i) for i, sp in enumerate(species_set)}

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for sp in species_set:
        mask = [s == sp for s in species]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            coords[mask, 2],
            color=color_map[sp],
            label=species_common_name(sp),
            s=30,
            alpha=0.8,
        )
    ax.set_title("UMAP of Acoustic Descriptions (3D)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

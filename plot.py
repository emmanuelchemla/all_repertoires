"""
Static plotting utilities for the repertoire database.

Outputs:
- acoustic_umap.png  : UMAP of acoustic descriptions (by species color)
- semantic_umap.png  : UMAP of semantic descriptions with ontology keyword markers
- calls_per_species.png : bar chart of #calls per species
- keyword_freq.png      : top-N ontology keyword frequencies
- keyword_heatmap.png   : species x keyword presence heatmap (counts)

Usage (script):
  python plot.py --json-path database.json --embedding-model sentence-transformers/all-MiniLM-L6-v2

Usage (notebook):
  from plot import generate_all_plots
  generate_all_plots("database.json")

Dependencies: sentence-transformers, umap-learn, matplotlib, numpy
"""

from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import re


def species_common_name(name: str) -> str:
    """Return display-friendly common name.

    If the species string ends with a parenthetical (e.g. "Lion (Panthera leo)"),
    strip the trailing parenthetical for display in legends/axes.
    """
    s = str(name)
    return re.sub(r"\s*\([^)]*\)\s*$", "", s).strip()


def load_calls(json_path: Path) -> List[Dict[str, object]]:
    """Flatten species -> calls into a list of call dicts.

    Also carries optional taxonomy fields if present in the JSON under each species entry.
    Supported keys (all optional): taxonomy (dict) and the flattened keys genus/family/order/class/phylum/kingdom.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    calls: List[Dict[str, object]] = []
    for species_entry in data.get("species", []):
        species_name = species_entry.get("species_name", "unknown")
        # Optional taxonomy fields (keep everything optional / backward-compatible)
        taxonomy = species_entry.get("taxonomy", {}) or {}
        # allow either nested taxonomy dict or flattened keys
        tax_kingdom = taxonomy.get("kingdom", species_entry.get("kingdom", ""))
        tax_phylum = taxonomy.get("phylum", species_entry.get("phylum", ""))
        tax_class = taxonomy.get("class", species_entry.get("class", ""))
        tax_order = taxonomy.get("order", species_entry.get("order", ""))
        tax_family = taxonomy.get("family", species_entry.get("family", ""))
        tax_genus = taxonomy.get("genus", species_entry.get("genus", ""))
        for call in species_entry.get("calls", []):
            calls.append(
                {
                    "species": species_name,
                    "call_name": call.get("call_name", "unknown"),
                    "acoustic_description": call.get("acoustic_description", ""),
                    "semantic_description": call.get("semantic_description", ""),
                    "ontology_keywords": call.get("ontology_keywords", []),
                    "taxonomy": taxonomy,
                    "kingdom": tax_kingdom,
                    "phylum": tax_phylum,
                    "class": tax_class,
                    "order": tax_order,
                    "family": tax_family,
                    "genus": tax_genus,
                }
            )
    return calls


def filter_calls(
    calls: List[Dict[str, object]],
    species_allowlist: Sequence[str] | None = None,
    taxon_rank: str | None = None,
    taxon_values: Sequence[str] | None = None,
) -> List[Dict[str, object]]:
    """Filter calls by species and/or a taxonomy rank.

    - species_allowlist: if provided, keep only these species.
    - taxon_rank: one of kingdom/phylum/class/order/family/genus
    - taxon_values: if provided with taxon_rank, keep only calls whose rank value is in taxon_values.
    """
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


def available_taxa(
    calls: List[Dict[str, object]],
    rank: str,
) -> List[str]:
    """Return sorted unique values for a given taxonomy rank."""
    vals = {str(c.get(rank, "")).strip() for c in calls}
    vals.discard("")
    return sorted(vals)


def build_taxonomy_sunburst(
    calls: List[Dict[str, object]],
    title: str = "Taxonomy overview",
    species_color_map: Dict[str, str] | None = None,
):
    """Build a simple Plotly sunburst chart from the optional taxonomy fields.

    This is a lightweight alternative to a full phylogenetic tree, but it helps users
    filter/organize by taxa in a web UI.

    Requires: plotly
    """
    import plotly.graph_objects as go

    # Use unique species entries; pick the first call for each species.
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

    # Light neutral for internal taxonomy nodes; species leaves get their species color.
    internal_color = "#E6E6E6"

    # Build a simple tree using string IDs like "rank:value" to avoid collisions.
    def node_id(rank: str, value: str) -> str:
        return f"{rank}:{value}" if rank != "species" else f"species:{value}"

    added = set()

    # Root node
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

        # Determine the first non-empty rank to attach to root
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
    fig.update_layout(title=title, margin=dict(t=40, l=10, r=10, b=10))
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


def run(
    json_path: Path | str = "database.json",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache: Path | str = ".embedding_cache.json",
    out: Path | str = "umap_acoustic.png",
) -> Path:
    """Embed acoustic/semantic descriptions and save a UMAP plot.

    Returns the output path for convenience in notebooks.
    """
    json_path = Path(json_path)
    cache = Path(cache)
    out = Path(out)

    calls = load_calls(json_path)
    if not calls:
        raise SystemExit("No calls found in the provided JSON.")

    acoustic_texts = [c["acoustic_description"] for c in calls]
    semantic_texts = [c["semantic_description"] for c in calls]

    encoder = SentenceTransformer(embedding_model)

    acoustic_embeds, _ = embed_texts(acoustic_texts, embedding_model, cache, encoder)
    semantic_embeds, _ = embed_texts(semantic_texts, embedding_model, cache, encoder)

    # Simple check: print shapes for confirmation.
    print(f"Acoustic embeddings: {acoustic_embeds.shape}")
    print(f"Semantic embeddings: {semantic_embeds.shape}")

    plot_umap(acoustic_embeds, [c["species"] for c in calls], out)
    print(f"Saved UMAP plot to {out}")
    return out


def _color_map(species: Sequence[str]):
    species_set = sorted(set(species))
    palette = plt.cm.get_cmap("tab20", len(species_set))
    return {sp: palette(i) for i, sp in enumerate(species_set)}


def plot_semantic_with_keywords(
    semantic_embeds: np.ndarray,
    species: List[str],
    call_names: List[str],
    ontology_keywords: List[List[str]],
    encoder: SentenceTransformer,
    reducer: UMAP,
    out_path: Path,
):
    coords = reducer.transform(semantic_embeds)
    color_map = _color_map(species)

    plt.figure(figsize=(10, 7))
    for sp in sorted(set(species)):
        mask = [s == sp for s in species]
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            color=color_map[sp],
            label=species_common_name(sp),
            s=30,
            alpha=0.8,
        )

    # embed unique ontology keywords and project
    all_keywords = sorted({kw for kws in ontology_keywords for kw in (kws or [])})
    if all_keywords:
        kw_embeds = encoder.encode(
            all_keywords, convert_to_numpy=True, normalize_embeddings=True
        )
        kw_coords = reducer.transform(kw_embeds)
        plt.scatter(
            kw_coords[:, 0],
            kw_coords[:, 1],
            marker="^",
            c="black",
            s=50,
            alpha=0.8,
            label="Ontology keyword",
        )
        for (x, y), kw in zip(kw_coords, all_keywords):
            plt.text(x, y, kw, fontsize=7, ha="left", va="bottom")

    plt.title("UMAP of Semantic Descriptions with Ontology Keywords")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(fontsize="small", bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_semantic_with_keywords_3d(
    semantic_embeds: np.ndarray,
    species: List[str],
    call_names: List[str],
    ontology_keywords: List[List[str]],
    encoder: SentenceTransformer,
    reducer: UMAP,
    out_path: Path,
):
    coords = reducer.transform(semantic_embeds)
    color_map = _color_map(species)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for sp in sorted(set(species)):
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

    all_keywords = sorted({kw for kws in ontology_keywords for kw in (kws or [])})
    if all_keywords:
        kw_embeds = encoder.encode(
            all_keywords, convert_to_numpy=True, normalize_embeddings=True
        )
        kw_coords = reducer.transform(kw_embeds)
        ax.scatter(
            kw_coords[:, 0],
            kw_coords[:, 1],
            kw_coords[:, 2],
            marker="^",
            c="black",
            s=50,
            alpha=0.8,
            label="Ontology keyword",
        )
        for (x, y, z), kw in zip(kw_coords, all_keywords):
            ax.text(x, y, z, kw, fontsize=7)

    ax.set_title("UMAP of Semantic Descriptions with Ontology Keywords (3D)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
    ax.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_calls_per_species(species: List[str], out_path: Path) -> None:
    counts = Counter(species)
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)
    labels_disp = [species_common_name(s) for s in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(labels)), values, color="steelblue")
    plt.xticks(range(len(labels)), labels_disp, rotation=60, ha="right")
    plt.ylabel("# calls")
    plt.title("Number of calls per species")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_keyword_frequency(
    ontology_keywords: List[List[str]],
    orders: List[str],
    out_path: Path,
    top_n: int = 20,
) -> None:
    # Count keyword occurrences per taxonomic order
    order_kw_counts: Dict[str, Counter] = defaultdict(Counter)
    total_counter: Counter[str] = Counter()

    for kws, order in zip(ontology_keywords, orders):
        if not kws:
            continue
        for kw in kws:
            order_kw_counts[order].update([kw])
            total_counter.update([kw])

    if not total_counter:
        return

    # Select top-N keywords overall
    top_keywords = [kw for kw, _ in total_counter.most_common(top_n)]

    orders_unique = sorted(order_kw_counts.keys())

    # Prepare stacked data
    data = {
        order: [order_kw_counts[order][kw] for kw in top_keywords]
        for order in orders_unique
    }

    # Plot
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(top_keywords))

    palette = plt.cm.get_cmap("tab20", len(orders_unique))

    for i, order in enumerate(orders_unique):
        values = np.array(data[order])
        plt.bar(
            range(len(top_keywords)),
            values,
            bottom=bottom,
            label=order,
            color=palette(i),
        )
        bottom += values

    plt.xticks(range(len(top_keywords)), top_keywords, rotation=60, ha="right")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} ontology keywords (stacked by taxonomic order)")
    plt.legend(
        title="Order",
        fontsize="small",
        title_fontsize="small",
        bbox_to_anchor=(1.04, 1),
        loc="upper left",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_keyword_heatmap(
    species: List[str],
    ontology_keywords: List[List[str]],
    out_path: Path,
    top_n: int = 20,
) -> None:
    # build species x keyword count
    species_set = sorted(set(species))
    counter = defaultdict(Counter)
    for sp, kws in zip(species, ontology_keywords):
        counter[sp].update(kws or [])

    # pick top keywords overall
    total = Counter()
    for sp in species_set:
        total.update(counter[sp])
    keywords = [kw for kw, _ in total.most_common(top_n)]
    if not keywords:
        return

    mat = np.zeros((len(species_set), len(keywords)), dtype=float)
    for i, sp in enumerate(species_set):
        for j, kw in enumerate(keywords):
            mat[i, j] = counter[sp][kw]

    plt.figure(figsize=(1.2 * len(keywords) + 2, 0.6 * len(species_set) + 2))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Count")
    plt.xticks(range(len(keywords)), keywords, rotation=60, ha="right")
    species_disp = [species_common_name(s) for s in species_set]
    plt.yticks(range(len(species_set)), species_disp)
    plt.title(f"Ontology keyword counts per species (top {top_n})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_keyword_heatmap_binned(
    species: List[str],
    ontology_keywords: List[List[str]],
    out_path: Path,
    top_n: int = 20,
) -> None:
    """Binned heatmap with fixed colors: 0=white,1=green,2-5=red,>5=blue."""
    species_set = sorted(set(species))
    counter = defaultdict(Counter)
    for sp, kws in zip(species, ontology_keywords):
        counter[sp].update(kws or [])

    total = Counter()
    for sp in species_set:
        total.update(counter[sp])
    keywords = [kw for kw, _ in total.most_common(top_n)]
    if not keywords:
        return

    mat = np.zeros((len(species_set), len(keywords)), dtype=int)
    for i, sp in enumerate(species_set):
        for j, kw in enumerate(keywords):
            mat[i, j] = counter[sp][kw]

    # bin values
    bins = np.zeros_like(mat)
    bins[mat == 0] = 0  # white
    bins[mat == 1] = 1  # green
    bins[(mat >= 2) & (mat <= 5)] = 2  # red
    bins[mat > 5] = 3  # blue

    # custom colormap
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["white", "green", "red", "blue"])

    plt.figure(figsize=(1.2 * len(keywords) + 2, 0.6 * len(species_set) + 2))
    im = plt.imshow(bins, aspect="auto", cmap=cmap, vmin=0, vmax=3)
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["0 calls", "1 call", "2–5 calls", ">5 calls"])
    plt.xticks(range(len(keywords)), keywords, rotation=60, ha="right")
    species_disp = [species_common_name(s) for s in species_set]
    plt.yticks(range(len(species_set)), species_disp)
    plt.title(f"Binned keyword counts per species (top {top_n})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_all_plots(
    json_path: Path | str = "database.json",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache: Path | str = ".embedding_cache.json",
    out_dir: Path | str = "plots",
) -> Dict[str, Path]:
    """Create acoustic UMAP, semantic UMAP with keywords, and descriptive plots."""
    json_path = Path(json_path)
    cache = Path(cache)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calls = load_calls(json_path)
    if not calls:
        raise SystemExit("No calls found in the provided JSON.")

    acoustic_texts = [c["acoustic_description"] for c in calls]
    semantic_texts = [c["semantic_description"] for c in calls]
    species = [c["species"] for c in calls]
    call_names = [c["call_name"] for c in calls]
    ontology_keywords = [c.get("ontology_keywords", []) for c in calls]

    encoder = SentenceTransformer(embedding_model)
    acoustic_embeds, _ = embed_texts(acoustic_texts, embedding_model, cache, encoder)
    semantic_embeds, _ = embed_texts(semantic_texts, embedding_model, cache, encoder)

    # reducers
    reducer2d = UMAP(random_state=42, n_components=2)
    reducer2d.fit(semantic_embeds)
    reducer3d = UMAP(random_state=42, n_components=3)
    reducer3d.fit(semantic_embeds)

    outputs: Dict[str, Path] = {}
    outputs["acoustic_umap"] = out_dir / "acoustic_umap.png"
    plot_umap(acoustic_embeds, species, outputs["acoustic_umap"])
    outputs["acoustic_umap3d"] = out_dir / "acoustic_umap3d.png"
    plot_umap_3d(acoustic_embeds, species, outputs["acoustic_umap3d"])

    outputs["semantic_umap"] = out_dir / "semantic_umap.png"
    plot_semantic_with_keywords(
        semantic_embeds,
        species,
        call_names,
        ontology_keywords,
        encoder,
        reducer2d,
        outputs["semantic_umap"],
    )
    outputs["semantic_umap3d"] = out_dir / "semantic_umap3d.png"
    plot_semantic_with_keywords_3d(
        semantic_embeds,
        species,
        call_names,
        ontology_keywords,
        encoder,
        reducer3d,
        outputs["semantic_umap3d"],
    )

    outputs["calls_per_species"] = out_dir / "calls_per_species.png"
    plot_calls_per_species(species, outputs["calls_per_species"])

    outputs["keyword_freq"] = out_dir / "keyword_freq.png"
    orders = [c.get("order", "") for c in calls]
    plot_keyword_frequency(ontology_keywords, orders, outputs["keyword_freq"])

    outputs["keyword_heatmap"] = out_dir / "keyword_heatmap.png"
    plot_keyword_heatmap(species, ontology_keywords, outputs["keyword_heatmap"])

    outputs["keyword_heatmap_bin"] = out_dir / "keyword_heatmap_bin.png"
    plot_keyword_heatmap_binned(
        species, ontology_keywords, outputs["keyword_heatmap_bin"]
    )

    for name, path in outputs.items():
        print(f"Saved {name}: {path}")
    return outputs


def run_interactive(
    json_path: Path | str = "database.json",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache: Path | str = ".embedding_cache.json",
    width: int | None = None,
    height: int | None = None,
    n_components: int = 2,
):
    """
    Return a Plotly FigureWidget with click-to-connect behavior.

    Click a point to draw a line to its nearest semantic neighbor in another species;
    click background to clear; clicking a new point replaces the connection.
    """
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    json_path = Path(json_path)
    cache = Path(cache)

    calls = load_calls(json_path)
    if not calls:
        raise SystemExit("No calls found in the provided JSON.")

    acoustic_texts = [c["acoustic_description"] for c in calls]
    semantic_texts = [c["semantic_description"] for c in calls]
    species = [c["species"] for c in calls]
    call_names = [c["call_name"] for c in calls]

    encoder = SentenceTransformer(embedding_model)
    acoustic_embeds, _ = embed_texts(acoustic_texts, embedding_model, cache, encoder)
    semantic_embeds, _ = embed_texts(semantic_texts, embedding_model, cache, encoder)

    coords = reduce_umap(acoustic_embeds, n_components=n_components)

    species_unique = sorted(set(species))
    species_to_idx = {sp: np.where(np.array(species) == sp)[0] for sp in species_unique}
    color_map = {
        sp: qualitative.Plotly[i % len(qualitative.Plotly)]
        for i, sp in enumerate(species_unique)
    }
    colors = [color_map[s] for s in species]

    is_3d = n_components == 3
    ScatterCls = go.Scatter3d if is_3d else go.Scatter
    scatter_kwargs = dict(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker=dict(color=colors, size=4 if is_3d else 8, opacity=0.8),
        text=[f"{c} ({s})" for c, s in zip(call_names, species)],
        hoverinfo="text",
        showlegend=False,
        uid="scatter",
    )
    if is_3d:
        scatter_kwargs["z"] = coords[:, 2]
    scatter = ScatterCls(**scatter_kwargs)
    # one line trace per species for colored/transparent connections
    line_traces = []
    for sp in species_unique:
        line_kwargs = dict(
            x=[],
            y=[],
            mode="lines",
            line=dict(color=color_map[sp], width=2),
            opacity=0.45,
            hoverinfo="skip",
            showlegend=False,
            name=f"{sp} connection",
            uid=f"line-{sp}",
        )
        if is_3d:
            line_kwargs["z"] = []
            line_kwargs["mode"] = "lines"
            line_traces.append(go.Scatter3d(**line_kwargs))
        else:
            line_traces.append(go.Scatter(**line_kwargs))

    label_kwargs = dict(
        x=[],
        y=[],
        mode="markers+text",
        marker=dict(color="black", size=1),
        text=[],
        textposition="top center",
        hoverinfo="skip",
        showlegend=False,
        uid="labels",
    )
    if is_3d:
        label_kwargs["z"] = []
        label_trace = go.Scatter3d(**label_kwargs)
    else:
        label_trace = go.Scatter(**label_kwargs)

    # legend-only markers for species color key
    legend_traces = []
    for sp in species_unique:
        leg_kwargs = dict(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color=color_map[sp], size=8),
            name=species_common_name(sp),
            hoverinfo="skip",
            showlegend=True,
            uid=f"legend-{sp}",
        )
        if is_3d:
            leg_kwargs["z"] = [None]
            legend_traces.append(go.Scatter3d(**leg_kwargs))
        else:
            legend_traces.append(go.Scatter(**leg_kwargs))

    layout = go.Layout(
        title="UMAP of Acoustic Descriptions (interactive)",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        showlegend=True,
        width=width,
        height=height,
    )
    if is_3d:
        layout.scene = dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        )
    fig = go.FigureWidget(
        data=[scatter, *line_traces, label_trace, *legend_traces],
        layout=layout,
    )

    # Ensure every trace has a uid (avoids ipywidgets KeyError on state diffs)
    for i, trace in enumerate(fig.data):
        if trace.uid is None:
            trace.uid = f"trace-{i}"

    species_arr = np.array(species)
    semantic = np.array(semantic_embeds)
    line_start = 1  # scatter is at index 0
    label_idx = 1 + len(species_unique)

    def on_click(trace, points, state):
        if not points.point_inds:
            for i in range(len(species_unique)):
                if is_3d:
                    fig.data[line_start + i].update(x=[], y=[], z=[])
                else:
                    fig.data[line_start + i].update(x=[], y=[])
            if is_3d:
                fig.data[label_idx].update(x=[], y=[], z=[], text=[])
            else:
                fig.data[label_idx].update(x=[], y=[], text=[])
            return

        idx = points.point_inds[0]
        base_species = species_arr[idx]

        label_x: List[float] = []
        label_y: List[float] = []
        label_z: List[float] = []
        label_texts: List[str] = []

        # clicked call label
        label_x.append(coords[idx, 0])
        label_y.append(coords[idx, 1])
        if is_3d:
            label_z.append(coords[idx, 2])
        label_texts.append(call_names[idx])

        # clear existing lines first
        for i in range(len(species_unique)):
            if is_3d:
                fig.data[line_start + i].update(x=[], y=[], z=[])
            else:
                fig.data[line_start + i].update(x=[], y=[])

        for sp in species_unique:
            if sp == base_species:
                continue
            candidates = species_to_idx.get(sp, [])
            if len(candidates) == 0:
                continue
            dists = np.linalg.norm(semantic[candidates] - semantic[idx], axis=1)
            nn_idx = candidates[dists.argmin()]

            # line segment for this species
            if is_3d:
                fig.data[line_start + species_unique.index(sp)].update(
                    x=[coords[idx, 0], coords[nn_idx, 0]],
                    y=[coords[idx, 1], coords[nn_idx, 1]],
                    z=[coords[idx, 2], coords[nn_idx, 2]],
                )
            else:
                fig.data[line_start + species_unique.index(sp)].update(
                    x=[coords[idx, 0], coords[nn_idx, 0]],
                    y=[coords[idx, 1], coords[nn_idx, 1]],
                )

            # labels for neighbor
            label_x.append(coords[nn_idx, 0])
            label_y.append(coords[nn_idx, 1])
            if is_3d:
                label_z.append(coords[nn_idx, 2])
            label_texts.append(call_names[nn_idx])

        if is_3d:
            fig.data[label_idx].update(
                x=label_x, y=label_y, z=label_z, text=label_texts
            )
        else:
            fig.data[label_idx].update(x=label_x, y=label_y, text=label_texts)

    fig.data[0].on_click(on_click)
    return fig


def run_dash_app(
    json_path: Path | str = "database.json",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    cache: Path | str = ".embedding_cache.json",
    host: str = "0.0.0.0",
    port: int = 8050,
    n_components: int = 2,
):
    """Run an internal Dash web app.

    Features:
    - Species selection (multi)
    - Taxonomy filtering (if taxonomy fields exist in JSON)
    - Click a point to show call details
    - Optional taxonomy sunburst overview

    Notes:
    - Requires: dash, plotly
    - Intended for internal use (e.g., Cloud Run/GCE) and access over Tailscale.
    """
    from pathlib import Path as _Path

    import base64 as _base64
    import numpy as _np
    import plotly.graph_objects as go
    import html as _html

    try:
        from dash import Dash, dcc, html, Input, Output
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Dash is required for run_dash_app(). Install with: pip install dash"
        ) from e

    json_path = _Path(json_path)
    cache = _Path(cache)

    calls_all = load_calls(json_path)
    if not calls_all:
        raise SystemExit("No calls found in the provided JSON.")

    # Build embeddings once
    acoustic_texts = [str(c.get("acoustic_description", "")) for c in calls_all]
    semantic_texts = [str(c.get("semantic_description", "")) for c in calls_all]
    species_all = [str(c.get("species", "unknown")) for c in calls_all]
    call_names_all = [str(c.get("call_name", "unknown")) for c in calls_all]

    encoder = SentenceTransformer(embedding_model)
    acoustic_embeds, _ = embed_texts(acoustic_texts, embedding_model, cache, encoder)
    semantic_embeds, _ = embed_texts(semantic_texts, embedding_model, cache, encoder)

    coords = reduce_umap(acoustic_embeds, n_components=n_components)
    coords = _np.asarray(coords)

    # Stable integer index for each call
    for i, c in enumerate(calls_all):
        c["_idx"] = i

    species_unique = sorted(set(species_all))

    # --- Color map (species -> color) for consistent coloring across plots/lines ---
    from plotly.colors import qualitative as _qual

    color_map = {
        sp: _qual.Plotly[i % len(_qual.Plotly)] for i, sp in enumerate(species_unique)
    }

    def _fmt_hover(i: int) -> str:
        """Minimal hover: species + call name."""
        c = calls_all[i]
        species = _html.escape(str(c.get("species", "")))
        call_name = _html.escape(str(c.get("call_name", "")))
        return f"{call_name} ({species})"

    def make_scatter(
        filtered_calls: List[Dict[str, object]],
        clicked_idx: int | None = None,
    ):
        """Build scatter colored by species.

        If clicked_idx is provided (stable global call index), draw one line per
        other species to its nearest neighbor in *semantic* space.
        """
        idxs = [int(c["_idx"]) for c in filtered_calls]
        if not idxs:
            return go.Figure()

        # Species present in the current filtered view
        species_in_view = sorted({species_all[i] for i in idxs})

        fig = go.Figure()

        # Scatter: one trace per species, so each dot is colored by its species
        for sp in species_in_view:
            sp_idxs = [i for i in idxs if species_all[i] == sp]
            if not sp_idxs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=coords[sp_idxs, 0],
                    y=coords[sp_idxs, 1],
                    mode="markers",
                    marker=dict(size=9, opacity=0.8, color=color_map.get(sp)),
                    customdata=sp_idxs,
                    text=["" for _ in sp_idxs],
                    hovertemplate="<extra></extra>",
                    name=species_common_name(sp),
                    showlegend=True,
                )
            )

        # Lines: when a point is clicked, connect to nearest semantic neighbor in each other species
        if clicked_idx is not None:
            try:
                clicked_idx = int(clicked_idx)
            except Exception:
                clicked_idx = None

        if clicked_idx is not None and clicked_idx in set(idxs):
            base_species = species_all[clicked_idx]

            # Precompute candidate lists per species within current view
            sp_to_candidates: Dict[str, List[int]] = {}
            for sp in species_in_view:
                if sp == base_species:
                    continue
                sp_to_candidates[sp] = [i for i in idxs if species_all[i] == sp]

            # Draw one line per other species (colored by the destination species)
            base_vec = semantic_embeds[clicked_idx]
            for sp, candidates in sp_to_candidates.items():
                if not candidates:
                    continue
                cand_vecs = semantic_embeds[candidates]
                dists = _np.linalg.norm(cand_vecs - base_vec, axis=1)
                nn_idx = candidates[int(dists.argmin())]

                fig.add_trace(
                    go.Scatter(
                        x=[coords[clicked_idx, 0], coords[nn_idx, 0]],
                        y=[coords[clicked_idx, 1], coords[nn_idx, 1]],
                        mode="lines",
                        line=dict(color=color_map.get(sp), width=2),
                        opacity=0.55,
                        hoverinfo="skip",
                        showlegend=False,
                        name=f"{sp} connection",
                    )
                )

        fig.update_layout(
            title="UMAP of Acoustic Descriptions (interactive)",
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            margin=dict(t=45, l=10, r=10, b=10),
        )
        return fig

    # Initial figures
    init_calls = calls_all
    fig_scatter = make_scatter(init_calls, clicked_idx=None)
    fig_tax = None
    try:
        fig_tax = build_taxonomy_sunburst(calls_all, species_color_map=color_map)
    except Exception:
        fig_tax = go.Figure()

    app = Dash(__name__)

    # Styling: CSS lives in ./assets/style.css (Dash auto-loads assets).
    # Prefer className hooks over inline style dicts.

    # --- Static plot gallery (images generated by generate_all_plots) ---
    plots_dir = _Path("plots")

    def _img_card(title: str, filename: str, caption: str):
        p = plots_dir / filename
        if not p.exists():
            return html.Div(
                className="img-card img-card--missing",
                children=[
                    html.H4(title, className="img-card__title"),
                    html.Div(
                        f"Missing: {p}",
                        className="img-card__subtle",
                    ),
                ],
            )

        try:
            b64 = _base64.b64encode(p.read_bytes()).decode("ascii")
            src = f"data:image/png;base64,{b64}"
        except Exception as e:
            return html.Div(
                className="img-card img-card--error",
                children=[
                    html.H4(title, className="img-card__title"),
                    html.Div(
                        f"Could not load {p}: {e}",
                        className="img-card__subtle",
                    ),
                ],
            )

        return html.Div(
            className="img-card",
            children=[
                html.H4(title, className="img-card__title"),
                html.Div(caption, className="img-card__caption"),
                html.Img(src=src, className="img-card__img"),
            ],
        )

    static_gallery = html.Div(
        className="static-gallery",
        children=[
            html.H3("Static plots"),
            html.Div(
                "A few summary plots (looked up in ./plots).",
                className="subtle",
            ),
            html.Div(
                className="static-gallery__grid",
                children=[
                    _img_card(
                        "Acoustic UMAP",
                        "acoustic_umap.png",
                        "2D UMAP of acoustic descriptions colored by species.",
                    ),
                    _img_card(
                        "Semantic UMAP",
                        "semantic_umap.png",
                        "Semantic description embedding with ontology keyword markers.",
                    ),
                    _img_card(
                        "Calls per species",
                        "calls_per_species.png",
                        "How many call types are currently recorded per species.",
                    ),
                    _img_card(
                        "Keyword frequency",
                        "keyword_freq.png",
                        "Most common ontology keywords across the dataset.",
                    ),
                    _img_card(
                        "Keyword heatmap (binned)",
                        "keyword_heatmap_bin.png",
                        "Species × keyword presence, binned into frequency buckets.",
                    ),
                ],
            ),
        ],
    )

    # Store user selection (species allowlist). Empty means "all species".
    # Taxonomy graph clicks will toggle species in/out of the selection.
    from dash import State

    # Map taxonomy node id (e.g., "family:Hominidae") -> set of species names.
    tax_ranks = ["kingdom", "phylum", "class", "order", "family", "genus"]
    tax_to_species: Dict[str, set[str]] = {}

    # Use unique species entries (first call per species) to avoid duplicates.
    first_by_species: Dict[str, Dict[str, object]] = {}
    for c in calls_all:
        sp = str(c.get("species", "unknown"))
        if sp not in first_by_species:
            first_by_species[sp] = c

    for sp, c in first_by_species.items():
        for r in tax_ranks:
            v = str(c.get(r, "")).strip()
            if not v:
                continue
            nid = f"{r}:{v}"
            tax_to_species.setdefault(nid, set()).add(sp)

    # A convenience node for "all species"
    tax_to_species["root:Life"] = set(first_by_species.keys())

    app.layout = html.Div(
        className="app-root",
        children=[
            html.Div(
                className="hero",
                children=[
                    html.H1("Many species repertoires", className="hero__title"),
                    html.Div(
                        [
                            html.P(
                                "This page offers an interactive explorer of a cross-species database of animal vocal repertoires.",
                                className="hero__paragraph",
                            ),
                            html.P(
                                "Each point in the UMAP corresponds to a specific call type, embedded based on its acoustic description. "
                                "Colors indicate species.",
                                className="hero__paragraph",
                            ),
                            html.P(
                                "How to use the interface:",
                                className="hero__paragraph hero__paragraph--lead",
                            ),
                            html.Ul(
                                [
                                    html.Li(
                                        "Use the taxonomy panel on the left to filter species or higher taxonomic groups (play around with, it does not work perfectly)."
                                    ),
                                    html.Li(
                                        "Hover over points to inspect individual calls."
                                    ),
                                    html.Li(
                                        "Click a point to select a call and view its nearest semantic equivalent in every other species:"
                                    ),
                                    html.Li(
                                        "> The connections in the plot show these 'translations'."
                                    ),
                                    html.Li(
                                        "> The Translations panel below provides all details, it also indicates whether these 'translations' map back to the original call (back-translation)."
                                    ),
                                ],
                                className="hero__list",
                            ),
                            html.P(
                                "Static plots below",
                                className="hero__paragraph hero__paragraph--lead",
                            ),
                            html.P(
                                "At the bottom of the page, you will find some plots summarizing the whole data.",
                                className="hero__paragraph",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="app-row",
                children=[
                    # Left: taxonomy tree (click to filter)
                    html.Div(
                        className="col col--left",
                        children=[
                            dcc.Store(id="species-store", data=[]),
                            # --- BEGIN: Selection mode toggle ---
                            html.H3("Taxonomy"),
                            html.Div(
                                "Click taxonomy nodes to filter species (add a group or focus on a group)",
                                className="subtle",
                            ),
                            html.Div(
                                className="taxonomy-controls",
                                children=[
                                    html.Div(
                                        "Selection mode",
                                        className="taxonomy-controls__label",
                                    ),
                                    dcc.RadioItems(
                                        id="selection-mode",
                                        options=[
                                            {"label": "Additive", "value": "add"},
                                            {"label": "Replace", "value": "replace"},
                                        ],
                                        value="add",
                                        inline=True,
                                        className="taxonomy-controls__radio",
                                    ),
                                ],
                            ),
                            # --- END: Selection mode toggle ---
                            dcc.Graph(id="taxonomy", figure=fig_tax),
                        ],
                    ),
                    # Middle: UMAP
                    html.Div(
                        className="col col--middle",
                        children=[
                            html.H3("UMAP"),
                            dcc.Graph(
                                id="scatter", figure=fig_scatter, clear_on_unhover=True
                            ),
                        ],
                    ),
                    # Right: hovered call only (selected call panel removed)
                    html.Div(
                        className="col col--right",
                        children=[
                            html.H3("Hovered call"),
                            html.Div(
                                id="overlayed",
                                children="Hover a point to see its details here.",
                                className="panel panel--hovered",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="below-row",
                children=[
                    html.Div(
                        className="below-row__selected",
                        children=[
                            html.H3("Selected call"),
                            html.Div(
                                id="selected-call",
                                children="Click a point to select a call.",
                                className="panel panel--selected",
                            ),
                        ],
                    ),
                    html.Div(
                        className="below-row__translations",
                        children=[
                            html.H3("Translations"),
                            html.Div(
                                id="translations-strip",
                                children="Click a point to see nearest semantic neighbors in other displayed species.",
                                className="selection-strip",
                            ),
                        ],
                    ),
                ],
            ),
            static_gallery,
        ],
    )

    # --- New callbacks for species/taxonomy selection ---

    # --- Reusable call-card renderer for selected/hovered call panels and translation strip ---
    def _render_call_card(c: Dict[str, object]) -> html.Div:
        species = str(c.get("species", ""))
        call_name = str(c.get("call_name", ""))
        acoustic = str(c.get("acoustic_description", "") or "").strip()
        semantic = str(c.get("semantic_description", "") or "").strip()

        kws = c.get("ontology_keywords", []) or []
        if isinstance(kws, list):
            kws_list = [str(x) for x in kws if str(x).strip()]
        else:
            kws_list = [str(kws).strip()] if str(kws).strip() else []
        kw_text = ", ".join(kws_list) if kws_list else "None"

        return html.Div(
            className="call-card",
            children=[
                html.Div(
                    [
                        html.Strong(species, className="call-card__title-strong"),
                        html.Span(" — "),
                        html.Strong(call_name, className="call-card__title-strong"),
                    ],
                    className="call-card__title",
                ),
                html.Hr(className="call-card__hr"),
                html.Div(
                    [
                        html.Span("Acoustic: ", className="call-card__label"),
                        acoustic or "None",
                    ],
                    className="call-card__row",
                ),
                html.Div(
                    [
                        html.Span("Semantic: ", className="call-card__label"),
                        semantic or "None",
                    ],
                    className="call-card__row",
                ),
                html.Div(
                    [
                        html.Span("Keywords: ", className="call-card__label"),
                        kw_text,
                    ],
                    className="call-card__row",
                ),
            ],
        )

    @app.callback(
        Output("species-store", "data"),
        Input("taxonomy", "clickData"),
        Input("selection-mode", "value"),
        State("species-store", "data"),
        prevent_initial_call=True,
    )
    def _toggle_species_from_taxonomy(clickData, mode, current):
        current_set = set(current or [])
        if not clickData or not clickData.get("points"):
            return sorted(current_set)

        mode = mode or "add"
        is_replace = mode == "replace"

        pt = clickData["points"][0]
        node_id = pt.get("id") or pt.get("label")
        if not node_id:
            return sorted(current_set)

        # Click root to reset to "all" (empty store)
        if node_id == "root:Life" or node_id == "Life":
            return []

        node_id = str(node_id)
        if ":" not in node_id:
            return sorted(current_set)

        rank, value = node_id.split(":", 1)

        if rank == "species":
            sp = value
            if is_replace:
                # Replace selection with this single species; clicking again clears
                if current_set == {sp}:
                    return []
                return [sp]
            # Additive: toggle
            if sp in current_set:
                current_set.remove(sp)
            else:
                current_set.add(sp)
            return sorted(current_set)

        species_under = tax_to_species.get(node_id, set())
        if not species_under:
            return sorted(current_set)

        if is_replace:
            # Replace selection with the species under this node; clicking again clears
            if current_set == set(species_under):
                return []
            return sorted(species_under)

        # Additive (existing): toggle union/subtract
        if species_under.issubset(current_set):
            current_set -= set(species_under)
        else:
            current_set |= set(species_under)

        return sorted(current_set)

    @app.callback(
        Output("scatter", "figure"),
        Input("species-store", "data"),
        Input("scatter", "clickData"),
    )
    def _update_scatter_from_store(species_sel, clickData):
        species_sel = species_sel or []
        filtered = filter_calls(
            calls_all,
            species_allowlist=species_sel if species_sel else None,
        )

        clicked_idx = None
        if clickData and clickData.get("points"):
            clicked_idx = clickData["points"][0].get("customdata")

        return make_scatter(filtered, clicked_idx=clicked_idx)

    @app.callback(
        Output("selected-call", "children"),
        Output("translations-strip", "children"),
        Input("species-store", "data"),
        Input("scatter", "clickData"),
    )
    def _show_selected_and_translations(species_sel, clickData):
        # Build the currently displayed set (same as scatter filtering)
        species_sel = species_sel or []
        filtered = filter_calls(
            calls_all,
            species_allowlist=species_sel if species_sel else None,
        )
        idxs = [int(c["_idx"]) for c in filtered]

        if not idxs:
            return (
                dcc.Markdown("No calls in the current view.", className="subtle"),
                dcc.Markdown("No calls in the current view.", className="subtle"),
            )

        placeholder_selected = dcc.Markdown(
            "Click a point to select a call.",
            className="subtle",
        )
        placeholder_trans = dcc.Markdown(
            "Click a point to see nearest semantic neighbors in other displayed species.",
            className="subtle",
        )

        if not clickData or not clickData.get("points"):
            return (placeholder_selected, placeholder_trans)

        clicked_idx = clickData["points"][0].get("customdata")
        if clicked_idx is None:
            return (placeholder_selected, placeholder_trans)

        try:
            clicked_idx = int(clicked_idx)
        except Exception:
            return (placeholder_selected, placeholder_trans)

        if clicked_idx not in set(idxs):
            return (
                dcc.Markdown(
                    "The selected point is not in the current filtered view.",
                    className="subtle",
                ),
                placeholder_trans,
            )

        base_species = species_all[clicked_idx]
        base_vec = semantic_embeds[clicked_idx]

        # Nearest semantic neighbor per other species within the CURRENT view
        neighbors: List[Tuple[float, int]] = []
        species_in_view = sorted({species_all[i] for i in idxs})
        for sp in species_in_view:
            if sp == base_species:
                continue
            candidates = [i for i in idxs if species_all[i] == sp]
            if not candidates:
                continue
            cand_vecs = semantic_embeds[candidates]
            dists = _np.linalg.norm(cand_vecs - base_vec, axis=1)
            j = int(dists.argmin())
            nn_idx = candidates[j]
            neighbors.append((float(dists[j]), nn_idx))

        neighbors.sort(key=lambda x: x[0])

        selected_card = _render_call_card(calls_all[clicked_idx])

        # Translation cards ordered by closeness, with similarity + back-translation check
        trans_cards = []

        # Candidates for back-translation: calls in the original (base) species within current view
        base_candidates = [i for i in idxs if species_all[i] == base_species]

        for dist, nn_idx in neighbors:
            # Similarity between selected call and the translation (embeddings are normalized)
            cos_sim = 1.0 - (dist * dist) / 2.0

            # Back-translation: from the translation back into the original species
            bt_idx = None
            bt_sim = None
            if base_candidates:
                bt_dists = _np.linalg.norm(
                    semantic_embeds[base_candidates] - semantic_embeds[nn_idx], axis=1
                )
                j_bt = int(bt_dists.argmin())
                bt_idx = base_candidates[j_bt]
                bt_dist = float(bt_dists[j_bt])
                bt_sim = 1.0 - (bt_dist * bt_dist) / 2.0

            if bt_idx is not None:
                bt_call_name = str(calls_all[bt_idx].get("call_name", ""))
                icon = "✅" if int(bt_idx) == int(clicked_idx) else "❌"
                back_line = f"Back-translation: {icon} {bt_call_name} (similarity: {bt_sim:.3f})"
            else:
                back_line = "Back-translation: (none)"

            trans_cards.append(
                html.Div(
                    className="translation-card",
                    children=[
                        html.Div(
                            f"Similarlity: {cos_sim:.3f}",
                            className="subtle translation-metric",
                        ),
                        html.Div(
                            back_line,
                            className="subtle translation-metric",
                        ),
                        _render_call_card(calls_all[nn_idx]),
                    ],
                )
            )

        translations_inner = html.Div(
            className="selection-strip__inner",
            children=trans_cards,
        )

        return (selected_card, translations_inner)

    @app.callback(
        Output("overlayed", "children"),
        Input("scatter", "hoverData"),
    )
    def _show_hovered(hoverData):
        if not hoverData or not hoverData.get("points"):
            return dcc.Markdown(
                "Hover a point in the UMAP to see its details here.",
                className="subtle",
            )

        idx = hoverData["points"][0].get("customdata")
        if idx is None:
            return dcc.Markdown(
                "Hover a point in the UMAP to see its details here.",
                className="subtle",
            )

        c = calls_all[int(idx)]
        return _render_call_card(c)

    # Dash 2.14+ prefers app.run(); keep a fallback for older versions.
    try:
        app.run(host=host, port=port, debug=False)
    except Exception:
        app.run_server(host=host, port=port, debug=False)


if __name__ == "__main__":
    # Default behavior: generate static plots.
    # generate_all_plots()
    # To run the internal web app instead, uncomment:
    run_dash_app()

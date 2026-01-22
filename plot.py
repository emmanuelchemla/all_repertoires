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


def load_calls(json_path: Path) -> List[Dict[str, object]]:
    """Flatten species -> calls into a list of call dicts."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    calls: List[Dict[str, object]] = []
    for species_entry in data.get("species", []):
        species_name = species_entry.get("species_name", "unknown")
        for call in species_entry.get("calls", []):
            calls.append(
                {
                    "species": species_name,
                    "call_name": call.get("call_name", "unknown"),
                    "acoustic_description": call.get("acoustic_description", ""),
                    "semantic_description": call.get("semantic_description", ""),
                    "ontology_keywords": call.get("ontology_keywords", []),
                }
            )
    return calls


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


def reduce_umap(embeddings: np.ndarray) -> np.ndarray:
    reducer = UMAP(random_state=42)
    return reducer.fit_transform(embeddings)


def plot_umap(embeddings: np.ndarray, species: List[str], out_path: Path) -> None:
    coords = reduce_umap(embeddings)

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
            label=sp,
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
            label=sp,
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


def plot_calls_per_species(species: List[str], out_path: Path) -> None:
    counts = Counter(species)
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*items)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(labels)), values, color="steelblue")
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.ylabel("# calls")
    plt.title("Number of calls per species")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_keyword_frequency(
    ontology_keywords: List[List[str]], out_path: Path, top_n: int = 20
) -> None:
    counter: Counter[str] = Counter()
    for kws in ontology_keywords:
        counter.update(kws or [])
    if not counter:
        return
    most = counter.most_common(top_n)
    labels, values = zip(*most)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(labels)), values, color="darkgreen")
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} ontology keywords")
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
    plt.yticks(range(len(species_set)), species_set)
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
    plt.yticks(range(len(species_set)), species_set)
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

    # shared reducer for semantic projections (so keywords align)
    reducer = UMAP(random_state=42)
    reducer.fit(semantic_embeds)

    outputs: Dict[str, Path] = {}
    outputs["acoustic_umap"] = out_dir / "acoustic_umap.png"
    plot_umap(acoustic_embeds, species, outputs["acoustic_umap"])

    outputs["semantic_umap"] = out_dir / "semantic_umap.png"
    plot_semantic_with_keywords(
        semantic_embeds,
        species,
        call_names,
        ontology_keywords,
        encoder,
        reducer,
        outputs["semantic_umap"],
    )

    outputs["calls_per_species"] = out_dir / "calls_per_species.png"
    plot_calls_per_species(species, outputs["calls_per_species"])

    outputs["keyword_freq"] = out_dir / "keyword_freq.png"
    plot_keyword_frequency(ontology_keywords, outputs["keyword_freq"])

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

    coords = reduce_umap(acoustic_embeds)

    species_unique = sorted(set(species))
    species_to_idx = {sp: np.where(np.array(species) == sp)[0] for sp in species_unique}
    color_map = {
        sp: qualitative.Plotly[i % len(qualitative.Plotly)]
        for i, sp in enumerate(species_unique)
    }
    colors = [color_map[s] for s in species]

    scatter = go.Scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers",
        marker=dict(color=colors, size=8, opacity=0.8),
        text=[f"{c} ({s})" for c, s in zip(call_names, species)],
        hoverinfo="text",
        showlegend=False,
        uid="scatter",
    )
    # one line trace per species for colored/transparent connections
    line_traces = [
        go.Scatter(
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
        for sp in species_unique
    ]
    label_trace = go.Scatter(
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

    # legend-only markers for species color key
    legend_traces = [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color=color_map[sp], size=8),
            name=sp,
            hoverinfo="skip",
            showlegend=True,
            uid=f"legend-{sp}",
        )
        for sp in species_unique
    ]

    fig = go.FigureWidget(
        data=[scatter, *line_traces, label_trace, *legend_traces],
        layout=go.Layout(
            title="UMAP of Acoustic Descriptions (interactive)",
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            showlegend=True,
            width=width,
            height=height,
        ),
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
            # clear all line traces
            for i in range(len(species_unique)):
                fig.data[line_start + i].update(x=[], y=[])
            fig.data[label_idx].update(x=[], y=[], text=[])
            return

        idx = points.point_inds[0]
        base_species = species_arr[idx]

        label_x: List[float] = []
        label_y: List[float] = []
        label_texts: List[str] = []

        # always include the clicked call's label
        label_x.append(coords[idx, 0])
        label_y.append(coords[idx, 1])
        label_texts.append(call_names[idx])

        # clear existing lines first
        for i in range(len(species_unique)):
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
            fig.data[line_start + species_unique.index(sp)].update(
                x=[coords[idx, 0], coords[nn_idx, 0]],
                y=[coords[idx, 1], coords[nn_idx, 1]],
            )

            # labels for neighbor
            label_x.append(coords[nn_idx, 0])
            label_y.append(coords[nn_idx, 1])
            label_texts.append(call_names[nn_idx])

        fig.data[label_idx].update(x=label_x, y=label_y, text=label_texts)

    fig.data[0].on_click(on_click)
    return fig


if __name__ == "__main__":
    generate_all_plots()

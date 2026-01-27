from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import warnings

import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP

import re

# Silence UMAP warning about n_jobs being forced to 1 when random_state is set.
warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value 1 overridden to 1 by setting random_state",
    category=UserWarning,
)


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
    # Keep margins tight so the wheel fills the available panel space
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
        title="UMAP of Acoustic Descriptions. Click on a dot (a call) to display semantic nearest neighbors, with details below.",
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
        from dash.dependencies import ALL
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

    # --- Fixed group colors for consistency across plots ---
    GROUP_COLOR_PRESET: Dict[str, str] = {
        "Bats": "#6b7280",  # gray
        "Amphibia": "#22c55e",  # green
        "Primates": "#a855f7",  # purple
        "Apes": "#7c3aed",  # deeper purple
        "Birds": "#0ea5e9",  # sky blue
        "Carnivores": "#f97316",  # orange
        "Elephants": "#10b981",  # teal/green
        "Ungulates": "#f59e0b",  # amber
        "Other": "#94a3b8",  # slate
    }
    FALLBACK_COLORS = _qual.Plotly

    def _group_label(c: Dict[str, object]) -> str:
        cls = str(c.get("class", "")).lower()
        order = str(c.get("order", "")).lower()
        family = str(c.get("family", "")).lower()
        if "hominidae" in family:
            return "Apes"
        if "primates" in order:
            return "Primates"
        if "aves" in cls:
            return "Birds"
        if "amphibia" in cls:
            return "Amphibia"
        if "chiroptera" in order:
            return "Bats"
        if "carnivora" in order:
            return "Carnivores"
        if "proboscidea" in order:
            return "Elephants"
        if order in {"cetartiodactyla", "artiodactyla", "perissodactyla"}:
            return "Ungulates"
        return cls.title() if cls else "Other"

    # --- Helper function: taxonomic icon ---
    def _taxon_icon(c: Dict[str, object]) -> str:
        """Return a small emoji icon based on broad taxonomy."""
        cls = str(c.get("class", "")).lower()
        order = str(c.get("order", "")).lower()

        if "aves" in cls:
            return "🐦"
        if "amphibia" in cls:
            return "🐸"
        if "mammalia" in cls:
            if "primates" in order:
                family = str(c.get("family", "")).lower()
                genus = str(c.get("genus", "")).lower()

                # Great apes (Hominidae)
                if "hominidae" in family or genus in ["pan", "gorilla", "pongo"]:
                    return "🦧"

                # Other primates
                return "🐒"
            if any(x in order for x in ["chiroptera"]):
                return "🦇"
            if any(x in order for x in ["proboscidea"]):
                return "🐘"
            if any(x in order for x in ["carnivora"]):
                return "🐺"
            return "🐾"
        return ""

    # Species -> icon map (first occurrence per species)
    species_icon_map: Dict[str, str] = {}
    for sp, c in zip(species_all, calls_all):
        if sp not in species_icon_map:
            species_icon_map[sp] = _taxon_icon(c)

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
                    name=f"{_taxon_icon(calls_all[sp_idxs[0]])} {species_common_name(sp)}",
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
            title=dict(
                text="UMAP of Acoustic Descriptions<br><span style='font-size:12px; color:#6b7280;'>Click a dot (call) to see semantic nearest neighbors; with full details displayed below</span>",
                y=0.97,
            ),
            margin=dict(t=58, l=10, r=10, b=10),
        )
        fig.update_xaxes(
            title=None,
            showticklabels=False,
            showgrid=True,
            zeroline=False,
            gridcolor="rgba(200,200,200,0.6)",
            gridwidth=1,
            dtick=3,
        )
        fig.update_yaxes(
            title=None,
            showticklabels=False,
            showgrid=True,
            zeroline=False,
            gridcolor="rgba(200,200,200,0.6)",
            gridwidth=1,
            dtick=3,
            scaleanchor="x",
        )
        return fig

    # Initial figures
    init_calls = calls_all
    fig_scatter = make_scatter(init_calls, clicked_idx=None)
    fig_tax = None
    try:
        fig_tax = build_taxonomy_sunburst(
            calls_all,
            title="",
            species_color_map=color_map,
        )
    except Exception:
        fig_tax = go.Figure()

    # Precompute UMAPs for static plots (two random seeds for variety)
    def _reduce_umap_static(embeds, n_components: int, seed: int = 42):
        reducer = UMAP(random_state=seed, n_components=n_components)
        return reducer.fit_transform(embeds)

    acoustic_umap2d_a = _reduce_umap_static(acoustic_embeds, 2, seed=42)
    acoustic_umap3d_a = _reduce_umap_static(acoustic_embeds, 3, seed=42)
    semantic_umap2d = _reduce_umap_static(semantic_embeds, 2, seed=42)
    semantic_umap3d = _reduce_umap_static(semantic_embeds, 3, seed=42)

    app = Dash(__name__)

    # Styling: CSS lives in ./assets/style.css (Dash auto-loads assets).
    # Prefer className hooks over inline style dicts.

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
            html.Nav(
                className="topnav",
                children=[
                    html.Div(
                        "Cross-species Repertoire Explorer & Translator",
                        className="topnav__brand",
                    ),
                    html.Div(
                        className="topnav__links",
                        children=[
                            html.A("Home", href="#home", className="topnav__link"),
                            html.A(
                                "Summary plots",
                                href="#static-plots",
                                className="topnav__link",
                            ),
                            html.A(
                                "Single species repertoire exploration",
                                href="#one-species-explore",
                                className="topnav__link",
                            ),
                            html.A(
                                "Single call translated to every other species",
                                href="#translations",
                                className="topnav__link",
                            ),
                            html.A(
                                "Full species repertoire translated to all species",
                                href="#one-species",
                                className="topnav__link",
                            ),
                            html.A(
                                "Full repertoire bilingual mapping",
                                href="#pair-species",
                                className="topnav__link",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="home",
                className="section section--home",
                children=[
                    html.Div(
                        className="section__header",
                        children=[
                            html.Div(
                                [
                                    html.P("Overview", className="section__kicker"),
                                    html.H2("Home", className="section__title"),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="hero__body",
                        children=[
                            html.P(
                                "Interactive explorer of a cross-species database of animal vocal repertoires.",
                                className="hero__paragraph",
                            ),
                            html.P(
                                f"{len(species_unique)} species · {len(calls_all)} calls currently in the database",
                                className="hero__paragraph hero__paragraph--meta",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "TODO: assess quality of LLM generated database"
                                    ),
                                    html.Div("TODO: review database manually"),
                                    html.Div(
                                        "TODO: make descriptions homogeneous by asking a full rewrite (certainly if we import expert descriptions)"
                                    ),
                                    html.Div(
                                        "TODO: expand database (more species, more calls)"
                                    ),
                                    html.Div(
                                        "TODO: provide reviewing/editing tools to the database for external users?"
                                    ),
                                    html.Div(
                                        "TODO: make the bilingual plot clickable (display the card of a call clicked on, and of its translation)"
                                    ),
                                    html.Div(
                                        "TODO: fix the issue with automatic scrolling for alignment"
                                    ),
                                    html.Div("TODO: better visualization"),
                                    html.Div(
                                        "TODO: see if text embeddings are doing a good job"
                                    ),
                                    html.Div(
                                        "TODO: explain 'translation' caveats. One of them: predator does not mean the same thing for all species (compare: president/king)"
                                    ),
                                ],
                                className="hero__todo",
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="static-plots",
                className="section section--static",
                children=[
                    html.Div(
                        className="section__header",
                        children=[
                            html.Div(
                                [
                                    html.P(
                                        "Descriptive plots", className="section__kicker"
                                    ),
                                    html.H2(
                                        "Quick summaries for the whole dataset",
                                        className="section__title",
                                    ),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="static-layout",
                        children=[
                            html.Div(
                                className="static-left",
                                children=[
                                    dcc.Store(id="static-species-store", data=[]),
                                    html.H3("Taxonomy selection"),
                                    html.Div(
                                        className="panel panel--taxonomy",
                                        children=[
                                            html.Div(
                                                "Click taxonomy nodes to filter species",
                                                className="subtle subtle--tight",
                                            ),
                                            html.Div(
                                                className="taxonomy-controls taxonomy-controls--compact taxonomy-controls--inline",
                                                children=[
                                                    html.Span(
                                                        "Selection mode",
                                                        className="taxonomy-controls__label",
                                                    ),
                                                    dcc.RadioItems(
                                                        id="static-selection-mode",
                                                        options=[
                                                            {
                                                                "label": "Focus",
                                                                "value": "replace",
                                                            },
                                                            {
                                                                "label": "Add",
                                                                "value": "add",
                                                            },
                                                        ],
                                                        value="replace",
                                                        inline=True,
                                                        className="taxonomy-controls__radio",
                                                    ),
                                                ],
                                            ),
                                            dcc.Graph(
                                                id="static-taxonomy",
                                                figure=fig_tax,
                                                style={
                                                    "height": "360px",
                                                    "margin": "0",
                                                },
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="static-right",
                                children=[
                                    html.Div(
                                        className="static-row",
                                        children=[
                                            dcc.Graph(
                                                id="static-calls-bar",
                                                className="static-graph static-graph--tall",
                                            ),
                                            dcc.Graph(
                                                id="static-kw-bar",
                                                className="static-graph static-graph--tall",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="static-row",
                                        children=[
                                            dcc.Graph(
                                                id="static-heat-acoustic",
                                                className="static-graph",
                                            ),
                                            dcc.Graph(
                                                id="static-heat-semantic",
                                                className="static-graph",
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="static-row",
                                        children=[
                                            html.Div(
                                                id="static-mantel",
                                                className="panel panel--tight",
                                                style={
                                                    "gridColumn": "span 2",
                                                },
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="one-species-explore",
                className="section section--onespecies",
                children=[
                    html.Div(
                        className="section__header",
                        children=[
                            html.Div(
                                [
                                    html.P(
                                        "Single species repertoire exploration",
                                        className="section__kicker",
                                    ),
                                    html.H2(
                                        "Single species repertoire exploration",
                                        className="section__title",
                                    ),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="one-species__controls",
                        children=[
                            html.Label(
                                "Choose a species",
                                className="taxonomy-controls__label",
                            ),
                            dcc.Dropdown(
                                id="one-species-explore-dropdown",
                                options=[
                                    {"label": species_common_name(sp), "value": sp}
                                    for sp in species_unique
                                ],
                                placeholder="Select a species",
                                clearable=True,
                            ),
                        ],
                    ),
                    html.Div(
                        id="one-species-explore-list",
                        className="one-species-explore__list",
                        children=html.Div(
                            "Pick a species to see its calls.",
                            className="subtle",
                        ),
                    ),
                ],
            ),
            html.Div(
                id="translations",
                className="section section--translations",
                children=[
                    html.Div(
                        className="translations-container",
                        children=[
                            html.Div(
                                className="section__header",
                                children=[
                                    html.Div(
                                        [
                                            html.P(
                                                "Single call translation",
                                                className="section__kicker",
                                            ),
                                            html.H2(
                                                "Single call translated to a call in every other species",
                                                className="section__title",
                                            ),
                                        ]
                                    )
                                ],
                            ),
                            html.Div(
                                className="static-layout",
                                children=[
                                    html.Div(
                                        className="static-left",
                                        children=[
                                            dcc.Store(
                                                id="trans-species-store", data=[]
                                            ),
                                            html.H3("Taxonomy selection"),
                                            html.Div(
                                                className="panel panel--taxonomy",
                                                children=[
                                                    html.Div(
                                                        "Click taxonomy nodes to filter species",
                                                        className="subtle subtle--tight",
                                                    ),
                                                    html.Div(
                                                        className="taxonomy-controls taxonomy-controls--compact taxonomy-controls--inline",
                                                        children=[
                                                            html.Span(
                                                                "Selection mode",
                                                                className="taxonomy-controls__label",
                                                            ),
                                                            dcc.RadioItems(
                                                                id="trans-selection-mode",
                                                                options=[
                                                                    {
                                                                        "label": "Focus",
                                                                        "value": "replace",
                                                                    },
                                                                    {
                                                                        "label": "Add",
                                                                        "value": "add",
                                                                    },
                                                                ],
                                                                value="replace",
                                                                inline=True,
                                                                className="taxonomy-controls__radio",
                                                            ),
                                                        ],
                                                    ),
                                                    dcc.Graph(
                                                        id="trans-taxonomy",
                                                        figure=fig_tax,
                                                        style={
                                                            "height": "360px",
                                                            "margin": "0",
                                                        },
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                id="trans-legend",
                                                className="legend legend--stacked",
                                                style={
                                                    "maxHeight": "160px",
                                                    "overflowY": "auto",
                                                    "fontSize": "12px",
                                                    "padding": "6px 8px",
                                                    "border": "1px solid var(--border-2)",
                                                    "borderRadius": "var(--radius)",
                                                    "background": "var(--surface)",
                                                    "marginTop": "8px",
                                                },
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                "●",
                                                                style={
                                                                    "color": color_map.get(
                                                                        sp, "#555"
                                                                    ),
                                                                    "fontSize": "12px",
                                                                    "marginRight": "6px",
                                                                },
                                                            ),
                                                            html.Span(
                                                                f"{species_icon_map.get(sp, '')} {species_common_name(sp)}",
                                                                className="legend__label",
                                                            ),
                                                        ],
                                                        className="legend__item",
                                                    )
                                                    for sp in species_unique
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(
                                        className="static-right",
                                        children=[
                                            html.Div(
                                                className="static-row",
                                                children=[
                                                    dcc.Graph(
                                                        id="trans-umap2d-a",
                                                        className="static-graph",
                                                    ),
                                                    dcc.Graph(
                                                        id="trans-umap3d-a",
                                                        className="static-graph",
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className="static-row",
                                                children=[
                                                    dcc.Graph(
                                                        id="trans-umap2d-b",
                                                        className="static-graph",
                                                    ),
                                                    dcc.Graph(
                                                        id="trans-umap3d-b",
                                                        className="static-graph",
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                            html.Div(
                                className="translations-bottom",
                                children=[
                                    dcc.Store(id="trans-selected-idx", data=None),
                                    dcc.Store(id="trans-scroll-trigger", data=None),
                                    dcc.Store(id="trans-focus-species", data=None),
                                    html.Div(
                                        id="trans-selected-call",
                                        className="panel panel--tight",
                                        children="Click a point in any UMAP to select a call.",
                                        style={
                                            "flex": "0 0 28%",
                                            "maxWidth": "360px",
                                            "minWidth": "260px",
                                        },
                                    ),
                                    html.Div(
                                        style={
                                            "flex": "1 1 auto",
                                            "display": "flex",
                                            "flexDirection": "column",
                                            "gap": "10px",
                                            "minWidth": "0",
                                        },
                                        children=[
                                            html.Div(
                                                id="trans-translations-strip",
                                                className="panel panel--tight selection-strip selection-strip--horizontal",
                                                children="Semantic translations will appear here.",
                                                style={"maxHeight": "340px"},
                                            ),
                                            html.Div(
                                                id="trans-translations-acoustic",
                                                className="panel panel--tight selection-strip selection-strip--horizontal",
                                                children="Acoustic translations will appear here.",
                                                style={"maxHeight": "340px"},
                                            ),
                                        ],
                                    ),
                                ],
                                style={"maxWidth": "1200px", "margin": "0 auto"},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                id="one-species",
                className="section section--onespecies",
                children=[
                    html.Div(
                        className="section__header",
                        children=[
                            html.Div(
                                [
                                    html.P(
                                        "Full species repertoire translated",
                                        className="section__kicker",
                                    ),
                                    html.H2(
                                        "Mapping every call of a species to a call in every other species",
                                        className="section__title",
                                    ),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="one-species__controls",
                        children=[
                            html.Label(
                                "Choose a species", className="taxonomy-controls__label"
                            ),
                            dcc.Dropdown(
                                id="one-species-dropdown",
                                options=[
                                    {"label": species_common_name(sp), "value": sp}
                                    for sp in species_unique
                                ],
                                placeholder="Select a species",
                                clearable=True,
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Show translations",
                                        className="taxonomy-controls__label",
                                        style={"marginTop": "8px"},
                                    ),
                                    dcc.Checklist(
                                        id="one-species-mode",
                                        options=[
                                            {"label": "Semantic", "value": "semantic"},
                                            {"label": "Acoustic", "value": "acoustic"},
                                        ],
                                        value=["semantic", "acoustic"],
                                        inline=True,
                                        className="taxonomy-controls__radio taxonomy-controls__radio--compact",
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(
                        id="one-species-list",
                        className="one-species__list",
                        children=html.Div(
                            "Pick a species to see its calls and translations.",
                            className="subtle",
                        ),
                    ),
                ],
            ),
            html.Div(
                id="pair-species",
                className="section section--pair",
                children=[
                    html.Div(
                        className="section__header",
                        children=[
                            html.Div(
                                [
                                    html.P(
                                        "Full repertoire bilingual mapping",
                                        className="section__kicker",
                                    ),
                                    html.H2(
                                        "FInd best repertoire alignment across two species",
                                        className="section__title",
                                    ),
                                ]
                            )
                        ],
                    ),
                    html.Div(
                        className="pair-controls",
                        children=[
                            dcc.Dropdown(
                                id="pair-species-1",
                                options=[
                                    {"label": species_common_name(sp), "value": sp}
                                    for sp in species_unique
                                ],
                                placeholder="Species 1",
                                clearable=False,
                                value=species_unique[0] if species_unique else None,
                                className="pair-dropdown",
                            ),
                            dcc.RadioItems(
                                id="pair-space",
                                options=[
                                    {
                                        "label": "Semantic nearest neighbors",
                                        "value": "semantic",
                                    },
                                    {
                                        "label": "Acoustic nearest neighbors",
                                        "value": "acoustic",
                                    },
                                ],
                                value="semantic",
                                inline=False,
                                className="taxonomy-controls__radio taxonomy-controls__radio--compact pair-space-toggle",
                            ),
                            dcc.Dropdown(
                                id="pair-species-2",
                                options=[
                                    {"label": species_common_name(sp), "value": sp}
                                    for sp in species_unique
                                ],
                                placeholder="Species 2",
                                clearable=False,
                                value=(
                                    species_unique[1]
                                    if len(species_unique) > 1
                                    else None
                                ),
                                className="pair-dropdown pair-dropdown--right",
                            ),
                        ],
                    ),
                    html.Div(
                        className="pair-layout",
                        children=[
                            html.Div(
                                id="pair-hover-left",
                                className="pair-hover__col pair-hover__col--left",
                                children=html.Div(
                                    "Click a call to see details.",
                                    className="subtle",
                                ),
                            ),
                            dcc.Graph(
                                id="pair-graph",
                                className="pair-graph",
                                figure=go.Figure(),
                            ),
                            html.Div(
                                id="pair-hover-right",
                                className="pair-hover__col pair-hover__col--right",
                                children=html.Div(
                                    "Click a call to see details.",
                                    className="subtle",
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    # --- New callbacks for species/taxonomy selection ---

    # --- New callbacks for species/taxonomy selection ---

    # --- Reusable call-card renderer for selected/hovered call panels and translation strip ---
    def _strip_parenthetical(text: str) -> str:
        """Drop trailing parenthetical from a call name, if present."""
        return re.sub(r"\s*\([^)]*\)\s*$", "", str(text)).strip()

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

        # Split species into common name and scientific name if present
        sp = species
        sci = ""
        if "(" in species and species.endswith(")"):
            sp = species[: species.rfind("(")].strip()
            sci = species[species.rfind("(") + 1 : -1].strip()

        return html.Div(
            className="call-card",
            children=[
                html.Div(
                    [
                        html.Span(_taxon_icon(c), className="call-card__icon"),
                        html.Span(" "),
                        html.Strong(sp, className="call-card__title-strong"),
                    ],
                    className="call-card__title",
                ),
                (
                    html.Div(
                        html.Em(sci),
                        className="call-card__subtitle",
                    )
                    if sci
                    else None
                ),
                html.Div(
                    html.Strong(call_name, className="call-card__callname"),
                    className="call-card__callname-wrapper",
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
        Output("selected-idx-store", "data"),
        Input("scatter", "clickData"),
        Input("clear-selection", "n_clicks"),
        State("selected-idx-store", "data"),
        prevent_initial_call=True,
    )
    def _update_selected_idx(clickData, n_clear, current):
        from dash import ctx

        if ctx.triggered_id == "clear-selection":
            return None

        if clickData and clickData.get("points"):
            idx = clickData["points"][0].get("customdata")
            try:
                return int(idx) if idx is not None else None
            except Exception:
                return None

        return current

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
        Output("static-species-store", "data"),
        Input("static-taxonomy", "clickData"),
        Input("static-selection-mode", "value"),
        State("static-species-store", "data"),
        prevent_initial_call=True,
    )
    def _toggle_species_static(clickData, mode, current):
        # same behavior as main taxonomy selector
        return _toggle_species_from_taxonomy(clickData, mode, current)

    @app.callback(
        Output("static-calls-bar", "figure"),
        Output("static-kw-bar", "figure"),
        Output("static-heat-acoustic", "figure"),
        Output("static-heat-semantic", "figure"),
        Output("static-mantel", "children"),
        Input("static-species-store", "data"),
    )
    def _update_static_figs(species_sel):
        fig_calls = make_calls_bar(species_sel or [])
        fig_kw = make_keyword_bar(species_sel or [])
        fig_heat_a = make_similarity_heatmap(
            acoustic_embeds, species_sel or [], "Pairwise acoustic similarities"
        )
        fig_heat_s = make_similarity_heatmap(
            semantic_embeds, species_sel or [], "Pairwise semantic similarities"
        )
        mantel_all = _mantel_stats(
            acoustic_embeds, semantic_embeds, _idxs_for_species(species_sel or [])
        )
        mantel_cross = _mantel_stats(
            acoustic_embeds,
            semantic_embeds,
            _idxs_for_species(species_sel or []),
            cross_only=True,
        )
        mantel_strong_all = _mantel_partial(
            acoustic_embeds, semantic_embeds, _idxs_for_species(species_sel or [])
        )

        def _fmt(res, label):
            if res is None:
                return f"{label}: n/a"
            r, p = res
            return f"{label}: r={r:.3f}, p={p:.3f}"

        return (
            fig_calls,
            fig_kw,
            fig_heat_a,
            fig_heat_s,
            html.Div(
                [
                    html.Div(
                        "Correlations between acoustic and semantic (Mantel tests):",
                        style={"marginBottom": "4px"},
                    ),
                    html.Table(
                        [
                            html.Tr(
                                [
                                    html.Td(html.Strong("Mantel (weak)")),
                                    html.Td("All pairs of calls"),
                                    html.Td(
                                        html.I(f"r = {mantel_all[0]:.3f}")
                                        if mantel_all
                                        else "n/a"
                                    ),
                                    html.Td(
                                        html.I(f"p = {mantel_all[1]:.3f}")
                                        if mantel_all
                                        else "n/a"
                                    ),
                                ],
                                style={"borderTop": "2px solid var(--border-strong)"},
                            ),
                            html.Tr(
                                [
                                    html.Td(""),
                                    html.Td("Cross-species only"),
                                    html.Td(
                                        html.I(f"r = {mantel_cross[0]:.3f}")
                                        if mantel_cross
                                        else "n/a"
                                    ),
                                    html.Td(
                                        html.I(f"p = {mantel_cross[1]:.3f}")
                                        if mantel_cross
                                        else "n/a"
                                    ),
                                ]
                            ),
                            html.Tr(
                                [],
                                style={"borderTop": "1px solid var(--border-2)"},
                            ),
                            html.Tr(
                                [
                                    html.Td(html.Strong("Mantel (strong)")),
                                    html.Td("All pairs of calls"),
                                    html.Td(
                                        html.I(f"r = {mantel_strong_all[0]:.3f}")
                                        if mantel_strong_all
                                        else "n/a"
                                    ),
                                    html.Td(
                                        html.I(f"p = {mantel_strong_all[1]:.3f}")
                                        if mantel_strong_all
                                        else "n/a"
                                    ),
                                ],
                                style={
                                    "borderTop": "2px solid var(--border-strong)",
                                    "borderBottom": "2px solid var(--border-strong)",
                                },
                            ),
                        ],
                        style={
                            "width": "auto",
                            "borderCollapse": "collapse",
                            "fontSize": "13px",
                        },
                        className="mantel-table",
                    ),
                ]
            ),
        )

    def _translations_for_idx_and_species(
        selected_idx: int, species_sel: list[str], focus_species: str | None = None
    ):
        """Return (selected_card, semantic_cards, acoustic_cards) for a call within a species subset."""
        species_sel = species_sel or []
        idxs = _idxs_for_species(species_sel)
        if selected_idx is None or selected_idx not in idxs:
            return None, None, None

        selected_card = _render_call_card(calls_all[selected_idx])
        base_species = species_all[selected_idx]
        species_in_view = sorted({species_all[i] for i in idxs})

        # precompute nearest semantic and acoustic per other species
        best_sem = {}
        best_ac = {}
        for sp in species_in_view:
            if sp == base_species:
                continue
            candidates = [i for i in idxs if species_all[i] == sp]
            if not candidates:
                continue
            d_sem = np.linalg.norm(
                semantic_embeds[candidates] - semantic_embeds[selected_idx], axis=1
            )
            best_sem[sp] = candidates[int(d_sem.argmin())]
            d_ac = np.linalg.norm(
                acoustic_embeds[candidates] - acoustic_embeds[selected_idx], axis=1
            )
            best_ac[sp] = candidates[int(d_ac.argmin())]

        def nearest_panel(
            embeds_from, other_embeds, label_prefix, secondary_label, secondary_best_map
        ):
            base_vec = embeds_from[selected_idx]
            neighbors = []
            for sp in species_in_view:
                if sp == base_species:
                    continue
                candidates = [i for i in idxs if species_all[i] == sp]
                if not candidates:
                    continue
                cand_vecs = embeds_from[candidates]
                dists = np.linalg.norm(cand_vecs - base_vec, axis=1)
                j = int(dists.argmin())
                neighbors.append((dists[j], candidates[j]))
            neighbors.sort(
                key=lambda x: (sp != focus_species if focus_species else False, x[0])
            )

            cards = []
            for dist, nn_idx in neighbors:
                sim = 1.0 - (dist * dist) / 2.0
                other_dist = float(
                    np.linalg.norm(other_embeds[nn_idx] - other_embeds[selected_idx])
                )
                other_sim = 1.0 - (other_dist * other_dist) / 2.0

                # back-translation in same space
                candidates_back = [i for i in idxs if species_all[i] == base_species]
                bt_idx = None
                bt_sim = None
                bt_ok = False
                if candidates_back:
                    bt_dists = np.linalg.norm(
                        embeds_from[candidates_back] - embeds_from[nn_idx], axis=1
                    )
                    j_bt = int(bt_dists.argmin())
                    bt_idx = candidates_back[j_bt]
                    bt_sim = 1.0 - (float(bt_dists[j_bt]) ** 2) / 2.0
                    if selected_idx in candidates_back:
                        sel_pos = candidates_back.index(selected_idx)
                        sel_dist = bt_dists[sel_pos]
                        min_dist = bt_dists[j_bt]
                        bt_ok = np.isclose(sel_dist, min_dist, atol=1e-6)
                    bt_ok = bt_ok or (bt_idx == selected_idx)
                    selected_call_name = _strip_parenthetical(
                        calls_all[selected_idx].get("call_name", "")
                    )
                    bt_call_name_check = _strip_parenthetical(
                        calls_all[bt_idx].get("call_name", "")
                    )
                    if selected_call_name and bt_call_name_check:
                        bt_ok = bt_ok or (
                            bt_call_name_check.lower() == selected_call_name.lower()
                        )

                if bt_idx is not None:
                    bt_call_name = _strip_parenthetical(
                        calls_all[bt_idx].get("call_name", "")
                    )
                    icon = "✅" if bt_ok else "❌"
                    back_line = f"Back-translation: {icon} {bt_call_name} (similarity: {bt_sim:.3f})"
                else:
                    back_line = "Back-translation: (none)"

                cards.append(
                    html.Div(
                        className="translation-card",
                        id={
                            "type": "trans-card",
                            "space": label_prefix.lower(),
                            "species": species_all[nn_idx],
                            "idx": int(nn_idx),
                        },
                        n_clicks=0,
                        **{"data-species": species_all[nn_idx]},
                        children=[
                            html.Div(
                                className="translation-meta",
                                children=[
                                    html.Div(
                                        f"{label_prefix} similarity: {sim:.3f}",
                                        className="subtle translation-metric",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                (
                                                    "✅ "
                                                    if nn_idx
                                                    == secondary_best_map.get(
                                                        species_all[nn_idx]
                                                    )
                                                    else "❌ "
                                                ),
                                                style={
                                                    "color": (
                                                        "#10b981"
                                                        if nn_idx
                                                        == secondary_best_map.get(
                                                            species_all[nn_idx]
                                                        )
                                                        else "#ef4444"
                                                    ),
                                                    "fontWeight": "700",
                                                },
                                            ),
                                            f"{secondary_label} similarity: {other_sim:.3f}",
                                        ],
                                        className="subtle translation-metric",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "✅ " if "✅" in back_line else "❌ ",
                                                style={
                                                    "color": (
                                                        "#10b981"
                                                        if "✅" in back_line
                                                        else "#ef4444"
                                                    ),
                                                    "fontWeight": "700",
                                                },
                                            ),
                                            back_line.replace("✅ ", "").replace(
                                                "❌ ", ""
                                            ),
                                        ],
                                        className="subtle translation-metric",
                                    ),
                                ],
                            ),
                            _render_call_card(calls_all[nn_idx]),
                        ],
                    )
                )
            return cards

        semantic_cards = nearest_panel(
            semantic_embeds, acoustic_embeds, "Semantic", "Acoustic", best_ac
        )
        acoustic_cards = nearest_panel(
            acoustic_embeds, semantic_embeds, "Acoustic", "Semantic", best_sem
        )

        return selected_card, semantic_cards, acoustic_cards

    @app.callback(
        Output("trans-species-store", "data"),
        Input("trans-taxonomy", "clickData"),
        Input("trans-selection-mode", "value"),
        State("trans-species-store", "data"),
        prevent_initial_call=True,
    )
    def _toggle_species_trans(clickData, mode, current):
        return _toggle_species_from_taxonomy(clickData, mode, current)

    @app.callback(
        Output("trans-umap2d-a", "figure"),
        Output("trans-umap3d-a", "figure"),
        Output("trans-umap2d-b", "figure"),
        Output("trans-umap3d-b", "figure"),
        Input("trans-species-store", "data"),
        Input("trans-selected-idx", "data"),
    )
    def _update_trans_figs(species_sel, selected_idx):
        fig_u2a = make_umap_fig(
            acoustic_umap2d_a,
            species_sel or [],
            "UMAP of Acoustic Descriptions<br><span style='font-size:12px; color:#6b7280;'>Click on a dot (i.e. on a call) to reveal its semantic nearest neighbors</span>",
            showlegend=False,
            selected_idx=selected_idx,
        )
        fig_u3a = make_umap_fig(
            acoustic_umap3d_a,
            species_sel or [],
            "UMAP of Acoustic Descriptions<br><span style='font-size:12px; color:#6b7280;'>Click on a dot (i.e. on a call) to reveal its semantic nearest neighbors</span>",
            showlegend=False,
            selected_idx=selected_idx,
        )
        fig_u2b = make_umap_fig(
            semantic_umap2d,
            species_sel or [],
            "UMAP of Semantic Descriptions<br><span style='font-size:12px; color:#6b7280;'>Click on a dot (i.e. on a call) to reveal its acoustic nearest neighbors</span>",
            showlegend=False,
            selected_idx=selected_idx,
        )
        fig_u3b = make_umap_fig(
            semantic_umap3d,
            species_sel or [],
            "UMAP of Semantic Descriptions<br><span style='font-size:12px; color:#6b7280;'>Click on a dot (i.e. on a call) to reveal its acoustic nearest neighbors</span>",
            showlegend=False,
            selected_idx=selected_idx,
        )

        # In acoustic UMAPs, draw segments to the nearest *semantic* neighbor
        # in each other species (same behavior as in the Explorer scatter).
        if selected_idx is not None:
            idxs = _idxs_for_species(species_sel or [])
            if selected_idx in idxs:
                base_species = species_all[selected_idx]
                sel2d = acoustic_umap2d_a[selected_idx]
                sel3d = acoustic_umap3d_a[selected_idx]
                for sp in sorted({species_all[i] for i in idxs}):
                    if sp == base_species:
                        continue
                    candidates = [i for i in idxs if species_all[i] == sp]
                    if not candidates:
                        continue
                    # choose nearest in SEMANTIC space
                    d_sem = np.linalg.norm(
                        semantic_embeds[candidates] - semantic_embeds[selected_idx],
                        axis=1,
                    )
                    nn_idx = candidates[int(d_sem.argmin())]
                    nn2d = acoustic_umap2d_a[nn_idx]
                    nn3d = acoustic_umap3d_a[nn_idx]
                    color = color_map.get(sp, "#888")
                    fig_u2a.add_trace(
                        go.Scatter(
                            x=[sel2d[0], nn2d[0]],
                            y=[sel2d[1], nn2d[1]],
                            mode="lines",
                            line=dict(color=color, width=2),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    fig_u3a.add_trace(
                        go.Scatter3d(
                            x=[sel3d[0], nn3d[0]],
                            y=[sel3d[1], nn3d[1]],
                            z=[sel3d[2], nn3d[2]],
                            mode="lines",
                            line=dict(color=color, width=3),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                # In semantic UMAPs, draw segments to the nearest *acoustic* neighbor
                # in each other species.
                sel2d_sem = semantic_umap2d[selected_idx]
                sel3d_sem = semantic_umap3d[selected_idx]
                for sp in sorted({species_all[i] for i in idxs}):
                    if sp == base_species:
                        continue
                    candidates = [i for i in idxs if species_all[i] == sp]
                    if not candidates:
                        continue
                    d_ac = np.linalg.norm(
                        acoustic_embeds[candidates] - acoustic_embeds[selected_idx],
                        axis=1,
                    )
                    nn_idx = candidates[int(d_ac.argmin())]
                    nn2d_sem = semantic_umap2d[nn_idx]
                    nn3d_sem = semantic_umap3d[nn_idx]
                    color = color_map.get(sp, "#888")
                    fig_u2b.add_trace(
                        go.Scatter(
                            x=[sel2d_sem[0], nn2d_sem[0]],
                            y=[sel2d_sem[1], nn2d_sem[1]],
                            mode="lines",
                            line=dict(color=color, width=2),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    fig_u3b.add_trace(
                        go.Scatter3d(
                            x=[sel3d_sem[0], nn3d_sem[0]],
                            y=[sel3d_sem[1], nn3d_sem[1]],
                            z=[sel3d_sem[2], nn3d_sem[2]],
                            mode="lines",
                            line=dict(color=color, width=3),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

        return fig_u2a, fig_u3a, fig_u2b, fig_u3b

    @app.callback(
        Output("trans-selected-idx", "data"),
        Input("trans-umap2d-a", "clickData"),
        Input("trans-umap3d-a", "clickData"),
        Input("trans-umap2d-b", "clickData"),
        Input("trans-umap3d-b", "clickData"),
        State("trans-selected-idx", "data"),
        prevent_initial_call=True,
    )
    def _select_trans_idx(cd2a, cd3a, cd2b, cd3b, current):
        from dash import ctx

        trigger = ctx.triggered_id
        clickData = {
            "trans-umap2d-a": cd2a,
            "trans-umap3d-a": cd3a,
            "trans-umap2d-b": cd2b,
            "trans-umap3d-b": cd3b,
        }.get(trigger)
        if clickData and clickData.get("points"):
            idx = clickData["points"][0].get("customdata")
            try:
                return int(idx) if idx is not None else current
            except Exception:
                return current
        return current

    @app.callback(
        Output("trans-focus-species", "data"),
        Input(
            {"type": "trans-card", "space": ALL, "idx": ALL, "species": ALL}, "n_clicks"
        ),
        State(
            {"type": "trans-card", "space": ALL, "idx": ALL, "species": ALL}, "species"
        ),
        State("trans-focus-species", "data"),
        prevent_initial_call=True,
    )
    def _focus_species_from_card(n_clicks_list, species_list, current):
        from dash import ctx

        trig = ctx.triggered_id
        if isinstance(trig, dict) and trig.get("species"):
            return trig.get("species")
        return current

    # Client-side scroll alignment: when focus species changes, scroll matching cards into view
    app.clientside_callback(
        """
        function(focus){
            if(!focus){return null;}
            const panels = ["trans-translations-strip", "trans-translations-acoustic"];
            panels.forEach(id => {
                const panel = document.getElementById(id);
                if(!panel) return;
                const card = panel.querySelector('[data-species=\"' + focus + '\"]');
                if(card && card.scrollIntoView){
                    card.scrollIntoView({behavior:"smooth", inline:"start", block:"nearest"});
                }
            });
            return null;
        }
        """,
        Output("trans-scroll-trigger", "data"),
        Input("trans-focus-species", "data"),
    )

    @app.callback(
        Output("trans-selected-call", "children"),
        Output("trans-translations-strip", "children"),
        Output("trans-translations-acoustic", "children"),
        Input("trans-selected-idx", "data"),
        Input("trans-species-store", "data"),
        Input("trans-focus-species", "data"),
    )
    def _update_trans_selected(selected_idx, species_sel, focus_species):
        species_sel = species_sel or []
        idxs = _idxs_for_species(species_sel)
        if selected_idx is None or selected_idx not in idxs:
            msg = "Click a point in the UMAPs to see details."
            return msg, msg, msg

        # selected call card
        selected_card = _render_call_card(calls_all[selected_idx])

        base_species = species_all[selected_idx]
        species_in_view = sorted({species_all[i] for i in idxs})

        # precompute nearest semantic and acoustic per other species
        best_sem = {}
        best_ac = {}
        for sp in species_in_view:
            if sp == base_species:
                continue
            candidates = [i for i in idxs if species_all[i] == sp]
            if not candidates:
                continue
            # semantic
            d_sem = _np.linalg.norm(
                semantic_embeds[candidates] - semantic_embeds[selected_idx], axis=1
            )
            best_sem[sp] = candidates[int(d_sem.argmin())]
            # acoustic
            d_ac = _np.linalg.norm(
                acoustic_embeds[candidates] - acoustic_embeds[selected_idx], axis=1
            )
            best_ac[sp] = candidates[int(d_ac.argmin())]

        def nearest_panel(
            embeds_from, other_embeds, label_prefix, secondary_label, secondary_best_map
        ):
            base_vec = embeds_from[selected_idx]
            neighbors = []
            for sp in species_in_view:
                if sp == base_species:
                    continue
                candidates = [i for i in idxs if species_all[i] == sp]
                if not candidates:
                    continue
                cand_vecs = embeds_from[candidates]
                dists = _np.linalg.norm(cand_vecs - base_vec, axis=1)
                j = int(dists.argmin())
                neighbors.append((dists[j], candidates[j]))
            neighbors.sort(
                key=lambda x: (sp != focus_species if focus_species else False, x[0])
            )

            cards = []
            for dist, nn_idx in neighbors:
                sim = 1.0 - (dist * dist) / 2.0
                # secondary similarity (other space)
                other_dist = float(
                    np.linalg.norm(other_embeds[nn_idx] - other_embeds[selected_idx])
                )
                other_sim = 1.0 - (other_dist * other_dist) / 2.0
                # back-translation in same space
                candidates_back = [i for i in idxs if species_all[i] == base_species]
                bt_idx = None
                bt_sim = None
                bt_ok = False
                if candidates_back:
                    bt_dists = _np.linalg.norm(
                        embeds_from[candidates_back] - embeds_from[nn_idx], axis=1
                    )
                    j_bt = int(bt_dists.argmin())
                    bt_idx = candidates_back[j_bt]
                    bt_sim = 1.0 - (float(bt_dists[j_bt]) ** 2) / 2.0
                    if selected_idx in candidates_back:
                        sel_pos = candidates_back.index(selected_idx)
                        sel_dist = bt_dists[sel_pos]
                        min_dist = bt_dists[j_bt]
                        # treat selected call as successful back-translation if it's tied (within tol) for best distance
                        bt_ok = abs(sel_dist - min_dist) < 1e-9
                    bt_ok = bt_ok or (bt_idx == selected_idx)

                if bt_idx is not None:
                    bt_call_name = _strip_parenthetical(
                        calls_all[bt_idx].get("call_name", "")
                    )
                    icon = "✅" if bt_ok else "❌"
                    back_line = f"Back-translation: {icon} {bt_call_name} (similarity: {bt_sim:.3f})"
                else:
                    back_line = "Back-translation: (none)"

                cards.append(
                    html.Div(
                        className="translation-card",
                        id={
                            "type": "trans-card",
                            "space": label_prefix.lower(),
                            "species": species_all[nn_idx],
                            "idx": int(nn_idx),
                        },
                        n_clicks=0,
                        **{"data-species": species_all[nn_idx]},
                        children=[
                            html.Div(
                                className="translation-meta",
                                children=[
                                    html.Div(
                                        f"{label_prefix} similarity: {sim:.3f}",
                                        className="subtle translation-metric",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                (
                                                    "✅ "
                                                    if nn_idx
                                                    == secondary_best_map.get(
                                                        species_all[nn_idx]
                                                    )
                                                    else "❌ "
                                                ),
                                                style={
                                                    "color": (
                                                        "#10b981"
                                                        if nn_idx
                                                        == secondary_best_map.get(
                                                            species_all[nn_idx]
                                                        )
                                                        else "#ef4444"
                                                    ),
                                                    "fontWeight": "700",
                                                },
                                            ),
                                            f"{secondary_label} similarity: {other_sim:.3f}",
                                        ],
                                        className="subtle translation-metric",
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "✅ " if "✅" in back_line else "❌ ",
                                                style={
                                                    "color": (
                                                        "#10b981"
                                                        if "✅" in back_line
                                                        else "#ef4444"
                                                    ),
                                                    "fontWeight": "700",
                                                },
                                            ),
                                            back_line.replace("✅ ", "").replace(
                                                "❌ ", ""
                                            ),
                                        ],
                                        className="subtle translation-metric",
                                    ),
                                ],
                            ),
                            _render_call_card(calls_all[nn_idx]),
                        ],
                    )
                )
            return cards

        semantic_cards = nearest_panel(
            semantic_embeds, acoustic_embeds, "Semantic", "Acoustic", best_ac
        )
        acoustic_cards = nearest_panel(
            acoustic_embeds, semantic_embeds, "Acoustic", "Semantic", best_sem
        )

        return selected_card, semantic_cards, acoustic_cards

    @app.callback(
        Output("pair-graph", "figure"),
        Input("pair-space", "value"),
        Input("pair-species-1", "value"),
        Input("pair-species-2", "value"),
    )
    def _update_pair_graph(space, sp1, sp2):
        fig = _make_pair_plot(space or "semantic", sp1, sp2)
        return fig

    @app.callback(
        Output("scatter", "figure"),
        Input("species-store", "data"),
        Input("selected-idx-store", "data"),
    )
    def _update_scatter_from_store(species_sel, selected_idx):
        species_sel = species_sel or []
        filtered = filter_calls(
            calls_all,
            species_allowlist=species_sel if species_sel else None,
        )

        clicked_idx = None
        if selected_idx is not None:
            try:
                clicked_idx = int(selected_idx)
            except Exception:
                clicked_idx = None

        return make_scatter(filtered, clicked_idx=clicked_idx)

    @app.callback(
        Output("selected-call", "children"),
        Output("translations-strip", "children"),
        Input("species-store", "data"),
        Input("selected-idx-store", "data"),
    )
    def _show_selected_and_translations(species_sel, selected_idx):
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

        if selected_idx is None:
            return (placeholder_selected, placeholder_trans)

        try:
            clicked_idx = int(selected_idx)
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

        # Fixed-height meta block (placeholder for Selected call)
        selected_meta = html.Div(
            className="translation-meta translation-meta--placeholder",
            children=[
                html.Div("", className="subtle translation-metric"),
                html.Div("", className="subtle translation-metric"),
            ],
        )

        selected_with_meta = html.Div(
            className="translation-card translation-card--selected",
            children=[selected_meta, selected_card],
        )

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
                bt_call_name = _strip_parenthetical(
                    calls_all[bt_idx].get("call_name", "")
                )
                icon = "✅" if int(bt_idx) == int(clicked_idx) else "❌"
                back_line = f"Back-translation: {icon} {bt_call_name} (similarity: {bt_sim:.3f})"
            else:
                back_line = "Back-translation: (none)"

            trans_cards.append(
                html.Div(
                    className="translation-card",
                    children=[
                        html.Div(
                            className="translation-meta",
                            children=[
                                html.Div(
                                    f"Similarlity: {cos_sim:.3f}",
                                    className="subtle translation-metric",
                                ),
                                html.Div(
                                    back_line,
                                    className="subtle translation-metric",
                                ),
                            ],
                        ),
                        _render_call_card(calls_all[nn_idx]),
                    ],
                )
            )

        translations_inner = html.Div(
            className="selection-strip__inner",
            children=trans_cards,
        )

        return (selected_with_meta, translations_inner)

    def _translations_for_idx(idx: int) -> Tuple[html.Div, html.Div]:
        """Return (base_call_card_with_meta, translations_strip_inner) for a given call index."""
        base_species = species_all[idx]
        base_vec = semantic_embeds[idx]

        # nearest SEMANTIC neighbor per other species (global)
        neighbors: List[Tuple[float, int]] = []
        best_ac: Dict[str, int] = {}
        for sp in species_unique:
            if sp == base_species:
                continue
            candidates = [i for i, s in enumerate(species_all) if s == sp]
            if not candidates:
                continue
            cand_vecs = semantic_embeds[candidates]
            dists = _np.linalg.norm(cand_vecs - base_vec, axis=1)
            j = int(dists.argmin())
            nn_idx = candidates[j]
            neighbors.append((float(dists[j]), nn_idx))

            # store acoustic nearest for this species (for secondary line check)
            d_ac = np.linalg.norm(
                acoustic_embeds[candidates] - acoustic_embeds[idx], axis=1
            )
            best_ac[sp] = candidates[int(d_ac.argmin())]

        neighbors.sort(key=lambda x: x[0])

        selected_card = _render_call_card(calls_all[idx])
        selected_meta = html.Div(
            className="translation-meta translation-meta--placeholder",
            children=[
                html.Div("", className="subtle translation-metric"),
                html.Div("", className="subtle translation-metric"),
            ],
        )
        selected_with_meta = html.Div(
            className="translation-card translation-card--selected",
            children=[selected_meta, selected_card],
        )

        base_candidates = [i for i, s in enumerate(species_all) if s == base_species]
        trans_cards = []
        for dist, nn_idx in neighbors:
            sim_sem = 1.0 - (dist * dist) / 2.0

            # Secondary (acoustic) similarity + check if also acoustic nearest for that species
            other_dist = float(
                np.linalg.norm(acoustic_embeds[nn_idx] - acoustic_embeds[idx])
            )
            other_sim = 1.0 - (other_dist * other_dist) / 2.0
            sp = species_all[nn_idx]
            is_acoustic_best = nn_idx == best_ac.get(sp)

            # Back-translation in semantic space
            bt_idx = None
            bt_sim = None
            bt_ok = False
            if base_candidates:
                bt_dists = _np.linalg.norm(
                    semantic_embeds[base_candidates] - semantic_embeds[nn_idx], axis=1
                )
                j_bt = int(bt_dists.argmin())
                bt_idx = base_candidates[j_bt]
                bt_dist = float(bt_dists[j_bt])
                bt_sim = 1.0 - (bt_dist * bt_dist) / 2.0
                if idx in base_candidates:
                    sel_pos = base_candidates.index(idx)
                    sel_dist = bt_dists[sel_pos]
                    min_dist = bt_dists[j_bt]
                    bt_ok = np.isclose(sel_dist, min_dist, atol=1e-6)
                bt_ok = bt_ok or (bt_idx == idx)
                if bt_idx is not None:
                    sel_name = _strip_parenthetical(
                        calls_all[idx].get("call_name", "")
                    ).lower()
                    bt_name = _strip_parenthetical(
                        calls_all[bt_idx].get("call_name", "")
                    ).lower()
                    if sel_name and bt_name and sel_name == bt_name:
                        bt_ok = True

            if bt_idx is not None:
                bt_call_name = _strip_parenthetical(
                    calls_all[bt_idx].get("call_name", "")
                )
                back_line = f"Back-translation: {'✅' if bt_ok else '❌'} {bt_call_name} (similarity: {bt_sim:.3f})"
            else:
                back_line = "Back-translation: (none)"

            trans_cards.append(
                html.Div(
                    className="translation-card",
                    children=[
                        html.Div(
                            className="translation-meta",
                            children=[
                                html.Div(
                                    f"Semantic similarity: {sim_sem:.3f}",
                                    className="subtle translation-metric",
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            "✅ " if is_acoustic_best else "❌ ",
                                            style={
                                                "color": (
                                                    "#10b981"
                                                    if is_acoustic_best
                                                    else "#ef4444"
                                                ),
                                                "fontWeight": "700",
                                            },
                                        ),
                                        f"Acoustic similarity: {other_sim:.3f}",
                                    ],
                                    className="subtle translation-metric",
                                ),
                                html.Div(
                                    [
                                        html.Span(
                                            (
                                                "✅ "
                                                if back_line.startswith(
                                                    "Back-translation: ✅"
                                                )
                                                else "❌ "
                                            ),
                                            style={
                                                "color": (
                                                    "#10b981"
                                                    if back_line.startswith(
                                                        "Back-translation: ✅"
                                                    )
                                                    else "#ef4444"
                                                ),
                                                "fontWeight": "700",
                                            },
                                        ),
                                        back_line.replace(
                                            "Back-translation: ✅ ",
                                            "Back-translation: ",
                                        ).replace(
                                            "Back-translation: ❌ ",
                                            "Back-translation: ",
                                        ),
                                    ],
                                    className="subtle translation-metric",
                                ),
                            ],
                        ),
                        _render_call_card(calls_all[nn_idx]),
                    ],
                )
            )

        translations_inner = html.Div(
            className="selection-strip__inner",
            children=trans_cards,
        )
        return selected_with_meta, translations_inner

    # --- Static plots helpers ---
    def _idxs_for_species(species_sel: List[str] | None) -> List[int]:
        if not species_sel:
            return list(range(len(calls_all)))
        allow = set(species_sel)
        return [i for i, sp in enumerate(species_all) if sp in allow]

    def make_calls_bar(species_sel: List[str] | None):
        idxs = _idxs_for_species(species_sel)
        if not idxs:
            return go.Figure().update_layout(
                title="Number of calls per species (none)",
                template="plotly_white",
                height=480,
            )
        counts = Counter(species_all[i] for i in idxs)
        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        species_icons = [species_icon_map.get(sp, "") for sp, _ in items]
        species_labels = [
            f"{species_common_name(sp)} {icon}".strip()
            for (sp, _), icon in zip(items, species_icons)
        ]
        values = [v for _, v in items]
        # assign fixed group colors with fallback
        group_colors: Dict[str, str] = {}
        fallback_idx = 0
        for sp in species_unique:
            g = _group_label(first_by_species.get(sp, {}))
            if g not in group_colors:
                group_colors[g] = GROUP_COLOR_PRESET.get(
                    g, FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
                )
                fallback_idx += 1

        colors = []
        for sp, _ in items:
            g = _group_label(first_by_species.get(sp, {}))
            colors.append(
                group_colors.get(g, FALLBACK_COLORS[len(colors) % len(FALLBACK_COLORS)])
            )

        fig = go.Figure(
            go.Bar(
                x=values,
                y=species_labels,
                orientation="h",
                marker=dict(color=colors),
                text=[f"{v}" for v in values],
                textposition="outside",
                insidetextanchor="middle",
            )
        )
        fig.update_layout(
            title="Number of calls per species",
            height=380,
            margin=dict(l=170, r=10, t=50, b=30),
            template="plotly_white",
        )
        return fig

    # --- Pair-of-species helper ---
    def _pair_rows(space: str, sp1: str, sp2: str):
        if not sp1 or not sp2 or sp1 == sp2:
            return []

        idxs1 = [i for i, sp in enumerate(species_all) if sp == sp1]
        idxs2 = [i for i, sp in enumerate(species_all) if sp == sp2]
        if not idxs1 or not idxs2:
            return []

        emb = acoustic_embeds if space == "acoustic" else semantic_embeds
        v1 = emb[idxs1]
        v2 = emb[idxs2]

        # normalize to be safe
        v1n = v1 / (np.linalg.norm(v1, axis=1, keepdims=True) + 1e-8)
        v2n = v2 / (np.linalg.norm(v2, axis=1, keepdims=True) + 1e-8)
        sims = np.dot(v1n, v2n.T)  # shape (m,n)

        best12 = sims.argmax(axis=1)
        best21 = sims.argmax(axis=0)

        used1 = set()
        used2 = set()
        rows: List[Tuple[int | None, int | None, float, float, bool]] = []

        # mutual best first
        mutual = []
        for i, j in enumerate(best12):
            if best21[j] == i:
                mutual.append((i, j, sims[i, j]))
        mutual.sort(key=lambda x: x[2], reverse=True)
        for i, j, sim in mutual:
            used1.add(i)
            used2.add(j)
            rows.append((i, j, sim, sim, True))

        # remaining: pair each unused i with its best j not yet used if possible
        remaining = []
        for i in range(len(idxs1)):
            if i in used1:
                continue
            j = best12[i]
            sim_ij = sims[i, j]
            sim_ji = sims[best21[j], j] if best21[j] < len(idxs1) else 0.0
            remaining.append((i, j, sim_ij, sim_ji))
        remaining.sort(key=lambda x: x[2], reverse=True)

        for i, j, sim_ij, sim_ji in remaining:
            if j in used2:
                rows.append((i, None, sim_ij, sim_ji, False))
                used1.add(i)
                continue
            rows.append((i, j, sim_ij, sim_ji, False))
            used1.add(i)
            used2.add(j)

        # leftover only-in-species2
        for j in range(len(idxs2)):
            if j in used2:
                continue
            rows.append((None, j, 0.0, 0.0, False))

        # to reduce crossings, keep current order
        return rows, idxs1, idxs2, sims

    def _make_pair_plot(space: str, sp1: str, sp2: str) -> go.Figure:
        rows, idxs1, idxs2, sims = _pair_rows(space, sp1, sp2)
        if not rows:
            return go.Figure().update_layout(
                title="Select two different species",
                margin=dict(l=20, r=20, t=40, b=20),
                height=220,
            )

        row_h = 2.0
        shapes = []
        annotations = []

        left_center_x = 0
        right_center_x = 8
        box_w = 1.8
        box_h = 1.6

        def _wrap_two_lines(text: str, width: int = 18) -> str:
            """Wrap text to at most two lines using <br>, trimming if needed."""
            words = str(text).split()
            if not words:
                return ""
            lines = []
            line = words[0]
            for w in words[1:]:
                if len(line) + 1 + len(w) <= width:
                    line += " " + w
                else:
                    lines.append(line)
                    line = w
                    if len(lines) == 1:  # already have first line
                        break
            lines.append(line)
            if len(lines) > 2:
                lines = lines[:2]
            # If there were leftover words after second line, add ellipsis
            remaining = len(words) - len(" ".join(lines).split())
            if remaining > 0:
                lines[-1] = lines[-1] + "…"
            return "<br>".join(lines)

        left_row_map: Dict[int, float] = {}
        right_row_map: Dict[int, float] = {}

        def add_box(is_left: bool, idx_local: int, row_idx: int):
            cx = left_center_x if is_left else right_center_x
            cy = -row_idx * row_h
            x0 = cx - box_w / 2
            x1 = cx + box_w / 2
            y0 = cy - box_h / 2
            y1 = cy + box_h / 2

            line_width = 1
            line_dash = "solid"
            line_color = "rgba(0,0,0,0.2)"
            shapes.append(
                dict(
                    type="rect",
                    x0=x0,
                    x1=x1,
                    y0=y0,
                    y1=y1,
                    line=dict(color=line_color, width=line_width, dash=line_dash),
                    fillcolor="white",
                )
            )
            if is_left:
                left_row_map[idx_local] = cy
                c = calls_all[idxs1[idx_local]]
                txt = _wrap_two_lines(_strip_parenthetical(c.get("call_name", "")))
            else:
                right_row_map[idx_local] = cy
                c = calls_all[idxs2[idx_local]]
                txt = _wrap_two_lines(_strip_parenthetical(c.get("call_name", "")))
            annotations.append(
                dict(
                    x=cx,
                    y=cy,
                    text=txt,
                    showarrow=False,
                    font=dict(size=12),
                )
            )

        # assign rows
        for r_idx, (i_local, j_local, _, _, _) in enumerate(rows):
            if i_local is not None:
                add_box(True, i_local, r_idx)
            if j_local is not None:
                add_box(False, j_local, r_idx)

        # arrows left -> right (red) for every call in species 1
        for i_local in range(len(idxs1)):
            j_local = sims[i_local].argmax()
            sim_ij = sims[i_local, j_local]
            y0 = left_row_map.get(i_local)
            y1 = right_row_map.get(j_local)
            if y0 is None or y1 is None:
                continue
            y0 += 0.35
            y1 += 0.35
            tail_x = left_center_x + box_w / 2
            head_x = right_center_x - box_w / 2
            annotations.append(
                dict(
                    x=head_x,
                    y=y1,
                    ax=tail_x,
                    ay=y0,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=4,
                    arrowsize=1.6,
                    arrowwidth=1.4,
                    arrowcolor="rgba(220,38,38,0.7)",
                    text=f"{sim_ij:.3f}",
                    font=dict(size=10, color="rgba(220,38,38,0.9)"),
                )
            )

        # arrows right -> left (blue)
        for j_local in range(len(idxs2)):
            i_local = sims[:, j_local].argmax()
            sim = sims[i_local, j_local]
            if i_local not in left_row_map or j_local not in right_row_map:
                continue
            y0 = right_row_map[j_local] - 0.35
            y1 = left_row_map[i_local] - 0.35
            tail_x = right_center_x - box_w / 2
            head_x = left_center_x + box_w / 2
            annotations.append(
                dict(
                    x=head_x,
                    y=y1,
                    ax=tail_x,
                    ay=y0,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=4,
                    arrowsize=1.6,
                    arrowwidth=1.4,
                    arrowcolor="rgba(37,99,235,0.7)",
                    text=f"{sim:.3f}",
                    font=dict(size=10, color="rgba(37,99,235,0.9)"),
                )
            )

        max_rows = len(rows)
        fig = go.Figure()

        # clickable invisible scatter points at box centers
        if left_row_map:
            fig.add_trace(
                go.Scatter(
                    x=[left_center_x] * len(left_row_map),
                    y=[left_row_map[i] for i in range(len(left_row_map))],
                    mode="markers",
                    marker=dict(size=30, color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    customdata=[idxs1[i] for i in range(len(left_row_map))],
                    name="left-boxes",
                )
            )
        if right_row_map:
            fig.add_trace(
                go.Scatter(
                    x=[right_center_x] * len(right_row_map),
                    y=[right_row_map[j] for j in range(len(right_row_map))],
                    mode="markers",
                    marker=dict(size=30, color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    customdata=[idxs2[j] for j in range(len(right_row_map))],
                    name="right-boxes",
                )
            )
        fig.update_layout(
            title=f"{space.title()} nearest neighbors<br>{species_common_name(sp1)} vs {species_common_name(sp2)}",
            title_x=0.5,
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(visible=False, range=[-2.5, 10]),
            yaxis=dict(visible=False, range=[-max_rows * row_h - 0.5, row_h]),
            shapes=shapes,
            annotations=annotations,
            height=max(240, int(max_rows * 60 + 80)),
        )
        return fig

    def make_keyword_bar(species_sel: List[str] | None, top_n: int = 20):
        idxs = _idxs_for_species(species_sel)

        # Collect keyword counts per group
        kw_counts: Dict[str, Counter] = {}
        for i in idxs:
            c = calls_all[i]
            group = _group_label(c)
            kw_counts.setdefault(group, Counter()).update(
                c.get("ontology_keywords") or []
            )

        # Top keywords overall
        total = Counter()
        for c in kw_counts.values():
            total.update(c)
        top = total.most_common(top_n)
        if not top:
            return go.Figure().update_layout(
                title="Keyword frequencies (none)",
                template="plotly_white",
                height=330,
            )

        keywords = [k for k, _ in top]

        fig = go.Figure()
        groups = sorted(kw_counts.keys())
        group_icons: Dict[str, str] = {}
        # stable group -> color map using preset with fallback
        group_colors: Dict[str, str] = {}
        fallback_idx = 0
        for sp in species_unique:
            g = _group_label(first_by_species.get(sp, {}))
            if g not in group_colors:
                group_colors[g] = GROUP_COLOR_PRESET.get(
                    g, FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]
                )
                if g not in GROUP_COLOR_PRESET:
                    fallback_idx += 1

        for group in groups:
            counts = [kw_counts.get(group, Counter()).get(k, 0) for k in keywords]
            if all(v == 0 for v in counts):
                continue
            color = group_colors.get(group, FALLBACK_COLORS[0])
            # pick an icon from any species in this group if available
            icon = ""
            for i in idxs:
                if _group_label(calls_all[i]) == group:
                    icon = species_icon_map.get(species_all[i], "")
                    break
            group_icons[group] = icon
            fig.add_trace(
                go.Bar(
                    x=counts,
                    y=keywords,
                    orientation="h",
                    name=f"{icon} {group}".strip(),
                    marker=dict(color=color),
                    customdata=[group] * len(counts),
                )
            )

        # add total labels aligned to the bar end (left edge of text at bar end)
        totals_per_kw = [sum(kw_counts[g].get(k, 0) for g in groups) for k in keywords]
        total_ann = []
        for y, t in zip(keywords, totals_per_kw):
            if t <= 0:
                continue
            total_ann.append(
                dict(
                    x=t,
                    y=y,
                    xanchor="left",
                    yanchor="middle",
                    text=str(t),
                    showarrow=False,
                    font=dict(color="black"),
                )
            )

        fig.update_layout(
            barmode="stack",
            title="Keyword frequencies",
            height=360,
            margin=dict(l=220, r=10, t=50, b=30),
            template="plotly_white",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            annotations=total_ann,
        )
        return fig

    def _mantel_stats(
        emb_a: np.ndarray,
        emb_s: np.ndarray,
        idxs: List[int],
        cross_only: bool = False,
        n_perm: int = 199,
    ):
        """Compute Mantel-like Pearson r between acoustic and semantic similarity matrices.
        If cross_only=True, ignore within-species pairs."""
        if len(idxs) < 3:
            return None
        # normalize embeddings
        ea = emb_a[idxs]
        es = emb_s[idxs]
        ea = ea / (np.linalg.norm(ea, axis=1, keepdims=True) + 1e-9)
        es = es / (np.linalg.norm(es, axis=1, keepdims=True) + 1e-9)
        Sa = ea @ ea.T
        Ss = es @ es.T

        # upper triangle indices
        tri = np.triu_indices(len(idxs), k=1)
        mask = np.ones_like(tri[0], dtype=bool)
        if cross_only:
            species_idxs = [species_all[i] for i in idxs]
            mask = np.array(
                [species_idxs[i] != species_idxs[j] for i, j in zip(tri[0], tri[1])]
            )
            if not mask.any():
                return None

        a_vec = Sa[tri][mask]
        s_vec = Ss[tri][mask]
        if a_vec.size < 3:
            return None

        r_obs = float(np.corrcoef(a_vec, s_vec)[0, 1])

        # permutation test (shuffle semantic labels)
        rs = []
        for _ in range(n_perm):
            perm = np.random.permutation(len(idxs))
            Ss_perm = Ss[perm][:, perm]
            s_vec_perm = Ss_perm[tri][mask]
            rs.append(np.corrcoef(a_vec, s_vec_perm)[0, 1])
        rs = np.array(rs)
        p = (np.sum(np.abs(rs) >= abs(r_obs)) + 1) / (n_perm + 1)
        return float(r_obs), float(p)

    def _mantel_partial(
        emb_a: np.ndarray,
        emb_s: np.ndarray,
        idxs: List[int],
        cross_only: bool = False,
        n_perm: int = 999,
    ):
        """Partial Mantel controlling for species identity matrix C."""
        if len(idxs) < 3:
            return None

        ea = emb_a[idxs]
        es = emb_s[idxs]
        ea = ea / (np.linalg.norm(ea, axis=1, keepdims=True) + 1e-9)
        es = es / (np.linalg.norm(es, axis=1, keepdims=True) + 1e-9)
        Sa = ea @ ea.T
        Ss = es @ es.T

        species_idxs = [species_all[i] for i in idxs]
        C = np.equal.outer(species_idxs, species_idxs).astype(float)

        tri = np.triu_indices(len(idxs), k=1)
        mask = np.ones_like(tri[0], dtype=bool)
        if cross_only:
            mask = np.array(
                [species_idxs[i] != species_idxs[j] for i, j in zip(tri[0], tri[1])]
            )
            if not mask.any():
                return None

        a_vec = Sa[tri][mask]
        s_vec = Ss[tri][mask]
        c_vec = C[tri][mask]
        if a_vec.size < 3:
            return None

        r_ab = np.corrcoef(a_vec, s_vec)[0, 1]
        r_ac = np.corrcoef(a_vec, c_vec)[0, 1]
        r_bc = np.corrcoef(s_vec, c_vec)[0, 1]
        denom = np.sqrt((1 - r_ac**2) * (1 - r_bc**2)) + 1e-12
        r_partial = (r_ab - r_ac * r_bc) / denom

        rs = []
        for _ in range(n_perm):
            perm = np.random.permutation(len(s_vec))
            s_perm = s_vec[perm]
            r_ab_p = np.corrcoef(a_vec, s_perm)[0, 1]
            r_bc_p = np.corrcoef(s_perm, c_vec)[0, 1]
            r_partial_p = (r_ab_p - r_ac * r_bc_p) / denom
            rs.append(r_partial_p)
        rs = np.array(rs)
        p = (np.sum(np.abs(rs) >= abs(r_partial)) + 1) / (n_perm + 1)
        return float(r_partial), float(p)

    def make_similarity_heatmap(
        embeds: np.ndarray,
        species_sel: List[str] | None,
        title: str,
    ):
        idxs = _idxs_for_species(species_sel)
        if not idxs:
            return go.Figure().update_layout(
                title=f"{title} (none)", template="plotly_white", height=400
            )

        # Simple ordering: species name then call name
        def _key(idx):
            return (
                species_common_name(species_all[idx]),
                calls_all[idx].get("call_name", ""),
            )

        # group by species (already ordered by group then species)
        blocks: List[List[int]] = []
        for sp in sorted({species_all[i] for i in idxs}, key=species_common_name):
            block = [i for i in sorted(idxs, key=_key) if species_all[i] == sp]
            if block:
                blocks.append(block)

        ordered: List[int] = []
        for b in blocks:
            ordered.extend(b)

        mat = embeds[ordered]
        mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = np.dot(mat, mat.T)

        # build labels per call (before separators)
        base_labels = [
            f"{species_icon_map.get(species_all[i], '')} {species_common_name(species_all[i])}: {_strip_parenthetical(calls_all[i].get('call_name',''))}"
            for i in ordered
        ]

        # no separators; use base labels and sims directly
        labels = base_labels

        fig = go.Figure(
            data=go.Heatmap(
                z=sims,
                x=labels,
                y=labels,
                colorscale=[
                    [0.0, "#ffff00"],  # yellow at -1
                    [0.25, "#ffffff"],  # white from -0.5 upward
                    [0.75, "#ffffff"],  # white until +0.5
                    [1.0, "#c81e1e"],  # red at +1
                ],
                zmin=-1,
                zmax=1,
                colorbar=dict(title="sim"),
                hovertemplate="<b>%{y}</b><br>%{x}<br>sim=%{z:.3f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=title,
            height=420,
            margin=dict(l=80, r=20, t=40, b=120),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            template="plotly_white",
        )
        return fig

    def make_umap_fig(
        coords: np.ndarray,
        species_sel: List[str] | None,
        title: str,
        showlegend: bool = True,
        selected_idx: int | None = None,
    ):
        idxs = _idxs_for_species(species_sel)
        if not idxs:
            return go.Figure().update_layout(title=f"{title} (none)")
        if coords.shape[1] == 3:
            fig = go.Figure()
            for sp in species_unique:
                sp_idxs = [i for i in idxs if species_all[i] == sp]
                if not sp_idxs:
                    continue
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[sp_idxs, 0],
                        y=coords[sp_idxs, 1],
                        z=coords[sp_idxs, 2],
                        mode="markers",
                        marker=dict(
                            color=color_map.get(sp),
                            size=[
                                (
                                    8
                                    if (selected_idx is not None and i == selected_idx)
                                    else 4
                                )
                                for i in sp_idxs
                            ],
                            opacity=0.85,
                        ),
                        name=f"{species_icon_map.get(sp, '')} {species_common_name(sp)}",
                        hovertext=[call_names_all[i] for i in sp_idxs],
                        hoverinfo="text",
                        customdata=sp_idxs,
                        showlegend=showlegend,
                    )
                )
            fig.update_layout(
                title=title,
                height=340,
                margin=dict(l=0, r=0, t=40, b=0),
                scene=dict(
                    xaxis_title="",
                    yaxis_title="",
                    zaxis_title="",
                    xaxis=dict(
                        showticklabels=False,
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.55)",
                        zeroline=False,
                    ),
                    yaxis=dict(
                        showticklabels=False,
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.55)",
                        zeroline=False,
                    ),
                    zaxis=dict(
                        showticklabels=False,
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.55)",
                        zeroline=False,
                    ),
                ),
                showlegend=showlegend,
            )
            return fig

        fig = go.Figure()
        for sp in species_unique:
            sp_idxs = [i for i in idxs if species_all[i] == sp]
            if not sp_idxs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=coords[sp_idxs, 0],
                    y=coords[sp_idxs, 1],
                    mode="markers",
                    marker=dict(
                        color=color_map.get(sp),
                        size=[
                            (
                                11
                                if (selected_idx is not None and i == selected_idx)
                                else 7
                            )
                            for i in sp_idxs
                        ],
                        opacity=0.85,
                    ),
                    name=f"{species_icon_map.get(sp, '')} {species_common_name(sp)}",
                    hovertext=[call_names_all[i] for i in sp_idxs],
                    hoverinfo="text",
                    customdata=sp_idxs,
                    showlegend=True,
                )
            )
        fig.update_layout(
            title=title,
            height=340,
            margin=dict(l=10, r=10, t=40, b=30),
            xaxis_title="",
            yaxis_title="",
            xaxis=dict(
                showticklabels=False,
                showgrid=True,
                gridcolor="rgba(255,255,255,0.55)",
                zeroline=False,
                scaleanchor="y",
                scaleratio=1,
                constrain="domain",
            ),
            yaxis=dict(
                showticklabels=False,
                showgrid=True,
                gridcolor="rgba(255,255,255,0.55)",
                zeroline=False,
                scaleanchor="x",
                scaleratio=1,
                constrain="domain",
            ),
            showlegend=showlegend,
        )
        return fig

    @app.callback(
        Output("one-species-list", "children"),
        Input("one-species-dropdown", "value"),
        Input("one-species-mode", "value"),
    )
    def _render_one_species(selected_species, modes):
        modes = modes or []
        if not selected_species:
            return html.Div(
                "Pick a species to see its calls and translations.",
                className="subtle",
            )

        idxs = [i for i, sp in enumerate(species_all) if sp == selected_species]
        if not idxs:
            return html.Div("No calls for that species.", className="subtle")

        cards = []
        for idx in idxs:
            # Use all species as reference set so translations are populated by default.
            sel_card, sem_cards, ac_cards = _translations_for_idx_and_species(
                idx, species_unique
            )
            strips = []
            if "semantic" in modes and sem_cards is not None:
                strips.append(
                    html.Div(
                        className="selection-strip selection-strip--horizontal",
                        children=sem_cards,
                    )
                )
            if "acoustic" in modes and ac_cards is not None:
                strips.append(
                    html.Div(
                        className="selection-strip selection-strip--horizontal",
                        children=ac_cards,
                    )
                )

            cards.append(
                html.Div(
                    className="onespecies-call",
                    children=[
                        html.Div(
                            className="onespecies-call__base",
                            children=html.Div(
                                className="panel panel--selected",
                                children=sel_card or _render_call_card(calls_all[idx]),
                            ),
                        ),
                        html.Div(
                            className="onespecies-call__translations",
                            children=strips,
                        ),
                    ],
                )
            )

        return cards

    @app.callback(
        Output("one-species-explore-list", "children"),
        Input("one-species-explore-dropdown", "value"),
    )
    def _render_one_species_explore(selected_species):
        if not selected_species:
            return html.Div(
                "Pick a species to see its calls.",
                className="subtle",
            )

        idxs = [i for i, sp in enumerate(species_all) if sp == selected_species]
        if not idxs:
            return html.Div("No calls for that species.", className="subtle")

        cards = []
        for idx in idxs:
            cards.append(
                html.Div(
                    className="onespecies-call-card",
                    children=html.Div(
                        className="panel panel--selected",
                        children=_render_call_card(calls_all[idx]),
                    ),
                )
            )

        return html.Div(cards, className="one-species-explore__grid")

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
    run_dash_app()

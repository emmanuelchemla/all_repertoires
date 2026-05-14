"""Plots for the main paper.

Each function takes the prepared call table + embeddings + metadata and writes a
single file under plots/. Visual style is shared via CLASS_COLORS and
SEMANTIC_CATEGORY.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

CLASS_COLORS = {"Mammalia": "#4C72B0", "Aves": "#55A868", "Amphibia": "#C44E52"}
CLASS_ORDER = ["Amphibia", "Aves", "Mammalia"]

# Group semantic keywords into broader communicative categories.
SEMANTIC_CATEGORY = {
    "cohesion":  ["contact", "group_coordination", "affiliation"],
    "agonistic": ["threat", "aggression", "submission"],
    "danger":    ["alarm", "predator"],
    "care":      ["distress", "begging", "caregiving"],
    "reproduction": ["courtship", "mating"],
    "resources": ["food", "recruitment"],
    "territory": ["territorial", "spacing"],
    "identity":  ["identity", "attention"],
    "meta":      ["play", "display", "combinatorial"],
}
CATEGORY_COLORS = {
    "cohesion": "#1f77b4",
    "agonistic": "#d62728",
    "danger": "#e377c2",
    "care": "#9467bd",
    "reproduction": "#e7ba52",
    "resources": "#2ca02c",
    "territory": "#8c564b",
    "identity": "#7f7f7f",
    "meta": "#17becf",
}

# Acoustic feature groupings (controlled vocab from schema.json).
ACOUSTIC_GROUPS = {
    "frequency":   ["high_frequency", "low_frequency", "frequency_modulated"],
    "spectral":    ["tonal", "broadband", "noisy", "harmonic"],
    "temporal":    ["short", "long", "abrupt", "repetitive", "pulsed", "multi_component"],
    "amplitude":   ["loud", "quiet"],
    "variation":   ["graded"],
}
ACOUSTIC_GROUP_COLORS = {
    "frequency": "#1f77b4",
    "spectral": "#2ca02c",
    "temporal": "#ff7f0e",
    "amplitude": "#9467bd",
    "variation": "#8c564b",
}


def _category_of(kw: str) -> str:
    for cat, kws in SEMANTIC_CATEGORY.items():
        if kw in kws:
            return cat
    return "other"


def _acoustic_group(kw: str) -> str:
    for g, kws in ACOUSTIC_GROUPS.items():
        if kw in kws:
            return g
    return "other"


def _flat_semantic_vocab() -> list[str]:
    return [k for cat in SEMANTIC_CATEGORY for k in SEMANTIC_CATEGORY[cat]]


def _flat_acoustic_vocab() -> list[str]:
    return [k for g in ACOUSTIC_GROUPS for k in ACOUSTIC_GROUPS[g]]


# ------------------------------------------------------------------ #
# Fig 1: dataset overview
# ------------------------------------------------------------------ #

def fig1_dataset_overview(calls, out: Path | None = None) -> Path:
    out = out or (PLOTS_DIR / "fig1_dataset_overview.png")

    sem_kws = _flat_semantic_vocab()
    sem_counts = Counter()
    for c in calls:
        sem_counts.update(set(c.semantic_keywords))
    ac_kws = _flat_acoustic_vocab()
    ac_counts = Counter()
    for c in calls:
        ac_counts.update(set(c.acoustic_keywords))

    sp = defaultdict(int)
    sp_meta = {}
    for c in calls:
        sp[c.species] += 1
        sp_meta[c.species] = (c.class_, c.family, c.common_name)
    sp_sorted = sorted(sp.keys(), key=lambda s: (CLASS_ORDER.index(sp_meta[s][0]), sp_meta[s][1], -sp[s]))

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], hspace=0.45, wspace=0.25)

    # (A) semantic keyword freq, ordered by category, colored by category
    ax = fig.add_subplot(gs[0, 0])
    sem_sorted = sorted(
        [(k, sem_counts[k]) for k in sem_kws if sem_counts[k] > 0],
        key=lambda t: (-t[1],),
    )
    colors = [CATEGORY_COLORS[_category_of(k)] for k, _ in sem_sorted]
    ax.barh([k for k, _ in sem_sorted], [v for _, v in sem_sorted], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Number of calls")
    ax.set_title("A.  Semantic keyword frequency", loc="left", fontweight="bold")
    # legend
    seen = set(); handles = []
    for k, _ in sem_sorted:
        cat = _category_of(k)
        if cat not in seen:
            handles.append(mpatches.Patch(color=CATEGORY_COLORS[cat], label=cat))
            seen.add(cat)
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)

    # (B) acoustic keyword freq, grouped
    ax = fig.add_subplot(gs[0, 1])
    ac_sorted = sorted(
        [(k, ac_counts[k]) for k in ac_kws if ac_counts[k] > 0],
        key=lambda t: -t[1],
    )
    colors = [ACOUSTIC_GROUP_COLORS[_acoustic_group(k)] for k, _ in ac_sorted]
    ax.barh([k for k, _ in ac_sorted], [v for _, v in ac_sorted], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Number of calls")
    ax.set_title("B.  Acoustic keyword frequency", loc="left", fontweight="bold")
    seen = set(); handles = []
    for k, _ in ac_sorted:
        g = _acoustic_group(k)
        if g not in seen:
            handles.append(mpatches.Patch(color=ACOUSTIC_GROUP_COLORS[g], label=g))
            seen.add(g)
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)

    # (C) calls per species, grouped by class & family
    ax = fig.add_subplot(gs[1, :])
    families_seen: dict[str, str] = {}     # family -> color cycle
    palette = plt.get_cmap("tab20").colors
    family_color: dict[str, tuple] = {}
    fam_index = 0
    for s in sp_sorted:
        fam = sp_meta[s][1]
        if fam not in family_color:
            family_color[fam] = palette[fam_index % len(palette)]
            fam_index += 1
    x = np.arange(len(sp_sorted))
    heights = [sp[s] for s in sp_sorted]
    colors = [family_color[sp_meta[s][1]] for s in sp_sorted]
    ax.bar(x, heights, color=colors,
           edgecolor=[CLASS_COLORS[sp_meta[s][0]] for s in sp_sorted], linewidth=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels([sp_meta[s][2] for s in sp_sorted], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Calls per species")
    ax.set_title("C.  Repertoire size per species (edge color = class, fill = family)",
                 loc="left", fontweight="bold")
    # class legend
    class_handles = [mpatches.Patch(facecolor="white", edgecolor=CLASS_COLORS[c],
                                    linewidth=2, label=c) for c in CLASS_ORDER]
    ax.legend(handles=class_handles, loc="upper right", fontsize=9, frameon=False)

    fig.suptitle(f"AnimalLex — {len(set(c.species for c in calls))} species, {len(calls)} calls",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ #
# Fig 2A/B: UMAP embeddings
# ------------------------------------------------------------------ #

def _umap(emb: np.ndarray, seed: int = 0) -> np.ndarray:
    import umap
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.25, metric="cosine",
                        random_state=seed)
    return reducer.fit_transform(emb)


def fig2_umaps(calls, sem_emb: np.ndarray, ac_emb: np.ndarray) -> tuple[Path, Path]:
    cats = [_category_of(c.semantic_keywords[0]) if c.semantic_keywords else "other"
            for c in calls]
    sem_xy = _umap(sem_emb, seed=0)
    ac_xy = _umap(ac_emb, seed=0)

    paths = []
    for name, xy in [("fig2A_semantic_umap.pdf", sem_xy),
                     ("fig2B_acoustic_umap.pdf", ac_xy)]:
        fig, ax = plt.subplots(figsize=(5.0, 5.0))
        for cat, col in CATEGORY_COLORS.items():
            mask = np.array([c == cat for c in cats])
            if mask.any():
                ax.scatter(xy[mask, 0], xy[mask, 1], s=18, alpha=0.85,
                           c=col, label=cat, linewidths=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        title = "Semantic UMAP" if "semantic" in name else "Acoustic UMAP"
        ax.set_title(title)
        ax.legend(fontsize=7, frameon=False, loc="best", ncol=2)
        p = PLOTS_DIR / name
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return tuple(paths)


# ------------------------------------------------------------------ #
# Fig 2C: Mantel scatter
# ------------------------------------------------------------------ #

def fig2c_mantel(dist_ac: np.ndarray, dist_sem: np.ndarray, species: list[str],
                 families: list[str], r_values: dict[str, float],
                 max_per_group: int = 1000, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = dist_ac.shape[0]
    iu = np.triu_indices(n, k=1)
    sp = np.array(species); fam = np.array(families)
    same_sp = sp[iu[0]] == sp[iu[1]]
    same_fam = (fam[iu[0]] == fam[iu[1]]) & ~same_sp
    cross_fam = ~(same_sp | same_fam)

    groups = [
        ("within species",     same_sp,  "#4C72B0"),
        ("same family",        same_fam, "#DD8452"),
        ("cross family",       cross_fam, "#8C8C8C"),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    x_all = dist_ac[iu]
    y_all = dist_sem[iu]
    for label, mask, color in groups:
        idx = np.where(mask)[0]
        if len(idx) > max_per_group:
            idx = rng.choice(idx, size=max_per_group, replace=False)
        r = r_values.get(label, float("nan"))
        ax.scatter(x_all[idx], y_all[idx], s=6, alpha=0.45, c=color,
                   label=f"{label} (r={r:.2f})", linewidths=0)
    ax.set_xlabel("Acoustic distance")
    ax.set_ylabel("Semantic distance")
    ax.set_title("Mantel: form ↔ meaning")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    p = PLOTS_DIR / "fig2C_mantel.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ #
# Supp Mantel matrix
# ------------------------------------------------------------------ #

def fig_supp_mantel_matrix(species_pair_r: dict[tuple[str, str], float],
                           species_meta: dict[str, tuple[str, str]]) -> Path:
    """species_meta: sp -> (class_, family)."""
    # order species: by class, hierarchical cluster within class
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    classes = sorted(set(m[0] for m in species_meta.values()),
                     key=lambda c: CLASS_ORDER.index(c) if c in CLASS_ORDER else 99)
    ordered: list[str] = []
    for cls in classes:
        sp_in = sorted([s for s, m in species_meta.items() if m[0] == cls])
        if len(sp_in) <= 2:
            ordered.extend(sp_in)
            continue
        idx = {s: i for i, s in enumerate(sp_in)}
        D = np.zeros((len(sp_in), len(sp_in)))
        for (a, b), r in species_pair_r.items():
            if a in idx and b in idx:
                D[idx[a], idx[b]] = D[idx[b], idx[a]] = 1.0 - r
        np.fill_diagonal(D, 0.0)
        D = np.clip(D, 0.0, None)
        try:
            Z = linkage(squareform(D, checks=False), method="average")
            ordered.extend([sp_in[i] for i in leaves_list(Z)])
        except Exception:
            ordered.extend(sp_in)

    n = len(ordered)
    M = np.full((n, n), np.nan)
    for i, a in enumerate(ordered):
        for j, b in enumerate(ordered):
            if i == j:
                M[i, j] = 1.0
            elif (a, b) in species_pair_r:
                M[i, j] = species_pair_r[(a, b)]
            elif (b, a) in species_pair_r:
                M[i, j] = species_pair_r[(b, a)]

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ordered, rotation=90, fontsize=6)
    ax.set_yticklabels(ordered, fontsize=6)
    # class bars
    bar_pos = []
    cur = ordered[0]; start = 0
    for i, s in enumerate(ordered):
        cls = species_meta[s][0]
        if cls != species_meta[cur][0]:
            bar_pos.append((start, i - 1, species_meta[cur][0]))
            start = i; cur = s
    bar_pos.append((start, n - 1, species_meta[cur][0]))
    for s, e, cls in bar_pos:
        ax.add_patch(mpatches.Rectangle((s - 0.5, -1.8), e - s + 1, 1.2,
                                        facecolor=CLASS_COLORS.get(cls, "gray"),
                                        clip_on=False))
        ax.add_patch(mpatches.Rectangle((-1.8, s - 0.5), 1.2, e - s + 1,
                                        facecolor=CLASS_COLORS.get(cls, "gray"),
                                        clip_on=False))
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson r (acoustic vs semantic)")
    ax.set_title("Species-pairwise acoustic–semantic correlation")
    p = PLOTS_DIR / "fig_supp_mantel_matrix.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ #
# PMI heatmap
# ------------------------------------------------------------------ #

def fig_pmi_heatmap(pmi: np.ndarray, ac_vocab: list[str], sem_vocab: list[str]) -> Path:
    # Group semantic columns by category (panels)
    cat_for = {k: _category_of(k) for k in sem_vocab}
    grp_for = {k: _acoustic_group(k) for k in ac_vocab}
    cat_order = list(SEMANTIC_CATEGORY.keys())
    sem_order = [k for cat in cat_order for k in SEMANTIC_CATEGORY[cat] if k in sem_vocab]
    ac_order = [k for g in ACOUSTIC_GROUPS for k in ACOUSTIC_GROUPS[g] if k in ac_vocab]

    # re-index pmi to ordered rows/cols
    a_idx = {k: i for i, k in enumerate(ac_vocab)}
    s_idx = {k: i for i, k in enumerate(sem_vocab)}
    M = np.array([[pmi[a_idx[a], s_idx[s]] for s in sem_order] for a in ac_order])

    fig, ax = plt.subplots(figsize=(14, 7))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(sem_order)))
    ax.set_xticklabels(sem_order, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(ac_order)))
    ax.set_yticklabels(ac_order, fontsize=9)

    # annotate top-2 and bottom-1 within each semantic-category panel
    for cat in cat_order:
        cols = [i for i, s in enumerate(sem_order) if cat_for[s] == cat]
        if not cols:
            continue
        sub = M[:, cols]
        flat = [(r, c, sub[r, c]) for r in range(sub.shape[0]) for c in range(sub.shape[1])
                if not np.isnan(sub[r, c])]
        flat.sort(key=lambda t: -t[2])
        for r, c, v in flat[:2]:
            ax.text(cols[c], r, f"{v:.1f}", ha="center", va="center",
                    fontsize=7, color="black")
        for r, c, v in flat[-1:]:
            ax.text(cols[c], r, f"{v:.1f}", ha="center", va="center",
                    fontsize=7, color="black")
        # category divider
        ax.axvline(cols[-1] + 0.5, color="white", lw=1.5)

    # acoustic group dividers + colored strip on left
    last_group = None
    for i, a in enumerate(ac_order):
        g = grp_for[a]
        if last_group is not None and g != last_group:
            ax.axhline(i - 0.5, color="white", lw=1.5)
        last_group = g
        ax.add_patch(mpatches.Rectangle((-1.4, i - 0.5), 0.8, 1.0,
                                        facecolor=ACOUSTIC_GROUP_COLORS[g],
                                        clip_on=False, lw=0))
    fig.colorbar(im, ax=ax, shrink=0.8, label="PMI (bits)")
    ax.set_title("Form × meaning: pointwise mutual information")
    out = PLOTS_DIR / "pmi_heatmap_paper.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------ #
# Fig 4: cross-species mutual-NN groups (landmark)
# ------------------------------------------------------------------ #

def _find_mutual_nn_groups(calls, sem_emb: np.ndarray, ac_emb: np.ndarray,
                            species_arr: np.ndarray) -> list[list[int]]:
    """For each ordered pair of species (A,B), find calls (i in A, j in B) such
    that j is the top-1 semantic *and* top-1 acoustic nearest neighbour of i in
    species B and vice versa. Build a graph where calls are nodes and edges
    connect such mutually-consistent pairs across species; extract maximal
    cliques whose calls live in distinct species."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(len(calls)))
    species_to_idx: dict[str, np.ndarray] = {
        s: np.where(species_arr == s)[0] for s in set(species_arr)
    }
    species_list = sorted(species_to_idx)
    for ai, A in enumerate(species_list):
        ia = species_to_idx[A]
        for B in species_list[ai + 1:]:
            ib = species_to_idx[B]
            sem_ab = sem_emb[ia] @ sem_emb[ib].T
            ac_ab = ac_emb[ia] @ ac_emb[ib].T
            # for each i in A: best j in B
            best_sem_b = sem_ab.argmax(axis=1)
            best_ac_b = ac_ab.argmax(axis=1)
            best_sem_a = sem_ab.argmax(axis=0)
            best_ac_a = ac_ab.argmax(axis=0)
            for i_local in range(len(ia)):
                j_sem = best_sem_b[i_local]
                j_ac = best_ac_b[i_local]
                if j_sem != j_ac:
                    continue
                j = j_sem
                if best_sem_a[j] != i_local or best_ac_a[j] != i_local:
                    continue
                G.add_edge(int(ia[i_local]), int(ib[j]))

    groups: list[list[int]] = []
    for clique in nx.find_cliques(G):
        species_in = {species_arr[i] for i in clique}
        if len(species_in) == len(clique) and len(clique) >= 2:
            groups.append(sorted(clique))
    # dedupe identical groups
    seen = set(); uniq = []
    for g in groups:
        t = tuple(g)
        if t not in seen:
            seen.add(t); uniq.append(g)
    return uniq


def _group_score(group: list[int], calls) -> tuple[int, int, int]:
    """Score: (n_classes, n_families, n_species) — higher = more diverse."""
    classes = {calls[i].class_ for i in group}
    families = {calls[i].family for i in group}
    species = {calls[i].species for i in group}
    return (len(classes), len(families), len(species))


def _dominant_category(group: list[int], calls) -> str:
    c = Counter()
    for i in group:
        for k in calls[i].semantic_keywords:
            c[_category_of(k)] += 1
    return c.most_common(1)[0][0] if c else "other"


def fig4_landmark(calls, sem_emb: np.ndarray, ac_emb: np.ndarray) -> Path:
    species_arr = np.array([c.species for c in calls])
    groups = _find_mutual_nn_groups(calls, sem_emb, ac_emb, species_arr)
    # pick four groups maximising diversity across distinct semantic categories
    groups.sort(key=lambda g: _group_score(g, calls), reverse=True)
    chosen: list[list[int]] = []
    used_cats: set[str] = set()
    for g in groups:
        cat = _dominant_category(g, calls)
        if cat in used_cats:
            continue
        chosen.append(g)
        used_cats.add(cat)
        if len(chosen) == 4:
            break
    # pad if fewer than 4
    for g in groups:
        if len(chosen) == 4:
            break
        if g not in chosen:
            chosen.append(g)

    if not chosen:
        # graceful fallback: draw an empty placeholder
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "No mutual-NN groups found", ha="center", va="center")
        ax.axis("off")
        p = PLOTS_DIR / "fig4_landmark_h.pdf"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        return p

    n = len(chosen)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.2), squeeze=False)
    for ax, group in zip(axes[0], chosen):
        cat = _dominant_category(group, calls)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(mpatches.Rectangle((0, 0.94), 1, 0.06,
                                        facecolor=CATEGORY_COLORS.get(cat, "#666"),
                                        transform=ax.transAxes))
        ax.text(0.5, 0.97, cat.upper(), ha="center", va="center",
                color="white", fontsize=11, fontweight="bold",
                transform=ax.transAxes)
        rows = len(group)
        for r, i in enumerate(group):
            c = calls[i]
            top = 0.92 - r * (0.92 / rows)
            bot = 0.92 - (r + 1) * (0.92 / rows)
            mid = (top + bot) / 2
            # class dot
            ax.scatter([0.05], [top - 0.04], s=80, c=CLASS_COLORS.get(c.class_, "gray"),
                       transform=ax.transAxes)
            ax.text(0.12, top - 0.04, c.common_name,
                    fontsize=9, fontweight="bold", transform=ax.transAxes)
            ax.text(0.12, top - 0.08, c.name, fontsize=8, style="italic",
                    color="#444", transform=ax.transAxes)
            ax.text(0.05, top - 0.13,
                    "Sem: " + ", ".join(c.semantic_keywords[:5]),
                    fontsize=7, color="#1f4e79", transform=ax.transAxes,
                    wrap=True)
            ax.text(0.05, top - 0.17,
                    "Acc: " + ", ".join(c.acoustic_keywords[:5]),
                    fontsize=7, color="#4f4f4f", transform=ax.transAxes,
                    wrap=True)
            if r < rows - 1:
                ax.plot([0.02, 0.98], [bot + 0.005, bot + 0.005],
                        color="#dddddd", lw=0.8, transform=ax.transAxes)

    # bottom-row legend for classes
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=CLASS_COLORS[c],
                          markersize=8, label=c) for c in CLASS_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cross-species call groups: mutual top-1 acoustic & semantic neighbours",
                 fontsize=12, y=1.02)
    p = PLOTS_DIR / "fig4_landmark_h.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p

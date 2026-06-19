"""
Figure 2: three independent panels assembled in a tabular layout.
  (A) Semantic UMAP   (B) Acoustic UMAP   (C) Mantel scatter

Layout rules:
- Row 0: panel titles only (A / B / C)
- Row 1: the plots, all same height
- A & B: UMAP with bounding box, no tick marks
- C: square axes, both distances 0–1, equal number of pairs per group,
     per-group regression lines, r/p in legend
"""

import json, sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
try:
    from paper_code.paths import DATABASE_PATH
except ModuleNotFoundError:
    from paths import DATABASE_PATH
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from umap import UMAP

# ------------------------------------------------------------------ #
# Colour schemes
# ------------------------------------------------------------------ #

SEM_CATEGORY = {
    "alarm": "danger",      "predator": "danger",   "threat": "danger",
    "distress": "distress", "infant": "distress",
    "contact": "cohesion",  "coordination": "cohesion",
    "long_distance": "cohesion", "affiliative": "cohesion",
    "aggression": "social", "display": "social",    "sex": "social",
    "dominance": "social",  "submission": "social", "territory": "social",
    "food": "foraging",     "recruitment": "foraging",
}
CAT_COLORS = {
    "danger":   "#d62728",
    "distress": "#ff7f0e",
    "cohesion": "#1f77b4",
    "social":   "#9467bd",
    "foraging": "#2ca02c",
    "other":    "#aaaaaa",
}
TAXON_GROUPS = ["within-species", "same family", "across families"]
TAXON_COLORS = {
    "within-species":   "#2ca02c",
    "same family":      "#ff7f0e",
    "across families":  "#9467bd",
}

# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def load():
    with open(DATABASE_PATH) as f:
        db = json.load(f)
    calls = []
    for s in db["species"]:
        for c in s.get("calls", []):
            calls.append({**c,
                          "species": s["species_name"],
                          "family":  s.get("family", ""),
                          "class":   s.get("class", "")})
    return calls


def primary_category(call):
    for kw in call.get("ontology_keywords", []):
        if kw in SEM_CATEGORY:
            return SEM_CATEGORY[kw]
    return "other"


# ------------------------------------------------------------------ #
# Panel A / B – UMAP
# ------------------------------------------------------------------ #

def compute_umap(emb, seed=42):
    reducer = UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.2)
    return reducer.fit_transform(emb)


def draw_umap(ax, coords, calls, title, add_legend=False):
    cats = [primary_category(c) for c in calls]
    for cat, color in CAT_COLORS.items():
        mask = [i for i, c in enumerate(cats) if c == cat]
        if not mask:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=20, alpha=0.78, linewidths=0,
                   label=cat, zorder=2)
    # force square data limits (UMAP has no intrinsic units)
    x, y = coords[:, 0], coords[:, 1]
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    half = max(x.max() - x.min(), y.max() - y.min()) / 2 * 1.08
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    # bounding box, no ticks
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#333333")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=7)
    if add_legend:
        ax.legend(fontsize=6.5, loc="lower right", framealpha=0.85,
                  title="Semantic category", title_fontsize=7,
                  markerscale=1.2, handletextpad=0.4, borderpad=0.5)


# ------------------------------------------------------------------ #
# Panel C – Mantel scatter (square, equal samples per group)
# ------------------------------------------------------------------ #

def taxon_label(i, j, sp, fam):
    if sp[i] == sp[j]:   return "within-species"
    if fam[i] == fam[j]: return "same family"
    return "across families"


# Mapping from figure group names to mantel_results.json keys
MANTEL_KEYS = {
    "within-species":  "within_species (pooled)",
    "same family":     "same_family_cross_species",
    "across families": "cross_family",
}


def draw_mantel(ax, ac_emb, se_emb, calls, mantel_stats, n_per_group=1000, seed=42):
    """mantel_stats: dict loaded from mantel_results.json (permutation-based r/p)."""
    rng = np.random.default_rng(seed)

    def norm(e):
        n = np.linalg.norm(e, axis=1, keepdims=True)
        n[n == 0] = 1
        return e / n

    ac  = norm(ac_emb.astype(float))
    se  = norm(se_emb.astype(float))
    n   = len(calls)
    sp  = np.array([c["species"] for c in calls])
    fam = np.array([c["family"]  for c in calls])

    ti, tj = np.triu_indices(n, k=1)

    # pre-compute distances for all pairs once
    ac_dist_all = 1 - (ac[ti] * ac[tj]).sum(axis=1)
    se_dist_all = 1 - (se[ti] * se[tj]).sum(axis=1)
    lbl_all     = np.array([taxon_label(ti[k], tj[k], sp, fam)
                             for k in range(len(ti))])

    for grp in TAXON_GROUPS:
        color = TAXON_COLORS[grp]
        mask  = lbl_all == grp
        idx_g = np.where(mask)[0]

        # equal-size sample (visualisation only)
        sample = rng.choice(idx_g, size=min(n_per_group, len(idx_g)), replace=False)
        x = ac_dist_all[sample]
        y = se_dist_all[sample]

        ax.scatter(x, y, c=color, s=4, alpha=0.25,
                   linewidths=0, rasterized=True, zorder=2)

        # regression line: slope from OLS on full group (visual guide only)
        xf = ac_dist_all[idx_g]
        yf = se_dist_all[idx_g]
        slope, intercept, _, _, _ = stats.linregress(xf, yf)
        xl = np.array([0.0, 1.0])
        ax.plot(xl, intercept + slope * xl, color=color, lw=1.8, zorder=3)

        # r/p from permutation Mantel test (not from linregress)
        ms  = mantel_stats[MANTEL_KEYS[grp]]
        r_m = ms["r"]
        p_m = ms["p"]
        n_m = ms["n_pairs"]
        p_str = "$p < 0.001$" if p_m < 0.001 else f"$p = {p_m:.3f}$"
        label = f"{grp}  ($r={r_m:.2f}$, {p_str},  $n={n_m:,}$)"
        ax.plot([], [], color=color, lw=2.5, label=label)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Acoustic distance", fontsize=10)
    ax.set_ylabel("Semantic distance", fontsize=10)
    ax.set_title("(C)  Form–meaning correlation", fontsize=11, fontweight="bold", pad=7)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    leg = ax.legend(fontsize=7.5, loc="upper left", framealpha=0.88,
                    title="Taxonomic relationship", title_fontsize=8,
                    handlelength=1.8)


# ------------------------------------------------------------------ #
# Compose
# ------------------------------------------------------------------ #

PANEL_SIZE = 4.2  # square panels — same for A, B, C
LEFT   = 0.10
RIGHT  = 0.96
BOTTOM = 0.08
TOP    = 0.94     # LEFT..RIGHT = RIGHT..TOP = 0.86 → square axes on square figure


def save_panel(fig, path):
    fig.savefig(path, dpi=200)   # fixed size — no bbox cropping
    plt.close(fig)
    print(f"Saved → {path}")


def main():
    with open(ROOT / "paper_code" / "mantel_results.json") as f:
        mantel_stats = json.load(f)

    calls  = load()
    ac_emb = np.load(ROOT / "paper_code" / "ac_emb.npy")
    se_emb = np.load(ROOT / "paper_code" / "se_emb.npy")
    print(f"{len(calls)} calls")

    print("Semantic UMAP …")
    se_coords = compute_umap(se_emb)
    print("Acoustic UMAP …")
    ac_coords = compute_umap(ac_emb)

    # ---- Panel A ----
    fig_A, ax_A = plt.subplots(figsize=(PANEL_SIZE, PANEL_SIZE))
    fig_A.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    draw_umap(ax_A, se_coords, calls, "(A)  Semantic space (UMAP)", add_legend=True)
    save_panel(fig_A, ROOT / "plots" / "fig2A_semantic_umap.pdf")

    # ---- Panel B ----
    fig_B, ax_B = plt.subplots(figsize=(PANEL_SIZE, PANEL_SIZE))
    fig_B.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    draw_umap(ax_B, ac_coords, calls, "(B)  Acoustic space (UMAP)")
    save_panel(fig_B, ROOT / "plots" / "fig2B_acoustic_umap.pdf")

    # ---- Panel C ----
    print("Mantel scatter …")
    fig_C, ax_C = plt.subplots(figsize=(PANEL_SIZE, PANEL_SIZE))
    fig_C.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    draw_mantel(ax_C, ac_emb, se_emb, calls, mantel_stats, n_per_group=1000)
    save_panel(fig_C, ROOT / "plots" / "fig2C_mantel.pdf")

    # ---- combined PNG (for quick preview) ----
    fig = plt.figure(figsize=(15, 5.2))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.38,
                   left=0.05, right=0.97, top=0.90, bottom=0.13)
    ax_A2 = fig.add_subplot(gs[0, 0])
    ax_B2 = fig.add_subplot(gs[0, 1])
    ax_C2 = fig.add_subplot(gs[0, 2])
    draw_umap(ax_A2, se_coords, calls, "(A)  Semantic space (UMAP)")
    draw_umap(ax_B2, ac_coords, calls, "(B)  Acoustic space (UMAP)")
    handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in CAT_COLORS.items()]
    fig.legend(handles=handles, bbox_to_anchor=(0.36, -0.01), loc="lower center",
               ncol=6, fontsize=8, framealpha=0.8,
               title="Semantic category", title_fontsize=8)
    draw_mantel(ax_C2, ac_emb, se_emb, calls, mantel_stats, n_per_group=1000)
    save_panel(fig, ROOT / "plots" / "fig2_embeddings.png")


if __name__ == "__main__":
    main()

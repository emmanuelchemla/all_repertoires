"""
Merged figure: left column = panels A/B/C from Fig 2; right column = PMI heatmap.

Layout (GridSpec 3×2, width_ratios [1, 2]):
  row 0, col 0 │ (A) Semantic UMAP
  row 1, col 0 │ (B) Acoustic UMAP
  row 2, col 0 │ (C) Mantel scatter
  rows 0-2, col 1 │ (D) PMI heatmap

Output: plots/fig_merged.pdf
"""

import json, sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
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
TAXON_GROUPS  = ["within-species", "same family", "across families"]
TAXON_COLORS  = {
    "within-species":  "#2ca02c",
    "same family":     "#ff7f0e",
    "across families": "#9467bd",
}
MANTEL_KEYS = {
    "within-species":  "within_species (pooled)",
    "same family":     "same_family_cross_species",
    "across families": "cross_family",
}

ACOUSTIC = [
    ("high-frequency",      ["high-frequency","high frequency","high-pitched","high pitched","ultrasonic"]),
    ("low-frequency",       ["low-frequency","low frequency","low-pitched","low pitched","infrasound"]),
    ("tonal",               ["tonal","pure tone","narrowband","narrow-band","sinusoidal"]),
    ("broadband / noisy",   ["broadband","broad-band","noisy","noise","atonal","wideband"]),
    ("frequency-modulated", ["frequency-modulated","frequency modulated","fm sweep","sweep","upsweep","downsweep","modulated"]),
    ("harmonic",            ["harmonic","overtone","formant"]),
    ("pulsed",              ["pulsed","pulse","click","burst","staccato"]),
    ("repetitive",          ["repetitive","repeated","series","bout","sequence of"]),
    ("short",               ["short ","brief","abrupt"]),
    ("long / sustained",    ["long ","prolonged","sustained","extended"]),
    ("loud",                ["loud","intense","powerful","far-carrying"]),
    ("soft / quiet",        ["soft ","quiet","low amplitude","low-amplitude","subtle"]),
]
SEMANTIC = [
    "alarm","predator","threat","aggression",
    "distress","infant",
    "contact","coordination","long_distance","affiliative",
    "display","sex","territory","food",
]
SEMANTIC_LABELS = {"long_distance": "long-distance"}

# ------------------------------------------------------------------ #
# Data helpers
# ------------------------------------------------------------------ #

def load():
    with open(ROOT / "database.json") as f:
        db = json.load(f)
    calls = []
    for s in db["species"]:
        for c in s.get("calls", []):
            calls.append({**c,
                          "species": s["species_name"],
                          "family":  s.get("family", ""),
                          "class":   s.get("class", "")})
    return calls


def normalize(e):
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0] = 1
    return e / n


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


def draw_umap(ax, coords, calls, title):
    cats = [primary_category(c) for c in calls]
    for cat, color in CAT_COLORS.items():
        mask = [i for i, c in enumerate(cats) if c == cat]
        if not mask:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=14, alpha=0.78, linewidths=0, zorder=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("#333333")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)


# ------------------------------------------------------------------ #
# Panel C – Mantel scatter
# ------------------------------------------------------------------ #

def taxon_label(i, j, sp, fam):
    if sp[i] == sp[j]:   return "within-species"
    if fam[i] == fam[j]: return "same family"
    return "across families"


def draw_mantel(ax, ac_emb, se_emb, calls, mantel_stats, n_per_group=800, seed=42):
    rng = np.random.default_rng(seed)
    ac  = normalize(ac_emb.astype(float))
    se  = normalize(se_emb.astype(float))
    n   = len(calls)
    sp  = np.array([c["species"] for c in calls])
    fam = np.array([c["family"]  for c in calls])
    ti, tj = np.triu_indices(n, k=1)
    ac_dist_all = 1 - (ac[ti] * ac[tj]).sum(axis=1)
    se_dist_all = 1 - (se[ti] * se[tj]).sum(axis=1)
    lbl_all = np.array([taxon_label(ti[k], tj[k], sp, fam) for k in range(len(ti))])

    for grp in TAXON_GROUPS:
        color = TAXON_COLORS[grp]
        idx_g = np.where(lbl_all == grp)[0]
        sample = rng.choice(idx_g, size=min(n_per_group, len(idx_g)), replace=False)
        ax.scatter(ac_dist_all[sample], se_dist_all[sample],
                   c=color, s=3, alpha=0.22, linewidths=0, rasterized=True, zorder=2)
        xf, yf = ac_dist_all[idx_g], se_dist_all[idx_g]
        slope, intercept, _, _, _ = stats.linregress(xf, yf)
        xl = np.array([0.0, 1.0])
        ax.plot(xl, intercept + slope * xl, color=color, lw=1.6, zorder=3)
        ms    = mantel_stats[MANTEL_KEYS[grp]]
        r_m, p_m = ms["r"], ms["p"]
        p_str = "$p<0.001$" if p_m < 0.001 else f"$p={p_m:.3f}$"
        ax.plot([], [], color=color, lw=2.2,
                label=f"{grp}  ($r={r_m:.2f}$, {p_str})")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Acoustic distance", fontsize=9)
    ax.set_ylabel("Semantic distance", fontsize=9)
    ax.set_title("(C)  Form–meaning correlation", fontsize=10, fontweight="bold", pad=5)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.85,
              title="Taxonomic group", title_fontsize=7, handlelength=1.5)


# ------------------------------------------------------------------ #
# Panel D – PMI heatmap
# ------------------------------------------------------------------ #

def compute_pmi(calls):
    n = len(calls)
    ac_labels  = [label for label, _ in ACOUSTIC]
    ac_mat  = np.zeros((n, len(ACOUSTIC)),  dtype=float)
    sem_mat = np.zeros((n, len(SEMANTIC)), dtype=float)
    for i, c in enumerate(calls):
        desc = c.get("acoustic_description", "").lower()
        for j, (_, patterns) in enumerate(ACOUSTIC):
            ac_mat[i, j] = float(any(p in desc for p in patterns))
        kws = set(c.get("ontology_keywords", []))
        for j, sk in enumerate(SEMANTIC):
            sem_mat[i, j] = float(sk in kws)
    p_ac   = ac_mat.mean(axis=0)
    p_sem  = sem_mat.mean(axis=0)
    p_joint = (ac_mat[:, :, None] * sem_mat[:, None, :]).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(
            (p_joint > 0) & (p_ac[:, None] > 0) & (p_sem[None, :] > 0),
            np.log2(p_joint / (p_ac[:, None] * p_sem[None, :])),
            0.0,
        )
    return pmi, ac_labels, SEMANTIC


def cluster_order(mat):
    if mat.shape[0] < 2:
        return list(range(mat.shape[0]))
    dist = pdist(mat, metric="euclidean")
    Z = linkage(dist, method="average")
    return list(leaves_list(Z))


def draw_pmi(ax, fig, pmi, ac_labels, sem_labels):
    row_order = cluster_order(pmi)
    col_order = cluster_order(pmi.T)
    pmi_c     = pmi[np.ix_(row_order, col_order)]
    ac_c      = [ac_labels[i] for i in row_order]
    sem_c     = [sem_labels[j] for j in col_order]
    sem_disp  = [SEMANTIC_LABELS.get(s, s) for s in sem_c]
    n_rows, n_cols = pmi_c.shape

    vmax = max(abs(pmi_c.max()), abs(pmi_c.min()), 1.0)
    im = ax.imshow(pmi_c, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(sem_disp, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ac_c, fontsize=9)
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    # annotate strongest cells
    flat = pmi_c.flatten()
    for idx in list(np.argsort(flat)[::-1][:6]) + list(np.argsort(flat)[:3]):
        r, c = divmod(idx, n_cols)
        v = pmi_c[r, c]
        col = "white" if abs(v) > vmax * 0.55 else "black"
        ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                fontsize=8, fontweight="bold", color=col)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("PMI (bits)", fontsize=9)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Semantic function", fontsize=10, labelpad=6)
    ax.set_ylabel("Acoustic feature", fontsize=10, labelpad=6)
    ax.set_title("(D)  Acoustic–semantic PMI", fontsize=10, fontweight="bold", pad=7)

    # x-axis semantic group brackets
    alarm_c   = [i for i, s in enumerate(sem_c) if s in {"alarm","predator","threat","aggression"}]
    infant_c  = [i for i, s in enumerate(sem_c) if s in {"distress","infant"}]
    contact_c = [i for i, s in enumerate(sem_c) if s in {"contact","coordination","long_distance","affiliative"}]

    def bracket(cols, label):
        if not cols:
            return
        x0, x1 = min(cols) - 0.45, max(cols) + 0.45
        ax.annotate("", xy=(x1, -0.06), xytext=(x0, -0.06),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="#444", lw=1.4))
        ax.text((x0 + x1) / 2, -0.10, label, ha="center", va="top",
                fontsize=8, color="#444", transform=ax.get_xaxis_transform())

    bracket(alarm_c,   "danger / threat")
    bracket(infant_c,  "infant / distress")
    bracket(contact_c, "contact / cohesion")


# ------------------------------------------------------------------ #
# Compose
# ------------------------------------------------------------------ #

def main():
    calls  = load()
    ac_emb = np.load(ROOT / "paper_code" / "ac_emb.npy")
    se_emb = np.load(ROOT / "paper_code" / "se_emb.npy")
    with open(ROOT / "paper_code" / "mantel_results.json") as f:
        mantel_stats = json.load(f)
    print(f"{len(calls)} calls")

    print("Semantic UMAP …")
    se_coords = compute_umap(se_emb)
    print("Acoustic UMAP …")
    ac_coords = compute_umap(ac_emb)
    print("PMI …")
    pmi, ac_labels, sem_labels = compute_pmi(calls)

    fig = plt.figure(figsize=(15, 10))
    gs  = GridSpec(3, 2, figure=fig,
                   width_ratios=[1, 2],
                   hspace=0.45, wspace=0.38,
                   left=0.06, right=0.97, top=0.95, bottom=0.12)

    ax_A   = fig.add_subplot(gs[0, 0])
    ax_B   = fig.add_subplot(gs[1, 0])
    ax_C   = fig.add_subplot(gs[2, 0])
    ax_pmi = fig.add_subplot(gs[:, 1])

    draw_umap(ax_A, se_coords, calls, "(A)  Semantic space (UMAP)")
    draw_umap(ax_B, ac_coords, calls, "(B)  Acoustic space (UMAP)")
    draw_mantel(ax_C, ac_emb, se_emb, calls, mantel_stats)
    draw_pmi(ax_pmi, fig, pmi, ac_labels, sem_labels)

    # Shared semantic-category legend anchored below panels A & B
    handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in CAT_COLORS.items()]
    # get bounding box of left column to position legend
    fig.legend(handles=handles,
               bbox_to_anchor=(0.005, 0.01, 0.34, 0.06),
               bbox_transform=fig.transFigure,
               loc="lower left", ncol=3, fontsize=7,
               framealpha=0.85, title="Semantic category", title_fontsize=7)

    out = ROOT / "plots" / "fig_merged.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

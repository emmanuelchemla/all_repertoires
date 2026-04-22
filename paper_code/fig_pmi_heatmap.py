"""
Generate Figure 3: PMI heatmap (acoustic × semantic keywords).

Design principles:
- Hierarchical clustering on both axes so related keywords group together
- Annotate the top-N highest and lowest PMI cells
- Readable font sizes, compact cells
- Saved to plots/pmi_heatmap_paper.png
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import pdist

# ------------------------------------------------------------------ #
# Keyword definitions
# ------------------------------------------------------------------ #

# Acoustic terms: (display label, list of substrings to match in description)
ACOUSTIC = [
    ("high-frequency",      ["high-frequency", "high frequency", "high-pitched", "high pitched", "ultrasonic"]),
    ("low-frequency",       ["low-frequency", "low frequency", "low-pitched", "low pitched", "infrasound"]),
    ("tonal",               ["tonal", "pure tone", "narrowband", "narrow-band", "sinusoidal"]),
    ("broadband / noisy",   ["broadband", "broad-band", "noisy", "noise", "atonal", "wideband"]),
    ("frequency-modulated", ["frequency-modulated", "frequency modulated", "fm sweep", "sweep", "upsweep", "downsweep", "modulated"]),
    ("harmonic",            ["harmonic", "overtone", "formant"]),
    ("pulsed",              ["pulsed", "pulse", "click", "burst", "staccato"]),
    ("repetitive",          ["repetitive", "repeated", "series", "bout", "sequence of"]),
    ("short",               ["short ", "brief", "abrupt"]),
    ("long / sustained",    ["long ", "prolonged", "sustained", "extended"]),
    ("loud",                ["loud", "intense", "powerful", "far-carrying"]),
    ("soft / quiet",        ["soft ", "quiet", "low amplitude", "low-amplitude", "subtle"]),
]

# Semantic keywords to include (filter to those with ≥5 calls)
SEMANTIC = [
    "alarm", "predator", "threat", "aggression",
    "distress", "infant",
    "contact", "coordination", "long_distance", "affiliative",
    "display", "sex", "dominance", "submission", "territory",
    "food", "recruitment",
    "individual_identity", "learning", "referential",
]

SEMANTIC_LABELS = {
    "long_distance":      "long-distance",
    "individual_identity": "indiv. identity",
}

# ------------------------------------------------------------------ #
# Load data and compute PMI
# ------------------------------------------------------------------ #

def load_calls():
    with open(ROOT / "database.json") as f:
        db = json.load(f)
    return [c for s in db["species"] for c in s.get("calls", [])]


def extract_acoustic_flags(desc: str, patterns: list[str]) -> bool:
    desc = desc.lower()
    return any(p in desc for p in patterns)


def compute_pmi(calls, min_calls=4):
    n = len(calls)
    ac_labels  = [label for label, _ in ACOUSTIC]

    ac_mat  = np.zeros((n, len(ACOUSTIC)),  dtype=float)
    sem_mat = np.zeros((n, len(SEMANTIC)), dtype=float)

    for i, c in enumerate(calls):
        desc = c.get("acoustic_description", "")
        for j, (_, patterns) in enumerate(ACOUSTIC):
            ac_mat[i, j] = float(extract_acoustic_flags(desc, patterns))
        kws = set(c.get("ontology_keywords", []))
        for j, sk in enumerate(SEMANTIC):
            sem_mat[i, j] = float(sk in kws)

    # filter semantic keywords with too few calls
    sem_counts = sem_mat.sum(axis=0)
    keep = sem_counts >= min_calls
    sem_mat = sem_mat[:, keep]
    sem_labels = [s for s, k in zip(SEMANTIC, keep) if k]
    print(f"  Semantic keywords kept (≥{min_calls} calls): {sem_labels}")

    p_ac    = ac_mat.mean(axis=0)
    p_sem   = sem_mat.mean(axis=0)
    p_joint = (ac_mat[:, :, None] * sem_mat[:, None, :]).mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(
            (p_joint > 0) & (p_ac[:, None] > 0) & (p_sem[None, :] > 0),
            np.log2(p_joint / (p_ac[:, None] * p_sem[None, :])),
            0.0,
        )

    return pmi, ac_labels, sem_labels, p_ac, p_sem, p_joint


# ------------------------------------------------------------------ #
# Clustering
# ------------------------------------------------------------------ #

def cluster_order(mat):
    """Return leaf order from complete-linkage hierarchical clustering."""
    if mat.shape[0] < 2:
        return list(range(mat.shape[0]))
    dist = pdist(mat, metric="euclidean")
    Z = linkage(dist, method="average")
    return list(leaves_list(Z))


# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #

def plot_pmi(pmi, ac_labels, sem_labels, p_ac, p_sem, out_path):
    # Acoustic features as rows (fewer = 12), semantic functions as columns (more = ~18).
    # Both axes ordered by hierarchical clustering so structure emerges bottom-up;
    # semantic category brackets overlay the data-driven column order.
    row_order = cluster_order(pmi)      # cluster acoustic rows
    col_order = cluster_order(pmi.T)    # cluster semantic columns

    pmi_c = pmi[np.ix_(row_order, col_order)]
    ac_c  = [ac_labels[i]  for i in row_order]
    sem_c = [sem_labels[j] for j in col_order]
    sem_c_display = [SEMANTIC_LABELS.get(s, s) for s in sem_c]

    n_rows, n_cols = pmi_c.shape        # n_rows ≤ 12, n_cols ≤ 18
    cell = 0.52                         # inches per cell — square cells
    margin_left   = 1.9                 # acoustic feature row labels
    margin_right  = 1.0                 # colorbar
    margin_top    = 0.5
    margin_bottom = 1.8                 # rotated semantic column labels + brackets
    fig_w = n_cols * cell + margin_left + margin_right
    fig_h = n_rows * cell + margin_top  + margin_bottom

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = max(abs(pmi_c.max()), abs(pmi_c.min()), 1.0)
    im = ax.imshow(pmi_c, aspect=1, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    # ticks: semantic on x (columns), acoustic on y (rows)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(sem_c_display, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ac_c, fontsize=10)

    # thin white grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    # annotate top-6 positive and top-3 negative cells
    flat = pmi_c.flatten()
    for idx in list(np.argsort(flat)[::-1][:6]) + list(np.argsort(flat)[:3]):
        r, c = divmod(idx, n_cols)
        v = pmi_c[r, c]
        col = "white" if abs(v) > vmax * 0.55 else "black"
        ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=col)

    # colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("PMI (bits)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Semantic function", fontsize=11, labelpad=8)
    ax.set_ylabel("Acoustic feature",  fontsize=11, labelpad=8)

    # semantic category brackets on x-axis — overlaid on the clustering order
    alarm_cols   = [i for i, s in enumerate(sem_c)
                    if s in {"alarm","predator","threat","aggression"}]
    infant_cols  = [i for i, s in enumerate(sem_c)
                    if s in {"distress","infant"}]
    contact_cols = [i for i, s in enumerate(sem_c)
                    if s in {"contact","coordination","long_distance","affiliative"}]
    social_cols  = [i for i, s in enumerate(sem_c)
                    if s in {"display","sex","dominance","submission","territory"}]

    def bracket(cols, label, color="#444"):
        if not cols:
            return
        x0, x1 = min(cols) - 0.45, max(cols) + 0.45
        ax.annotate("", xy=(x1, -0.07), xytext=(x0, -0.07),
                    xycoords=("data", "axes fraction"),
                    textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color=color, lw=1.5))
        ax.text((x0+x1)/2, -0.12, label, ha="center", va="top",
                fontsize=8.5, color=color, transform=ax.get_xaxis_transform())

    bracket(alarm_cols,   "danger / threat")
    bracket(infant_cols,  "distress")
    bracket(contact_cols, "cohesion")
    bracket(social_cols,  "social")

    fig.tight_layout()
    # save as both PNG and PDF
    png_path = Path(str(out_path).replace(".pdf", ".png")) if str(out_path).endswith(".pdf") else out_path
    pdf_path = Path(str(out_path).replace(".png", ".pdf")) if str(out_path).endswith(".png") else out_path
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    calls = load_calls()
    print(f"{len(calls)} calls")
    pmi, ac_labels, sem_labels, p_ac, p_sem, p_joint = compute_pmi(calls)

    print("\nTop 10 PMI associations:")
    flat_idx = np.argsort(pmi.flatten())[::-1]
    for idx in flat_idx[:10]:
        r, c = divmod(idx, len(sem_labels))
        print(f"  {ac_labels[r]:25s} × {sem_labels[c]:20s}  PMI={pmi[r,c]:.2f}")

    print("\nBottom 5 PMI associations:")
    for idx in flat_idx[-5:]:
        r, c = divmod(idx, len(sem_labels))
        print(f"  {ac_labels[r]:25s} × {sem_labels[c]:20s}  PMI={pmi[r,c]:.2f}")

    out = ROOT / "plots" / "pmi_heatmap_paper.png"
    plot_pmi(pmi, ac_labels, sem_labels, p_ac, p_sem, out)


if __name__ == "__main__":
    main()

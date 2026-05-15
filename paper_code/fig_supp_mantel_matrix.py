"""
Supplementary figure: Species-pairwise acoustic–semantic correlation matrix.

For each pair of species (A, B), compute the Pearson r between all
cross-species acoustic cosine similarities and semantic cosine similarities.
Display as a symmetric heatmap sorted by taxonomic class, with hierarchical clustering within each class.

Output: plots/fig_supp_mantel_matrix.pdf
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, leaves_list

from paper_code.data_sources import DATA_SOURCES, load_calls, load_embedding_artifacts

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

CLASS_ORDER = {"Amphibia": 0, "Aves": 1, "Mammalia": 2}
CLASS_COLORS = {
    "Amphibia": "#C44E52",
    "Aves": "#55A868",
    "Mammalia": "#4C72B0",
}
MIN_PAIRS = 6  # skip species pairs with fewer cross-species pairs than this
MIN_CALLS_PER_SPECIES = 6  # drop species with fewer calls than this


def normalize(e):
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0] = 1
    return e / n


def common_name(species_name):
    """Return just the common name, dropping '(Scientific name)' suffix."""
    return species_name.split("(")[0].strip()


# ------------------------------------------------------------------ #
# Main computation
# ------------------------------------------------------------------ #


def compute_pairwise_r(calls, Sa, Ss):
    """
    For each pair of species (i, j) compute Pearson r between their
    cross-species acoustic and semantic similarity vectors.
    Returns (species_list, r_matrix, class_list).
    """
    sp_names = np.array([c["species"] for c in calls])
    sp_classes = {c["species"]: c["class"] for c in calls}

    # Drop species with too few calls
    from collections import Counter
    call_counts = Counter(sp_names)
    eligible = {s for s, n in call_counts.items() if n >= MIN_CALLS_PER_SPECIES}
    dropped = set(call_counts) - eligible
    if dropped:
        print(f"  Dropped {len(dropped)} species with <{MIN_CALLS_PER_SPECIES} calls: "
              + ", ".join(sorted(dropped)[:5]) + ("…" if len(dropped) > 5 else ""))

    # Sorted species list: by (class_order, species_name)
    all_species = sorted(
        eligible,
        key=lambda s: (CLASS_ORDER.get(sp_classes.get(s, ""), 9), s),
    )
    n_sp = len(all_species)

    # Index arrays: sp_idx[species] = array of call indices for that species
    sp_idx = {s: np.where(sp_names == s)[0] for s in all_species}

    r_matrix = np.full((n_sp, n_sp), np.nan)
    np.fill_diagonal(r_matrix, np.nan)  # diagonal left as NaN (same species)

    for i in range(n_sp):
        for j in range(i + 1, n_sp):
            idx_i = sp_idx[all_species[i]]
            idx_j = sp_idx[all_species[j]]

            # All cross-species pairs (Cartesian product)
            # rows from species i, cols from species j
            ac_pairs = Sa[np.ix_(idx_i, idx_j)].ravel()
            se_pairs = Ss[np.ix_(idx_i, idx_j)].ravel()

            if len(ac_pairs) < MIN_PAIRS:
                continue

            r = np.corrcoef(ac_pairs, se_pairs)[0, 1]
            r_matrix[i, j] = r
            r_matrix[j, i] = r

    classes = [sp_classes.get(s, "") for s in all_species]
    return all_species, r_matrix, classes


# ------------------------------------------------------------------ #
# Plotting helpers
# ------------------------------------------------------------------ #


def reorder_within_classes(species, R, classes):
    """Reorder species within each class block by hierarchical clustering on their r-row profiles."""
    spans = class_spans(classes)
    new_order = []
    for start, end, _cls in spans:
        if end - start <= 2:
            new_order.extend(range(start, end))
            continue
        # Use full row of R (correlation profile with all species), NaN → 0
        block = np.nan_to_num(R[start:end, :], nan=0.0)
        Z = linkage(block, method="average", metric="euclidean")
        leaf_order = leaves_list(Z)
        new_order.extend(start + i for i in leaf_order)
    return new_order


def orient_class_blocks(R, classes, n_boundary=3):
    """For each class block, choose keep-or-flip to maximise mean r at class boundaries.

    After hierarchical clustering any subtree can be reflected without breaking the
    topology, so we try both orientations and keep whichever puts more similar species
    at each inter-class boundary.  Returns a permutation of range(len(classes)).
    """
    spans = class_spans(classes)
    perm = list(range(len(classes)))

    def _boundary_mean(left_indices, right_indices):
        vals = [R[i, j] for i in left_indices for j in right_indices
                if not np.isnan(R[i, j])]
        return float(np.mean(vals)) if vals else 0.0

    for k, (start, end, _cls) in enumerate(spans):
        if end - start <= 1:
            continue
        nb = min(n_boundary, end - start)

        # Indices of the neighbour bands (already in current perm order)
        left_band  = list(range(max(0, spans[k-1][1] - nb), spans[k-1][1])) if k > 0 else []
        right_band = list(range(spans[k+1][0], min(len(classes), spans[k+1][0] + nb))) if k < len(spans) - 1 else []

        # Forward: first nb of this block face left_band; last nb face right_band
        fwd_first = list(range(start, start + nb))
        fwd_last  = list(range(end - nb, end))
        score_fwd  = _boundary_mean(left_band, fwd_first) + _boundary_mean(fwd_last, right_band)

        # Reversed: last nb face left_band; first nb face right_band
        score_rev  = _boundary_mean(left_band, fwd_last) + _boundary_mean(fwd_first, right_band)

        if score_rev > score_fwd:
            perm[start:end] = list(range(end - 1, start - 1, -1))

    return perm


def class_spans(classes):
    """
    Return list of (start_idx, end_idx_exclusive, class_name) for each
    contiguous block of the same class.
    """
    spans = []
    if not classes:
        return spans
    current = classes[0]
    start = 0
    for k, cls in enumerate(classes[1:], 1):
        if cls != current:
            spans.append((start, k, current))
            current = cls
            start = k
    spans.append((start, len(classes), current))
    return spans


# ------------------------------------------------------------------ #
# Main figure
# ------------------------------------------------------------------ #


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="old")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading calls and embeddings …")
    calls = load_calls(args.data_source)
    ac_emb, se_emb = load_embedding_artifacts(args.data_source, len(calls))
    print(f"  {len(calls)} calls, {len({c['species'] for c in calls})} species, source={args.data_source}")

    # L2-normalise and compute cosine similarity matrices
    ac = normalize(ac_emb.astype(float))
    se = normalize(se_emb.astype(float))
    Sa = ac @ ac.T
    Ss = se @ se.T

    print("Computing pairwise Pearson r …")
    species, R, classes = compute_pairwise_r(calls, Sa, Ss)
    n_sp = len(species)
    print(f"  {n_sp} species in matrix")

    # Reorder within each class block by hierarchical clustering
    order = reorder_within_classes(species, R, classes)
    species = [species[i] for i in order]
    R = R[np.ix_(order, order)]
    classes = [classes[i] for i in order]

    # Orient each class block (keep or flip) to maximise cross-class boundary similarity
    flip = orient_class_blocks(R, classes)
    species = [species[i] for i in flip]
    R = R[np.ix_(flip, flip)]
    classes = [classes[i] for i in flip]

    # Summary statistics (off-diagonal non-NaN cells)
    off_diag = R[~np.eye(n_sp, dtype=bool)]
    valid = off_diag[~np.isnan(off_diag)]
    mean_r = np.mean(valid)
    std_r = np.std(valid)
    print(f"  Mean r = {mean_r:.3f} ± {std_r:.3f}  (N={len(valid)} pairs)")

    # ---------------------------------------------------------------- #
    # Figure layout
    # ---------------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.subplots_adjust(left=0.20, right=0.88, top=0.93, bottom=0.20)

    # Heatmap
    im = ax.imshow(R, vmin=-0.3, vmax=0.3, cmap="RdBu_r", aspect="equal")

    # Tick labels: common names
    tick_labels = [common_name(s) for s in species]
    ax.set_xticks(range(n_sp))
    ax.set_yticks(range(n_sp))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6.5, ha="right")
    ax.set_yticklabels(tick_labels, fontsize=6.5)

    # Grid lines between cells for readability
    ax.set_xticks(np.arange(-0.5, n_sp, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_sp, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ---------------------------------------------------------------- #
    # Class-coloured sidebar rectangles
    # ---------------------------------------------------------------- #
    # Bar thickness and offset in data coordinates
    bar_w = 1.2   # width (height for x-axis bar) in data units
    bar_gap = 0.6  # gap between heatmap edge and bar

    spans = class_spans(classes)

    for (start, end, cls) in spans:
        color = CLASS_COLORS.get(cls, "#aaaaaa")
        mid = (start + end) / 2.0 - 0.5

        # --- Y-axis (left) bar ---
        rect_y = Rectangle(
            (-0.5 - bar_gap - bar_w, start - 0.5),
            bar_w,
            end - start,
            transform=ax.transData,
            clip_on=False,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect_y)

        # --- X-axis (top, since origin is upper-left for imshow) bar ---
        rect_x = Rectangle(
            (start - 0.5, -0.5 - bar_gap - bar_w),
            end - start,
            bar_w,
            transform=ax.transData,
            clip_on=False,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect_x)

        # Class label on the left bar
        ax.text(
            -0.5 - bar_gap - bar_w / 2,
            mid,
            cls,
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="white",
            rotation=90,
            transform=ax.transData,
            clip_on=False,
        )

    # ---------------------------------------------------------------- #
    # Colorbar
    # ---------------------------------------------------------------- #
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Pearson r (acoustic vs. semantic)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # ---------------------------------------------------------------- #
    # Legend for class colors
    # ---------------------------------------------------------------- #
    import matplotlib.patches as mpatches

    legend_handles = [
        mpatches.Patch(facecolor=CLASS_COLORS[cls], edgecolor="grey",
                       linewidth=0.5, label=cls)
        for cls in ["Amphibia", "Aves", "Mammalia"]
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        title="Taxonomic class",
        title_fontsize=8,
        framealpha=0.85,
        bbox_to_anchor=(1.0, 1.15),
    )

    # ---------------------------------------------------------------- #
    # Titles and annotations
    # ---------------------------------------------------------------- #
    ax.set_title(
        "Species-pairwise acoustic–semantic correlation",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    # Overall mean r annotation (bottom-left of axes)
    ax.text(
        0.01,
        -0.28,
        f"Overall mean r = {mean_r:.3f} ± {std_r:.3f}  (N = {len(valid)} species pairs)",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        va="top",
    )

    # ---------------------------------------------------------------- #
    # Save
    # ---------------------------------------------------------------- #
    out = out_dir / "fig_supp_mantel_matrix.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

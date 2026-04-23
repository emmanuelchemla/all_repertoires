"""
Supplementary figure: Sensitivity analyses.

Panel A: Mantel r (acoustic vs semantic) broken down by taxonomic class pairs
         (within-Mammalia, within-Aves, within-Amphibia, and all cross-class pairs).
         95% bootstrap CI shown as error bars.

Panel B: Sensitivity of the SPNN clique analysis to the top-k NN threshold (k=1,2,3).
         Shows n_edges, n_cliques(>=3 species), n_3-class-cliques as k varies.

Output: plots/fig_supp_sensitivity.pdf
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

CLASS_COLORS = {
    "Mammalia": "#4C72B0",
    "Aves": "#55A868",
    "Amphibia": "#C44E52",
    "mixed": "#888888",
}

N_BOOT = 500
BOOT_SEED = 42

# ------------------------------------------------------------------ #
# Data helpers
# ------------------------------------------------------------------ #


def load():
    with open(ROOT / "database.json") as f:
        db = json.load(f)
    calls = []
    for s in db["species"]:
        for c in s.get("calls", []):
            calls.append(
                {
                    **c,
                    "species": s["species_name"],
                    "family": s.get("family", ""),
                    "class": s.get("class", ""),
                }
            )
    return calls


def normalize(e):
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0] = 1
    return e / n


# ------------------------------------------------------------------ #
# Panel A helpers
# ------------------------------------------------------------------ #


def pearson_r(x, y):
    """Fast Pearson r for 1-D arrays."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.linalg.norm(xc) * np.linalg.norm(yc)
    if denom == 0:
        return np.nan
    return float(np.dot(xc, yc) / denom)


def bootstrap_ci(x, y, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap 95% CI for Pearson r by resampling pairs with replacement."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rs[b] = pearson_r(x[idx], y[idx])
    return float(np.nanpercentile(rs, 2.5)), float(np.nanpercentile(rs, 97.5))


def compute_mantel_subsets(calls, Sa, Ss):
    """
    Compute Pearson r between upper-triangle acoustic and semantic
    similarities for each class-pair subset.

    Returns list of dicts with keys: label, r, ci_lo, ci_hi, color, n_pairs.
    """
    sp = np.array([c["species"] for c in calls])
    cls = np.array([c["class"] for c in calls])
    n = len(calls)

    # All upper-triangle indices (cross-species only)
    ti, tj = np.triu_indices(n, k=1)
    cross_mask = sp[ti] != sp[tj]
    ti_cs, tj_cs = ti[cross_mask], tj[cross_mask]

    ac_flat = Sa[ti_cs, tj_cs]
    se_flat = Ss[ti_cs, tj_cs]
    cls_i = cls[ti_cs]
    cls_j = cls[tj_cs]

    subsets = [
        ("Mammalia x Mammalia", "Mammalia", "Mammalia", CLASS_COLORS["Mammalia"]),
        ("Aves x Aves",         "Aves",     "Aves",     CLASS_COLORS["Aves"]),
        ("Amphibia x Amphibia", "Amphibia", "Amphibia", CLASS_COLORS["Amphibia"]),
        ("Mammalia x Aves",     "Mammalia", "Aves",     CLASS_COLORS["mixed"]),
        ("Mammalia x Amphibia", "Mammalia", "Amphibia", CLASS_COLORS["mixed"]),
        ("Aves x Amphibia",     "Aves",     "Amphibia", CLASS_COLORS["mixed"]),
        ("All cross-species",   None,       None,       CLASS_COLORS["mixed"]),
    ]

    results = []
    for label, clsA, clsB, color in subsets:
        if clsA is None:
            # All cross-species pairs
            mask = np.ones(len(ti_cs), dtype=bool)
        elif clsA == clsB:
            # Within-class, cross-species
            mask = (cls_i == clsA) & (cls_j == clsA)
        else:
            # Between two different classes
            mask = ((cls_i == clsA) & (cls_j == clsB)) | (
                (cls_i == clsB) & (cls_j == clsA)
            )

        x = ac_flat[mask]
        y = se_flat[mask]
        n_pairs = int(mask.sum())

        if n_pairs < 10:
            print(f"  [{label}] skipped — only {n_pairs} pairs")
            continue

        r = pearson_r(x, y)
        ci_lo, ci_hi = bootstrap_ci(x, y)
        results.append(
            dict(
                label=label,
                r=r,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                color=color,
                n_pairs=n_pairs,
            )
        )
        print(
            f"  [{label:25s}]  r={r:+.3f}  95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]"
            f"  n={n_pairs:,}"
        )

    return results


# ------------------------------------------------------------------ #
# Panel B helpers
# ------------------------------------------------------------------ #


def build_spnn_graph_topk(ac_sim, se_sim, calls, k):
    """
    Build the species-pairwise NN graph with top-k threshold.

    Edge (i, j) between call i in species A and call j in species B iff:
      - There EXISTS a call a in A that is in top-k acoustic NNs of j
        (within A) AND in top-k semantic NNs of j (within A).
        That is: a is in ac_topk[j, A] AND a is in se_topk[j, A].
      - Symmetrically: j (= b) is in ac_topk[i, B] AND se_topk[i, B].

    For k=1 this matches the original build_spnn_graph exactly.
    """
    sp = np.array([c["species"] for c in calls])
    species = sorted(set(sp))
    sp_idx = {s: np.where(sp == s)[0] for s in species}

    G = nx.Graph()
    G.add_nodes_from(range(len(calls)))

    for si, sA in enumerate(species):
        idxA = sp_idx[sA]
        for sB in species[si + 1 :]:
            idxB = sp_idx[sB]

            # For each call b in B, find top-k candidates in A
            # (calls in A that are simultaneously in ac top-k and se top-k for b)
            for b in idxB:
                ac_scores_b = ac_sim[b][idxA]
                se_scores_b = se_sim[b][idxA]
                # top-k indices within idxA (local indices)
                ac_topk_local = set(
                    np.argsort(ac_scores_b)[::-1][:k].tolist()
                )
                se_topk_local = set(
                    np.argsort(se_scores_b)[::-1][:k].tolist()
                )
                # intersection: must appear in both top-k lists
                candidates_local = ac_topk_local & se_topk_local
                if not candidates_local:
                    continue

                for a_local in candidates_local:
                    a = idxA[a_local]
                    # Check symmetry: b must be in top-k of a within B
                    ac_scores_a = ac_sim[a][idxB]
                    se_scores_a = se_sim[a][idxB]
                    b_local = np.where(idxB == b)[0][0]
                    ac_topk_B = set(np.argsort(ac_scores_a)[::-1][:k].tolist())
                    se_topk_B = set(np.argsort(se_scores_a)[::-1][:k].tolist())
                    if b_local in ac_topk_B and b_local in se_topk_B:
                        w = float(0.5 * (ac_sim[a, b] + se_sim[a, b]))
                        if not G.has_edge(a, b):
                            G.add_edge(a, b, weight=w)

    return G


def analyze_graph(G, calls, k):
    """
    Count edges, cliques with >=3 distinct species, cliques with 3 distinct classes.
    Returns dict.
    """
    n_edges = G.number_of_edges()
    print(f"  k={k}: {n_edges} edges — finding cliques …")

    all_cliques = list(nx.find_cliques(G))
    n_cliques_3sp = 0
    n_cliques_3cls = 0
    for clique in all_cliques:
        n_sp = len({calls[i]["species"] for i in clique})
        n_cls = len({calls[i]["class"] for i in clique})
        if n_sp >= 3:
            n_cliques_3sp += 1
        if n_cls == 3:
            n_cliques_3cls += 1

    print(
        f"         {len(all_cliques)} total cliques | "
        f"{n_cliques_3sp} with >=3 species | "
        f"{n_cliques_3cls} with 3 classes"
    )
    return dict(
        k=k,
        n_edges=n_edges,
        n_cliques_total=len(all_cliques),
        n_cliques_3sp=n_cliques_3sp,
        n_cliques_3cls=n_cliques_3cls,
    )


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #


def draw_panel_A(ax, results):
    """Horizontal bar chart with 95% CI error bars."""
    labels = [r["label"] for r in results]
    rs = np.array([r["r"] for r in results])
    ci_lo = np.array([r["ci_lo"] for r in results])
    ci_hi = np.array([r["ci_hi"] for r in results])
    colors = [r["color"] for r in results]
    n_pairs = [r["n_pairs"] for r in results]

    y_pos = np.arange(len(results))
    xerr_lo = rs - ci_lo
    xerr_hi = ci_hi - rs

    bars = ax.barh(
        y_pos,
        rs,
        xerr=[xerr_lo, xerr_hi],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        height=0.55,
        capsize=4,
        error_kw=dict(elinewidth=1.2, ecolor="#333333", capthick=1.2),
    )

    # Annotate with n_pairs
    x_max = max(ci_hi) * 1.05
    for i, (r_val, n) in enumerate(zip(rs, n_pairs)):
        ax.text(
            max(ci_hi[i], 0) + 0.003,
            y_pos[i],
            f"n={n:,}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#555",
        )

    ax.axvline(0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Pearson r (acoustic vs. semantic similarity)", fontsize=9)
    ax.set_title(
        "A. Mantel r by taxonomic class pair",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=8,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)

    # Legend patches for class colours
    legend_handles = [
        mpatches.Patch(color=CLASS_COLORS["Mammalia"], label="Mammalia"),
        mpatches.Patch(color=CLASS_COLORS["Aves"], label="Aves"),
        mpatches.Patch(color=CLASS_COLORS["Amphibia"], label="Amphibia"),
        mpatches.Patch(color=CLASS_COLORS["mixed"], label="Mixed / all"),
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=7.5,
        loc="lower right",
        framealpha=0.85,
        title="Class pair",
        title_fontsize=7.5,
    )

    # Add 95% CI note
    ax.text(
        0.01,
        -0.14,
        "Error bars: 95% bootstrap CI (n=500 resamples)",
        transform=ax.transAxes,
        fontsize=7,
        color="#666",
    )


def draw_panel_B(ax, stats_list):
    """Grouped line plot with markers for k=1,2,3."""
    ks = [s["k"] for s in stats_list]

    metrics = [
        ("n_edges",      "Edges",                 "#4C72B0", "o"),
        ("n_cliques_3sp","Cliques (>=3 species)",  "#55A868", "s"),
        ("n_cliques_3cls","Cliques (3 classes)",   "#C44E52", "^"),
    ]

    ax2 = ax.twinx()

    lines = []
    # Draw edges on primary axis (larger scale), clique counts on secondary
    for key, label, color, marker in metrics:
        values = [s[key] for s in stats_list]
        if key == "n_edges":
            ln, = ax.plot(
                ks,
                values,
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=8,
                label=label,
                zorder=3,
            )
            for x, y in zip(ks, values):
                ax.text(
                    x,
                    y + max(values) * 0.015,
                    str(y),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=color,
                )
        else:
            ln, = ax2.plot(
                ks,
                values,
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=8,
                label=label,
                linestyle="--",
                zorder=3,
            )
            for x, y in zip(ks, values):
                ax2.text(
                    x,
                    y + max(values) * 0.03 if max(values) > 0 else y + 0.5,
                    str(y),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=color,
                )
        lines.append(ln)

    ax.set_xlabel("Top-k nearest-neighbour threshold (k)", fontsize=9)
    ax.set_ylabel("Number of edges", fontsize=9, color=CLASS_COLORS["Mammalia"])
    ax2.set_ylabel("Number of cliques", fontsize=9, color="#555")
    ax.set_xticks(ks)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8, colors=CLASS_COLORS["Mammalia"])
    ax2.tick_params(axis="y", labelsize=8, colors="#555")
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)

    # Combined legend
    labels_all = [ln.get_label() for ln in lines]
    ax.legend(lines, labels_all, fontsize=8, loc="upper left", framealpha=0.85)

    ax.set_title(
        "B. SPNN graph sensitivity to top-k threshold",
        fontsize=10,
        fontweight="bold",
        loc="left",
        pad=8,
    )

    # Add note about solid=edges (left axis) vs dashed=cliques (right axis)
    ax.text(
        0.01,
        -0.14,
        "Solid line (left axis): edges; dashed lines (right axis): clique counts",
        transform=ax.transAxes,
        fontsize=7,
        color="#666",
    )


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main():
    print("Loading calls and embeddings ...")
    calls = load()
    ac_emb = np.load(ROOT / "paper_code" / "ac_emb.npy")
    se_emb = np.load(ROOT / "paper_code" / "se_emb.npy")
    print(
        f"  {len(calls)} calls, {len({c['species'] for c in calls})} species"
    )

    # L2-normalise and compute cosine similarity matrices
    ac = normalize(ac_emb.astype(float))
    se = normalize(se_emb.astype(float))
    Sa = ac @ ac.T
    Ss = se @ se.T

    # ---- Panel A --------------------------------------------------- #
    print("\nPanel A: computing Mantel r by class pair ...")
    mantel_results = compute_mantel_subsets(calls, Sa, Ss)

    # ---- Panel B --------------------------------------------------- #
    print("\nPanel B: SPNN graph sensitivity to top-k threshold ...")
    k_stats = []
    for k in [1, 2, 3]:
        print(f"\n  Building graph for k={k} ...")
        G = build_spnn_graph_topk(Sa, Ss, calls, k)
        stats = analyze_graph(G, calls, k)
        k_stats.append(stats)

    # ---- Figure ---------------------------------------------------- #
    print("\nRendering figure ...")
    fig, (ax_A, ax_B) = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=False
    )
    fig.subplots_adjust(left=0.17, right=0.93, top=0.90, bottom=0.18, wspace=0.45)

    draw_panel_A(ax_A, mantel_results)
    draw_panel_B(ax_B, k_stats)

    out = ROOT / "plots" / "fig_supp_sensitivity.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()

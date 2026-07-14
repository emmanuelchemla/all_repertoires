"""
Figure 5: Landmark vocalizations.

A "landmark" call is one that has many consistent cross-species neighbors —
calls from other species that rank in the top-k of BOTH acoustic and semantic
cosine similarity simultaneously.  These are calls where acoustic form and
semantic function co-vary across the tree of life.

Each panel ("rosette") shows one landmark call at the centre, connected to its
consistent cross-species neighbors.  Line width ∝ mean of acoustic and semantic
similarity; node color = semantic category.

Output: plots/fig5_landmark_calls.pdf
"""

import json, sys
from pathlib import Path

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

# ------------------------------------------------------------------ #
# Colour scheme (shared with other figures)
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

# ------------------------------------------------------------------ #
# Data helpers
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


def normalize(e):
    n = np.linalg.norm(e, axis=1, keepdims=True)
    n[n == 0] = 1
    return e / n


def primary_category(call):
    for kw in call.get("ontology_keywords", []):
        if kw in SEM_CATEGORY:
            return SEM_CATEGORY[kw]
    return "other"


def species_abbrev(name):
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}.\u202f{parts[1]}"
    return name[:14]


def call_short(name):
    """Truncate call name to ≤22 chars for labels."""
    if not name:
        return "?"
    return name if len(name) <= 22 else name[:21].rstrip() + "\u2026"


# ------------------------------------------------------------------ #
# Landmark scoring
# ------------------------------------------------------------------ #

def score_landmarks(ac, se, calls, k=20):
    """
    For each call i, find cross-species calls that rank in the top-k of BOTH
    acoustic and semantic similarity.  Score = number of distinct species
    represented in that consistent-neighbor set.

    Returns
    -------
    scores : (N,) int array
    neighbors : list of lists  (consistent cross-species neighbor indices)
    """
    n = len(calls)
    sp = np.array([c["species"] for c in calls])
    scores = np.zeros(n, dtype=int)
    neighbors = [[] for _ in range(n)]

    for i in range(n):
        ac_sim = ac @ ac[i]
        se_sim = se @ se[i]
        # exclude same-species calls by pushing their similarity to -2
        same = sp == sp[i]
        ac_sim = np.where(same, -2.0, ac_sim)
        se_sim = np.where(same, -2.0, se_sim)

        ac_top = set(np.argsort(ac_sim)[::-1][:k])
        se_top = set(np.argsort(se_sim)[::-1][:k])
        consistent = list(ac_top & se_top)

        neighbors[i] = consistent
        scores[i] = len({calls[j]["species"] for j in consistent})

    return scores, neighbors


def pick_diverse_landmarks(scores, neighbors, calls, n_panels=6):
    """
    Return indices of up to n_panels landmark calls, prioritising:
      1. High score (many consistent cross-species neighbors)
      2. One call per semantic category (diversity of function)
      3. One call per species (no double-counting)
    """
    order = np.argsort(scores)[::-1]
    picked, used_cats, used_species = [], set(), set()

    # First pass: one per category
    for i in order:
        if scores[i] < 2:
            break
        cat = primary_category(calls[i])
        sp_i = calls[i]["species"]
        if cat not in used_cats and sp_i not in used_species:
            picked.append(i)
            used_cats.add(cat)
            used_species.add(sp_i)
            if len(picked) == n_panels:
                return picked

    # Second pass: fill any remaining slots (relax category constraint)
    for i in order:
        if scores[i] < 2 or i in picked:
            continue
        sp_i = calls[i]["species"]
        if sp_i not in used_species:
            picked.append(i)
            used_species.add(sp_i)
            if len(picked) == n_panels:
                return picked

    return picked


# ------------------------------------------------------------------ #
# Rosette panel
# ------------------------------------------------------------------ #

def draw_rosette(ax, idx, ac_mat, se_mat, calls, nbrs):
    sp_i       = calls[idx]["species"]
    cat_i      = primary_category(calls[idx])
    center_col = CAT_COLORS.get(cat_i, "#aaa")

    ac_sim_all = ac_mat @ ac_mat[idx]
    se_sim_all = se_mat @ se_mat[idx]

    # Cross-species consistent neighbors, sorted by mean similarity
    cross = [(j, 0.5 * ac_sim_all[j] + 0.5 * se_sim_all[j])
             for j in nbrs if calls[j]["species"] != sp_i]
    cross.sort(key=lambda x: -x[1])
    top = cross[:8]

    if not top:
        ax.axis("off")
        return

    n = len(top)
    # Start at top (π/2) and go clockwise
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    R = 1.0

    for (j, sim), angle in zip(top, angles):
        x, y = R * np.cos(angle), R * np.sin(angle)
        cat_j  = primary_category(calls[j])
        color_j = CAT_COLORS.get(cat_j, "#aaa")

        # spoke
        lw = 0.5 + sim * 2.2
        ax.plot([0, x], [0, y], color="#aaaaaa", lw=lw, alpha=0.6, zorder=1,
                solid_capstyle="round")

        # neighbor node
        ax.scatter([x], [y], c=color_j, s=140, zorder=4,
                   edgecolors="black", linewidths=0.7)

        # label: species abbreviation + call name
        lbl    = species_abbrev(calls[j]["species"])
        cname  = call_short(calls[j].get("call_name", ""))
        ha  = "left"   if x >  0.12 else ("right"  if x < -0.12 else "center")
        va  = "bottom" if y >  0.12 else ("top"    if y < -0.12 else "center")
        ax.text(x * 1.38, y * 1.38, f"{lbl}\n{cname}",
                ha=ha, va=va, fontsize=5.5, linespacing=1.25, zorder=5)

    # Central landmark node (star) + call name below it
    ax.scatter([0], [0], c=center_col, s=380, marker="*", zorder=6,
               edgecolors="black", linewidths=1.5)
    center_name = call_short(calls[idx].get("call_name", ""))
    ax.text(0, -0.18, center_name, ha="center", va="top",
            fontsize=6.5, fontweight="bold", zorder=7)

    # Panel title: species + n_species
    sp_label = species_abbrev(sp_i)
    n_sp     = len({calls[j]["species"] for j in nbrs if calls[j]["species"] != sp_i})
    ax.set_title(f"{sp_label}  ({n_sp} species)",
                 fontsize=8, fontweight="bold", pad=5)

    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.75, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    calls  = load()
    ac_emb = np.load(ROOT / "paper_code" / "ac_emb.npy")
    se_emb = np.load(ROOT / "paper_code" / "se_emb.npy")
    print(f"{len(calls)} calls")

    ac = normalize(ac_emb.astype(float))
    se = normalize(se_emb.astype(float))

    print("Scoring landmark calls (k=20) …")
    scores, neighbors = score_landmarks(ac, se, calls, k=20)
    landmarks = pick_diverse_landmarks(scores, neighbors, calls, n_panels=6)

    print(f"\nTop {len(landmarks)} landmark calls selected:")
    for i in landmarks:
        kws = calls[i].get("ontology_keywords", [])
        n_sp = len({calls[j]["species"] for j in neighbors[i]
                    if calls[j]["species"] != calls[i]["species"]})
        print(f"  score={scores[i]}  {calls[i]['species']}: {kws}  "
              f"({n_sp} unique nbr-species)")

    fig = plt.figure(figsize=(14, 9.5))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.28,
                   left=0.03, right=0.97, top=0.88, bottom=0.10)

    for k_idx, i in enumerate(landmarks):
        row, col = divmod(k_idx, 3)
        ax = fig.add_subplot(gs[row, col])
        draw_rosette(ax, i, ac, se, calls, neighbors[i])

    # Shared legend
    handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in CAT_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               fontsize=8, framealpha=0.8, title="Semantic category",
               title_fontsize=8, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Landmark vocalizations: calls with consistent cross-species neighbors\n"
        "in both acoustic and semantic space  (★ = landmark call; "
        "● = consistent neighbor; line width ∝ mean similarity)",
        fontsize=10, fontweight="bold", linespacing=1.5
    )

    out = ROOT / "plots" / "fig5_landmark_calls.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()

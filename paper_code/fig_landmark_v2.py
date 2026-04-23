"""
Figure 4 v2: Cross-species groups with species-pairwise nearest-neighbor consistency.

One panel per semantic category (Cohesion, Social, Foraging, Danger).
Within each panel, calls are sorted by taxonomic class.
Selection criterion: most taxonomic classes, then most species.

Output: plots/fig4_landmark_v2.pdf
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
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.legend_handler import HandlerBase
from collections import Counter

_ICON_CACHE: dict = {}


def _load_icon(name: str):
    if name not in _ICON_CACHE:
        _ICON_CACHE[name] = plt.imread(
            str(ROOT / "paper_code" / "icons" / f"{name}.png")
        )
    return _ICON_CACHE[name]


def _place_icon(ax, name: str, x: float, y: float, zoom: float = 0.20):
    ab = AnnotationBbox(
        OffsetImage(_load_icon(name), zoom=zoom),
        (x, y),
        xycoords="axes fraction",
        frameon=False,
        box_alignment=(0.0, 0.5),
    )
    ax.add_artist(ab)


class _IconLegendHandler(HandlerBase):
    def __init__(self, img):
        self._img = img
        super().__init__()

    def create_artists(
        self,
        _legend,
        _orig_handle,
        xdescent,
        ydescent,
        _width,
        _height,
        _fontsize,
        trans,
    ):
        s = _fontsize * 2
        y_bottom = (_fontsize - s) / 2  # center on font baseline, ignore ydescent
        bb = TransformedBbox(Bbox.from_bounds(xdescent, y_bottom, s, s), trans)
        img_artist = BboxImage(bb, data=self._img, origin="upper")
        return [img_artist]


# ------------------------------------------------------------------ #
# Colour schemes
# ------------------------------------------------------------------ #

CLASS_COLORS = {
    "Mammalia": "#4C72B0",
    "Aves": "#55A868",
    "Amphibia": "#C44E52",
}
SEM_CATEGORY = {
    "alarm": "danger",
    "predator": "danger",
    "threat": "danger",
    "distress": "distress",
    "infant": "distress",
    "contact": "cohesion",
    "coordination": "cohesion",
    "long_distance": "cohesion",
    "affiliative": "cohesion",
    "aggression": "social",
    "display": "social",
    "sex": "social",
    "dominance": "social",
    "submission": "social",
    "territory": "social",
    "food": "foraging",
    "recruitment": "foraging",
}
CAT_COLORS = {
    "danger": "#d62728",
    "distress": "#ff7f0e",
    "cohesion": "#1f77b4",
    "social": "#9467bd",
    "foraging": "#2ca02c",
    "other": "#aaaaaa",
}

# Keywords that are too generic / meta to show as "shared function"
_KW_SKIP = {
    "affective",
    "sequence",
    "turn_taking",
    "sexual_selection",
    "individual_identity",
    "learning",
    "referential",
}

# Unified dark header background (avoids confusion with taxonomic-class dot colours)
_HEADER_BG = "#2d3436"
_ROUNDING = 0.03  # corner radius in axes-fraction units
_CAT_ICONS = {
    "danger": "⚠",
    "cohesion": "↔",
    "social": "⊕",
    "foraging": "★",
}

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


def primary_category(call):
    for kw in call.get("ontology_keywords", []):
        if kw in SEM_CATEGORY:
            return SEM_CATEGORY[kw]
    return "other"


def dominant_category(group, calls):
    cats = [primary_category(calls[i]) for i in group]
    return Counter(cats).most_common(1)[0][0]


def call_short(name, maxlen=40):
    if not name:
        return "—"
    return name if len(name) <= maxlen else name[: maxlen - 1].rstrip() + "…"


def desc_short(text, maxlen=88):
    if not text:
        return ""
    return text if len(text) <= maxlen else text[: maxlen - 1].rstrip() + "…"


# ------------------------------------------------------------------ #
# Species-pairwise mutual nearest-neighbour graph
# ------------------------------------------------------------------ #


def build_spnn_graph(ac_sim, se_sim, calls):
    """
    Edge (i, j) iff both the acoustic and semantic nearest neighbour
    within each species point to the same call, and the relation is mutual.
    """
    sp = np.array([c["species"] for c in calls])
    species = sorted(set(sp))
    sp_idx = {s: np.where(sp == s)[0].tolist() for s in species}

    G = nx.Graph()
    G.add_nodes_from(range(len(calls)))

    for si, sA in enumerate(species):
        idxA = sp_idx[sA]
        for sB in species[si + 1 :]:
            idxB = sp_idx[sB]
            for b in idxB:
                ac_nn_a = idxA[int(np.argmax(ac_sim[b][idxA]))]
                se_nn_a = idxA[int(np.argmax(se_sim[b][idxA]))]
                if ac_nn_a != se_nn_a:
                    continue
                a = ac_nn_a
                ac_nn_b = idxB[int(np.argmax(ac_sim[a][idxB]))]
                se_nn_b = idxB[int(np.argmax(se_sim[a][idxB]))]
                if ac_nn_b == b and se_nn_b == b:
                    w = float(0.5 * (ac_sim[a, b] + se_sim[a, b]))
                    G.add_edge(a, b, weight=w)

    return G


# ------------------------------------------------------------------ #
# Group selection — one best group per semantic category
# ------------------------------------------------------------------ #


def diversity_score(group, calls):
    sp = [calls[i]["species"] for i in group]
    if len(set(sp)) < len(group):
        return -1
    return len({calls[i]["class"] for i in group}) * 10 + len(
        {calls[i]["family"] for i in group}
    )


def find_best_per_category(
    G,
    calls,
    target_cats=("cohesion", "social", "foraging", "danger"),
    min_size=3,
    max_size=8,
):
    print("Finding cliques …")
    all_cliques = list(nx.find_cliques(G))
    print(f"  {len(all_cliques)} maximal cliques")

    result = {}
    for cat in target_cats:
        best_score = (-1, 0, 0)
        best_clique = None
        for c in all_cliques:
            if not (min_size <= len(c) <= max_size):
                continue
            d = diversity_score(c, calls)
            if d < 0:
                continue
            if dominant_category(c, calls) != cat:
                continue
            # Primary: n classes  Secondary: n species  Tertiary: diversity score
            n_cls = len({calls[i]["class"] for i in c})
            score = (n_cls, len(c), d)
            if score > best_score:
                best_score = score
                best_clique = c
        if best_clique is not None:
            result[cat] = best_clique
            n_cls = best_score[0]
            print(f"  [{cat:10s}] {len(best_clique)} calls, {n_cls} classes")

    return [result[cat] for cat in target_cats if cat in result]


# ------------------------------------------------------------------ #
# Panel drawing — compact card layout
# ------------------------------------------------------------------ #

_CLS_ORDER = {"Amphibia": 0, "Aves": 1, "Mammalia": 2}


def draw_card(ax, group, calls):
    cat = dominant_category(group, calls)
    cat_col = CAT_COLORS.get(cat, "#555")

    # Sort calls: Amphibia → Aves → Mammalia, then alphabetically
    order = sorted(
        group,
        key=lambda i: (
            _CLS_ORDER.get(calls[i].get("class", ""), 9),
            calls[i].get("species", ""),
        ),
    )
    n = len(order)
    n_cls = len({calls[i]["class"] for i in order})
    n_fam = len({calls[i]["family"] for i in order})

    # ---- card chrome ------------------------------------------------
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    # Drop shadow (offset slightly down-right)
    ax.add_patch(
        FancyBboxPatch(
            (0.020, 0.002),
            0.980,
            0.982,
            boxstyle=f"round,pad=0,rounding_size={_ROUNDING}",
            facecolor="#555555",
            edgecolor="none",
            alpha=0.18,
            zorder=0,
            transform=ax.transAxes,
            clip_on=False,
        )
    )

    # White card background with light border
    ax.add_patch(
        FancyBboxPatch(
            (0.008, 0.012),
            0.984,
            0.980,
            boxstyle=f"round,pad=0,rounding_size={_ROUNDING}",
            facecolor="white",
            edgecolor="#d0d0d0",
            linewidth=0.9,
            zorder=1,
            transform=ax.transAxes,
            clip_on=False,
        )
    )

    # Dark header band — rounded top to match card, flat bottom
    ax.add_patch(
        FancyBboxPatch(
            (0.008, 0.908),
            0.984,
            0.084,
            boxstyle=f"round,pad=0,rounding_size={_ROUNDING}",
            facecolor=_HEADER_BG,
            edgecolor="none",
            zorder=2,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    # Fill rounded-bottom corners of the header to make bottom edge flat
    ax.axhspan(
        0.908,
        0.908 + _ROUNDING + 0.002,
        xmin=0.008,
        xmax=0.992,
        color=_HEADER_BG,
        zorder=2,
    )

    # ---- header content ---------------------------------------------
    ax.text(
        0.50,
        0.963,
        cat.upper(),
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="white",
        transform=ax.transAxes,
        zorder=3,
    )
    fam_str = f"{n_fam} famil{'ies' if n_fam > 1 else 'y'}"
    cls_str = f"{n_cls} class{'es' if n_cls > 1 else ''}"
    ax.text(
        0.50,
        0.933,
        f"{n} species · {fam_str} · {cls_str}",
        ha="center",
        va="center",
        fontsize=8,
        color="#aaaaaa",
        transform=ax.transAxes,
        zorder=3,
    )

    # ---- call rows --------------------------------------------------
    y_top = 0.900
    y_bot = 0.016
    slot = (y_top - y_bot) / n
    line_h = min(slot * 0.27, 0.055)

    for k, idx in enumerate(order):
        yc = y_top - k * slot - slot / 2
        cls = calls[idx].get("class", "")
        color = CLASS_COLORS.get(cls, "#aaa")
        y1 = yc + line_h
        y2 = yc
        y3 = yc - line_h

        # Coloured class dot
        ax.scatter(
            [0.025],
            [y1],
            c=color,
            s=100,
            zorder=4,
            edgecolors="white",
            linewidths=0.7,
            transform=ax.transAxes,
            clip_on=False,
        )

        # Line 1: species — call name
        sp = calls[idx].get("species", "")
        cn = call_short(calls[idx].get("call_name", ""), maxlen=30)
        ax.text(
            0.068,
            y1,
            f"{sp}  —  {cn}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#111",
            transform=ax.transAxes,
        )

        # Line 2: Sem
        kws = [
            kw.replace("_", " ")
            for kw in calls[idx].get("ontology_keywords", [])
            if kw not in _KW_SKIP
        ]
        sem_str = ", ".join(kws) if kws else "—"
        ax.text(
            0.068,
            y2,
            f"Sem: {sem_str}",
            ha="left",
            va="center",
            fontsize=8.5,
            color="#333",
            transform=ax.transAxes,
        )

        # Line 3: Acc
        ac = calls[idx].get("acoustic_description") or "—"
        ax.text(
            0.068,
            y3,
            f"Acc: {desc_short(ac, maxlen=88)}",
            ha="left",
            va="center",
            fontsize=8,
            color="#666",
            style="italic",
            transform=ax.transAxes,
        )


# ------------------------------------------------------------------ #
# Narrow card — horizontal (1 × 4) layout
# ------------------------------------------------------------------ #


def _common_name(species_name):
    """Return just the common name, dropping '(Scientific name)' suffix."""
    return species_name.split("(")[0].strip() if "(" in species_name else species_name


def draw_card_narrow(ax, group, calls):
    cat = dominant_category(group, calls)
    cat_col = CAT_COLORS.get(cat, "#555")

    order = sorted(
        group,
        key=lambda i: (
            _CLS_ORDER.get(calls[i].get("class", ""), 9),
            calls[i].get("species", ""),
        ),
    )
    n = len(order)
    n_cls = len({calls[i]["class"] for i in order})
    n_fam = len({calls[i]["family"] for i in order})

    # ---- card chrome (same as 2×2 version) ----
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.add_patch(
        FancyBboxPatch(
            (0.020, 0.002),
            0.980,
            0.982,
            boxstyle=f"round,pad=0,rounding_size={_ROUNDING}",
            facecolor="#555555",
            edgecolor="none",
            alpha=0.18,
            zorder=0,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.008, 0.012),
            0.984,
            0.980,
            boxstyle=f"round,pad=0,rounding_size={_ROUNDING}",
            facecolor="white",
            edgecolor="#d0d0d0",
            linewidth=0.9,
            zorder=1,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.008, 0.908),
            0.984,
            0.084,
            boxstyle=f"round,pad=0,rounding_size={_ROUNDING}",
            facecolor=_HEADER_BG,
            edgecolor="none",
            zorder=2,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.axhspan(
        0.908,
        0.908 + _ROUNDING + 0.002,
        xmin=0.008,
        xmax=0.992,
        color=_HEADER_BG,
        zorder=2,
    )

    # ---- header text (compact, no icon) ----
    ax.text(
        0.50,
        0.963,
        cat.upper(),
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color="white",
        transform=ax.transAxes,
        zorder=3,
    )
    ax.text(
        0.50,
        0.933,
        f"{n} species · {n_fam} families · {n_cls} classes",
        ha="center",
        va="center",
        fontsize=14,
        color="white",
        transform=ax.transAxes,
        zorder=3,
    )

    # ---- call rows (4 lines each, top-aligned) ----
    # Convert absolute point values → axes fraction
    fig = ax.get_figure()
    ax_h_pt = ax.get_position().height * fig.get_size_inches()[1] * 72

    def pt(p):
        return p / ax_h_pt

    lh_pt = 16  # line spacing within a call (points)
    call_gap_pt = 32  # blank gap between call blocks (points)

    lh = pt(lh_pt)
    call_gap = pt(call_gap_pt)
    y_cur = 0.88  # start just below the header (axes fraction)

    indent1 = 0.05
    indent2 = 0.17
    indent3 = 0.25

    for k, idx in enumerate(order):
        cls = calls[idx].get("class", "")

        y1 = y_cur  # common name
        y2 = y_cur - lh  # call name
        y3 = y_cur - 2 * lh  # semantic line
        y4 = y_cur - 3 * lh  # acoustic line

        # Line 1: [indent1] class icon  [indent2] common name (bold)
        cls_icon = {
            "Mammalia": "cls_mammalia",
            "Aves": "cls_aves",
            "Amphibia": "cls_amphibia",
        }.get(cls, "cls_mammalia")
        _place_icon(ax, cls_icon, indent1, y1, zoom=0.12)
        ax.text(
            indent2,
            y1,
            _common_name(calls[idx].get("species", "")),
            ha="left",
            va="center",
            fontsize=16,
            fontweight="bold",
            color="#111",
            transform=ax.transAxes,
        )

        # Line 2: [indent2] call name
        cn = call_short(calls[idx].get("call_name", ""), maxlen=36)
        ax.text(
            indent2,
            y2,
            cn,
            ha="left",
            va="center",
            fontsize=14,
            color="#444",
            transform=ax.transAxes,
        )

        # Line 3: [indent2] bubble icon  [indent3] semantic keywords
        kws = [
            kw.replace("_", " ")
            for kw in calls[idx].get("ontology_keywords", [])
            if kw not in _KW_SKIP
        ]
        sem_str = ", ".join(kws) if kws else "—"
        _place_icon(ax, "sem_bubble", indent2, y3, zoom=0.07)
        ax.text(
            indent3,
            y3,
            desc_short(sem_str, maxlen=38),
            ha="left",
            va="center",
            fontsize=13,
            color="#333",
            transform=ax.transAxes,
        )

        # Line 4: [indent2] speaker icon  [indent3] acoustic description
        ac = calls[idx].get("acoustic_description") or "—"
        _place_icon(ax, "acc_speaker", indent2, y4, zoom=0.07)
        ax.text(
            indent3,
            y4,
            desc_short(ac, maxlen=38),
            ha="left",
            va="center",
            fontsize=13,
            color="#555",
            style="italic",
            transform=ax.transAxes,
        )

        y_cur = y4 - call_gap


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main():
    calls = load()
    ac_emb = np.load(ROOT / "paper_code" / "ac_emb.npy")
    se_emb = np.load(ROOT / "paper_code" / "se_emb.npy")
    print(f"{len(calls)} calls from {len({c['species'] for c in calls})} species")

    ac = normalize(ac_emb.astype(float))
    se = normalize(se_emb.astype(float))
    ac_sim = ac @ ac.T
    se_sim = se @ se.T

    print("Building species-pairwise NN graph …")
    G = build_spnn_graph(ac_sim, se_sim, calls)
    print(f"  {G.number_of_edges()} edges")

    groups = find_best_per_category(
        G,
        calls,
        target_cats=("cohesion", "social", "foraging", "danger"),
    )

    print(f"\n{len(groups)} groups selected:")
    for g in groups:
        cat = dominant_category(g, calls)
        cls = {calls[i]["class"] for i in g}
        fam = {calls[i]["family"] for i in g}
        print(f"  [{cat}]  {len(g)} calls  classes={cls}  families={fam}")
        for i in g:
            print(
                f"    {calls[i]['species']:48s}  "
                f"{calls[i].get('call_name', ''):32s}  "
                f"{calls[i].get('ontology_keywords', [])}"
            )

    # ---- figure: 2 rows × 2 cols ----
    fig = plt.figure(figsize=(15, 11.3))
    gs = GridSpec(
        2,
        2,
        figure=fig,
        hspace=0.10,
        wspace=0.08,
        left=0.02,
        right=0.98,
        top=0.95,
        bottom=0.06,
    )

    for k, group in enumerate(groups):
        row, col = divmod(k, 2)
        ax = fig.add_subplot(gs[row, col])
        draw_card(ax, group, calls)

    # ---- legend: taxonomic class only ----
    cls_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=col,
            markersize=9,
            markeredgecolor="#eeeeee",
            markeredgewidth=0.5,
            label=cls,
        )
        for cls, col in CLASS_COLORS.items()
    ]
    fig.legend(
        handles=cls_handles,
        loc="lower center",
        ncol=3,
        fontsize=14,
        framealpha=0.92,
        frameon=True,
        title="Taxonomic class (dot colour)",
        title_fontsize=14,
        bbox_to_anchor=(0.5, 0.004),
        handletextpad=0.4,
        columnspacing=1.2,
    )

    out = ROOT / "plots" / "fig4_landmark_v2.pdf"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    # ---- horizontal figure: 1 row × 4 cols ----
    fig2 = plt.figure(figsize=(20, 10))
    gs2 = GridSpec(
        1,
        4,
        figure=fig2,
        hspace=0.0,
        wspace=0.07,
        left=0.01,
        right=0.99,
        top=0.97,
        bottom=0.06,
    )

    for k, group in enumerate(groups):
        ax2 = fig2.add_subplot(gs2[0, k])
        draw_card_narrow(ax2, group, calls)

    _CLS_ICON_NAMES = {
        "Mammalia": "cls_mammalia",
        "Aves": "cls_aves",
        "Amphibia": "cls_amphibia",
    }
    cls_handles2 = [mpatches.Patch(label=cls, alpha=0) for cls in CLASS_COLORS]
    cls_handler_map = {
        h: _IconLegendHandler(_load_icon(_CLS_ICON_NAMES[h.get_label()]))
        for h in cls_handles2
    }
    fig2.legend(
        handles=cls_handles2,
        handler_map=cls_handler_map,
        loc="lower center",
        ncol=3,
        fontsize=14,
        framealpha=0.92,
        frameon=True,
        title="Taxonomic class",
        title_fontsize=14,
        bbox_to_anchor=(0.5, 0.004),
        handletextpad=0.6,
        handleheight=2,
    )

    out2 = ROOT / "plots" / "fig4_landmark_h.pdf"
    fig2.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {out2}")


if __name__ == "__main__":
    main()

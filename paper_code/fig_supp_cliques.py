"""
Supplementary figure: distribution of all maximal SPNN cliques.

Outputs:
  paper_code/all_cliques.csv          — CSV of all cliques (size >= 2, distinct species)
  plots/fig_supp_cliques.pdf          — summary figure (3 panels)
  paper_code/supp_cliques_table.tex   — longtable LaTeX fragment
"""

import json
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
try:
    from paper_code.paths import DATABASE_PATH
except ModuleNotFoundError:
    from paths import DATABASE_PATH
sys.path.insert(0, str(ROOT))

import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# ------------------------------------------------------------------ #
# Colour scheme (consistent with fig_landmark_v2.py)
# ------------------------------------------------------------------ #

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

# ------------------------------------------------------------------ #
# Data helpers
# ------------------------------------------------------------------ #


def load():
    with open(DATABASE_PATH) as f:
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


# ------------------------------------------------------------------ #
# SPNN graph (identical to fig_landmark_v2.py)
# ------------------------------------------------------------------ #


def build_spnn_graph(ac_sim, se_sim, calls):
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
                    G.add_edge(a, b, weight=float(0.5 * (ac_sim[a, b] + se_sim[a, b])))

    return G


# ------------------------------------------------------------------ #
# Clique filtering & annotation
# ------------------------------------------------------------------ #


CLASS_ORDER_IDX = {"Amphibia": 0, "Aves": 1, "Mammalia": 2}


def _sci_name(species_name):
    """Extract scientific name from 'Common name (Genus species)' or return as-is."""
    import re
    m = re.search(r'\(([^)]+)\)', species_name)
    return m.group(1) if m else species_name


def annotate_clique(clique, calls):
    """Return a dict with derived properties for one clique."""
    species_set = {calls[i]["species"] for i in clique}
    if len(species_set) < len(clique):
        return None
    n_classes = len({calls[i]["class"] for i in clique})
    n_families = len({calls[i]["family"] for i in clique})
    cat = dominant_category(clique, calls)
    # Per-member data: species_name -> (class, call_name), sorted by class then name
    members = {}
    for i in clique:
        sp = calls[i]["species"]
        members[sp] = (calls[i]["class"], calls[i].get("call_name", "") or "")
    members_sorted = sorted(
        members.items(),
        key=lambda kv: (CLASS_ORDER_IDX.get(kv[1][0], 9), kv[0]),
    )
    return {
        "clique": clique,
        "size": len(clique),
        "category": cat,
        "n_classes": n_classes,
        "n_families": n_families,
        "species": [sp for sp, _ in members_sorted],
        "calls": [call for _, (_, call) in members_sorted],
        "members": members_sorted,  # [(species_name, (class, call_name)), ...]
    }


def collect_cliques(G, calls, min_size=2):
    print("Finding all maximal cliques …")
    raw = list(nx.find_cliques(G))
    print(f"  {len(raw)} raw maximal cliques found")

    annotated = []
    for c in raw:
        if len(c) < min_size:
            continue
        info = annotate_clique(c, calls)
        if info is not None:
            annotated.append(info)

    # Sort: n_classes desc, n_families desc, size desc
    annotated.sort(key=lambda x: (-x["n_classes"], -x["n_families"], -x["size"]))
    print(f"  {len(annotated)} cliques with size >= {min_size} and distinct species")
    return annotated


# ------------------------------------------------------------------ #
# CSV output
# ------------------------------------------------------------------ #


def save_csv(cliques, path):
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n_classes", "n_families", "n_species", "category", "species", "calls"],
        )
        writer.writeheader()
        for info in cliques:
            writer.writerow(
                {
                    "n_classes": info["n_classes"],
                    "n_families": info["n_families"],
                    "n_species": info["size"],
                    "category": info["category"],
                    "species": ";".join(info["species"]),
                    "calls": ";".join(info["calls"]),
                }
            )
    print(f"Saved CSV → {path}")


# ------------------------------------------------------------------ #
# Summary figure
# ------------------------------------------------------------------ #


def make_figure(cliques, out_path):
    cats_all = sorted(CAT_COLORS.keys())  # stable ordering for legend

    # ---- Panel A: histogram of clique sizes, coloured by dominant category ----
    sizes = [info["size"] for info in cliques]
    max_size = max(sizes) if sizes else 8
    size_range = range(2, max_size + 1)

    # For each (size, category) count cliques
    size_cat: dict[int, Counter] = {}
    for info in cliques:
        s = info["size"]
        c = info["category"]
        size_cat.setdefault(s, Counter())[c] += 1

    # ---- Panel B: stacked bar of n_classes per clique, grouped by category ----
    cat_cls_count: dict[str, Counter] = {cat: Counter() for cat in cats_all}
    for info in cliques:
        cat_cls_count[info["category"]][info["n_classes"]] += 1

    # ---- Panel C: top-15 cliques as a table ----
    top15 = cliques[:15]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12, wspace=0.35)

    # ---- Panel A ----
    ax = axes[0]
    bottoms = {s: 0 for s in size_range}
    for cat in cats_all:
        heights = [size_cat.get(s, Counter()).get(cat, 0) for s in size_range]
        ax.bar(
            list(size_range),
            heights,
            bottom=[bottoms[s] for s in size_range],
            color=CAT_COLORS[cat],
            label=cat,
            edgecolor="white",
            linewidth=0.5,
        )
        for s, h in zip(size_range, heights):
            bottoms[s] += h

    ax.set_xlabel("Clique size (# species)", fontsize=11)
    ax.set_ylabel("Number of cliques", fontsize=11)
    ax.set_title("A — Clique size distribution", fontsize=12, fontweight="bold", loc="left")
    ax.set_xticks(list(size_range))
    ax.legend(fontsize=8, title="Category", title_fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel B ----
    ax = axes[1]
    all_n_classes = sorted({nc for info in cliques for nc in [info["n_classes"]]})
    x = np.arange(len(cats_all))
    width = 0.65 / max(len(all_n_classes), 1)
    cmap = plt.cm.Blues
    nc_colors = {nc: cmap(0.35 + 0.55 * i / max(len(all_n_classes) - 1, 1))
                 for i, nc in enumerate(all_n_classes)}

    bottom_arr = np.zeros(len(cats_all))
    for nc in all_n_classes:
        heights = [cat_cls_count[cat].get(nc, 0) for cat in cats_all]
        ax.bar(
            x,
            heights,
            bottom=bottom_arr,
            color=nc_colors[nc],
            edgecolor="white",
            linewidth=0.5,
            label=f"{nc} class{'es' if nc > 1 else ''}",
        )
        bottom_arr += np.array(heights)

    ax.set_xticks(x)
    ax.set_xticklabels(cats_all, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of cliques", fontsize=11)
    ax.set_title("B — Taxonomic diversity per category", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=8, title="# classes", title_fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Panel C: table of top-15 cliques ----
    ax = axes[2]
    ax.set_axis_off()
    ax.set_title("C — Top 15 cliques", fontsize=12, fontweight="bold", loc="left")

    col_labels = ["Cat", "N", "Cls", "Fam", "Species (abbreviated)"]
    table_data = []
    row_colors = []
    for info in top15:
        sp_str = ", ".join(sp.split("(")[0].strip().split()[-1] for sp in info["species"])
        if len(sp_str) > 38:
            sp_str = sp_str[:37] + "…"
        table_data.append([
            info["category"],
            str(info["size"]),
            str(info["n_classes"]),
            str(info["n_families"]),
            sp_str,
        ])
        base = CAT_COLORS.get(info["category"], "#aaaaaa")
        row_colors.append([base + "33"] * 5)  # 20% alpha hex approximation

    the_table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(7.5)
    the_table.auto_set_column_width([0, 1, 2, 3, 4])

    # Style header row
    for j in range(len(col_labels)):
        cell = the_table[0, j]
        cell.set_facecolor("#2d3436")
        cell.set_text_props(color="white", fontweight="bold")

    # Style data rows with category colour tint
    for row_i, info in enumerate(top15):
        base = CAT_COLORS.get(info["category"], "#aaaaaa")
        for j in range(len(col_labels)):
            cell = the_table[row_i + 1, j]
            cell.set_facecolor(base + "22")

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {out_path}")


# ------------------------------------------------------------------ #
# LaTeX fragment
# ------------------------------------------------------------------ #


def _tex_escape(text):
    """Escape LaTeX special characters in a string."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


_CLASS_ICON_TEX = {
    "Amphibia": r"\includegraphics[height=0.9em]{../paper_code/icons/cls_amphibia.png}",
    "Aves":     r"\includegraphics[height=0.9em]{../paper_code/icons/cls_aves.png}",
    "Mammalia": r"\includegraphics[height=0.9em]{../paper_code/icons/cls_mammalia.png}",
}

_HEADER = r"Classes & Families & Species & Category & Members \\"
_COL_SPEC = r"{ccclp{9.5cm}}"


def save_latex(cliques, path):
    path = Path(path)
    lines = []
    lines.append(r"% Auto-generated by fig_supp_cliques.py — do not edit by hand")
    lines.append(rf"\begin{{longtable}}{_COL_SPEC}")
    # Caption + label (first-page header)
    lines.append(
        r"\caption{All cross-species call groups (maximal cliques in the mutual nearest-neighbour"
        r" graph with $\geq 2$ distinct species). Sorted by number of taxonomic classes, then"
        r" families, then group size. Icons denote taxonomic class"
        r" (\protect\includegraphics[height=0.9em]{../paper_code/icons/cls_amphibia.png}~Amphibia,"
        r" \protect\includegraphics[height=0.9em]{../paper_code/icons/cls_aves.png}~Aves,"
        r" \protect\includegraphics[height=0.9em]{../paper_code/icons/cls_mammalia.png}~Mammalia).}"
        r"\label{tab:supp_cliques}\\"
    )
    lines.append(r"\toprule")
    lines.append(_HEADER)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    # Continuation header
    lines.append(r"\multicolumn{5}{l}{\textit{(continued from previous page)}} \\")
    lines.append(r"\toprule")
    lines.append(_HEADER)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{r}{\textit{(continued on next page)}} \\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for info in cliques:
        # Build members cell: one line per species
        member_lines = []
        for sp, (cls, call_name) in info["members"]:
            icon = _CLASS_ICON_TEX.get(cls, "")
            sci = _tex_escape(_sci_name(sp))
            call = _tex_escape(call_name)
            member_lines.append(
                rf"{icon}~\textit{{{sci}}} ({call})" if call else rf"{icon}~\textit{{{sci}}}"
            )
        members_cell = r"\newline ".join(member_lines)

        cat = _tex_escape(info["category"])
        row = (
            f"{info['n_classes']} & {info['n_families']} & {info['size']} & "
            f"{cat} & {members_cell} \\\\"
        )
        lines.append(row)

    lines.append(r"\end{longtable}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved LaTeX → {path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #


def main():
    # --- Load data ---
    calls = load()
    print(f"Loaded {len(calls)} calls from {len({c['species'] for c in calls})} species")

    ac_emb = np.load(ROOT / "paper_code" / "ac_emb.npy")
    se_emb = np.load(ROOT / "paper_code" / "se_emb.npy")
    print(f"Embeddings: ac={ac_emb.shape}, se={se_emb.shape}")

    ac = normalize(ac_emb.astype(float))
    se = normalize(se_emb.astype(float))
    ac_sim = ac @ ac.T
    se_sim = se @ se.T

    # --- Build graph ---
    print("Building SPNN graph …")
    G = build_spnn_graph(ac_sim, se_sim, calls)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- Find and filter cliques ---
    cliques = collect_cliques(G, calls, min_size=2)

    # --- Size distribution summary ---
    size_counter = Counter(info["size"] for info in cliques)
    print("\nClique size distribution:")
    for s in sorted(size_counter):
        print(f"  size {s}: {size_counter[s]} cliques")

    cat_counter = Counter(info["category"] for info in cliques)
    print("\nCategory distribution:")
    for cat, n in cat_counter.most_common():
        print(f"  {cat}: {n}")

    print(f"\nTop 5 cliques:")
    for i, info in enumerate(cliques[:5], 1):
        print(
            f"  #{i}: size={info['size']}, cat={info['category']}, "
            f"classes={info['n_classes']}, families={info['n_families']}, "
            f"species={info['species']}"
        )

    # --- Save outputs ---
    save_csv(cliques, ROOT / "paper_code" / "all_cliques.csv")
    make_figure(cliques, ROOT / "plots" / "fig_supp_cliques.pdf")
    save_latex(cliques, ROOT / "paper_code" / "supp_cliques_table.tex")

    print("\nDone.")


if __name__ == "__main__":
    main()

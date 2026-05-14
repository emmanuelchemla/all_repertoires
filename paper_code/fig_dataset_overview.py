"""
Figure 1: Dataset overview — 4 panels.
  (A) Semantic keyword frequency, colored by communicative category
  (B) Acoustic keyword frequency, extracted from free-text descriptions
  (C) Taxonomic breakdown (class × order)
  (D) Calls per species, colored by class

Output: plots/fig1_dataset_overview.png
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ------------------------------------------------------------------ #
# Colour palette
# ------------------------------------------------------------------ #

CLASS_COLORS = {
    "Mammalia":  "#4C72B0",
    "Aves":      "#55A868",
    "Amphibia":  "#C44E52",
}

# Semantic keyword → broad category for colouring panel A
SEM_CATEGORY = {
    "alarm":     "danger",   "predator": "danger",  "threat": "danger",
    "distress":  "distress", "infant":   "distress",
    "contact":   "cohesion", "coordination": "cohesion",
    "long_distance": "cohesion", "affiliative": "cohesion",
    "aggression":"social",   "display":  "social",  "sex": "social",
    "dominance": "social",   "submission":"social",  "territory": "social",
    "food":      "foraging", "recruitment":"foraging",
    "learning":  "other",    "sequence": "other",   "individual_identity": "other",
    "referential":"other",   "syntax":   "other",   "turn_taking": "other",
    "group_identity": "other",
}

CAT_COLORS = {
    "danger":   "#d62728",
    "distress": "#ff7f0e",
    "cohesion": "#1f77b4",
    "social":   "#9467bd",
    "foraging": "#2ca02c",
    "other":    "#7f7f7f",
}

# Acoustic patterns → display label + colour category
ACOUSTIC_PATTERNS = [
    ("high-frequency",     ["high-frequency","high frequency","high-pitched","high pitched","ultrasonic"],        "frequency"),
    ("low-frequency",      ["low-frequency","low frequency","low-pitched","low pitched","infrasound"],            "frequency"),
    ("tonal",              ["tonal","pure tone","narrowband","narrow-band","sinusoidal"],                          "spectral"),
    ("broadband",          ["broadband","broad-band","noisy","wideband","atonal"],                                 "spectral"),
    ("freq-modulated",     ["frequency-modulated","frequency modulated","fm sweep","sweep","upsweep","downsweep"], "spectral"),
    ("harmonic",           ["harmonic","overtone","formant"],                                                      "spectral"),
    ("pulsed",             ["pulsed","pulse","click","burst","staccato"],                                          "temporal"),
    ("repetitive",         ["repetitive","repeated","series","bout","sequence of"],                                "temporal"),
    ("short",              ["short ","brief","abrupt"],                                                            "temporal"),
    ("long / sustained",   ["long ","prolonged","sustained","extended"],                                           "temporal"),
    ("loud",               ["loud","intense","powerful","far-carrying"],                                           "amplitude"),
    ("soft / quiet",       ["soft ","quiet","low amplitude","low-amplitude"],                                      "amplitude"),
]

AC_CAT_COLORS = {
    "frequency": "#e377c2",
    "spectral":  "#bcbd22",
    "temporal":  "#17becf",
    "amplitude": "#8c564b",
}

# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #

def load():
    with open(ROOT / "database.json") as f:
        db = json.load(f)
    species_list = db["species"]
    calls = []
    for s in species_list:
        for c in s.get("calls", []):
            calls.append({**c,
                          "species": s["species_name"],
                          "class":   s.get("class", ""),
                          "order":   s.get("order", ""),
                          "family":  s.get("family", "")})
    return species_list, calls

# ------------------------------------------------------------------ #
# Panel A – semantic keywords
# ------------------------------------------------------------------ #

def panel_A(ax, calls, top_n=16):
    kw_counts = Counter(kw for c in calls for kw in c.get("ontology_keywords", []))
    top = kw_counts.most_common(top_n)
    labels = [k.replace("_", " ") for k, _ in top]
    values = [v for _, v in top]
    cats   = [SEM_CATEGORY.get(k, "other") for k, _ in top]
    colors = [CAT_COLORS[cat] for cat in cats]

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.7)
    ax.set_xlabel("Number of calls", fontsize=10)
    ax.set_title("(A)  Semantic functions", fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(values) * 1.15)
    for bar, val in zip(bars[::-1], values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=7.5)

    legend_patches = [mpatches.Patch(color=c, label=lbl)
                      for lbl, c in CAT_COLORS.items()]
    ax.legend(handles=legend_patches, fontsize=7.5, loc="lower right",
              framealpha=0.8, title="category", title_fontsize=7.5)

# ------------------------------------------------------------------ #
# Panel B – acoustic keywords
# ------------------------------------------------------------------ #

def panel_B(ax, calls, top_n=12):
    counts = {}
    cats   = {}
    for label, patterns, cat in ACOUSTIC_PATTERNS:
        c = sum(1 for call in calls
                if any(p in call.get("acoustic_description","").lower()
                       for p in patterns))
        counts[label] = c
        cats[label]   = cat

    items = sorted(counts.items(), key=lambda x: -x[1])[:top_n]
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    colors = [AC_CAT_COLORS[cats[k]] for k in labels]

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.7)
    ax.set_xlabel("Number of calls", fontsize=10)
    ax.set_title("(B)  Acoustic features", fontsize=11, fontweight="bold", pad=6)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(values) * 1.15)
    for bar, val in zip(bars[::-1], values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=7.5)

    legend_patches = [mpatches.Patch(color=c, label=lbl)
                      for lbl, c in AC_CAT_COLORS.items()]
    ax.legend(handles=legend_patches, fontsize=7.5, loc="lower right",
              framealpha=0.8, title="feature type", title_fontsize=7.5)

# ------------------------------------------------------------------ #
# Panel CD – calls per species, grouped by taxonomic group
# ------------------------------------------------------------------ #

# Ordered groups: (display label, family membership test)
GROUPS = [
    ("Apes",                 lambda s: s.get("family") in {"Hominidae", "Hylobatidae"}),
    ("Old World\nmonkeys",   lambda s: s.get("family") == "Cercopithecidae"),
    ("Other\nprimates",      lambda s: s.get("order") == "Primates"
                                        and s.get("family") not in {"Hominidae","Hylobatidae","Cercopithecidae"}),
    ("Carnivora",            lambda s: s.get("order") == "Carnivora"),
    ("Other\nmammals",       lambda s: s.get("class") == "Mammalia"
                                        and s.get("order") not in {"Primates","Carnivora"}),
    ("Passeriformes",        lambda s: s.get("order") == "Passeriformes"),
    ("Psittaciformes",       lambda s: s.get("order") == "Psittaciformes"),
    ("Amphibia",             lambda s: s.get("class") == "Amphibia"),
]

# One colour per family
FAMILY_COLORS = {
    "Hominidae":        "#2166ac",
    "Hylobatidae":      "#74add1",
    "Cercopithecidae":  "#4393c3",
    "Lemuridae":        "#92c5de",
    "Indriidae":        "#d1e5f0",
    "Cebidae":          "#b2abd2",
    "Callitrichidae":   "#8073ac",
    "Atelidae":         "#542788",
    "Cheirogaleidae":   "#c2a5cf",
    "Herpestidae":      "#f4a582",
    "Canidae":          "#d6604d",
    "Hyaenidae":        "#b2182b",
    "Pteropodidae":     "#fdae61",
    "Emballonuridae":   "#fee090",
    "Elephantidae":     "#878787",
    "Paridae":          "#1a9850",
    "Estrildidae":      "#66bd63",
    "Troglodytidae":    "#a6d96a",
    "Ploceidae":        "#d9ef8b",
    "Sturnidae":        "#006837",
    "Pomatostomidae":   "#31a354",
    "Psittacidae":      "#756bb1",
    "Dendrobatidae":    "#e08214",
    "Hylidae":          "#fdb863",
    "Leptodactylidae":  "#b35806",
    "Ranidae":          "#7f3b08",
}

GROUP_BG = ["#f7f7f7", "#ffffff"]  # alternating backgrounds


def panel_CD(ax, species_list):
    # Assign each species to a group
    grouped = {g: [] for g, _ in GROUPS}
    for s in species_list:
        for g_label, test in GROUPS:
            if test(s):
                grouped[g_label].append(s)
                break

    # Build ordered bar data
    bars_data = []   # (x_pos, n_calls, color, short_name)
    group_spans = [] # (g_label, x_start, x_end)
    x = 0
    for g_idx, (g_label, _) in enumerate(GROUPS):
        species_in_group = sorted(grouped[g_label], key=lambda s: -len(s.get("calls",[])))
        if not species_in_group:
            continue
        x_start = x
        for s in species_in_group:
            n = len(s.get("calls", []))
            family = s.get("family", "")
            color = FAMILY_COLORS.get(family, "#aaaaaa")
            name = s["species_name"].split("(")[0].strip()
            # shorten long names
            parts = name.split()
            short = parts[0] if len(parts) == 1 else f"{parts[0][0]}. {' '.join(parts[1:])}"
            bars_data.append((x, n, color, short, family))
            x += 1
        group_spans.append((g_label, x_start, x - 1))
        x += 0.8  # gap between groups

    xs     = [d[0] for d in bars_data]
    values = [d[1] for d in bars_data]
    colors = [d[2] for d in bars_data]
    names  = [d[3] for d in bars_data]

    # Alternating background bands
    for i, (g_label, x0, x1) in enumerate(group_spans):
        ax.axvspan(x0 - 0.5, x1 + 0.5, color=GROUP_BG[i % 2], zorder=0)

    ax.bar(xs, values, color=colors, width=0.75, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=90, fontsize=6, ha="center")
    ax.set_ylabel("Calls in database", fontsize=10)
    ax.set_title("(C)  Calls per species by taxonomic group", fontsize=11,
                 fontweight="bold", pad=18)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.8, max(xs) + 0.8)
    ax.set_ylim(0, max(values) * 1.22)

    # Group labels above bars
    for g_label, x0, x1 in group_spans:
        mid = (x0 + x1) / 2
        top = ax.get_ylim()[1]
        ax.text(mid, top * 0.97, g_label.replace("\n", " "),
                ha="center", va="top", fontsize=8.5, fontweight="bold", color="#333")
        # bracket line
        ax.annotate("", xy=(x1 + 0.4, top * 0.90), xytext=(x0 - 0.4, top * 0.90),
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=1.0))

    # Family colour legend (compact, two columns)
    seen_families = list(dict.fromkeys(d[4] for d in bars_data))
    legend_patches = [mpatches.Patch(color=FAMILY_COLORS.get(f, "#aaa"), label=f)
                      for f in seen_families]
    ax.legend(handles=legend_patches, fontsize=6.5, loc="upper right",
              framealpha=0.85, ncol=2, title="Family", title_fontsize=7)

# ------------------------------------------------------------------ #
# Compose figure
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots")
    args = parser.parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    species_list, calls = load()
    print(f"{len(calls)} calls, {len(species_list)} species")

    fig = plt.figure(figsize=(16, 11))
    gs = GridSpec(2, 2, figure=fig,
                  hspace=0.55, wspace=0.35,
                  left=0.07, right=0.97, top=0.95, bottom=0.18)

    ax_A  = fig.add_subplot(gs[0, 0])
    ax_B  = fig.add_subplot(gs[0, 1])
    ax_CD = fig.add_subplot(gs[1, :])   # merged bottom row

    panel_A(ax_A, calls)
    panel_B(ax_B, calls)
    panel_CD(ax_CD, species_list)

    out = out_dir / "fig1_dataset_overview.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()

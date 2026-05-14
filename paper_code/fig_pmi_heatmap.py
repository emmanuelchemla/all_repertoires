"""
Generate Figure 3: PMI heatmap (acoustic × semantic keywords).

Design:
- Acoustic features in meaningful groups (frequency, duration, amplitude, texture),
  clearest groups at top.
- Semantic functions split into separate panels per category (one subplot each).
- Fixed meaningful ordering on both axes — no clustering scramble.
- Saved to plots/pmi_heatmap_paper.pdf/.png
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

from paper_code.data_sources import DATA_SOURCES, load_calls

# ------------------------------------------------------------------ #
# Acoustic features — meaningful order, grouped
# ------------------------------------------------------------------ #

OLD_ACOUSTIC = [
    ("high-frequency", "high-frequency", ["high-frequency", "high frequency", "high-pitched", "high pitched", "ultrasonic"]),
    ("low-frequency", "low-frequency", ["low-frequency", "low frequency", "low-pitched", "low pitched", "infrasound"]),
    ("frequency-modulated", "frequency-modulated", ["frequency-modulated", "frequency modulated", "fm sweep", "sweep", "upsweep", "downsweep", "modulated"]),
    ("short", "short", ["short ", "brief", "abrupt"]),
    ("long / sustained", "long / sustained", ["long ", "prolonged", "sustained", "extended"]),
    ("loud", "loud", ["loud", "intense", "powerful", "far-carrying"]),
    ("soft / quiet", "soft / quiet", ["soft ", "quiet", "low amplitude", "low-amplitude", "subtle"]),
    ("broadband / noisy", "broadband / noisy", ["broadband", "broad-band", "noisy", "noise", "atonal", "wideband"]),
    ("tonal", "tonal", ["tonal", "pure tone", "narrowband", "narrow-band", "sinusoidal"]),
    ("harmonic", "harmonic", ["harmonic", "overtone", "formant"]),
    ("pulsed", "pulsed", ["pulsed", "pulse", "click", "burst", "staccato"]),
    ("repetitive", "repetitive", ["repetitive", "repeated", "series", "bout", "sequence of"]),
]

NEW_ACOUSTIC = [
    # ── Frequency (strongest overall signal) ──────────────────────
    ("high_frequency", "high-frequency", ["high-frequency", "high frequency", "high-pitched", "high pitched", "ultrasonic"]),
    ("low_frequency", "low-frequency", ["low-frequency", "low frequency", "low-pitched", "low pitched", "infrasound"]),
    ("frequency_modulated", "frequency-modulated", ["frequency-modulated", "frequency modulated", "fm sweep", "sweep", "upsweep", "downsweep", "modulated"]),
    # ── Duration ──────────────────────────────────────────────────
    ("short", "short", ["short ", "brief"]),
    ("long", "long / sustained", ["long ", "prolonged", "sustained", "extended"]),
    ("abrupt", "abrupt", ["abrupt", "rapid onset"]),
    # ── Amplitude ─────────────────────────────────────────────────
    ("loud", "loud", ["loud", "intense", "powerful", "far-carrying"]),
    ("quiet", "soft / quiet", ["soft ", "quiet", "low amplitude", "low-amplitude", "subtle"]),
    # ── Texture / temporal pattern ────────────────────────────────
    ("broadband", "broadband", ["broadband", "broad-band", "wideband"]),
    ("noisy", "noisy", ["noisy", "noise", "atonal", "harsh"]),
    ("tonal", "tonal", ["tonal", "pure tone", "narrowband", "narrow-band", "sinusoidal"]),
    ("harmonic", "harmonic", ["harmonic", "overtone", "formant"]),
    ("pulsed", "pulsed", ["pulsed", "pulse", "click", "burst", "staccato"]),
    ("repetitive", "repetitive", ["repetitive", "repeated", "series", "bout", "sequence of"]),
    ("multi_component", "multi-component", ["multi-component", "multi component", "multi-note", "multi note", "multi-syllable", "complex"]),
    ("graded", "graded", ["graded", "variable", "gradation"]),
]

# (group label, number of consecutive rows)
OLD_ACOUSTIC_GROUPS = [
    ("Frequency",          3),
    ("Duration",           2),
    ("Amplitude",          2),
    ("Texture / pattern",  5),
]

NEW_ACOUSTIC_GROUPS = [
    ("Frequency",          3),
    ("Duration",           3),
    ("Amplitude",          2),
    ("Texture / pattern",  8),
]

GROUP_BG_COLORS = {
    "Frequency":         "#AED6F1",  # pastel blue
    "Duration":          "#A9DFBF",  # pastel green
    "Amplitude":         "#FAD7A0",  # pastel orange
    "Texture / pattern": "#D7BDE2",  # pastel purple
}

# ------------------------------------------------------------------ #
# Semantic functions — split into panels, ordered within each
# ------------------------------------------------------------------ #

OLD_SEMANTIC_PANELS = [
    ("Danger / threat",  ["alarm", "predator", "threat", "aggression"]),
    ("Distress / Infant",["distress", "infant"]),
    ("Cohesion",         ["contact", "coordination", "long_distance", "affiliative"]),
    ("Social",           ["display", "sex", "territory"]),
    ("Foraging",         ["food", "recruitment"]),
    ("Cognitive",        ["individual_identity", "learning", "referential"]),
]

NEW_SEMANTIC_PANELS = [
    ("Danger / threat",  ["alarm", "predator", "threat", "aggression"]),
    ("Distress / care",  ["distress", "infant", "begging", "caregiving"]),
    ("Cohesion",         ["contact", "coordination", "group_coordination", "long_distance", "affiliative", "affiliation", "spacing"]),
    ("Social",           ["display", "sex", "courtship", "mating", "territory", "territorial", "submission", "identity", "attention", "play", "combinatorial"]),
    ("Foraging",         ["food", "recruitment"]),
    ("Cognitive",        ["individual_identity", "learning", "referential"]),
]

OLD_SEMANTIC = [s for _, panel in OLD_SEMANTIC_PANELS for s in panel]
NEW_SEMANTIC = [s for _, panel in NEW_SEMANTIC_PANELS for s in panel]

SEMANTIC_LABELS = {
    "distress":            "adult distress",
    "infant":              "infant calls",
    "long_distance":       "long-distance",
    "group_coordination":  "coordination",
    "individual_identity": "indiv. identity",
    "affiliative":         "affiliation",
    "territory":           "territorial",
    "sex":                 "mating",
    "learning":            "vocal learning",
    "multi_component":     "multi-component",
}

def extract_keyword_flags(desc: str, patterns: list[str]) -> bool:
    desc = desc.lower()
    return any(p in desc for p in patterns)


def _semantic_patterns():
    return {
        "alarm": ["alarm"],
        "predator": ["predator", "raptor", "snake"],
        "threat": ["threat", "defensive"],
        "aggression": ["aggression", "aggressive", "agonistic"],
        "distress": ["distress", "disturbed", "capture", "pain"],
        "infant": ["infant", "juvenile", "nestling", "pup", "calf"],
        "begging": ["begging", "solicitation", "solicit"],
        "caregiving": ["caregiving", "parent", "maternal", "offspring"],
        "contact": ["contact"],
        "coordination": ["coordination", "coordinate"],
        "group_coordination": ["group coordination", "coordinate", "cohesion"],
        "long_distance": ["long-distance", "long distance", "far"],
        "affiliative": ["affiliative", "affiliation"],
        "affiliation": ["affiliation", "affiliative"],
        "spacing": ["spacing", "separation"],
        "display": ["display"],
        "sex": ["sex", "sexual"],
        "courtship": ["courtship"],
        "mating": ["mating", "mate", "sexual"],
        "territory": ["territory", "territorial"],
        "territorial": ["territorial", "territory"],
        "submission": ["submission", "submissive"],
        "identity": ["identity", "individual", "recognition"],
        "attention": ["attention", "alert"],
        "play": ["play"],
        "combinatorial": ["combinatorial", "combination"],
        "food": ["food", "foraging", "feeding"],
        "recruitment": ["recruitment", "recruit"],
        "individual_identity": ["individual identity", "individual recognition"],
        "learning": ["learning", "learned"],
        "referential": ["referential"],
    }


def _provided_keyword_matrix(calls, field, labels):
    mat = np.zeros((len(calls), len(labels)), dtype=float)
    missing = 0
    for i, call in enumerate(calls):
        kws = set(call.get(field) or [])
        if not kws:
            missing += 1
        for j, key in enumerate(labels):
            mat[i, j] = float(key in kws)
    return mat, missing


def _extracted_keyword_matrix(calls, desc_field, specs):
    mat = np.zeros((len(calls), len(specs)), dtype=float)
    for i, call in enumerate(calls):
        desc = str(call.get(desc_field, ""))
        for j, (_key, _label, patterns) in enumerate(specs):
            mat[i, j] = float(extract_keyword_flags(desc, patterns))
    return mat


def compute_pmi(calls, data_source, extract_keywords_from_text, min_calls=4):
    n = len(calls)
    acoustic_specs = OLD_ACOUSTIC if data_source == "old" else NEW_ACOUSTIC
    acoustic_groups = OLD_ACOUSTIC_GROUPS if data_source == "old" else NEW_ACOUSTIC_GROUPS
    semantic_keys_all = OLD_SEMANTIC if data_source == "old" else NEW_SEMANTIC
    semantic_panels = OLD_SEMANTIC_PANELS if data_source == "old" else NEW_SEMANTIC_PANELS
    ac_keys = [key for key, _label, _patterns in acoustic_specs]
    ac_labels = [label for _key, label, _patterns in acoustic_specs]

    if extract_keywords_from_text:
        ac_mat = _extracted_keyword_matrix(calls, "acoustic_description", acoustic_specs)
        print("  Acoustic keywords: extracted from acoustic_description")
    else:
        ac_mat, missing_ac = _provided_keyword_matrix(calls, "acoustic_keywords", ac_keys)
        if missing_ac:
            raise ValueError(
                f"{data_source!r} data has no provided acoustic_keywords for "
                f"{missing_ac}/{n} calls. Re-run with --extract-keywords-from-text."
            )
        print("  Acoustic keywords: provided by data source")

    if data_source == "old" and extract_keywords_from_text:
        sem_keys = semantic_keys_all
        sem_mat, _ = _provided_keyword_matrix(calls, "ontology_keywords", sem_keys)
        print("  Semantic keywords: provided ontology_keywords (old pipeline)")
    elif extract_keywords_from_text:
        sem_keys = semantic_keys_all
        patterns = _semantic_patterns()
        sem_specs = [(key, key, patterns.get(key, [key.replace("_", " ")])) for key in sem_keys]
        sem_mat = _extracted_keyword_matrix(calls, "semantic_description", sem_specs)
        print("  Semantic keywords: extracted from semantic_description")
    else:
        sem_keys = semantic_keys_all
        sem_mat, missing_sem = _provided_keyword_matrix(calls, "semantic_keywords", sem_keys)
        if missing_sem:
            raise ValueError(
                f"{data_source!r} data has no provided semantic_keywords for {missing_sem}/{n} calls."
            )
        print("  Semantic keywords: provided by data source")

    sem_counts = sem_mat.sum(axis=0)
    keep = sem_counts >= min_calls
    sem_mat = sem_mat[:, keep]
    sem_labels = [s for s, k in zip(sem_keys, keep) if k]
    print(f"  Semantic keywords kept (≥{min_calls} calls): {sem_labels}")
    if not sem_labels:
        raise ValueError(f"No semantic keywords met min_calls={min_calls}")

    p_ac    = ac_mat.mean(axis=0)
    p_sem   = sem_mat.mean(axis=0)
    p_joint = (ac_mat[:, :, None] * sem_mat[:, None, :]).mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(
            (p_joint > 0) & (p_ac[:, None] > 0) & (p_sem[None, :] > 0),
            np.log2(p_joint / (p_ac[:, None] * p_sem[None, :])),
            0.0,
        )

    return pmi, ac_labels, sem_labels, p_ac, p_sem, p_joint, acoustic_groups, semantic_panels


# ------------------------------------------------------------------ #
# Plot
# ------------------------------------------------------------------ #

def plot_pmi(pmi, ac_labels, sem_labels, p_ac, p_sem, out_path, acoustic_groups, semantic_panels):
    import matplotlib.patches as mpatches

    n_rows = pmi.shape[0]

    # Build panels, filtering to keywords that survived min_calls
    panels = []
    for panel_name, features in semantic_panels:
        idxs = [sem_labels.index(f) for f in features if f in sem_labels]
        if idxs:
            disp = [SEMANTIC_LABELS.get(sem_labels[i], sem_labels[i]) for i in idxs]
            panels.append((panel_name, idxs, disp))
    if not panels:
        raise ValueError("No semantic panels contain retained keywords")

    panel_sizes = [len(p[1]) for p in panels]
    n_panels    = len(panels)

    # Font sizes
    FS_X      = 18   # x-tick labels
    FS_Y      = 18   # y-tick labels
    FS_TITLE  = 18   # panel titles
    FS_ANNOT  = 13   # cell value annotations
    FS_AXIS   = 20   # axis labels
    FS_CBAR   = 17   # colorbar label
    FS_CBAR_T = 14   # colorbar tick labels
    FS_GRP    = 13   # group name in colored strip

    # ── Figure geometry (inches) ────────────────────────────────────
    cell         = 0.72   # inches per heatmap cell
    gap          = 0.22   # gap between panels
    strip_w      = 0.55   # width of acoustic-group colored strip
    strip_gap    = 0.08   # gap between strip right edge and heatmap left edge
    margin_left  = 4.30   # room for strip + y-tick labels
    margin_right = 1.20   # colorbar
    margin_top   = 0.85   # panel titles
    margin_bot   = 2.50   # rotated x-tick labels

    heatmap_w = sum(s * cell for s in panel_sizes) + gap * (n_panels - 1)
    fig_w = margin_left + heatmap_w + margin_right
    fig_h = n_rows * cell + margin_top + margin_bot

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ── Compute subplot positions in figure fractions ───────────────
    left0      = margin_left / fig_w
    bot0       = margin_bot  / fig_h
    ax_h_frac  = n_rows * cell / fig_h
    gap_frac   = gap / fig_w

    axes = []
    x = left0
    for _, idxs, _ in panels:
        w = len(idxs) * cell / fig_w
        axes.append(fig.add_axes([x, bot0, w, ax_h_frac]))
        x += w + gap_frac

    vmax   = max(abs(pmi.max()), abs(pmi.min()), 1.0)
    im_ref = None

    # Extra pad pushes y-tick labels left, making room for the colored strip
    y_pad_pts = int((strip_w + strip_gap) * 72) + 4

    for k, (ax, (panel_name, sem_idxs, sem_display)) in enumerate(zip(axes, panels)):
        pmi_panel = pmi[:, sem_idxs]
        im = ax.imshow(pmi_panel, aspect=1, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        im_ref = im

        # x-axis labels
        ax.set_xticks(range(len(sem_idxs)))
        ax.set_xticklabels(sem_display, rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=FS_X)
        ax.tick_params(axis="x", which="major", pad=2)

        # y-axis labels only on leftmost panel
        if k == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(ac_labels, fontsize=FS_Y)
            ax.tick_params(axis="y", which="major", pad=y_pad_pts)
        else:
            ax.set_yticks([])

        # Minor grid (white lines between cells)
        ax.set_xticks(np.arange(-0.5, len(sem_idxs)), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

        # Panel title
        ax.set_title(panel_name, fontsize=FS_TITLE, fontweight="bold", pad=6)

        # Annotate top-2 positive and top-1 negative per panel (if |v| ≥ 1.0)
        flat = pmi_panel.flatten()
        for idx in set(list(np.argsort(flat)[::-1][:2]) + list(np.argsort(flat)[:1])):
            r, c = divmod(idx, pmi_panel.shape[1])
            v = pmi_panel[r, c]
            if abs(v) >= 1.0:
                col = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                        fontsize=FS_ANNOT, fontweight="bold", color=col)

        # Horizontal dashed separators between acoustic groups
        row = 0
        for _, grp_count in acoustic_groups:
            row += grp_count
            if row < n_rows:
                ax.axhline(row - 0.5, color="#888888", linewidth=0.8,
                           linestyle="--", alpha=0.55)

    # ── Acoustic group colored strips (y-panels) ────────────────────
    # Drawn in ax0's data coordinates, to the left of the heatmap.
    # strip_gap separates the strip from the heatmap edge (x = -0.5 in data coords).
    # The extra tick pad above ensures y-tick labels clear the strip.
    ax0 = axes[0]
    strip_right_data = -0.5 - strip_gap / cell
    strip_left_data  = strip_right_data - strip_w / cell

    row = 0
    for grp_name, grp_count in acoustic_groups:
        y_top = row - 0.5
        y_bot = row + grp_count - 0.5
        color = GROUP_BG_COLORS[grp_name]
        rect = mpatches.Rectangle(
            (strip_left_data, y_top),
            strip_w / cell,
            grp_count,
            transform=ax0.transData,
            clip_on=False,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=0,
        )
        ax0.add_patch(rect)
        mid_x = (strip_left_data + strip_right_data) / 2
        mid_y = (y_top + y_bot) / 2
        ax0.text(mid_x, mid_y, grp_name,
                 ha="center", va="center", fontsize=FS_GRP,
                 fontweight="bold", color="#333333", rotation=90,
                 transform=ax0.transData, clip_on=False, zorder=3)
        row += grp_count

    # ── Shared axis labels ──────────────────────────────────────────
    x_center = left0 + heatmap_w / fig_w / 2

    # Position "Semantic function" just below the rotated x-tick labels.
    # At 45°, label text extends downward ≈ text_width × sin(45°) from the axis.
    max_label_len = max(len(d) for _, _, disp in panels for d in disp)
    tick_down_in = max_label_len * FS_X / 72 * 0.45 * 0.707
    sem_label_y = bot0 - (tick_down_in + 0.30) / fig_h
    fig.text(x_center, max(0.02, sem_label_y), "Semantic function",
             ha="center", va="top", fontsize=FS_AXIS)
    fig.text(0.01, bot0 + ax_h_frac / 2, "Acoustic feature",
             ha="left", va="center", fontsize=FS_AXIS, rotation=90)

    # ── Colorbar ────────────────────────────────────────────────────
    cbar_left = (margin_left + heatmap_w + 0.12) / fig_w
    cbar_ax   = fig.add_axes([cbar_left, bot0, 0.022, ax_h_frac])
    cbar      = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.set_label("PMI (bits)", fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_CBAR_T)

    # ── Save ────────────────────────────────────────────────────────
    out_path = Path(out_path)
    png_path = out_path.with_suffix(".png")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="old")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plots")
    parser.add_argument(
        "--extract-keywords-from-text",
        action="store_true",
        help="Extract acoustic/semantic keywords from descriptions instead of using provided keyword fields.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = load_calls(args.data_source)
    print(f"{len(calls)} calls from {args.data_source!r} data")
    pmi, ac_labels, sem_labels, p_ac, p_sem, p_joint, acoustic_groups, semantic_panels = compute_pmi(
        calls,
        data_source=args.data_source,
        extract_keywords_from_text=args.extract_keywords_from_text,
    )

    print("\nTop 10 PMI associations:")
    flat_idx = np.argsort(pmi.flatten())[::-1]
    for idx in flat_idx[:10]:
        r, c = divmod(idx, len(sem_labels))
        print(f"  {ac_labels[r]:25s} × {sem_labels[c]:20s}  PMI={pmi[r,c]:.2f}")

    print("\nBottom 5 PMI associations:")
    for idx in flat_idx[-5:]:
        r, c = divmod(idx, len(sem_labels))
        print(f"  {ac_labels[r]:25s} × {sem_labels[c]:20s}  PMI={pmi[r,c]:.2f}")

    out = out_dir / "pmi_heatmap_paper.png"
    plot_pmi(pmi, ac_labels, sem_labels, p_ac, p_sem, out, acoustic_groups, semantic_panels)


if __name__ == "__main__":
    main()

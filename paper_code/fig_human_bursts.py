"""
Generate three appendix figures for human vocal bursts.

Figure H1  plots/fig_human_correspondents.pdf/.png
    Heatmap: 30 human burst types × animal species, cell = max semantic cosine sim.

Figure H2  plots/fig_human_pmi.pdf/.png
    PMI heatmap for human bursts (same style as fig_pmi_heatmap.py / Figure 4).

Figure H3  plots/fig_human_mantel_row.pdf/.png
    Per-species Pearson r (human × animal acoustic–semantic cross-similarity).

Run from project root:
    python paper_code/fig_human_bursts.py
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
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr

# ------------------------------------------------------------------ #
# Shared constants (reused from fig_pmi_heatmap.py)
# ------------------------------------------------------------------ #

ACOUSTIC = [
    ("high-frequency",      ["high-frequency", "high frequency", "high-pitched", "high pitched", "ultrasonic"]),
    ("low-frequency",       ["low-frequency", "low frequency", "low-pitched", "low pitched", "infrasound"]),
    ("frequency-modulated", ["frequency-modulated", "frequency modulated", "fm sweep", "sweep", "upsweep", "downsweep", "modulated"]),
    ("short",               ["short ", "brief", "abrupt"]),
    ("long / sustained",    ["long ", "prolonged", "sustained", "extended"]),
    ("loud",                ["loud", "intense", "powerful", "far-carrying"]),
    ("soft / quiet",        ["soft ", "quiet", "low amplitude", "low-amplitude", "subtle"]),
    ("broadband / noisy",   ["broadband", "broad-band", "noisy", "noise", "atonal", "wideband"]),
    ("tonal",               ["tonal", "pure tone", "narrowband", "narrow-band", "sinusoidal"]),
    ("harmonic",            ["harmonic", "overtone", "formant"]),
    ("pulsed",              ["pulsed", "pulse", "click", "burst", "staccato"]),
    ("repetitive",          ["repetitive", "repeated", "series", "bout", "sequence of"]),
]

ACOUSTIC_GROUPS = [
    ("Frequency",         3),
    ("Duration",          2),
    ("Amplitude",         2),
    ("Texture / pattern", 5),
]

GROUP_BG_COLORS = {
    "Frequency":         "#AED6F1",
    "Duration":          "#A9DFBF",
    "Amplitude":         "#FAD7A0",
    "Texture / pattern": "#D7BDE2",
}

CLASS_ORDER = {"Amphibia": 0, "Aves": 1, "Mammalia": 2}
CLASS_COLORS = {
    "Amphibia": "#C44E52",
    "Aves":     "#55A868",
    "Mammalia": "#4C72B0",
}

# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #

def load_animals():
    """Load animal calls with species metadata, in database.json order."""
    with open(ROOT / "database.json") as f:
        db = json.load(f)
    calls = []
    for s in db["species"]:
        for c in s.get("calls", []):
            calls.append({
                **c,
                "species":      s["species_name"],
                "family":       s.get("family", ""),
                "class":        s.get("class", ""),
            })
    return calls


def load_humans():
    """Load human burst calls from JSON."""
    with open(ROOT / "human_bursts" / "manual" / "human_vocal_bursts.json") as f:
        hdb = json.load(f)
    # Single "species" entry for humans
    return hdb["species"][0]["calls"]


def common_name(species_name: str) -> str:
    """Strip '(Scientific name)' parenthetical."""
    return species_name.split("(")[0].strip()


# ------------------------------------------------------------------ #
# Embedding helpers
# ------------------------------------------------------------------ #

def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts, model):
    return model.encode(texts, normalize_embeddings=True)


def load_or_compute_human_embeddings(human_calls):
    """Load cached human embeddings or compute and cache them."""
    ac_path = ROOT / "paper_code" / "human_ac_emb.npy"
    se_path = ROOT / "paper_code" / "human_se_emb.npy"

    if ac_path.exists() and se_path.exists():
        print("Loading cached human embeddings …")
        h_ac = np.load(ac_path)
        h_se = np.load(se_path)
        return h_ac, h_se

    print("Computing human embeddings (first run) …")
    model = get_model()
    ac_texts = [c.get("acoustic_description", "") for c in human_calls]
    se_texts = [c.get("semantic_description", "") for c in human_calls]
    h_ac = embed_texts(ac_texts, model)
    h_se = embed_texts(se_texts, model)
    np.save(ac_path, h_ac)
    np.save(se_path, h_se)
    print(f"  Saved {ac_path}  {se_path}")
    return h_ac, h_se


# ------------------------------------------------------------------ #
# PMI helpers (inlined from fig_pmi_heatmap.py)
# ------------------------------------------------------------------ #

def extract_acoustic_flags(desc: str, patterns: list) -> bool:
    desc = desc.lower()
    return any(p in desc for p in patterns)


def compute_pmi(calls, min_calls=4):
    """Compute PMI matrix for the given calls list."""
    # We need to determine SEMANTIC from the calls dynamically
    # but we use the full ACOUSTIC list always.
    # For the semantic axis we collect all keywords then filter by min_calls.
    from collections import Counter
    kw_counter = Counter(kw for c in calls for kw in c.get("ontology_keywords", []))
    sem_labels = [kw for kw, cnt in sorted(kw_counter.items()) if cnt >= min_calls]

    n = len(calls)
    ac_mat  = np.zeros((n, len(ACOUSTIC)), dtype=float)
    sem_mat = np.zeros((n, len(sem_labels)), dtype=float)

    for i, c in enumerate(calls):
        desc = c.get("acoustic_description", "")
        for j, (_, patterns) in enumerate(ACOUSTIC):
            ac_mat[i, j] = float(extract_acoustic_flags(desc, patterns))
        kws = set(c.get("ontology_keywords", []))
        for j, sk in enumerate(sem_labels):
            sem_mat[i, j] = float(sk in kws)

    p_ac    = ac_mat.mean(axis=0)
    p_sem   = sem_mat.mean(axis=0)
    p_joint = (ac_mat[:, :, None] * sem_mat[:, None, :]).mean(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(
            (p_joint > 0) & (p_ac[:, None] > 0) & (p_sem[None, :] > 0),
            np.log2(p_joint / (p_ac[:, None] * p_sem[None, :])),
            0.0,
        )

    ac_labels = [label for label, _ in ACOUSTIC]
    print(f"  Semantic keywords kept (≥{min_calls} calls): {sem_labels}")
    return pmi, ac_labels, sem_labels, p_ac, p_sem, p_joint


# ================================================================== #
# Figure H1 — semantic correspondents heatmap
# ================================================================== #

def fig_h1_correspondents(human_calls, animal_calls, h_se, se_emb):
    """
    Heatmap: rows = 30 human burst types, cols = animal species.
    Cell = max semantic cosine sim between human burst and any call of that species.
    """
    print("\n=== Figure H1: Semantic correspondents ===")

    # Build species list sorted by (class_order, species_name)
    sp_set = {}
    for c in animal_calls:
        sp = c["species"]
        if sp not in sp_set:
            sp_set[sp] = c["class"]
    species_sorted = sorted(
        sp_set.keys(),
        key=lambda s: (CLASS_ORDER.get(sp_set[s], 9), s)
    )
    n_sp = len(species_sorted)
    n_human = len(human_calls)

    # Map species -> call indices
    sp_idx = {s: [] for s in species_sorted}
    for idx, c in enumerate(animal_calls):
        sp_idx[c["species"]].append(idx)

    # h_se: (30, 384), se_emb: (373, 384) — already L2 normalised
    # Compute cross-similarity: (30, 373)
    sim_matrix = h_se @ se_emb.T  # cosine sim (both normalised)

    # For each human burst × species: max sim over species' calls
    heat = np.zeros((n_human, n_sp))
    best_call = [["" for _ in range(n_sp)] for _ in range(n_human)]
    for j, sp in enumerate(species_sorted):
        idxs = sp_idx[sp]
        if not idxs:
            continue
        sims = sim_matrix[:, idxs]   # (n_human, n_calls_of_sp)
        argmax = np.argmax(sims, axis=1)
        heat[:, j] = sims[np.arange(n_human), argmax]
        for i in range(n_human):
            call_idx = idxs[argmax[i]]
            best_call[i][j] = animal_calls[call_idx].get("call_name", "")

    # ── Row labels ──────────────────────────────────────────────────
    row_labels = []
    for c in human_calls:
        name = c["call_name"]
        # strip trailing " burst" or "(burst)" for brevity
        name = name.replace(" burst", "").replace("(", "").replace(")", "").strip()
        row_labels.append(name)

    # ── Column labels (common names) ────────────────────────────────
    col_labels = [common_name(s) for s in species_sorted]
    col_classes = [sp_set[s] for s in species_sorted]

    # ── Figure geometry ─────────────────────────────────────────────
    cell_w     = 0.55   # inches per column
    cell_h     = 0.45   # inches per row
    margin_left  = 2.2    # for row labels
    margin_right = 1.0    # colorbar
    margin_top   = 0.9    # class bars + rotated species names
    margin_bot   = 0.3

    fig_w = margin_left + n_sp * cell_w + margin_right
    fig_h = margin_top + n_human * cell_h + margin_bot

    fig = plt.figure(figsize=(fig_w, fig_h))

    left_frac  = margin_left / fig_w
    bot_frac   = margin_bot / fig_h
    ax_w_frac  = n_sp * cell_w / fig_w
    ax_h_frac  = n_human * cell_h / fig_h

    ax = fig.add_axes([left_frac, bot_frac, ax_w_frac, ax_h_frac])

    im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                   interpolation="nearest")

    # ── Axis ticks ──────────────────────────────────────────────────
    ax.set_yticks(range(n_human))
    ax.set_yticklabels(row_labels, fontsize=8.5)
    ax.set_xticks(range(n_sp))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7.5,
                       rotation_mode="anchor")
    ax.tick_params(axis="x", which="major", pad=2)

    # Minor grid
    ax.set_xticks(np.arange(-0.5, n_sp),   minor=True)
    ax.set_yticks(np.arange(-0.5, n_human), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    # ── Cell annotations (≥ 0.75) ───────────────────────────────────
    THRESH = 0.75
    for i in range(n_human):
        for j in range(n_sp):
            if heat[i, j] >= THRESH:
                label = best_call[i][j][:12]
                col = "white" if heat[i, j] > 0.85 else "black"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=4.5, color=col, fontweight="bold")

    # ── Class color bars along the top ──────────────────────────────
    # Bar drawn in data coords above the heatmap
    bar_h = 0.35  # in data units (rows)
    bar_gap = 0.15
    bar_y0 = -0.5 - bar_gap - bar_h

    # Group consecutive columns by class
    class_spans_col = []
    cur_cls = col_classes[0]
    cur_start = 0
    for jj in range(1, n_sp):
        if col_classes[jj] != cur_cls:
            class_spans_col.append((cur_start, jj, cur_cls))
            cur_cls = col_classes[jj]
            cur_start = jj
    class_spans_col.append((cur_start, n_sp, cur_cls))

    for (start, end, cls) in class_spans_col:
        color = CLASS_COLORS.get(cls, "#aaaaaa")
        rect = Rectangle(
            (start - 0.5, bar_y0),
            end - start,
            bar_h,
            transform=ax.transData,
            clip_on=False,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)
        mid_x = (start + end) / 2.0 - 0.5
        ax.text(mid_x, bar_y0 + bar_h / 2, cls,
                ha="center", va="center", fontsize=6, fontweight="bold",
                color="white", transform=ax.transData, clip_on=False)

    # ── Colorbar ────────────────────────────────────────────────────
    cbar_left = left_frac + ax_w_frac + 0.01
    cbar_ax   = fig.add_axes([cbar_left, bot_frac, 0.018, ax_h_frac])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Max semantic cosine sim.", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── Title ───────────────────────────────────────────────────────
    fig.text(left_frac + ax_w_frac / 2, 1.0 - 0.01 / fig_h,
             "Semantic correspondents: human vocal bursts × animal species",
             ha="center", va="top", fontsize=10, fontweight="bold",
             transform=fig.transFigure)

    # ── Save ────────────────────────────────────────────────────────
    out_base = ROOT / "plots" / "fig_human_correspondents"
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_base}.pdf/.png")


# ================================================================== #
# Figure H2 — PMI heatmap (human bursts)
# ================================================================== #

# Human-specific semantic panels (keywords that survive min_calls=3)
HUMAN_SEMANTIC_PANELS = [
    ("Positive",      ["positive_affect", "affiliative"]),
    ("Social",        ["display", "contact"]),
    ("Negative",      ["negative_affect", "distress"]),
    ("Alarm / Ref.",  ["alarm", "referential"]),
]

# Keywords shared with animal PMI (for correlation)
SHARED_KEYWORDS = ["alarm", "affiliative", "display", "contact", "distress", "referential"]


def fig_h2_pmi(human_calls, animal_calls):
    """
    PMI heatmap for human bursts — same visual style as Figure 4.
    Also prints and annotates Pearson r with animal PMI on shared keywords.
    """
    print("\n=== Figure H2: Human PMI heatmap ===")

    # ── Human PMI ───────────────────────────────────────────────────
    h_pmi, ac_labels, h_sem_labels, *_ = compute_pmi(human_calls, min_calls=3)
    print(f"  Human PMI matrix: {h_pmi.shape}  ac={len(ac_labels)}  sem={len(h_sem_labels)}")

    # ── Animal PMI (for cross-correlation) ──────────────────────────
    a_pmi, _, a_sem_labels, *_ = compute_pmi(animal_calls, min_calls=4)
    print(f"  Animal PMI matrix: {a_pmi.shape}  sem={len(a_sem_labels)}")

    # ── Pearson r on shared keywords ────────────────────────────────
    shared = [kw for kw in SHARED_KEYWORDS if kw in h_sem_labels and kw in a_sem_labels]
    if shared:
        h_cols = [h_sem_labels.index(kw) for kw in shared]
        a_cols = [a_sem_labels.index(kw) for kw in shared]
        h_vec = h_pmi[:, h_cols].ravel()
        a_vec = a_pmi[:, a_cols].ravel()
        r_val, p_val = pearsonr(h_vec, a_vec)
        shared_str = ", ".join(shared)
        annot_text = f"PMI correlation with non-human animals: r = {r_val:.2f}  (shared keywords: {shared_str})"
        print(f"  {annot_text}")
    else:
        r_val = None
        annot_text = ""

    # ── Build panels from surviving human keywords ───────────────────
    panels = []
    for panel_name, features in HUMAN_SEMANTIC_PANELS:
        idxs = [h_sem_labels.index(f) for f in features if f in h_sem_labels]
        if idxs:
            panels.append((panel_name, idxs, [h_sem_labels[i] for i in idxs]))

    # ── Figure geometry (matching fig_pmi_heatmap.py style) ──────────
    n_rows = h_pmi.shape[0]
    panel_sizes = [len(p[1]) for p in panels]
    n_panels = len(panels)

    FS_X      = 16
    FS_Y      = 16
    FS_TITLE  = 16
    FS_ANNOT  = 11
    FS_AXIS   = 18
    FS_CBAR   = 15
    FS_CBAR_T = 12
    FS_GRP    = 12

    cell       = 0.72
    gap        = 0.22
    strip_w    = 0.55
    strip_gap  = 0.08
    margin_left  = 4.0
    margin_right = 1.20
    margin_top   = 0.85
    margin_bot   = 2.80   # extra for annotation text

    heatmap_w = sum(s * cell for s in panel_sizes) + gap * (n_panels - 1)
    fig_w = margin_left + heatmap_w + margin_right
    fig_h = n_rows * cell + margin_top + margin_bot

    fig = plt.figure(figsize=(fig_w, fig_h))

    left0     = margin_left / fig_w
    bot0      = margin_bot  / fig_h
    ax_h_frac = n_rows * cell / fig_h
    gap_frac  = gap / fig_w

    axes = []
    x = left0
    for _, idxs, _ in panels:
        w = len(idxs) * cell / fig_w
        axes.append(fig.add_axes([x, bot0, w, ax_h_frac]))
        x += w + gap_frac

    vmax   = max(abs(h_pmi.max()), abs(h_pmi.min()), 1.0)
    im_ref = None

    y_pad_pts = int((strip_w + strip_gap) * 72) + 4

    for k, (ax, (panel_name, sem_idxs, sem_display)) in enumerate(zip(axes, panels)):
        pmi_panel = h_pmi[:, sem_idxs]
        im = ax.imshow(pmi_panel, aspect=1, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        im_ref = im

        ax.set_xticks(range(len(sem_idxs)))
        ax.set_xticklabels(sem_display, rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=FS_X)
        ax.tick_params(axis="x", which="major", pad=2)

        if k == 0:
            ax.set_yticks(range(n_rows))
            ax.set_yticklabels(ac_labels, fontsize=FS_Y)
            ax.tick_params(axis="y", which="major", pad=y_pad_pts)
        else:
            ax.set_yticks([])

        ax.set_xticks(np.arange(-0.5, len(sem_idxs)), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.5)
        ax.tick_params(which="minor", length=0)

        ax.set_title(panel_name, fontsize=FS_TITLE, fontweight="bold", pad=6)

        # Annotate top-2 positive and top-1 negative
        flat = pmi_panel.flatten()
        for idx in set(list(np.argsort(flat)[::-1][:2]) + list(np.argsort(flat)[:1])):
            r, c = divmod(idx, pmi_panel.shape[1])
            v = pmi_panel[r, c]
            if abs(v) >= 1.0:
                col = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(c, r, f"{v:.1f}", ha="center", va="center",
                        fontsize=FS_ANNOT, fontweight="bold", color=col)

        # Horizontal separators between acoustic groups
        row = 0
        for _, grp_count in ACOUSTIC_GROUPS:
            row += grp_count
            if row < n_rows:
                ax.axhline(row - 0.5, color="#888888", linewidth=0.8,
                           linestyle="--", alpha=0.55)

    # ── Acoustic group colored strips ────────────────────────────────
    ax0 = axes[0]
    strip_right_data = -0.5 - strip_gap / cell
    strip_left_data  = strip_right_data - strip_w / cell

    row = 0
    for grp_name, grp_count in ACOUSTIC_GROUPS:
        y_top = row - 0.5
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
        mid_y = y_top + grp_count / 2
        ax0.text(mid_x, mid_y, grp_name,
                 ha="center", va="center", fontsize=FS_GRP,
                 fontweight="bold", color="#333333", rotation=90,
                 transform=ax0.transData, clip_on=False, zorder=3)
        row += grp_count

    # ── Axis labels ──────────────────────────────────────────────────
    x_center = left0 + heatmap_w / fig_w / 2
    max_label_len = max(len(d) for _, _, disp in panels for d in disp)
    tick_down_in = max_label_len * FS_X / 72 * 0.45 * 0.707
    sem_label_y = bot0 - (tick_down_in + 0.40) / fig_h
    fig.text(x_center, max(0.02, sem_label_y), "Semantic function",
             ha="center", va="top", fontsize=FS_AXIS)
    fig.text(0.01, bot0 + ax_h_frac / 2, "Acoustic feature",
             ha="left", va="center", fontsize=FS_AXIS, rotation=90)

    # PMI cross-correlation printed to stdout only; not annotated on figure

    # ── Colorbar ────────────────────────────────────────────────────
    cbar_left = (margin_left + heatmap_w + 0.12) / fig_w
    cbar_ax   = fig.add_axes([cbar_left, bot0, 0.022, ax_h_frac])
    cbar = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.set_label("PMI (bits)", fontsize=FS_CBAR)
    cbar.ax.tick_params(labelsize=FS_CBAR_T)

    # ── Save ────────────────────────────────────────────────────────
    out_base = ROOT / "plots" / "fig_human_pmi"
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_base}.pdf/.png")


# ================================================================== #
# Figure H3 — per-species Mantel bar chart
# ================================================================== #

OVERALL_MANTEL_R = 0.215   # non-human overall Mantel r (from paper)


def fig_h3_mantel_row(human_calls, animal_calls, h_ac, h_se, ac_emb, se_emb):
    """
    Horizontal bar chart: Pearson r between human–animal acoustic sim and semantic sim,
    one bar per animal species.
    """
    print("\n=== Figure H3: Per-species Mantel bar chart ===")

    # ── Build species list sorted by (class, species_name) ───────────
    sp_meta = {}
    for c in animal_calls:
        sp = c["species"]
        if sp not in sp_meta:
            sp_meta[sp] = c["class"]
    species_sorted = sorted(
        sp_meta.keys(),
        key=lambda s: (CLASS_ORDER.get(sp_meta[s], 9), s)
    )

    # Map species -> indices into animal_calls / ac_emb / se_emb
    sp_idx = {s: [] for s in species_sorted}
    for idx, c in enumerate(animal_calls):
        sp_idx[c["species"]].append(idx)

    # h_ac, h_se: (30, 384) — human embeddings
    # ac_emb, se_emb: (373, 384) — animal embeddings
    MIN_PAIRS = 6
    results = []
    for sp in species_sorted:
        idxs = np.array(sp_idx[sp])
        if len(idxs) == 0:
            continue
        ac_S = ac_emb[idxs]   # (n_sp_calls, 384)
        se_S = se_emb[idxs]

        # Cross-species pairs: all combinations of (human call, animal call)
        ac_pairs = (h_ac @ ac_S.T).ravel()   # (30 * n_sp_calls,)
        se_pairs = (h_se @ se_S.T).ravel()

        if len(ac_pairs) < MIN_PAIRS:
            print(f"  Skipping {sp}: only {len(ac_pairs)} pairs")
            continue

        r, p = pearsonr(ac_pairs, se_pairs)
        results.append({
            "species":  sp,
            "class":    sp_meta[sp],
            "r":        r,
            "p":        p,
            "n_pairs":  len(ac_pairs),
        })

    # ── Sort within each class by r value (descending) ──────────────
    results.sort(key=lambda x: (CLASS_ORDER.get(x["class"], 9), -x["r"]))

    # ── Summary statistics ───────────────────────────────────────────
    r_vals = np.array([x["r"] for x in results])
    n_pos  = int((r_vals > 0).sum())
    n_neg  = int((r_vals <= 0).sum())
    mean_r = float(r_vals.mean())
    top5   = sorted(results, key=lambda x: x["r"], reverse=True)[:5]

    # Per-class means
    for cls in ["Amphibia", "Aves", "Mammalia"]:
        cls_r = [x["r"] for x in results if x["class"] == cls]
        if cls_r:
            print(f"  Mean r ({cls}): {np.mean(cls_r):.3f}  (n={len(cls_r)})")

    print(f"\n  Species with r > 0:  {n_pos}")
    print(f"  Species with r ≤ 0:  {n_neg}")
    print(f"  Mean r across species: {mean_r:.3f}")
    print("  Top-5 species by r:")
    for x in top5:
        print(f"    {common_name(x['species']):35s}  r={x['r']:.3f}  (n_pairs={x['n_pairs']})")

    # ── Plot ─────────────────────────────────────────────────────────
    n_bars = len(results)
    fig, ax = plt.subplots(figsize=(7, 8))

    y_pos = np.arange(n_bars)
    bar_colors = [CLASS_COLORS.get(x["class"], "#aaaaaa") for x in results]
    r_arr = [x["r"] for x in results]
    labels = [common_name(x["species"]) for x in results]

    ax.barh(y_pos, r_arr, color=bar_colors, height=0.7, edgecolor="white",
            linewidth=0.4)

    # Class background strips
    class_spans_list = []
    cur_cls = results[0]["class"]
    cur_start = 0
    for ii in range(1, n_bars):
        if results[ii]["class"] != cur_cls:
            class_spans_list.append((cur_start, ii, cur_cls))
            cur_cls = results[ii]["class"]
            cur_start = ii
    class_spans_list.append((cur_start, n_bars, cur_cls))

    x_lim = (-0.4, 0.6)
    for (start, end, cls) in class_spans_list:
        color = CLASS_COLORS.get(cls, "#aaaaaa")
        ax.barh(np.arange(start, end), [x_lim[1] - x_lim[0]] * (end - start),
                left=x_lim[0], height=1.0, color=color, alpha=0.07,
                zorder=0, edgecolor="none")

    # r = 0 dashed line
    ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--", zorder=2)

    # Overall Mantel r solid red line
    ax.axvline(OVERALL_MANTEL_R, color="#CC2222", linewidth=1.4, linestyle="-",
               zorder=3, label=f"Non-human overall Mantel r = {OVERALL_MANTEL_R:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(x_lim)
    ax.set_xlabel("Pearson r  (acoustic × semantic cosine similarity)", fontsize=9)
    ax.set_title("Human–animal acoustic–semantic correlation\nper animal species",
                 fontsize=10, fontweight="bold")

    # Class legend
    legend_handles = [
        mpatches.Patch(facecolor=CLASS_COLORS[cls], edgecolor="grey",
                       linewidth=0.5, label=cls)
        for cls in ["Amphibia", "Aves", "Mammalia"]
    ]
    legend_handles.append(
        plt.Line2D([0], [0], color="#CC2222", linewidth=1.4,
                   label=f"Non-human overall Mantel r = {OVERALL_MANTEL_R:.3f}")
    )
    ax.legend(handles=legend_handles, fontsize=7.5, loc="lower right",
              framealpha=0.85)

    ax.invert_yaxis()   # top-to-bottom: Amphibia → Aves → Mammalia
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    out_base = ROOT / "plots" / "fig_human_mantel_row"
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {out_base}.pdf/.png")


# ================================================================== #
# Main
# ================================================================== #

def main():
    print("Loading data …")
    animal_calls = load_animals()
    human_calls  = load_humans()
    print(f"  Animal calls: {len(animal_calls)}")
    print(f"  Human burst calls: {len(human_calls)}")

    # Load pre-computed animal embeddings (L2-normalised)
    ac_emb = np.load(ROOT / "paper_code" / "ac_emb.npy").astype(float)
    se_emb = np.load(ROOT / "paper_code" / "se_emb.npy").astype(float)
    # Ensure L2 normalised (should already be)
    norms = np.linalg.norm(ac_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    ac_emb = ac_emb / norms
    norms = np.linalg.norm(se_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    se_emb = se_emb / norms
    print(f"  ac_emb: {ac_emb.shape}  se_emb: {se_emb.shape}")

    # Compute (or load) human embeddings
    h_ac, h_se = load_or_compute_human_embeddings(human_calls)
    print(f"  h_ac: {h_ac.shape}  h_se: {h_se.shape}")

    # Ensure plots/ directory exists
    (ROOT / "plots").mkdir(exist_ok=True)

    # Generate figures
    fig_h1_correspondents(human_calls, animal_calls, h_se, se_emb)
    fig_h2_pmi(human_calls, animal_calls)
    fig_h3_mantel_row(human_calls, animal_calls, h_ac, h_se, ac_emb, se_emb)

    print("\nDone.")


if __name__ == "__main__":
    main()

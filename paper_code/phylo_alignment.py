"""
Explore whether cross-species acoustic--semantic alignment tracks evolutionary
divergence time.

This is intentionally exploratory. Divergence times are coarse, rank-based
heuristics so that we can inspect the shape of the result before investing in a
curated dated phylogeny or TimeTree/OpenTree lookup.

Outputs:
  plots/phylo_alignment_pairs.csv
  plots/phylo_alignment_summary.json
  plots/phylo_alignment_scatter.png
  plots/phylo_alignment_bins.png
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
try:
    from paper_code.paths import DATABASE_PATH
except ModuleNotFoundError:
    from paths import DATABASE_PATH
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import pearsonr, spearmanr

DB_PATH = DATABASE_PATH
OUT_DIR = ROOT / "plots"
TIMETREE_PAIR_PATH = OUT_DIR / "timetree_divergence_pairs.csv"
MIN_PAIRS = 6
N_PERM = 9999


CLASS_COLORS = {
    "Amphibia": "#C44E52",
    "Aves": "#55A868",
    "Mammalia": "#4C72B0",
    "Cross-class": "#8172B3",
}


@dataclass(frozen=True)
class DivergenceEstimate:
    mya: float
    relation: str
    source: str = "coarse_taxonomic_heuristic"


def normalize(emb: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return emb / norms


def load_calls(json_path: Path) -> list[dict[str, object]]:
    with open(json_path, "r", encoding="utf-8") as f:
        db = json.load(f)

    calls = []
    for species_entry in db.get("species", []):
        species_name = species_entry.get("species_name", "unknown")
        for call in species_entry.get("calls", []):
            calls.append(
                {
                    "species": species_name,
                    "call_name": call.get("call_name", "unknown"),
                    "acoustic_description": call.get("acoustic_description", ""),
                    "semantic_description": call.get("semantic_description", ""),
                    "class": species_entry.get("class", ""),
                    "order": species_entry.get("order", ""),
                    "family": species_entry.get("family", ""),
                    "genus": species_entry.get("genus", ""),
                }
            )
    return calls


def species_table(calls: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    seen = set()
    for c in calls:
        sp = str(c["species"])
        if sp in seen:
            continue
        seen.add(sp)
        rows.append(
            {
                "species": sp,
                "class": str(c.get("class", "") or ""),
                "order": str(c.get("order", "") or ""),
                "family": str(c.get("family", "") or ""),
                "genus": str(c.get("genus", "") or ""),
            }
        )
    return pd.DataFrame(rows).sort_values("species").reset_index(drop=True)


def estimate_divergence(a: pd.Series, b: pd.Series) -> DivergenceEstimate:
    """Return a coarse MYA estimate from shared taxonomic rank.

    These values are deliberately approximate midpoints. They should be replaced
    by pair-specific dated phylogenetic estimates before paper use.
    """
    if a["species"] == b["species"]:
        return DivergenceEstimate(0.0, "same_species")

    class_a, class_b = a["class"], b["class"]
    order_a, order_b = a["order"], b["order"]
    family_a, family_b = a["family"], b["family"]
    genus_a, genus_b = a["genus"], b["genus"]

    if genus_a and genus_a == genus_b:
        return DivergenceEstimate(3.0, "same_genus")
    if family_a and family_a == family_b:
        return DivergenceEstimate(12.0, "same_family")
    if order_a and order_a == order_b:
        if class_a == "Mammalia":
            return DivergenceEstimate(30.0, "same_order")
        if class_a == "Aves":
            return DivergenceEstimate(35.0, "same_order")
        if class_a == "Amphibia":
            return DivergenceEstimate(60.0, "same_order")
        return DivergenceEstimate(45.0, "same_order")
    if class_a and class_a == class_b:
        if class_a == "Mammalia":
            return DivergenceEstimate(95.0, "same_class")
        if class_a == "Aves":
            return DivergenceEstimate(80.0, "same_class")
        if class_a == "Amphibia":
            return DivergenceEstimate(160.0, "same_class")
        return DivergenceEstimate(100.0, "same_class")

    pair = {class_a, class_b}
    if pair == {"Mammalia", "Aves"}:
        return DivergenceEstimate(312.0, "cross_class")
    if "Amphibia" in pair:
        return DivergenceEstimate(355.0, "cross_class")
    return DivergenceEstimate(320.0, "cross_class")


def compute_species_pair_alignment(
    calls: list[dict[str, object]],
    ac_emb: np.ndarray,
    se_emb: np.ndarray,
    timetree_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    sp_names = np.array([str(c["species"]) for c in calls])
    meta = species_table(calls).set_index("species")

    ac = normalize(ac_emb.astype(float))
    se = normalize(se_emb.astype(float))
    sa = ac @ ac.T
    ss = se @ se.T

    species = sorted(set(sp_names))
    sp_idx = {sp: np.where(sp_names == sp)[0] for sp in species}
    timetree_lookup = {}
    if timetree_pairs is not None and len(timetree_pairs):
        for row in timetree_pairs.itertuples(index=False):
            age = getattr(row, "timetree_mya", np.nan)
            if pd.isna(age):
                continue
            key = tuple(sorted((row.species_a, row.species_b)))
            timetree_lookup[key] = row

    rows = []

    for i, sp_a in enumerate(species):
        idx_a = sp_idx[sp_a]
        for sp_b in species[i + 1 :]:
            idx_b = sp_idx[sp_b]
            ac_pairs = sa[np.ix_(idx_a, idx_b)].ravel()
            se_pairs = ss[np.ix_(idx_a, idx_b)].ravel()
            n_pairs = len(ac_pairs)
            if n_pairs < MIN_PAIRS:
                continue

            r = float(np.corrcoef(ac_pairs, se_pairs)[0, 1])
            z = float(np.arctanh(np.clip(r, -0.999999, 0.999999)))
            a = meta.loc[sp_a]
            b = meta.loc[sp_b]
            div = estimate_divergence(
                pd.Series({"species": sp_a, **a.to_dict()}),
                pd.Series({"species": sp_b, **b.to_dict()}),
            )
            divergence_mya = div.mya
            divergence_source = div.source
            timetree_study_count = np.nan
            timetree_ci_low = np.nan
            timetree_ci_high = np.nan
            timetree_name_a = ""
            timetree_name_b = ""
            timetree_key = tuple(sorted((sp_a, sp_b)))
            if timetree_key in timetree_lookup:
                tt = timetree_lookup[timetree_key]
                divergence_mya = float(tt.timetree_mya)
                divergence_source = "TimeTree API pairwise summaryjson"
                timetree_study_count = tt.timetree_study_count
                timetree_ci_low = tt.timetree_ci_low
                timetree_ci_high = tt.timetree_ci_high
                timetree_name_a = tt.timetree_name_a
                timetree_name_b = tt.timetree_name_b

            pair_class = (
                a["class"] if a["class"] and a["class"] == b["class"] else "Cross-class"
            )
            rows.append(
                {
                    "species_a": sp_a,
                    "species_b": sp_b,
                    "class_a": a["class"],
                    "class_b": b["class"],
                    "order_a": a["order"],
                    "order_b": b["order"],
                    "family_a": a["family"],
                    "family_b": b["family"],
                    "genus_a": a["genus"],
                    "genus_b": b["genus"],
                    "pair_class": pair_class,
                    "relation": div.relation,
                    "divergence_mya": divergence_mya,
                    "divergence_years": divergence_mya * 1_000_000,
                    "divergence_source": divergence_source,
                    "timetree_study_count": timetree_study_count,
                    "timetree_ci_low": timetree_ci_low,
                    "timetree_ci_high": timetree_ci_high,
                    "timetree_name_a": timetree_name_a,
                    "timetree_name_b": timetree_name_b,
                    "n_calls_a": len(idx_a),
                    "n_calls_b": len(idx_b),
                    "n_pairs": n_pairs,
                    "alignment_r": r,
                    "alignment_fisher_z": z,
                }
            )

    df = pd.DataFrame(rows)
    df["log10_divergence_mya"] = np.log10(df["divergence_mya"])
    return df


def matrix_permutation_pvalue(df: pd.DataFrame, seed: int = 42) -> float:
    """Permutation test preserving species-pair dependence approximately.

    We permute species labels on the divergence matrix and correlate the
    permuted upper triangle with the fixed alignment matrix.
    """
    rng = np.random.default_rng(seed)
    species = sorted(set(df["species_a"]) | set(df["species_b"]))
    n = len(species)
    idx = {sp: i for i, sp in enumerate(species)}

    align = np.full((n, n), np.nan)
    div = np.full((n, n), np.nan)
    for row in df.itertuples(index=False):
        i = idx[row.species_a]
        j = idx[row.species_b]
        align[i, j] = align[j, i] = row.alignment_fisher_z
        div[i, j] = div[j, i] = row.log10_divergence_mya

    tri = np.triu_indices(n, k=1)
    mask = ~np.isnan(align[tri]) & ~np.isnan(div[tri])
    observed = float(pearsonr(div[tri][mask], align[tri][mask]).statistic)

    hits = 0
    for _ in range(N_PERM):
        perm = rng.permutation(n)
        div_p = div[np.ix_(perm, perm)]
        mask_p = ~np.isnan(align[tri]) & ~np.isnan(div_p[tri])
        r_p = float(pearsonr(div_p[tri][mask_p], align[tri][mask_p]).statistic)
        if abs(r_p) >= abs(observed):
            hits += 1
    return (hits + 1) / (N_PERM + 1)


def summarize(df: pd.DataFrame) -> dict[str, object]:
    pear = pearsonr(df["log10_divergence_mya"], df["alignment_fisher_z"])
    spear = spearmanr(df["divergence_mya"], df["alignment_r"])

    wls = smf.wls(
        "alignment_fisher_z ~ log10_divergence_mya",
        data=df,
        weights=df["n_pairs"],
    ).fit()
    wls_class = smf.wls(
        "alignment_fisher_z ~ log10_divergence_mya + C(pair_class)",
        data=df,
        weights=df["n_pairs"],
    ).fit()

    by_relation = (
        df.groupby("relation")
        .agg(
            n_pairs_of_species=("alignment_r", "size"),
            median_alignment_r=("alignment_r", "median"),
            mean_alignment_r=("alignment_r", "mean"),
            median_divergence_mya=("divergence_mya", "median"),
        )
        .reset_index()
        .sort_values("median_divergence_mya")
        .to_dict(orient="records")
    )

    return {
        "n_species_pairs": int(len(df)),
        "n_species": int(len(set(df["species_a"]) | set(df["species_b"]))),
        "divergence_source_counts": {
            str(k): int(v) for k, v in df["divergence_source"].value_counts().items()
        },
        "pearson_log10_mya_vs_fisher_z": {
            "r": float(pear.statistic),
            "p": float(pear.pvalue),
        },
        "spearman_mya_vs_alignment_r": {
            "rho": float(spear.statistic),
            "p": float(spear.pvalue),
        },
        "species_label_matrix_permutation_p": matrix_permutation_pvalue(df),
        "weighted_linear_model": {
            "formula": "alignment_fisher_z ~ log10_divergence_mya",
            "slope": float(wls.params["log10_divergence_mya"]),
            "slope_p": float(wls.pvalues["log10_divergence_mya"]),
            "r2": float(wls.rsquared),
            "aic": float(wls.aic),
        },
        "weighted_linear_model_with_pair_class": {
            "formula": "alignment_fisher_z ~ log10_divergence_mya + C(pair_class)",
            "slope": float(wls_class.params["log10_divergence_mya"]),
            "slope_p": float(wls_class.pvalues["log10_divergence_mya"]),
            "r2": float(wls_class.rsquared),
            "aic": float(wls_class.aic),
        },
        "by_relation": by_relation,
    }


def add_regression_line(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    slope, intercept = np.polyfit(x, y, deg=1)
    xs = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    ax.plot(xs, intercept + slope * xs, color="black", lw=1.5, alpha=0.8)


def plot_scatter(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for pair_class, sub in df.groupby("pair_class"):
        ax.scatter(
            sub["log10_divergence_mya"],
            sub["alignment_r"],
            s=np.sqrt(sub["n_pairs"]) * 7,
            alpha=0.55,
            linewidth=0.4,
            edgecolor="white",
            color=CLASS_COLORS.get(pair_class, "#666666"),
            label=pair_class,
        )
    add_regression_line(
        ax,
        df["log10_divergence_mya"].to_numpy(),
        df["alignment_r"].to_numpy(),
    )
    ax.axhline(0, color="#777777", lw=0.8, ls=":")
    ax.set_xlabel("log10 divergence time (MYA)")
    ax.set_ylabel("Species-pair acoustic--semantic alignment r")
    ax.set_title("Does form-to-meaning alignment decay with evolutionary distance?")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.text(
        0.99,
        0.02,
        "Point size: cross-species call-pair count",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
        ha="right",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_bins(df: pd.DataFrame, out_path: Path) -> None:
    order = ["same_genus", "same_family", "same_order", "same_class", "cross_class"]
    plot_df = df[df["relation"].isin(order)].copy()
    data = [plot_df.loc[plot_df["relation"] == rel, "alignment_r"].values for rel in order]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bp = ax.boxplot(
        data,
        tick_labels=[rel.replace("_", "\n") for rel in order],
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#D9E3F0")
        patch.set_edgecolor("#44546A")
    for i, vals in enumerate(data, start=1):
        x = np.full(len(vals), i) + np.linspace(-0.08, 0.08, len(vals))
        ax.scatter(x, vals, s=12, color="#333333", alpha=0.35, linewidth=0)
    ax.axhline(0, color="#777777", lw=0.8, ls=":")
    ax.set_ylabel("Species-pair acoustic--semantic alignment r")
    ax.set_xlabel("Taxonomic divergence bin")
    ax.set_title("Alignment by coarse evolutionary-distance bin")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    calls = load_calls(DB_PATH)
    ac_emb = np.load(ROOT / "paper_code" / "ac_emb.npy")
    se_emb = np.load(ROOT / "paper_code" / "se_emb.npy")
    timetree_pairs = None
    if TIMETREE_PAIR_PATH.exists():
        timetree_pairs = pd.read_csv(TIMETREE_PAIR_PATH)

    if len(calls) != len(ac_emb) or len(calls) != len(se_emb):
        raise ValueError(
            f"Call/embedding length mismatch: calls={len(calls)}, "
            f"ac={len(ac_emb)}, se={len(se_emb)}"
        )

    df = compute_species_pair_alignment(calls, ac_emb, se_emb, timetree_pairs)
    summary = summarize(df)

    csv_path = OUT_DIR / "phylo_alignment_pairs.csv"
    json_path = OUT_DIR / "phylo_alignment_summary.json"
    scatter_path = OUT_DIR / "phylo_alignment_scatter.png"
    bins_path = OUT_DIR / "phylo_alignment_bins.png"

    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_scatter(df, scatter_path)
    plot_bins(df, bins_path)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {scatter_path}")
    print(f"Saved: {bins_path}")


if __name__ == "__main__":
    main()

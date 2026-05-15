"""
Isolation experiment: does the old-vs-new Mantel gap come from species selection or descriptions?

Four controlled conditions:
  A: Old species set (55 sp)  + Old descriptions   <- baseline old
  B: Common species only      + Old descriptions   <- isolates species selection
  C: Common species only      + New descriptions   <- isolates description style
  D: New species set (117 sp) + New descriptions   <- baseline new

Run from the project root:
    python paper_code/isolation_experiment.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from paper_code.data_sources import load_calls
from paper_code.mantel import (
    CACHE_PATH,
    EMBEDDING_MODEL,
    N_PERM,
    embed_texts,
    run_mantel_subset,
    similarity_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scientific_key(species_str: str) -> str:
    """Extract lower-cased scientific name from 'Common name (Genus species)' format."""
    m = re.search(r"\(([^)]+)\)", species_str)
    if m:
        return m.group(1).strip().lower()
    return species_str.strip().lower()


def build_matrices(calls: list[dict], encoder: SentenceTransformer):
    """Embed acoustic + semantic texts, return Sa, Ss similarity matrices."""
    acoustic_texts = [str(c["acoustic_description"]) for c in calls]
    semantic_texts = [str(c["semantic_description"]) for c in calls]
    ac_emb, _ = embed_texts(acoustic_texts, EMBEDDING_MODEL, CACHE_PATH, encoder)
    se_emb, _ = embed_texts(semantic_texts, EMBEDDING_MODEL, CACHE_PATH, encoder)
    Sa = similarity_matrix(ac_emb)
    Ss = similarity_matrix(se_emb)
    return Sa, Ss


def run_condition(label: str, calls: list[dict], encoder: SentenceTransformer) -> dict:
    """Run Mantel for all_pairs, within_species, same_family_cross_species, cross_family."""
    n = len(calls)
    print(f"\n  [{label}] Building matrices for {n} calls from "
          f"{len({c['species'] for c in calls})} species …")
    Sa, Ss = build_matrices(calls, encoder)

    sp_arr = np.array([c["species"] for c in calls])
    fa_arr = np.array([c["family"] for c in calls])

    mask_all = np.ones((n, n), dtype=bool)
    mask_within = sp_arr[:, None] == sp_arr[None, :]
    mask_same_fam = (fa_arr[:, None] == fa_arr[None, :]) & (sp_arr[:, None] != sp_arr[None, :])
    mask_cross_fam = fa_arr[:, None] != fa_arr[None, :]

    results: dict[str, dict] = {}
    for subset_label, mask in [
        ("all_pairs", mask_all),
        ("within_species", mask_within),
        ("same_family_cross_species", mask_same_fam),
        ("cross_family", mask_cross_fam),
    ]:
        r, p, k, _, _ = run_mantel_subset(Sa, Ss, mask)
        results[subset_label] = {"r": r, "p": p, "n_pairs": k}
        sig = "***" if (p is not None and p < 0.001) else (
              "**"  if (p is not None and p < 0.01)  else (
              "*"   if (p is not None and p < 0.05)  else "n.s."))
        r_str = f"{r:+.3f}" if r is not None else "  N/A "
        p_str = f"{p:.4f}" if p is not None else "  N/A"
        print(f"    {subset_label:<35s}  r={r_str}  p={p_str} {sig}  (n_pairs={k})")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ISOLATION EXPERIMENT: species selection vs description style")
    print("=" * 70)

    # Load raw calls
    old_calls = load_calls("old")
    new_calls = load_calls("new")

    print(f"\nOld database: {len(old_calls)} calls, "
          f"{len({c['species'] for c in old_calls})} species")
    print(f"New database: {len(new_calls)} calls, "
          f"{len({c['species'] for c in new_calls})} species")

    # -----------------------------------------------------------------------
    # Step 1: Find common species
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: Matching species across databases")
    print("-" * 70)

    old_by_key: dict[str, list[dict]] = {}
    for c in old_calls:
        k = scientific_key(c["species"])
        old_by_key.setdefault(k, []).append(c)

    new_by_key: dict[str, list[dict]] = {}
    for c in new_calls:
        k = scientific_key(c["species"])
        new_by_key.setdefault(k, []).append(c)

    common_keys = sorted(set(old_by_key) & set(new_by_key))
    print(f"\nCommon species: {len(common_keys)} "
          f"(out of {len(old_by_key)} old, {len(new_by_key)} new)")
    print(f"\n{'Scientific name':<40s}  {'Old calls':>9}  {'New calls':>9}")
    print("-" * 62)
    for k in common_keys:
        old_n = len(old_by_key[k])
        new_n = len(new_by_key[k])
        # Use the display name from old database
        display = old_by_key[k][0]["species"]
        print(f"  {display:<38s}  {old_n:>9}  {new_n:>9}")

    # Build common-species call lists
    common_old_calls = [c for k in common_keys for c in old_by_key[k]]
    common_new_calls = [c for k in common_keys for c in new_by_key[k]]
    print(f"\nCommon set — Old descriptions: {len(common_old_calls)} calls")
    print(f"Common set — New descriptions: {len(common_new_calls)} calls")

    # For condition C we need new descriptions but mapped so the call list
    # mirrors the new-db calls for the common species.
    # However, for a fair comparison with B we also want to note that the
    # number of calls per species may differ. We use the natural call lists
    # from each database for the common species.

    # -----------------------------------------------------------------------
    # Step 2: Four controlled conditions
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: Running four controlled Mantel conditions")
    print(f"        (n_perm={N_PERM} per subset)")
    print("-" * 70)

    encoder = SentenceTransformer(EMBEDDING_MODEL)

    print("\nCondition A: Old species set + Old descriptions (baseline old)")
    res_A = run_condition("A", old_calls, encoder)

    print("\nCondition B: Common species only + Old descriptions")
    res_B = run_condition("B", common_old_calls, encoder)

    print("\nCondition C: Common species only + New descriptions")
    res_C = run_condition("C", common_new_calls, encoder)

    print("\nCondition D: New species set + New descriptions (baseline new)")
    res_D = run_condition("D", new_calls, encoder)

    # -----------------------------------------------------------------------
    # Step 3: Summary table + interpretation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Summary table")
    print("=" * 70)

    conditions = [
        ("A  Old full    + Old desc", res_A),
        ("B  Common only + Old desc", res_B),
        ("C  Common only + New desc", res_C),
        ("D  New full    + New desc", res_D),
    ]
    subsets = ["all_pairs", "within_species", "same_family_cross_species", "cross_family"]

    # Header
    header = f"{'Condition':<28s}"
    for s in subsets:
        header += f"  {s[:22]:>22s}"
    print(header)
    print("-" * (28 + len(subsets) * 25))

    for cond_label, res in conditions:
        row = f"{cond_label:<28s}"
        for s in subsets:
            v = res.get(s, {})
            r = v.get("r")
            p = v.get("p")
            if r is None:
                row += f"  {'N/A':>22s}"
            else:
                sig = "***" if (p is not None and p < 0.001) else (
                      "**"  if (p is not None and p < 0.01)  else (
                      "*"   if (p is not None and p < 0.05)  else "n.s."))
                row += f"  {f'r={r:+.3f} {sig}':>22s}"
        print(row)

    # Focus on same_family_cross_species as the headline metric
    print("\n" + "-" * 70)
    print("Headline metric: same_family_cross_species  r")
    print("-" * 70)

    r_A = res_A["same_family_cross_species"]["r"]
    r_B = res_B["same_family_cross_species"]["r"]
    r_C = res_C["same_family_cross_species"]["r"]
    r_D = res_D["same_family_cross_species"]["r"]

    print(f"  A (old full,    old desc):  r = {r_A:+.3f}")
    print(f"  B (common only, old desc):  r = {r_B:+.3f}")
    print(f"  C (common only, new desc):  r = {r_C:+.3f}")
    print(f"  D (new full,    new desc):  r = {r_D:+.3f}")
    print()

    gap_AB = r_B - r_A if (r_A is not None and r_B is not None) else None
    gap_BC = r_C - r_B if (r_B is not None and r_C is not None) else None
    gap_AD = r_D - r_A if (r_A is not None and r_D is not None) else None

    if gap_AB is not None:
        direction = "higher" if gap_AB > 0 else "lower"
        print(f"  A -> B gap ({direction} = species selection effect): {gap_AB:+.3f}")
    if gap_BC is not None:
        direction = "higher" if gap_BC > 0 else "lower"
        print(f"  B -> C gap ({direction} = description style effect):  {gap_BC:+.3f}")
    if gap_AD is not None:
        print(f"  Total A -> D gap:                                {gap_AD:+.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    abs_AB = abs(gap_AB) if gap_AB is not None else 0
    abs_BC = abs(gap_BC) if gap_BC is not None else 0
    total  = abs_AB + abs_BC

    if total > 0:
        frac_sp   = abs_AB / total * 100
        frac_desc = abs_BC / total * 100
    else:
        frac_sp = frac_desc = 0

    print(f"\n  Species selection explains ~{frac_sp:.0f}% of the observed gap.")
    print(f"  Description style  explains ~{frac_desc:.0f}% of the observed gap.")
    print()

    if frac_sp > 60:
        dom = "SPECIES SELECTION dominates"
    elif frac_desc > 60:
        dom = "DESCRIPTION STYLE dominates"
    else:
        dom = "BOTH factors contribute roughly equally"
    print(f"  => {dom}.")
    print()

    print("  A->B: Restricting to common species changes r by "
          f"{gap_AB:+.3f} (species selection effect).")
    print("  B->C: Swapping descriptions for common species changes r by "
          f"{gap_BC:+.3f} (description style effect).")
    print()
    print("  The remaining A->D gap not explained by these two factors")
    gap_CD = r_D - r_C if (r_C is not None and r_D is not None) else None
    if gap_CD is not None:
        print(f"  (C->D = adding new-only species with new desc): {gap_CD:+.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

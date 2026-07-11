"""
Controlled isolation experiment: compare old vs new descriptions on MATCHED CALLS ONLY.

A matched call is one that exists in BOTH databases for the same species
(matched by normalized scientific name + normalized call name).

Four conditions:
  M-old       : old acoustic  + old semantic
  M-new       : new acoustic  + new semantic
  M-cross-ac  : old acoustic  + new semantic
  M-cross-se  : new acoustic  + old semantic

Run from the project root:
    python paper_code/controlled_isolation.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from paper_code.data_sources import load_calls
from paper_code.mantel import (
    CACHE_PATH,
    EMBEDDING_MODEL,
    embed_texts,
    run_mantel_subset,
    similarity_matrix,
)
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_scientific(species_str: str) -> str:
    """Pull scientific name from 'Common name (Genus species)' format."""
    m = re.search(r'\(([^)]+)\)', species_str)
    return m.group(1).lower().strip() if m else species_str.lower().strip()


def norm_call(name: str) -> str:
    """Lowercase, collapse whitespace/underscores/hyphens."""
    return re.sub(r'[\s_\-]+', ' ', name.lower().strip())


# ---------------------------------------------------------------------------
# Step 1-2: Load both databases and find matched calls
# ---------------------------------------------------------------------------

print("=" * 70)
print("STEP 1-2: Loading databases and matching calls")
print("=" * 70)

calls_old = load_calls("old")
calls_new = load_calls("new")
print(f"  Old database: {len(calls_old)} calls, "
      f"{len({c['species'] for c in calls_old})} species")
print(f"  New database: {len(calls_new)} calls, "
      f"{len({c['species'] for c in calls_new})} species")

# Build lookup: scientific_name -> {norm_call_name -> call_dict}
def build_lookup(calls: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for c in calls:
        sci = extract_scientific(c["species"])
        nc = norm_call(c["call_name"])
        lookup.setdefault(sci, {})[nc] = c
    return lookup

old_by_sp = build_lookup(calls_old)
new_by_sp = build_lookup(calls_new)

common_species = set(old_by_sp.keys()) & set(new_by_sp.keys())
print(f"\n  Common species: {len(common_species)}")

# Find matched calls
matched: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]] = []
unmatched_old_sample: Dict[str, List[str]] = {}
unmatched_new_sample: Dict[str, List[str]] = {}

for sci in sorted(common_species):
    old_calls_sp = old_by_sp[sci]
    new_calls_sp = new_by_sp[sci]
    common_nc = set(old_calls_sp.keys()) & set(new_calls_sp.keys())
    for nc in common_nc:
        matched.append((sci, nc, old_calls_sp[nc], new_calls_sp[nc]))
    # Collect unmatched for sample reporting
    only_old = set(old_calls_sp.keys()) - set(new_calls_sp.keys())
    only_new = set(new_calls_sp.keys()) - set(old_calls_sp.keys())
    if only_old:
        unmatched_old_sample[sci] = sorted(only_old)
    if only_new:
        unmatched_new_sample[sci] = sorted(only_new)

print(f"  Matched calls total: {len(matched)}\n")

# Report top contributing species
sp_counts = Counter(sci for sci, nc, o, n in matched)
print("  Top species by matched call count:")
for sp, cnt in sp_counts.most_common(15):
    print(f"    {sp:<35s} {cnt:3d} matched calls")

# Report a few unmatched samples
print("\n  Sample unmatched call names (first 3 species with unmatched):")
sample_sp = list(unmatched_old_sample.keys())[:3]
for sp in sample_sp:
    print(f"    {sp} — only in OLD: {unmatched_old_sample[sp][:4]}")
    if sp in unmatched_new_sample:
        print(f"    {sp} — only in NEW: {unmatched_new_sample[sp][:4]}")

print()

# ---------------------------------------------------------------------------
# Step 3: Build the 4 text lists and taxonomy arrays from matched calls
# ---------------------------------------------------------------------------

print("=" * 70)
print("STEP 3: Building condition text lists")
print("=" * 70)

ac_old_texts = [str(o["acoustic_description"]) for _, _, o, n in matched]
se_old_texts  = [str(o["semantic_description"]) for _, _, o, n in matched]
ac_new_texts  = [str(n["acoustic_description"]) for _, _, o, n in matched]
se_new_texts  = [str(n["semantic_description"]) for _, _, o, n in matched]

# Taxonomy from old (they refer to the same species, so either works)
species_arr = np.array([o["species"] for _, _, o, n in matched])
families_arr = np.array([str(o.get("family", "")) for _, _, o, n in matched])

print(f"  {len(matched)} matched calls across {len(set(sp for sp, nc, o, n in matched))} species")
print(f"  Unique families: {len(set(families_arr))}\n")

# ---------------------------------------------------------------------------
# Step 4: Embed all unique texts and compute Mantel r
# ---------------------------------------------------------------------------

print("=" * 70)
print("STEP 4: Embedding texts and running Mantel tests")
print("=" * 70)

encoder = SentenceTransformer(EMBEDDING_MODEL)

# Gather all texts to embed in one pass (uses cache efficiently)
all_texts = list(set(ac_old_texts + se_old_texts + ac_new_texts + se_new_texts))
print(f"  Embedding {len(all_texts)} unique texts …")
_, _ = embed_texts(all_texts, EMBEDDING_MODEL, CACHE_PATH, encoder)

def get_emb(texts: List[str]) -> np.ndarray:
    emb, _ = embed_texts(texts, EMBEDDING_MODEL, CACHE_PATH, encoder)
    return emb

# Condition embeddings
emb = {
    "ac_old": get_emb(ac_old_texts),
    "se_old": get_emb(se_old_texts),
    "ac_new": get_emb(ac_new_texts),
    "se_new": get_emb(se_new_texts),
}

# Condition definitions: (label, ac_key, se_key)
conditions = [
    ("M-old",      "ac_old", "se_old"),
    ("M-new",      "ac_new", "se_new"),
    ("M-cross-ac", "ac_old", "se_new"),  # old acoustic + new semantic
    ("M-cross-se", "ac_new", "se_old"),  # new acoustic + old semantic
]

n = len(matched)

# Build masks
mask_all    = np.ones((n, n), dtype=bool)
mask_within = species_arr[:, None] == species_arr[None, :]
mask_fam    = (
    (families_arr[:, None] == families_arr[None, :])
    & (species_arr[:, None] != species_arr[None, :])
)

print()
print(f"  n_calls={n}  n_pairs_all={int(n*(n-1)/2)}  "
      f"n_pairs_within={int(mask_within.sum()/2)}  "
      f"n_pairs_same_fam_cross_sp={int(mask_fam.sum()/2)}")
print()

# ---------------------------------------------------------------------------
# Step 5: Report results
# ---------------------------------------------------------------------------

print("=" * 70)
print("STEP 5: Results")
print("=" * 70)

header = f"{'Condition':<14}  {'Subset':<30}  {'r':>7}  {'p':>8}  {'n_pairs':>9}"
print(header)
print("-" * len(header))

results: Dict[str, Any] = {}

for label, ac_key, se_key in conditions:
    Sa = similarity_matrix(emb[ac_key])
    Ss = similarity_matrix(emb[se_key])

    subsets = [
        ("all_pairs",               mask_all),
        ("within_species",          mask_within),
        ("same_family_cross_sp",    mask_fam),
    ]

    results[label] = {}
    for subset_name, mask in subsets:
        r, p, k, _, _ = run_mantel_subset(Sa, Ss, mask)
        results[label][subset_name] = {"r": r, "p": p, "n_pairs": k}
        if r is None:
            print(f"  {label:<14}  {subset_name:<30}  {'N/A':>7}  {'N/A':>8}  {k:>9d}")
        else:
            sig = ("***" if p < 0.001
                   else "**" if p < 0.01
                   else "*" if p < 0.05
                   else "n.s.")
            print(f"  {label:<14}  {subset_name:<30}  {r:>+7.3f}  {p:>8.4f}  {k:>9d}  {sig}")

print()
print("=" * 70)
print("INTERPRETATION SUMMARY")
print("=" * 70)
same_fam_r = {label: results[label]["same_family_cross_sp"]["r"] for label in results}
print(f"  Same-family cross-species r:")
for label, r in same_fam_r.items():
    print(f"    {label:<14}: r = {r:+.3f}")

# Decompose the gap
r_old = same_fam_r["M-old"]
r_new = same_fam_r["M-new"]
r_cross_ac = same_fam_r["M-cross-ac"]  # old acoustic + new semantic
r_cross_se = same_fam_r["M-cross-se"]  # new acoustic + old semantic
total_gap = r_old - r_new

print(f"\n  Total gap (M-old - M-new) = {total_gap:+.3f}")
if total_gap != 0:
    # Swapping semantic old->new while keeping acoustic old:  cross_ac vs old
    acoustic_contribution  = r_old - r_cross_ac   # effect of swapping semantic (old->new)
    semantic_contribution  = r_old - r_cross_se   # effect of swapping acoustic (old->new)
    print(f"  Swapping semantic  (old->new), keep acoustic old: {r_old:+.3f} -> {r_cross_ac:+.3f}  (delta = {-acoustic_contribution:+.3f})")
    print(f"  Swapping acoustic  (old->new), keep semantic old: {r_old:+.3f} -> {r_cross_se:+.3f}  (delta = {-semantic_contribution:+.3f})")
    print()
    print("  Conclusion:")
    if abs(acoustic_contribution) > abs(semantic_contribution):
        print("    Semantic descriptions drive most of the gap (swapping semantic -> large change)")
    elif abs(semantic_contribution) > abs(acoustic_contribution):
        print("    Acoustic descriptions drive most of the gap (swapping acoustic -> large change)")
    else:
        print("    Both description types contribute roughly equally to the gap")

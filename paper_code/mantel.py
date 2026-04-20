"""
Compute Mantel test statistics for the paper.
Reuses the embedding cache and load_calls from app/utils.py.

Run from the project root:
    python paper_code/mantel.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils import load_calls, embed_texts

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_PATH = ROOT / ".embedding_cache.json"
DB_PATH = ROOT / "database.json"
N_PERM = 9999


def mantel(a_vec, s_vec, n_perm=N_PERM, seed=42):
    rng = np.random.default_rng(seed)
    r_obs = float(np.corrcoef(a_vec, s_vec)[0, 1])
    n = int(np.round((1 + np.sqrt(1 + 8 * len(a_vec))) / 2))  # recover matrix size
    # permutation on vector level (equivalent to row/col permutation on upper triangle)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(s_vec))
        r_p = float(np.corrcoef(a_vec, s_vec[perm])[0, 1])
        if r_p >= r_obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return r_obs, p


def similarity_matrix(emb):
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    return e @ e.T


def run_mantel_subset(Sa, Ss, mask_2d):
    """Extract upper-triangle pairs where mask_2d is True, then run Mantel."""
    n = Sa.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    pair_mask = mask_2d[tri_i, tri_j]
    a_vec = Sa[tri_i, tri_j][pair_mask]
    s_vec = Ss[tri_i, tri_j][pair_mask]
    if len(a_vec) < 3:
        return None, None, len(a_vec)
    r, p = mantel(a_vec, s_vec)
    return r, p, len(a_vec)


def main():
    calls = load_calls(DB_PATH)
    print(f"Loaded {len(calls)} calls from {len({c['species'] for c in calls})} species")

    acoustic_texts = [str(c["acoustic_description"]) for c in calls]
    semantic_texts = [str(c["semantic_description"]) for c in calls]
    species      = [str(c["species"]) for c in calls]
    families     = [str(c["family"]) for c in calls]

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    ac_emb, _ = embed_texts(acoustic_texts, EMBEDDING_MODEL, CACHE_PATH, encoder)
    se_emb, _ = embed_texts(semantic_texts, EMBEDDING_MODEL, CACHE_PATH, encoder)

    Sa = similarity_matrix(ac_emb)
    Ss = similarity_matrix(se_emb)
    n = len(calls)

    sp_arr = np.array(species)
    fa_arr = np.array(families)

    # ---- All pairs ----
    mask_all = np.ones((n, n), dtype=bool)
    r_all, p_all, k_all = run_mantel_subset(Sa, Ss, mask_all)

    # ---- Within-species pairs ----
    mask_within = sp_arr[:, None] == sp_arr[None, :]
    r_within, p_within, k_within = run_mantel_subset(Sa, Ss, mask_within)

    # ---- Same family, cross-species ----
    mask_same_fam = (fa_arr[:, None] == fa_arr[None, :]) & (sp_arr[:, None] != sp_arr[None, :])
    r_fam, p_fam, k_fam = run_mantel_subset(Sa, Ss, mask_same_fam)

    # ---- Cross-family pairs ----
    mask_cross_fam = fa_arr[:, None] != fa_arr[None, :]
    r_cross, p_cross, k_cross = run_mantel_subset(Sa, Ss, mask_cross_fam)

    results = {
        "all_pairs":           {"r": r_all,    "p": p_all,    "n_pairs": k_all},
        "within_species":      {"r": r_within, "p": p_within, "n_pairs": k_within},
        "same_family_cross_species": {"r": r_fam,   "p": p_fam,   "n_pairs": k_fam},
        "cross_family":        {"r": r_cross,  "p": p_cross,  "n_pairs": k_cross},
    }

    print(f"\nMantel test results (n_perm={N_PERM})\n{'='*55}")
    for label, v in results.items():
        r, p, k = v["r"], v["p"], v["n_pairs"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(f"  {label:<35s}  r={r:+.3f}  p={p:.4f} {sig}  (n={k})")

    out = ROOT / "paper_code" / "mantel_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()

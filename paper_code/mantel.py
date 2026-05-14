"""
Compute Mantel test statistics for the paper.
Reuses the embedding cache and shared paper data-source loader.

Run from the project root:
    python paper_code/mantel.py --data-source old
    python paper_code/mantel.py --data-source new
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from paper_code.data_sources import DATA_SOURCES, artifact_path, load_calls

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_PATH = ROOT / ".embedding_cache.json"
N_PERM = 9999


def load_cache(cache_path: Path, model: str) -> Dict[str, List[float]]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if payload.get("model") != model:
        return {}
    return payload.get("embeddings", {})


def save_cache(cache_path: Path, model: str, cache: Dict[str, List[float]]) -> None:
    cache_path.write_text(json.dumps({"model": model, "embeddings": cache}), encoding="utf-8")


def batch_iter(seq: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def embed_texts(
    texts: List[str],
    model_name: str,
    cache_path: Path,
    encoder: SentenceTransformer,
    batch_size: int = 64,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    cache = load_cache(cache_path, model_name)
    vectors: List[List[float]] = []
    missing: List[str] = [t for t in texts if t not in cache]

    for chunk in batch_iter(missing, batch_size):
        if not chunk:
            continue
        embeds = encoder.encode(
            chunk,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for text, emb in zip(chunk, embeds):
            cache[text] = emb.tolist()

    for text in texts:
        vectors.append(cache[text])

    save_cache(cache_path, model_name, cache)
    return np.array(vectors, dtype=np.float32), cache


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
    """Extract upper-triangle pairs where mask_2d is True, then run Mantel.

    Returns (r, p, n_pairs, a_vec, s_vec).
    """
    n = Sa.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    pair_mask = mask_2d[tri_i, tri_j]
    a_vec = Sa[tri_i, tri_j][pair_mask]
    s_vec = Ss[tri_i, tri_j][pair_mask]
    if len(a_vec) < 3:
        return None, None, len(a_vec), a_vec, s_vec
    r, p = mantel(a_vec, s_vec)
    return r, p, len(a_vec), a_vec, s_vec


def partial_mantel(Sa, Ss, C, n_perm=N_PERM, seed=42):
    """Partial Mantel: correlation between Sa and Ss controlling for C.

    Computes the partial correlation r(Sa, Ss | C) and tests it by
    permuting the Ss vector (equivalent to permuting Ss rows/cols).
    """
    rng = np.random.default_rng(seed)
    n = Sa.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    a_vec = Sa[tri_i, tri_j]
    s_vec = Ss[tri_i, tri_j]
    c_vec = C[tri_i, tri_j]

    r_ab = float(np.corrcoef(a_vec, s_vec)[0, 1])
    r_ac = float(np.corrcoef(a_vec, c_vec)[0, 1])
    r_bc = float(np.corrcoef(s_vec, c_vec)[0, 1])
    denom = np.sqrt((1 - r_ac**2) * (1 - r_bc**2)) + 1e-12
    r_partial = (r_ab - r_ac * r_bc) / denom

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Ss_p = Ss[np.ix_(perm, perm)]
        s_p  = Ss_p[tri_i, tri_j]
        r_ab_p = float(np.corrcoef(a_vec, s_p)[0, 1])
        r_bc_p = float(np.corrcoef(s_p, c_vec)[0, 1])
        r_p = (r_ab_p - r_ac * r_bc_p) / denom
        if r_p >= r_partial:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return float(r_partial), p


def bootstrap_ci(a_vec, s_vec, n_boot=1000, alpha=0.05, seed=0):
    """Bootstrap confidence interval for the Mantel correlation.

    Resamples pairs (a_vec[idx], s_vec[idx]) with replacement n_boot times
    and returns the (alpha/2, 1-alpha/2) percentile interval.
    """
    rng = np.random.default_rng(seed)
    m = len(a_vec)
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, m, size=m)
        rs[i] = float(np.corrcoef(a_vec[idx], s_vec[idx])[0, 1])
    ci_lo = float(np.percentile(rs, 100 * alpha / 2))
    ci_hi = float(np.percentile(rs, 100 * (1 - alpha / 2)))
    return ci_lo, ci_hi


def bootstrap_ci_partial(Sa, Ss, C, n_boot=500, seed=0):
    """Bootstrap confidence interval for the partial Mantel correlation.

    Resamples matrix rows/columns with replacement, recomputes r_partial
    each time, and returns the 2.5th and 97.5th percentile interval.
    """
    rng = np.random.default_rng(seed)
    n = Sa.shape[0]
    rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Sa_b = Sa[np.ix_(idx, idx)]
        Ss_b = Ss[np.ix_(idx, idx)]
        C_b  = C[np.ix_(idx, idx)]
        tri_i, tri_j = np.triu_indices(n, k=1)
        a_b = Sa_b[tri_i, tri_j]
        s_b = Ss_b[tri_i, tri_j]
        c_b = C_b[tri_i, tri_j]
        r_ab = float(np.corrcoef(a_b, s_b)[0, 1])
        r_ac = float(np.corrcoef(a_b, c_b)[0, 1])
        r_bc = float(np.corrcoef(s_b, c_b)[0, 1])
        denom = np.sqrt((1 - r_ac**2) * (1 - r_bc**2)) + 1e-12
        rs[i] = (r_ab - r_ac * r_bc) / denom
    ci_lo = float(np.percentile(rs, 2.5))
    ci_hi = float(np.percentile(rs, 97.5))
    return ci_lo, ci_hi


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-source", choices=DATA_SOURCES, default="old")
    return parser.parse_args()


def main():
    args = parse_args()
    calls = load_calls(args.data_source)
    print(
        f"Loaded {len(calls)} calls from {len({c['species'] for c in calls})} "
        f"species, source={args.data_source}"
    )

    acoustic_texts = [str(c["acoustic_description"]) for c in calls]
    semantic_texts = [str(c["semantic_description"]) for c in calls]
    species      = [str(c["species"]) for c in calls]
    families     = [str(c["family"]) for c in calls]

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    ac_emb, _ = embed_texts(acoustic_texts, EMBEDDING_MODEL, CACHE_PATH, encoder)
    se_emb, _ = embed_texts(semantic_texts, EMBEDDING_MODEL, CACHE_PATH, encoder)

    ac_out = artifact_path("ac_emb", args.data_source, "npy")
    se_out = artifact_path("se_emb", args.data_source, "npy")
    np.save(ac_out, ac_emb)
    np.save(se_out, se_emb)
    print(f"Saved embeddings → {ac_out}, {se_out}")

    Sa = similarity_matrix(ac_emb)
    Ss = similarity_matrix(se_emb)
    n = len(calls)

    sp_arr = np.array(species)
    fa_arr = np.array(families)

    # ---- All pairs ----
    mask_all = np.ones((n, n), dtype=bool)
    r_all, p_all, k_all, av_all, sv_all = run_mantel_subset(Sa, Ss, mask_all)
    ci_all = bootstrap_ci(av_all, sv_all)

    # ---- Within-species pairs ----
    mask_within = sp_arr[:, None] == sp_arr[None, :]
    r_within, p_within, k_within, av_within, sv_within = run_mantel_subset(Sa, Ss, mask_within)
    ci_within = bootstrap_ci(av_within, sv_within)

    # ---- Same family, cross-species ----
    mask_same_fam = (fa_arr[:, None] == fa_arr[None, :]) & (sp_arr[:, None] != sp_arr[None, :])
    r_fam, p_fam, k_fam, av_fam, sv_fam = run_mantel_subset(Sa, Ss, mask_same_fam)
    ci_fam = bootstrap_ci(av_fam, sv_fam)

    # ---- Cross-family pairs ----
    mask_cross_fam = fa_arr[:, None] != fa_arr[None, :]
    r_cross, p_cross, k_cross, av_cross, sv_cross = run_mantel_subset(Sa, Ss, mask_cross_fam)
    ci_cross = bootstrap_ci(av_cross, sv_cross)

    # ---- Partial Mantel controlling for species identity ----
    C_species = (sp_arr[:, None] == sp_arr[None, :]).astype(float)
    print("Running partial Mantel (controlling for species identity) …")
    r_partial, p_partial = partial_mantel(Sa, Ss, C_species)
    ci_partial = bootstrap_ci_partial(Sa, Ss, C_species)

    # ---- Partial Mantel controlling for family identity ----
    C_family = (fa_arr[:, None] == fa_arr[None, :]).astype(float)
    print("Running partial Mantel (controlling for family identity) …")
    r_partial_fam, p_partial_fam = partial_mantel(Sa, Ss, C_family)
    ci_partial_fam = bootstrap_ci_partial(Sa, Ss, C_family)

    results = {
        "all_pairs":                     {"r": r_all,         "p": p_all,         "n_pairs": k_all,     "ci_lo": ci_all[0],         "ci_hi": ci_all[1]},
        "within_species (pooled)":       {"r": r_within,      "p": p_within,      "n_pairs": k_within,  "ci_lo": ci_within[0],      "ci_hi": ci_within[1]},
        "same_family_cross_species":     {"r": r_fam,         "p": p_fam,         "n_pairs": k_fam,     "ci_lo": ci_fam[0],         "ci_hi": ci_fam[1]},
        "cross_family":                  {"r": r_cross,       "p": p_cross,       "n_pairs": k_cross,   "ci_lo": ci_cross[0],       "ci_hi": ci_cross[1]},
        "partial | species identity":    {"r": r_partial,     "p": p_partial,     "n_pairs": k_all,     "ci_lo": ci_partial[0],     "ci_hi": ci_partial[1]},
        "partial | family identity":     {"r": r_partial_fam, "p": p_partial_fam, "n_pairs": k_all,     "ci_lo": ci_partial_fam[0], "ci_hi": ci_partial_fam[1]},
    }

    print(f"\nMantel test results (n_perm={N_PERM})\n{'='*70}")
    for label, v in results.items():
        r, p, k = v["r"], v["p"], v["n_pairs"]
        ci_lo, ci_hi = v["ci_lo"], v["ci_hi"]
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(f"  {label:<35s}  r={r:+.3f}  [{ci_lo:+.3f}, {ci_hi:+.3f}]  p={p:.4f} {sig}  (n={k})")

    out = artifact_path("mantel_results", args.data_source, "json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()

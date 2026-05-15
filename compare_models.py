"""
Compare sentence-transformer models for Mantel r on the new animal call database.
Tests: all-MiniLM-L6-v2 (baseline), all-mpnet-base-v2, all-MiniLM-L12-v2
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from paper_code.data_sources import load_calls
from paper_code.mantel import similarity_matrix, run_mantel_subset

MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
]

SUBSETS = ["all_pairs", "within_species", "same_family_cross_species"]


def embed(texts, model_name):
    encoder = SentenceTransformer(model_name)
    emb = encoder.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.array(emb, dtype=np.float32)


def main():
    print("Loading new database calls...")
    calls = load_calls("new")
    print(f"Loaded {len(calls)} calls from {len({c['species'] for c in calls})} species")

    acoustic_texts = [str(c["acoustic_description"]) for c in calls]
    semantic_texts = [str(c["semantic_description"]) for c in calls]
    species  = np.array([str(c["species"]) for c in calls])
    families = np.array([str(c["family"]) for c in calls])

    n = len(calls)
    mask_all        = np.ones((n, n), dtype=bool)
    mask_within     = species[:, None] == species[None, :]
    mask_same_fam   = (families[:, None] == families[None, :]) & (species[:, None] != species[None, :])

    masks = {
        "all_pairs":                  mask_all,
        "within_species":             mask_within,
        "same_family_cross_species":  mask_same_fam,
    }

    results = {}

    for model_name in MODELS:
        short = model_name.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Model: {short}")
        print(f"{'='*60}")

        print("  Encoding acoustic descriptions...")
        ac_emb = embed(acoustic_texts, model_name)
        print("  Encoding semantic descriptions...")
        se_emb = embed(semantic_texts, model_name)

        Sa = similarity_matrix(ac_emb)
        Ss = similarity_matrix(se_emb)

        row = {}
        for subset_name, mask in masks.items():
            r, p, k, _, _ = run_mantel_subset(Sa, Ss, mask)
            row[subset_name] = (r, p, k)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            print(f"  {subset_name:<35s}  r={r:+.3f}  p={p:.4f} {sig}  (n={k})")
        results[short] = row

    # --- Clean comparison table ---
    print("\n\n" + "="*80)
    print("COMPARISON TABLE  (new database, 633 calls, 117 species)")
    print("="*80)
    header = f"{'Model':<30s}" + "".join(f"  {'Mantel r':>10s}" for _ in SUBSETS)
    print(f"{'Model':<30s}" + "".join(f"  {s:>32s}" for s in SUBSETS))
    print("-"*80)
    for model_short, row in results.items():
        line = f"{model_short:<30s}"
        for s in SUBSETS:
            r, p, k = row[s]
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            line += f"  {r:+.3f} ({sig}){'':<18s}"
        print(line)
    print("="*80)
    print("Significance: *** p<0.001  ** p<0.01  * p<0.05  n.s. not significant")
    print("Subsets: all_pairs | within_species | same_family_cross_species")


if __name__ == "__main__":
    main()

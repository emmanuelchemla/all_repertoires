"""
Compute semantic and acoustic embeddings for all calls in database.json,
run Mantel tests, and train/evaluate form-to-meaning prediction models.

Outputs:
  - paper_code/embeddings.npz  (acoustic, semantic, metadata arrays)
  - plots/mantel_scatter.png
  - plots/pmi_heatmap.png
  - paper_code/results.json
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
try:
    from paper_code.paths import DATABASE_PATH
except ModuleNotFoundError:
    from paths import DATABASE_PATH
DB_PATH = DATABASE_PATH
PLOTS_DIR = ROOT / "plots"
OUT_DIR = Path(__file__).parent
PLOTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------ #
# Load data
# ------------------------------------------------------------------ #

def load_calls():
    with open(DB_PATH) as f:
        db = json.load(f)
    records = []
    for species in db["species"]:
        for call in species.get("calls", []):
            records.append({
                "species": species["species_name"],
                "class": species.get("class", ""),
                "order": species.get("order", ""),
                "family": species.get("family", ""),
                "call_name": call.get("call_name", ""),
                "acoustic_description": call.get("acoustic_description", ""),
                "semantic_description": call.get("semantic_description", ""),
                "keywords": call.get("ontology_keywords", []),
                "reliability": call.get("subjective_reliability", ""),
            })
    return records


# ------------------------------------------------------------------ #
# Embeddings
# ------------------------------------------------------------------ #

def get_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True)


# ------------------------------------------------------------------ #
# Mantel test
# ------------------------------------------------------------------ #

def mantel_test(dist_x, dist_y, n_permutations=9999, seed=42):
    """Pearson Mantel test on flattened upper triangles."""
    rng = np.random.default_rng(seed)
    n = dist_x.shape[0]
    idx = np.triu_indices(n, k=1)
    x = dist_x[idx]
    y = dist_y[idx]
    # center
    xc = x - x.mean()
    yc = y - y.mean()
    observed_r = (xc * yc).sum() / (np.linalg.norm(xc) * np.linalg.norm(yc))
    # permutation
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        yp = dist_y[np.ix_(perm, perm)][idx]
        ypc = yp - yp.mean()
        rp = (xc * ypc).sum() / (np.linalg.norm(xc) * np.linalg.norm(ypc))
        if rp >= observed_r:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return float(observed_r), p_value


def cosine_dist_matrix(embeddings):
    # embeddings assumed L2-normalized
    sim = embeddings @ embeddings.T
    return 1 - sim


# ------------------------------------------------------------------ #
# PMI keyword analysis
# ------------------------------------------------------------------ #

def compute_pmi(records, acoustic_keywords, semantic_keywords):
    """Compute PMI between acoustic and semantic keywords via co-occurrence in calls."""
    n = len(records)
    # For acoustic keywords we extract them from acoustic descriptions using simple term matching
    acoustic_counts = Counter()
    semantic_counts = Counter()
    co_counts = Counter()

    for rec in records:
        ac_desc = rec["acoustic_description"].lower()
        sem_kws = set(rec["keywords"])
        present_ac = {kw for kw in acoustic_keywords if kw.lower() in ac_desc}
        for ac in present_ac:
            acoustic_counts[ac] += 1
        for sk in sem_kws:
            if sk in semantic_keywords:
                semantic_counts[sk] += 1
        for ac in present_ac:
            for sk in sem_kws:
                if sk in semantic_keywords:
                    co_counts[(ac, sk)] += 1

    pmi = {}
    for ac in acoustic_keywords:
        for sk in semantic_keywords:
            p_ac = acoustic_counts[ac] / n
            p_sk = semantic_counts[sk] / n
            p_co = co_counts[(ac, sk)] / n
            if p_co > 0 and p_ac > 0 and p_sk > 0:
                pmi[(ac, sk)] = np.log2(p_co / (p_ac * p_sk))
            else:
                pmi[(ac, sk)] = 0.0
    return pmi, acoustic_counts, semantic_counts


# ------------------------------------------------------------------ #
# Prediction: acoustic → semantic
# ------------------------------------------------------------------ #

def leave_one_species_out(records, acoustic_emb, semantic_emb):
    """For each species, train on the rest, predict on held-out species."""
    species_list = list({r["species"] for r in records})
    results = []
    for held_out in species_list:
        train_idx = [i for i, r in enumerate(records) if r["species"] != held_out]
        test_idx  = [i for i, r in enumerate(records) if r["species"] == held_out]
        if not test_idx:
            continue
        X_train = acoustic_emb[train_idx]
        Y_train = semantic_emb[train_idx]
        X_test  = acoustic_emb[test_idx]
        Y_test  = semantic_emb[test_idx]
        # Ridge regression
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0).fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)
        # Normalize predictions
        norms = np.linalg.norm(Y_pred, axis=1, keepdims=True)
        norms[norms == 0] = 1
        Y_pred = Y_pred / norms
        cos_sim = (Y_pred * Y_test).sum(axis=1).mean()
        results.append({"held_out_species": held_out, "cos_sim": float(cos_sim)})
    return results


def leave_one_family_out(records, acoustic_emb, semantic_emb):
    families = list({r["family"] for r in records})
    results = []
    for held_out in families:
        train_idx = [i for i, r in enumerate(records) if r["family"] != held_out]
        test_idx  = [i for i, r in enumerate(records) if r["family"] == held_out]
        if len(test_idx) < 2:
            continue
        X_train = acoustic_emb[train_idx]
        Y_train = semantic_emb[train_idx]
        X_test  = acoustic_emb[test_idx]
        Y_test  = semantic_emb[test_idx]
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1.0).fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)
        norms = np.linalg.norm(Y_pred, axis=1, keepdims=True)
        norms[norms == 0] = 1
        Y_pred = Y_pred / norms
        cos_sim = (Y_pred * Y_test).sum(axis=1).mean()
        results.append({"held_out_family": held_out, "cos_sim": float(cos_sim)})
    return results


# ------------------------------------------------------------------ #
# Figures
# ------------------------------------------------------------------ #

def plot_mantel_scatter(acoustic_dist, semantic_dist, records, out_path):
    import matplotlib.pyplot as plt
    n = len(records)
    idx = np.triu_indices(n, k=1)
    x = acoustic_dist[idx]
    y = semantic_dist[idx]
    # label by whether same species, same family, different family
    labels = []
    for i, j in zip(*idx):
        if records[i]["species"] == records[j]["species"]:
            labels.append("Within species")
        elif records[i]["family"] == records[j]["family"]:
            labels.append("Same family")
        else:
            labels.append("Across families")
    labels = np.array(labels)
    colors = {"Within species": "#2196F3", "Same family": "#4CAF50", "Across families": "#FF9800"}

    fig, ax = plt.subplots(figsize=(6, 5))
    for lbl, col in colors.items():
        mask = labels == lbl
        ax.scatter(x[mask], y[mask], c=col, s=3, alpha=0.3, label=lbl, rasterized=True)
    ax.set_xlabel("Acoustic distance")
    ax.set_ylabel("Semantic distance")
    ax.legend(markerscale=4, framealpha=0.8)
    ax.set_title("Form-to-meaning stability (Mantel test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_pmi_heatmap(pmi, acoustic_keywords, semantic_keywords, out_path):
    import matplotlib.pyplot as plt
    mat = np.array([[pmi.get((ac, sk), 0.0) for sk in semantic_keywords]
                    for ac in acoustic_keywords])
    fig, ax = plt.subplots(figsize=(max(8, len(semantic_keywords) * 0.7),
                                    max(4, len(acoustic_keywords) * 0.5)))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_xticks(range(len(semantic_keywords)))
    ax.set_xticklabels(semantic_keywords, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(acoustic_keywords)))
    ax.set_yticklabels(acoustic_keywords, fontsize=8)
    plt.colorbar(im, ax=ax, label="PMI (bits)")
    ax.set_title("Acoustic–semantic keyword PMI")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    records = load_calls()
    print(f"Loaded {len(records)} calls from {len({r['species'] for r in records})} species")

    acoustic_texts  = [r["acoustic_description"] for r in records]
    semantic_texts  = [r["semantic_description"]  for r in records]

    print("Computing embeddings...")
    acoustic_emb = get_embeddings(acoustic_texts)
    semantic_emb = get_embeddings(semantic_texts)

    np.savez(OUT_DIR / "embeddings.npz",
             acoustic=acoustic_emb, semantic=semantic_emb,
             species=[r["species"] for r in records],
             family=[r["family"] for r in records])

    acoustic_dist = cosine_dist_matrix(acoustic_emb)
    semantic_dist = cosine_dist_matrix(semantic_emb)

    print("Running Mantel test (all pairs)...")
    r_all, p_all = mantel_test(acoustic_dist, semantic_dist, n_permutations=999)
    print(f"  r = {r_all:.3f}, p = {p_all:.4f}")

    # Within-species pairs only
    n = len(records)
    same_species = np.array([[records[i]["species"] == records[j]["species"]
                               for j in range(n)] for i in range(n)])
    within_idx = np.where(np.triu(same_species, k=1))
    if len(within_idx[0]) > 10:
        sub_ac = np.zeros((n, n))
        sub_se = np.zeros((n, n))
        sub_ac[within_idx] = acoustic_dist[within_idx]
        sub_se[within_idx] = semantic_dist[within_idx]
        r_within, p_within = mantel_test(sub_ac, sub_se, n_permutations=999)
        print(f"  Within-species: r = {r_within:.3f}, p = {p_within:.4f}")
    else:
        r_within, p_within = None, None

    results = {
        "mantel_all": {"r": r_all, "p": p_all},
        "mantel_within_species": {"r": r_within, "p": p_within},
    }

    print("Running leave-one-species-out prediction...")
    loso = leave_one_species_out(records, acoustic_emb, semantic_emb)
    mean_loso = np.mean([x["cos_sim"] for x in loso])
    print(f"  Mean cosine similarity (across species): {mean_loso:.3f}")
    results["loso_mean_cos_sim"] = mean_loso

    print("Running leave-one-family-out prediction...")
    lofo = leave_one_family_out(records, acoustic_emb, semantic_emb)
    if lofo:
        mean_lofo = np.mean([x["cos_sim"] for x in lofo])
        print(f"  Mean cosine similarity (across families): {mean_lofo:.3f}")
        results["lofo_mean_cos_sim"] = mean_lofo

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_mantel_scatter(acoustic_dist, semantic_dist, records,
                        PLOTS_DIR / "mantel_scatter.png")

    # PMI analysis
    acoustic_keywords = [
        "tonal", "broadband", "high-frequency", "low-frequency",
        "pulsed", "repetitive", "short", "long", "loud",
        "harmonic", "noisy", "frequency-modulated",
    ]
    semantic_keywords = [
        "alarm", "contact", "affective", "coordination",
        "distress", "predator", "aggression", "infant",
        "display", "long_distance", "food", "territory",
    ]
    pmi, _, _ = compute_pmi(records, acoustic_keywords, semantic_keywords)
    plot_pmi_heatmap(pmi, acoustic_keywords, semantic_keywords,
                     PLOTS_DIR / "pmi_heatmap.png")

    print("Done. Results saved to paper_code/results.json")


if __name__ == "__main__":
    main()

"""
Table 1 experiments: acoustic → semantic prediction under three hold-out conditions.

Conditions:
  - Within repertoire : 10-fold random CV (calls split regardless of species)
  - Held-out species  : leave-one-species-out
  - Held-out family   : leave-one-family-out

Models:
  - Random            : shuffled acoustic-semantic pairings
  - Retrieval (k=1)   : nearest neighbour in acoustic space
  - Linear (ridge)    : ridge regression, λ via inner-CV
  - MLP               : 2-layer MLP

Metrics:
  - Cos.  : mean cosine similarity between predicted and true semantic embedding
  - MRR   : mean reciprocal rank (true call ranked among all test calls by
             cosine similarity to the prediction)

Run from project root:
    /Users/chemla/.venvs/apes_comparison/bin/python paper_code/predict.py
"""

import sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
try:
    from paper_code.paths import DATABASE_PATH, EMBEDDING_CACHE_PATH
except ModuleNotFoundError:
    from paths import DATABASE_PATH, EMBEDDING_CACHE_PATH
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from webapp.utils import load_calls, embed_texts

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_PATH = EMBEDDING_CACHE_PATH
DB_PATH = DATABASE_PATH
OUT_PATH   = Path(__file__).parent / "prediction_results.json"

SEED = 42
rng  = np.random.default_rng(SEED)


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

def cosine_sim(pred, true):
    """Mean cosine similarity; both arrays assumed L2-normalised."""
    return float((pred * true).sum(axis=1).mean())


def mrr(pred, true):
    """
    For each row i, rank all rows in `true` by cosine similarity to pred[i].
    The reciprocal rank is 1 / (rank of true[i]).
    """
    scores = pred @ true.T          # (n, n)
    n = len(true)
    rrs = []
    for i in range(n):
        order = np.argsort(-scores[i])
        rank  = int(np.where(order == i)[0][0]) + 1
        rrs.append(1.0 / rank)
    return float(np.mean(rrs))


def normalise(v):
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #

def predict_random(X_train, Y_train, X_test, Y_test):
    perm = rng.permutation(len(Y_test))
    return normalise(Y_test[perm].astype(float))


def predict_retrieval(X_train, Y_train, X_test, Y_test):
    sims = X_test @ X_train.T          # (n_test, n_train)
    nn   = np.argmax(sims, axis=1)
    return normalise(Y_train[nn].astype(float))


def predict_ridge(X_train, Y_train, X_test, Y_test):
    reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=min(5, len(X_train)))
    reg.fit(X_train, Y_train)
    return normalise(reg.predict(X_test).astype(float))


def predict_mlp(X_train, Y_train, X_test, Y_test):
    mlp = MLPRegressor(
        hidden_layer_sizes=(512, 512),
        activation="relu",
        max_iter=500,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    mlp.fit(X_train, Y_train)
    return normalise(mlp.predict(X_test).astype(float))


MODELS = {
    "Random":          predict_random,
    "Retrieval (k=1)": predict_retrieval,
    "Linear (ridge)":  predict_ridge,
    "MLP":             predict_mlp,
}


# ------------------------------------------------------------------ #
# Evaluation loop
# ------------------------------------------------------------------ #

def evaluate_folds(folds, ac_emb, se_emb):
    """
    folds: list of (train_idx, test_idx) pairs.
    Returns dict model_name -> {cos: float, mrr: float}.
    """
    results = {m: {"cos": [], "mrr": []} for m in MODELS}
    for train_idx, test_idx in folds:
        X_tr = ac_emb[train_idx];  Y_tr = se_emb[train_idx]
        X_te = ac_emb[test_idx];   Y_te = se_emb[test_idx]
        if len(X_te) == 0 or len(X_tr) < 2:
            continue
        for name, fn in MODELS.items():
            pred = fn(X_tr, Y_tr, X_te, Y_te)
            results[name]["cos"].append(cosine_sim(pred, Y_te))
            results[name]["mrr"].append(mrr(pred, Y_te))
    return {m: {"cos": float(np.mean(v["cos"])), "mrr": float(np.mean(v["mrr"]))}
            for m, v in results.items()}


def within_repertoire_folds(n, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    return list(kf.split(np.arange(n)))


def leave_one_species_out_folds(species):
    unique = sorted(set(species))
    sp_arr = np.array(species)
    folds  = []
    for sp in unique:
        test  = np.where(sp_arr == sp)[0]
        train = np.where(sp_arr != sp)[0]
        if len(test) > 0 and len(train) >= 2:
            folds.append((train, test))
    return folds


def leave_one_family_out_folds(families):
    unique = sorted(set(families))
    fa_arr = np.array(families)
    folds  = []
    for fa in unique:
        test  = np.where(fa_arr == fa)[0]
        train = np.where(fa_arr != fa)[0]
        if len(test) > 0 and len(train) >= 2:
            folds.append((train, test))
    return folds


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    calls   = load_calls(DB_PATH)
    species = [str(c["species"]) for c in calls]
    families= [str(c["family"])  for c in calls]
    n       = len(calls)
    print(f"Loaded {n} calls, {len(set(species))} species, {len(set(families))} families")

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    ac_emb, _ = embed_texts([c["acoustic_description"] for c in calls],
                             EMBEDDING_MODEL, CACHE_PATH, encoder)
    se_emb, _ = embed_texts([c["semantic_description"] for c in calls],
                             EMBEDDING_MODEL, CACHE_PATH, encoder)
    ac_emb = normalise(ac_emb.astype(float))
    se_emb = normalise(se_emb.astype(float))

    conditions = {
        "Within repertoire": within_repertoire_folds(n, k=10),
        "Held-out species":  leave_one_species_out_folds(species),
        "Held-out family":   leave_one_family_out_folds(families),
    }

    all_results = {}
    for cond_name, folds in conditions.items():
        print(f"\n{cond_name}  ({len(folds)} folds) …")
        res = evaluate_folds(folds, ac_emb, se_emb)
        all_results[cond_name] = res
        for model, v in res.items():
            print(f"  {model:<20s}  cos={v['cos']:.3f}  mrr={v['mrr']:.3f}")

    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {OUT_PATH}")

    # Print LaTeX table rows
    print("\n--- LaTeX table rows ---")
    conds = list(conditions.keys())
    for model in MODELS:
        row = model
        for cond in conds:
            v = all_results[cond][model]
            row += f" & {v['cos']:.2f} & {v['mrr']:.2f}"
        row += " \\\\"
        print(row)


if __name__ == "__main__":
    main()

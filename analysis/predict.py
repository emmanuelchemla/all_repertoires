"""Acoustic → semantic prediction under three hold-out conditions.

Models: random shuffle baseline, nearest-neighbor retrieval (k=1), ridge
regression, and a small MLP. Evaluated by cosine similarity and MRR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor


def _norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _metrics(pred: np.ndarray, true_full: np.ndarray, test_idx: np.ndarray) -> dict:
    pred = _norm(pred)
    truths = _norm(true_full[test_idx])
    cos = float((pred * truths).sum(axis=1).mean())
    # MRR: rank true call's semantic embedding among all test calls by sim to prediction
    sims = pred @ truths.T          # (n_test, n_test)
    # argsort descending: rank of correct neighbor (diagonal)
    order = np.argsort(-sims, axis=1)
    ranks = np.argmax(order == np.arange(sims.shape[0])[:, None], axis=1) + 1
    mrr = float((1.0 / ranks).mean())
    return {"cos": cos, "mrr": mrr}


def _fit_predict(model: str, X_tr: np.ndarray, Y_tr: np.ndarray,
                 X_te: np.ndarray, Y_full: np.ndarray, te_idx: np.ndarray,
                 rng: np.random.Generator) -> dict:
    if model == "random":
        perm = rng.permutation(len(te_idx))
        pred = Y_full[te_idx][perm]
    elif model == "retrieval":
        sims = _norm(X_te) @ _norm(X_tr).T
        nn = np.argmax(sims, axis=1)
        pred = Y_tr[nn]
    elif model == "ridge":
        m = Ridge(alpha=1.0)
        m.fit(X_tr, Y_tr)
        pred = m.predict(X_te)
    elif model == "mlp":
        m = MLPRegressor(hidden_layer_sizes=(512, 512), activation="relu",
                         max_iter=200, random_state=0, early_stopping=False)
        m.fit(X_tr, Y_tr)
        pred = m.predict(X_te)
    else:
        raise ValueError(model)
    return _metrics(pred, Y_full, te_idx)


@dataclass
class PredictionResults:
    within: dict[str, dict[str, float]]
    held_species: dict[str, dict[str, float]]
    held_family: dict[str, dict[str, float]]


def run(X: np.ndarray, Y: np.ndarray, species: list[str], families: list[str],
        seed: int = 0) -> PredictionResults:
    rng = np.random.default_rng(seed)
    models = ["random", "retrieval", "ridge", "mlp"]
    n = len(species)

    # ---- within-repertoire: 10 random folds ----
    idx = np.arange(n)
    folds = np.array_split(rng.permutation(idx), 10)
    within: dict[str, list[dict]] = {m: [] for m in models}
    for f in folds:
        te = np.array(f)
        tr = np.setdiff1d(idx, te)
        for m in models:
            within[m].append(_fit_predict(m, X[tr], Y[tr], X[te], Y, te, rng))

    # ---- leave-one-species-out ----
    held_sp: dict[str, list[dict]] = {m: [] for m in models}
    sp_arr = np.array(species)
    for sp in sorted(set(species)):
        te = np.where(sp_arr == sp)[0]
        tr = np.setdiff1d(idx, te)
        if len(te) < 2 or len(tr) < 2:
            continue
        for m in models:
            held_sp[m].append(_fit_predict(m, X[tr], Y[tr], X[te], Y, te, rng))

    # ---- leave-one-family-out ----
    held_fam: dict[str, list[dict]] = {m: [] for m in models}
    fam_arr = np.array(families)
    for fam in sorted(set(families)):
        te = np.where(fam_arr == fam)[0]
        tr = np.setdiff1d(idx, te)
        if len(te) < 2 or len(tr) < 2:
            continue
        for m in models:
            held_fam[m].append(_fit_predict(m, X[tr], Y[tr], X[te], Y, te, rng))

    def _mean(rs: list[dict]) -> dict[str, float]:
        return {k: float(np.mean([r[k] for r in rs])) for k in ("cos", "mrr")}

    return PredictionResults(
        within={m: _mean(within[m]) for m in models},
        held_species={m: _mean(held_sp[m]) for m in models},
        held_family={m: _mean(held_fam[m]) for m in models},
    )

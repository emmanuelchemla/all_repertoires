"""Mantel test, partial Mantel, and PMI."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np


def cosine_distance(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance on L2-normalized rows."""
    sim = emb @ emb.T
    np.clip(sim, -1.0, 1.0, out=sim)
    return 1.0 - sim


def mantel(dist_x: np.ndarray, dist_y: np.ndarray,
           n_perm: int = 9999, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = dist_x.shape[0]
    iu = np.triu_indices(n, k=1)
    x = dist_x[iu]
    y = dist_y[iu]
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.linalg.norm(xc) * np.linalg.norm(yc)
    r = float((xc * yc).sum() / denom) if denom > 0 else 0.0
    if n_perm <= 0:
        return r, float("nan")
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        yp = dist_y[np.ix_(perm, perm)][iu]
        ypc = yp - yp.mean()
        rp = (xc * ypc).sum() / (np.linalg.norm(xc) * np.linalg.norm(ypc))
        if rp >= r:
            ge += 1
    return r, (ge + 1) / (n_perm + 1)


def partial_mantel(dist_x: np.ndarray, dist_y: np.ndarray, dist_z: np.ndarray,
                   n_perm: int = 9999, seed: int = 42) -> tuple[float, float]:
    """Partial Mantel: correlation of x,y residuals after regressing out z."""
    rng = np.random.default_rng(seed)
    n = dist_x.shape[0]
    iu = np.triu_indices(n, k=1)
    x = dist_x[iu]; y = dist_y[iu]; z = dist_z[iu]

    def _partial_r(xv, yv, zv):
        zc = zv - zv.mean()
        beta_x = (xv * zc).sum() / (zc * zc).sum() if (zc * zc).sum() > 0 else 0.0
        beta_y = (yv * zc).sum() / (zc * zc).sum() if (zc * zc).sum() > 0 else 0.0
        rx = xv - beta_x * zv
        ry = yv - beta_y * zv
        rxc = rx - rx.mean(); ryc = ry - ry.mean()
        denom = np.linalg.norm(rxc) * np.linalg.norm(ryc)
        return float((rxc * ryc).sum() / denom) if denom > 0 else 0.0

    r = _partial_r(x, y, z)
    if n_perm <= 0:
        return r, float("nan")
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        yp = dist_y[np.ix_(perm, perm)][iu]
        rp = _partial_r(x, yp, z)
        if rp >= r:
            ge += 1
    return r, (ge + 1) / (n_perm + 1)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - a.mean(); bc = b - b.mean()
    d = np.linalg.norm(ac) * np.linalg.norm(bc)
    return float((ac * bc).sum() / d) if d > 0 else 0.0


def pmi_matrix(rows: list[tuple[tuple[str, ...], tuple[str, ...]]],
               acoustic_vocab: list[str],
               semantic_vocab: list[str],
               min_count: int = 3,
               smoothing: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """PMI between acoustic (rows) and semantic (cols) keywords.

    rows: list of (acoustic_kws, semantic_kws) per call.
    Returns (pmi, counts).
    """
    a_idx = {k: i for i, k in enumerate(acoustic_vocab)}
    s_idx = {k: i for i, k in enumerate(semantic_vocab)}
    n = len(rows)
    a_count = np.zeros(len(acoustic_vocab))
    s_count = np.zeros(len(semantic_vocab))
    co = np.zeros((len(acoustic_vocab), len(semantic_vocab)))
    for a_kws, s_kws in rows:
        a_set = {k for k in a_kws if k in a_idx}
        s_set = {k for k in s_kws if k in s_idx}
        for a in a_set:
            a_count[a_idx[a]] += 1
        for s in s_set:
            s_count[s_idx[s]] += 1
        for a in a_set:
            for s in s_set:
                co[a_idx[a], s_idx[s]] += 1
    pa = (a_count + smoothing) / (n + smoothing)
    ps = (s_count + smoothing) / (n + smoothing)
    pas = (co + smoothing) / (n + smoothing)
    pmi = np.log2(pas / (pa[:, None] * ps[None, :]))
    pmi = np.where(co < min_count, np.nan, pmi)
    return pmi, co

"""Ranking vertices by participation in higher-order triadic closure.

For a vertex $v$, let $c(v)$ be the number of cross-hyperedge triangles containing it. This is
obtainable exactly and for free from the per-edge counts, since every such triangle meets $v$
in exactly two $v$-incident edges:

    c(v) = (1/2) * sum_{e incident to v} t(e).

The mining question is whether a stream can identify the top of that ranking without
materialising it, and how much the predictor helps. The evaluation is the standard one:
precision at K against the exact ranking, plus the rank correlation over all vertices.
"""
import numpy as np

from .bench import _covbits

__all__ = ['vertex_scores_exact', 'simulate_vertex_scores', 'precision_at_k',
           'rank_correlation']


def vertex_scores_exact(hg):
    """Exact number of cross-hyperedge triangles at each vertex."""
    seg = hg.t.astype(float)[hg.slot_eid]
    return np.add.reduceat(seg, hg.indptr[:-1]) / 2.0


def simulate_vertex_scores(hg, w, k, reps=10, seed=0, cov=None):
    """Estimate the per-vertex scores by running the sampler.

    Each discovered cross triangle contributes 1 / Pr_w[path] to each of its three vertices,
    so one run of k instances produces the whole score vector.
    """
    rng = np.random.default_rng(seed)
    wp = np.asarray(w, dtype=float) + 1.0
    W = wp.sum()
    D = hg._D(wp)
    pe = wp / W
    cum = np.cumsum(wp[hg.slot_eid])
    block_start = np.concatenate([[0.0], cum[hg.indptr[1:-1] - 1]])
    cov = cov if cov is not None else _covbits(hg)
    packed = hg.eu.astype(np.int64) * hg.n + hg.ev.astype(np.int64)
    order = np.argsort(packed)
    ps = packed[order]

    out = np.zeros((reps, hg.n))
    for r in range(reps):
        base = rng.choice(hg.m, size=k, p=pe)
        u, v = hg.eu[base], hg.ev[base]
        lo = hg.rank[u] < hg.rank[v]
        x = np.where(lo, u, v)
        y = np.where(lo, v, u)
        target = block_start[x] + rng.random(k) * D[x]
        slot = np.searchsorted(cum, target, side='left')
        slot = np.clip(slot, hg.indptr[x], hg.indptr[x + 1] - 1)
        c = hg.indices[slot]
        xz = hg.slot_eid[slot]

        ok = (c != u) & (c != v) & (hg.rank[c] > hg.rank[y])
        a = np.minimum(y, c).astype(np.int64)
        b = np.maximum(y, c).astype(np.int64)
        pos = np.clip(np.searchsorted(ps, a * hg.n + b), 0, len(ps) - 1)
        ok &= ps[pos] == a * hg.n + b

        idx = np.flatnonzero(ok)
        keep, vals = [], []
        for j in idx:
            if not (cov[base[j]] >> int(c[j])) & 1:
                pr = (wp[base[j]] / W) * (wp[xz[j]] / D[x[j]])
                keep.append(j)
                vals.append(1.0 / pr)
        if keep:
            keep = np.array(keep)
            vals = np.array(vals) / k
            acc = out[r]
            np.add.at(acc, x[keep], vals)
            np.add.at(acc, y[keep], vals)
            np.add.at(acc, c[keep], vals)
    return out


def precision_at_k(est, exact, K=100):
    """Overlap of the top-K sets of the estimated and exact score vectors."""
    K = min(K, len(exact))
    a = set(np.argsort(-np.asarray(exact))[:K].tolist())
    b = set(np.argsort(-np.asarray(est))[:K].tolist())
    return len(a & b) / K


def rank_correlation(est, exact):
    """Spearman correlation over the vertices that bear at least one copy."""
    from scipy.stats import spearmanr
    exact = np.asarray(exact, dtype=float)
    mask = exact > 0
    if mask.sum() < 3:
        return np.nan
    return float(spearmanr(np.asarray(est)[mask], exact[mask])[0])

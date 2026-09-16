"""Reporting: the localization / prediction split, the permutation control, and a
Monte-Carlo check that the analytic success probabilities are the right ones.

`loc_x`            localized but unweighted sampler, over the prediction-free baseline
`perfect_over_loc` what a perfect predictor adds on top of localization
`permuted_*`       the same weight multiset shuffled across edges (the distributional null)
`structural_share` the fraction of the predictor's gain the shuffle does not reproduce
"""
import numpy as np

__all__ = ['evaluate', 'monte_carlo_check']


def evaluate(hg, n_perm=50, seed=0):
    """Localization, perfect predictor, permutation null, cheap heuristic, first-edge."""
    pb = hg.p_base()
    p_loc = hg.p_succ(hg.w_uniform())
    p_perf = hg.p_succ(hg.w_perfect())
    p_mind = hg.p_succ(hg.w_mindeg())
    p_first = hg.p_succ(hg.w_perfect(), first_edge_only=True)
    perms = np.array([hg.p_succ(hg.w_permuted(hg.w_perfect(), seed + b))
                      for b in range(n_perm)])
    s_perf, s_perm = p_perf / p_loc, perms / p_loc
    gain = s_perf - 1.0
    shares = (s_perf - s_perm) / gain if abs(gain) > 1e-12 else np.full(n_perm, np.nan)
    pval = (1 + (s_perm >= s_perf).sum()) / (1 + n_perm)
    return dict(
        n=hg.n, m=hg.m, kappa=hg.kappa, n_cross=hg.n_cross_total,
        copy_density=hg.n_cross_total / max(hg.m, 1),
        triple_fraction=hg.triple_fraction,
        loc_x=p_loc / pb, perfect_x=p_perf / pb, firstedge_x=p_first / pb,
        perfect_over_loc=s_perf,
        permuted_over_loc_mean=float(s_perm.mean()),
        permuted_over_loc_lo=float(np.percentile(s_perm, 2.5)),
        permuted_over_loc_hi=float(np.percentile(s_perm, 97.5)),
        mindeg_over_loc=p_mind / p_loc,
        mindeg_capture=(p_mind / p_loc - 1) / gain if abs(gain) > 1e-12 else np.nan,
        structural_share=float(np.mean(shares)),
        structural_share_lo=float(np.percentile(shares, 2.5)),
        structural_share_hi=float(np.percentile(shares, 97.5)),
        perm_pvalue=float(pval), n_perm=n_perm,
    )


def monte_carlo_check(hg, w, n_samples=200_000, seed=0):
    """Sample real paths from the weighted sampler; compare hit rate to the analytic p."""
    rng = np.random.default_rng(seed)
    wp = np.asarray(w, dtype=float) + 1.0
    pe = wp / wp.sum()
    D = hg._D(wp)
    picks = rng.choice(hg.m, size=n_samples, p=pe)
    hits = 0
    tset = {}
    for i in picks:
        u, v = int(hg.eu[i]), int(hg.ev[i])
        x, y = (u, v) if hg.rank[u] < hg.rank[v] else (v, u)
        a, b = hg.indptr[x], hg.indptr[x + 1]
        slots = hg.slot_eid[a:b]
        pr = wp[slots] / D[x]
        c = int(hg.indices[a + rng.choice(len(slots), p=pr / pr.sum())])
        if c == u or c == v or hg.rank[c] <= hg.rank[y]:
            continue
        key = (min(x, c), max(x, c))
        if key not in hg.eid or (min(y, c), max(y, c)) not in hg.eid:
            continue
        S = hg.memb[x] & hg.memb[y] & hg.memb[c]
        if not S:
            hits += 1
    return hits / n_samples

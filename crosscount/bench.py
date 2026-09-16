"""Sample complexity, competing estimators, and a simulation of the real estimator.

Every estimator here is unbiased for #cross, so the cost of reaching a target relative
error is governed by its variance:

    k(eps) = Var[X] / (eps * #cross)^2 ,

which is the number of independent instances the estimator needs.  Reporting all methods
in this single currency makes the comparison exact rather than a horse race between
implementations.
"""
import numpy as np

__all__ = ['second_moment', 'samples_for_error', 'weighted_cost', 'baseline_cost',
           'wedge_cost', 'edge_local_cost', 'heavy_storage_cost', 'simulate_estimator',
           'wedge_weighted_cost', 'wedge_report', 'normalizer_inflation']

EPS = 0.1


def _local_words(hg, mask=None):
    """Words of state a uniformly sampled edge costs a method that counts locally.

    Counting t(e) exactly requires both endpoint neighbourhoods, so the per-sample space
    is deg(u) + deg(v) rather than O(1).  Samplers that make a single closing-vertex draw
    cost one word per instance.
    """
    d = hg.deg[hg.eu] + hg.deg[hg.ev]
    return float(d[mask].mean() if mask is not None else d.mean())


def samples_for_error(var, truth, eps=EPS):
    """Independent instances needed for relative error eps, by Chebyshev."""
    if truth <= 0:
        return np.inf
    return var / (eps * truth) ** 2


# ------------------------------------------------- the weighted / localized estimator

def second_moment(hg, w):
    """E[X^2] for the importance-sampling estimator X = 1 / Pr_w[path], summed over copies.

    Pr_w[C] = (w(base)+1)/W * (w(xz)+1)/D_x, so E[X^2] = sum_C 1/Pr_w[C].
    """
    wp = np.asarray(w, dtype=float) + 1.0
    W = wp.sum()
    D = hg._D(wp)
    pr = (wp[hg.base] / W) * (wp[hg.xz] / D[hg.pivot])
    return float((1.0 / pr).sum()) * hg.triple_scale


def weighted_cost(hg, w, eps=EPS):
    """Variance and sample complexity of the weighted sampler under weights w."""
    truth = hg.n_cross_total
    m2 = second_moment(hg, w)
    var = max(m2 - truth ** 2, 0.0)
    k = samples_for_error(var, truth, eps)
    return dict(second_moment=m2, var=var, samples=k, words=k)


def baseline_cost(hg, eps=EPS):
    """The prediction-free Fichtenberger-Peng sampler: every copy found w.p. (2m)^{-3/2}."""
    truth = hg.n_cross_total
    p = (2 * hg.m) ** 1.5
    var = max(truth * p - truth ** 2, 0.0)
    k = samples_for_error(var, truth, eps)
    return dict(var=var, samples=k, words=k)


# ------------------------------------------------------------------- competing methods

def wedge_cost(hg, eps=EPS):
    """Uniform wedge sampling: draw one of the sum_v C(deg v, 2) wedges, test closure.

    X = (W_total / 3) * 1[the wedge closes into a cross triangle]; unbiased for #cross.
    """
    deg = hg.deg.astype(float)
    W_total = float((deg * (deg - 1) / 2).sum())
    truth = hg.n_cross_total
    if truth <= 0 or W_total <= 0:
        return dict(wedges=W_total, p=0.0, var=np.inf, samples=np.inf, words=np.inf)
    p = 3.0 * truth / W_total
    scale = W_total / 3.0
    var = scale ** 2 * p * (1 - p)
    k = samples_for_error(var, truth, eps)
    return dict(wedges=W_total, p=p, var=var, samples=k, words=k)


def edge_local_cost(hg, eps=EPS):
    """Uniform edge sampling with exact local counting: X = m * t(e) / 3.

    Stronger than our sampler per draw, since it needs a neighbourhood intersection rather
    than a single closing-vertex draw; we report it anyway and say so.
    """
    t = hg.t.astype(float)
    truth = hg.n_cross_total
    x = hg.m * t / 3.0
    var = float(x.var())
    k = samples_for_error(var, truth, eps)
    return dict(var=var, samples=k, words=k * _local_words(hg))


def heavy_storage_cost(hg, taus=None, eps=EPS):
    """Store the top-tau edges by true weight, sample the residual with edge+local counting.

    Total space is tau edge-slots plus the residual sample size; the grid is swept and the
    best operating point reported, which is the setting most favourable to the competitor.
    """
    t = hg.t.astype(float)
    truth = hg.n_cross_total
    order = np.argsort(-t)
    if taus is None:
        taus = np.unique(np.clip(
            np.round(np.geomspace(1, max(hg.m, 2), 25)).astype(int), 1, hg.m))
    best = None
    for tau in taus:
        heavy = order[:tau]
        mask = np.ones(hg.m, dtype=bool)
        mask[heavy] = False
        rest = t[mask]
        m_rest = int(mask.sum())
        if m_rest == 0:
            total = float(tau)
            if best is None or total < best:
                best, best_tau = total, int(tau)
            continue
        # the residual contribution, estimated by uniform sampling over the light edges
        x = m_rest * rest / 3.0
        var = float(x.var())
        k = samples_for_error(var, truth, eps)
        total = float(tau) + k * _local_words(hg, mask)
        if best is None or total < best:
            best, best_tau = total, int(tau)
    return dict(words=best, tau=best_tau if best is not None else None)


# --------------------------------------------------------- simulation of the estimator

def _covbits(hg):
    """Per-edge union of the hyperedges containing both endpoints (as a bitset)."""
    cov = [0] * hg.m
    for i in range(hg.m):
        u, v = int(hg.eu[i]), int(hg.ev[i])
        c = 0
        for h in hg.memb[u] & hg.memb[v]:
            c |= hg.hbits[h]
        cov[i] = c
    return cov


def simulate_estimator(hg, w, k, reps=20, seed=0, cov=None):
    """Run the actual sampler: draw base edge, draw closing vertex, verify, average.

    Returns the array of `reps` estimates of #cross, each from `k` independent instances.
    The estimator is X = 1 / Pr_w[path] on success and 0 otherwise, which is unbiased.
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
    packed_sorted = packed[order]

    out = np.empty(reps)
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
        # is {y, c} an edge?
        a = np.minimum(y, c).astype(np.int64)
        b = np.maximum(y, c).astype(np.int64)
        key = a * hg.n + b
        pos = np.searchsorted(packed_sorted, key)
        pos = np.clip(pos, 0, len(packed_sorted) - 1)
        ok &= packed_sorted[pos] == key

        idx = np.flatnonzero(ok)
        vals = np.zeros(k)
        for j in idx:                       # the cross test, O(1) per surviving candidate
            if not (cov[base[j]] >> int(c[j])) & 1:
                pr = (wp[base[j]] / W) * (wp[xz[j]] / D[x[j]])
                vals[j] = 1.0 / pr
        out[r] = vals.mean()
    return out


# ------------------------------------------------- the wedge estimator, with predictions

def _third_edge_ids(hg):
    """Edge ids of the third side (y,z) of every cross triangle in the triple arrays.

    The triple arrays give the base edge (x,y) and the closing edge (x,z); the remaining
    side is recovered by a packed-key lookup.  Cached on the CrossHG instance.
    """
    if getattr(hg, '_yz', None) is not None:
        return hg._yz
    b, xz, x = hg.base, hg.xz, hg.pivot
    y = np.where(hg.eu[b] == x, hg.ev[b], hg.eu[b])
    z = np.where(hg.eu[xz] == x, hg.ev[xz], hg.eu[xz])
    a = np.minimum(y, z).astype(np.int64)
    c = np.maximum(y, z).astype(np.int64)
    packed = hg.eu.astype(np.int64) * hg.n + hg.ev.astype(np.int64)
    order = np.argsort(packed)
    ps = packed[order]
    pos = np.searchsorted(ps, a * hg.n + c)
    pos = np.clip(pos, 0, len(ps) - 1)
    hg._yz = order[pos].astype(np.int32)
    return hg._yz


def wedge_weighted_cost(hg, w, eps=EPS):
    """Predictor-weighted uniform-wedge sampling.

    Ordered wedges (x; y, z) are drawn with probability proportional to
    wp(x,y) * wp(x,z), and X = T / (6 * wp(x,y) * wp(x,z)) on a closed cross triangle,
    where T = sum_x (D_x^2 - Q_x) with Q_x = sum_c wp(x,c)^2.  Setting w = 0 recovers
    prediction-free uniform wedge sampling.
    """
    wp = np.asarray(w, dtype=float) + 1.0
    D = hg._D(wp)
    Q = np.add.reduceat(wp[hg.slot_eid] ** 2, hg.indptr[:-1])
    T = float((D ** 2 - Q).sum())
    yz = _third_edge_ids(hg)
    a, b, c = wp[hg.base], wp[hg.xz], wp[yz]
    inv = 1.0 / (a * b) + 1.0 / (a * c) + 1.0 / (b * c)
    m2 = float(inv.sum()) * hg.triple_scale * T / 18.0
    truth = hg.n_cross_total
    var = max(m2 - truth ** 2, 0.0)
    k = samples_for_error(var, truth, eps)
    return dict(T=T, second_moment=m2, var=var, samples=k, words=k)


def wedge_report(hg, n_perm=50, seed=0, eps=EPS):
    """The prediction question for the wedge estimator: does a predictor help, and why?"""
    c_unif = wedge_weighted_cost(hg, hg.w_uniform(), eps)['words']
    c_perf = wedge_weighted_cost(hg, hg.w_perfect(), eps)['words']
    c_mind = wedge_weighted_cost(hg, hg.w_mindeg(), eps)['words']
    perms = np.array([wedge_weighted_cost(hg, hg.w_permuted(hg.w_perfect(), seed + b),
                                          eps)['words'] for b in range(n_perm)])
    s_perf = c_unif / c_perf                      # speedup = cost ratio
    s_perm = c_unif / perms
    gain = s_perf - 1.0
    shares = (s_perf - s_perm) / gain if abs(gain) > 1e-12 else np.full(n_perm, np.nan)
    return dict(
        m=hg.m, n_cross=hg.n_cross_total, copy_density=hg.n_cross_total / max(hg.m, 1),
        wedge_unif_words=c_unif, wedge_perfect_words=c_perf,
        perfect_over_unif=s_perf,
        permuted_over_unif_mean=float(s_perm.mean()),
        permuted_over_unif_lo=float(np.percentile(s_perm, 2.5)),
        permuted_over_unif_hi=float(np.percentile(s_perm, 97.5)),
        mindeg_over_unif=c_unif / c_mind,
        structural_share=float(np.mean(shares)),
        structural_share_lo=float(np.percentile(shares, 2.5)),
        structural_share_hi=float(np.percentile(shares, 97.5)),
        perm_pvalue=float((1 + (s_perm >= s_perf).sum()) / (1 + n_perm)),
    )



def normalizer_inflation(hg):
    """Which normalizer does a skewed weighting inflate?

    The weights are normalised to mean one so that only their skew remains.  W is linear and
    therefore unchanged; D_x is linear but local, and shrinks on average; T is quadratic and
    inflates by convexity.  That difference is why the augmented estimator profits from a
    predictor and the wedge estimator does not.
    """
    wp = hg.w_perfect() + 1.0
    wn = wp / wp.mean()
    one = np.ones(hg.m)

    def T(w):
        D = hg._D(w)
        Q = np.add.reduceat(w[hg.slot_eid] ** 2, hg.indptr[:-1])
        return float((D ** 2 - Q).sum())

    ratio = hg._D(wn) / np.maximum(hg._D(one), 1e-12)
    return dict(
        m=hg.m, weight_cv=float(wn.std()),
        W_inflation=float(wn.sum() / one.sum()),
        D_mean_inflation=float(ratio.mean()), D_max_inflation=float(ratio.max()),
        T_inflation=T(wn) / T(one),
        weighted_gain=weighted_cost(hg, hg.w_uniform())['words'] /
                      weighted_cost(hg, hg.w_perfect())['words'],
        wedge_gain=wedge_weighted_cost(hg, hg.w_uniform())['words'] /
                   wedge_weighted_cost(hg, hg.w_perfect())['words'],
    )

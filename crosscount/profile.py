"""The higher-order triadic profile, and its estimation from a stream.

Every triangle of the projection falls into exactly one of three classes:

  covered      some hyperedge contains all three vertices, so the triangle is an artefact
               of the projection rather than evidence of triadic structure;
  transient    a cross-hyperedge triangle at least one of whose pairs co-occurs in only one
               hyperedge -- a triad closed by a one-off encounter;
  persistent   a cross-hyperedge triangle all three of whose pairs recur in two or more
               hyperedges -- a triad closed by repeated relationships.

(Note that a cross triangle always needs three distinct hyperedges: a hyperedge covering two
of its edges would contain all three vertices, which is the covered case.  The classes are
therefore distinguished by pair multiplicity rather than by hyperedge count.)

The profile of a hypergraph is the pair

    f_cross      = #cross / #triangles
    f_persistent = #persistent / #cross

both of which are ratios of quantities the same sampler can estimate.
"""
import numpy as np

from .bench import _covbits, _third_edge_ids

__all__ = ['exact_profile', 'simulate_profile', 'domain_separation']


def exact_profile(hg):
    """Exact counts and profile, from the per-edge quantities already computed."""
    n_tri = int((hg.t + hg.covn).sum()) // 3
    n_cross = hg.n_cross_total
    yz = _third_edge_ids(hg)
    mult_min = np.minimum(np.minimum(hg.nhy[hg.base], hg.nhy[hg.xz]), hg.nhy[yz])
    n_pers = float((mult_min >= 2).sum()) * hg.triple_scale
    return dict(
        n_triangles=n_tri, n_cross=n_cross, n_persistent=n_pers,
        f_cross=n_cross / n_tri if n_tri else np.nan,
        f_persistent=n_pers / n_cross if n_cross else np.nan,
    )


def simulate_profile(hg, w, k, reps=20, seed=0, cov=None):
    """Estimate the profile by running the sampler, classifying each discovered triangle.

    One pass of instances yields all three counts at once, since the class is read off the
    triangle the instance already found.
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

    out = np.empty((reps, 3))
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
        key = a * hg.n + b
        pos = np.clip(np.searchsorted(ps, key), 0, len(ps) - 1)
        hit = ps[pos] == key
        ok &= hit
        yz_id = order[pos]

        tri = np.zeros(k)
        cro = np.zeros(k)
        per = np.zeros(k)
        for j in np.flatnonzero(ok):
            pr = (wp[base[j]] / W) * (wp[xz[j]] / D[x[j]])
            val = 1.0 / pr
            tri[j] = val
            if not (cov[base[j]] >> int(c[j])) & 1:
                cro[j] = val
                mm = min(hg.nhy[base[j]], hg.nhy[xz[j]], hg.nhy[yz_id[j]])
                if mm >= 2:
                    per[j] = val
        out[r] = [tri.mean(), cro.mean(), per.mean()]
    return out


def domain_separation(points, domains, seed=0):
    """Leave-one-out nearest-neighbour accuracy of the domain label in profile space.

    A crude but transparent measure of whether the profile carries domain information; with
    ten graphs in five domains, chance is 1/9 per neighbour under a random pairing.
    """
    P = np.asarray(points, dtype=float)
    P = (P - P.mean(0)) / (P.std(0) + 1e-12)
    dom = np.asarray(domains)
    correct = 0
    for i in range(len(P)):
        d = np.linalg.norm(P - P[i], axis=1)
        d[i] = np.inf
        correct += dom[int(np.argmin(d))] == dom[i]
    return correct / len(P)

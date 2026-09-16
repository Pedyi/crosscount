"""Projection of a hypergraph, with exact cross-hyperedge-triangle structure.

A triple {u, v, w} is a *cross-hyperedge triangle* when it is a triangle in the projection
and no single hyperedge contains all three vertices.  Per-edge counts come from the closed
form

    t({u,v}) = |N(u) & N(v)|  -  |union of hyperedges containing both u and v, minus {u,v}|

which avoids triangle enumeration; only the list of triples used to sum per-copy path
probabilities is (optionally) subsampled, and every predictor is scored on the same sample,
so the reported ratios are unaffected.
"""
import numpy as np

from .core import bits_to_idx, popcount, degeneracy_arrays, oracle_width

__all__ = ['CrossHG', 'diagnostic_crossK3']


class CrossHG:
    """Projected graph of a hypergraph + exact cross-triangle structure.

    Cross triangle: {u,v,w} pairwise adjacent in the projection, with no single
    hyperedge containing all three.  Canonical discovery follows the Fichtenberger-Peng
    path used by PredCount: order the three vertices by (degree, id) as x < y < z,
    base edge {x,y}, pivot x, closing vertex z.
    """

    def __init__(self, hyperedges, max_size=25, max_triples=4_000_000, seed=0, verbose=True):
        hs = [tuple(sorted(set(h))) for h in hyperedges]
        hs = [h for h in hs if 2 <= len(h) <= max_size]
        hs = list(dict.fromkeys(hs))
        verts = sorted({v for h in hs for v in h})
        idx = {v: i for i, v in enumerate(verts)}
        self.n = n = len(verts)
        self.nbytes = (n + 7) // 8
        self.hs = [tuple(idx[v] for v in h) for h in hs]
        self.hbits = [sum(1 << a for a in h) for h in self.hs]
        self.hsize = np.array([len(h) for h in self.hs], dtype=np.int32)

        memb = [[] for _ in range(n)]
        adj = [0] * n
        for hid, h in enumerate(self.hs):
            bits = self.hbits[hid]
            for a in h:
                memb[a].append(hid)
                adj[a] |= bits & ~(1 << a)
        self.memb = [set(m) for m in memb]
        self.adj = adj

        # ---- edges, CSR adjacency, edge ids
        nbr = [bits_to_idx(adj[u], self.nbytes) for u in range(n)]
        self.deg = np.array([len(x) for x in nbr], dtype=np.int64)
        self.indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(self.deg, out=self.indptr[1:])
        self.indices = np.concatenate(nbr) if n else np.empty(0, np.int32)

        eu, ev = [], []
        for u in range(n):
            w = nbr[u][nbr[u] > u]
            eu.append(np.full(len(w), u, dtype=np.int32))
            ev.append(w)
        self.eu = np.concatenate(eu) if n else np.empty(0, np.int32)
        self.ev = np.concatenate(ev) if n else np.empty(0, np.int32)
        self.m = len(self.eu)
        self.eid = {}
        for i in range(self.m):
            self.eid[(int(self.eu[i]), int(self.ev[i]))] = i
        # edge id for every CSR slot
        self.slot_eid = np.empty(len(self.indices), dtype=np.int64)
        for u in range(n):
            a, b = self.indptr[u], self.indptr[u + 1]
            for k in range(a, b):
                v = int(self.indices[k])
                self.slot_eid[k] = self.eid[(u, v)] if u < v else self.eid[(v, u)]

        self._core_numbers()
        self.rank = np.empty(n, dtype=np.int64)
        order = np.lexsort((np.arange(n), self.deg))
        self.rank[order] = np.arange(n)

        self._cross_and_triples(max_triples, seed, verbose)
        self._features()

    # ------------------------------------------------------------------ structure
    def _core_numbers(self):
        n = self.n
        deg = self.deg.copy()
        core = np.zeros(n, dtype=np.int64)
        removed = np.zeros(n, dtype=bool)
        maxd = int(deg.max()) if n else 0
        buckets = [set() for _ in range(maxd + 1)]
        for v in range(n):
            buckets[deg[v]].add(v)
        k, i = 0, 0
        for _ in range(n):
            while i <= maxd and not buckets[i]:
                i += 1
            if i > maxd:
                break
            v = buckets[i].pop()
            k = max(k, i)
            core[v] = k
            removed[v] = True
            for j in range(self.indptr[v], self.indptr[v + 1]):
                w = int(self.indices[j])
                if removed[w]:
                    continue
                d = int(deg[w])
                buckets[d].discard(w)
                deg[w] = d - 1
                buckets[d - 1].add(w)
                if d - 1 < i:
                    i = d - 1
        self.core = core
        self.kappa = int(k)

    def _cross_and_triples(self, max_triples, seed, verbose):
        """Per-edge cross-triangle counts (exact) and the triple arrays (possibly sampled)."""
        nb = self.nbytes
        t = np.zeros(self.m, dtype=np.int64)
        covn = np.zeros(self.m, dtype=np.int64)     # covered common neighbours
        nhy = np.zeros(self.m, dtype=np.int64)      # hyperedges containing the edge
        hsum = np.zeros(self.m, dtype=np.int64)     # sum over those of (|h|-2)
        hmax = np.zeros(self.m, dtype=np.int64)
        covbits = [0] * self.m
        for i in range(self.m):
            u, v = int(self.eu[i]), int(self.ev[i])
            common = self.adj[u] & self.adj[v]
            S = self.memb[u] & self.memb[v]
            cov = 0
            for h in S:
                cov |= self.hbits[h]
            covbits[i] = cov
            nhy[i] = len(S)
            if S:
                sz = self.hsize[list(S)]
                hsum[i] = int((sz - 2).sum())
                hmax[i] = int(sz.max())
                covn[i] = popcount(cov & common)
            t[i] = popcount(common & ~cov)
        self.t = t
        self.covn, self.nhy, self.hsum, self.hmax = covn, nhy, hsum, hmax
        self.n_cross_total = int(t.sum()) // 3

        # triples: each cross triangle is generated once, from its canonical base edge
        target = self.n_cross_total if self.n_cross_total else 1
        p = min(1.0, max_triples / target)
        rng = np.random.default_rng(seed)
        keep = np.ones(self.m, dtype=bool) if p >= 1.0 else (rng.random(self.m) < p)
        self.triple_scale = 1.0 / p
        self.triple_fraction = p

        base, pivot, xz = [], [], []
        rank = self.rank
        for i in np.flatnonzero(keep):
            i = int(i)
            if t[i] == 0:
                continue
            u, v = int(self.eu[i]), int(self.ev[i])
            cross = (self.adj[u] & self.adj[v]) & ~covbits[i]
            z = bits_to_idx(cross, nb)
            hi = max(rank[u], rank[v])
            z = z[rank[z] > hi]
            if not len(z):
                continue
            x, y = (u, v) if rank[u] < rank[v] else (v, u)
            base.append(np.full(len(z), i, dtype=np.int32))
            pivot.append(np.full(len(z), x, dtype=np.int32))
            xz.append(np.array([self.eid[(min(x, int(c)), max(x, int(c)))] for c in z],
                               dtype=np.int32))
        self.base = np.concatenate(base) if base else np.empty(0, np.int32)
        self.pivot = np.concatenate(pivot) if pivot else np.empty(0, np.int32)
        self.xz = np.concatenate(xz) if xz else np.empty(0, np.int32)
        if verbose:
            print(f'    n={self.n} m={self.m} kappa={self.kappa} '
                  f'#cross={self.n_cross_total} triples kept={len(self.base)} '
                  f'({100*self.triple_fraction:.1f}% of edges)')

    def _features(self):
        lg = np.log1p
        du, dv = self.deg[self.eu], self.deg[self.ev]
        cu, cv = self.core[self.eu], self.core[self.ev]
        ku = np.array([len(self.memb[u]) for u in self.eu], dtype=np.int64)
        kv = np.array([len(self.memb[v]) for v in self.ev], dtype=np.int64)
        A = np.column_stack([                       # Array-paper block: degrees + cores
            lg(np.minimum(du, dv)), lg(np.maximum(du, dv)),
            lg(du) + lg(dv), np.abs(lg(du) - lg(dv)),
            lg(np.minimum(cu, cv)), lg(np.maximum(cu, cv)), lg(cu) + lg(cv),
        ])
        B = np.column_stack([                       # cheap higher-order block
            lg(np.minimum(ku, kv)), lg(np.maximum(ku, kv)),
            lg(self.nhy), lg(self.hsum), lg(self.hmax),
        ])
        Cc = np.column_stack([lg(self.covn)])       # covered-neighbour block
        self.F = {'A': A, 'AB': np.hstack([A, B]), 'ABC': np.hstack([A, B, Cc])}
        self.y = np.log1p(self.t.astype(float))

    # ------------------------------------------------------------- success probability
    def _D(self, wp):
        """D_x = sum over neighbours c of (w({x,c})+1), as an array over vertices."""
        seg = wp[self.slot_eid]
        out = np.add.reduceat(seg, self.indptr[:-1])
        return out

    def p_succ(self, w, first_edge_only=False):
        wp = np.asarray(w, dtype=float) + 1.0
        W = wp.sum()
        D = self._D(wp)
        if first_edge_only:
            inner = 1.0 / self.deg[self.pivot]
        else:
            inner = wp[self.xz] / D[self.pivot]
        return float((wp[self.base] * inner).sum()) / W * self.triple_scale

    def p_base(self):
        return self.n_cross_total / (2 * self.m) ** 1.5

    # -------------------------------------------------------------------- predictors
    def w_perfect(self):
        return self.t.astype(float)

    def w_uniform(self):
        return np.zeros(self.m)

    def w_mindeg(self):
        return np.minimum(self.deg[self.eu], self.deg[self.ev]).astype(float)

    def w_permuted(self, w, seed):
        rng = np.random.default_rng(seed)
        return rng.permutation(np.asarray(w, dtype=float))


def diagnostic_crossK3(hg, time_budget=900.0):
    """kappa, kappa_copy, exact alpha_H, and the two ratios of the paper's Table 1."""
    import time as _time
    keep = hg.t > 0
    eu, ev = hg.eu[keep], hg.ev[keep]
    kappa_copy, _ = degeneracy_arrays(eu, ev)
    t0 = _time.time()
    alpha = oracle_width(eu, ev, time_budget=time_budget)
    return dict(
        m=hg.m, m_copy=int(keep.sum()), kappa=hg.kappa, kappa_copy=kappa_copy,
        alpha=alpha,
        alpha_lb=int(np.ceil(kappa_copy / 2)), alpha_ub=kappa_copy,
        alpha_over_kappa=(alpha / hg.kappa) if (alpha and hg.kappa) else None,
        kappa_copy_over_kappa=kappa_copy / hg.kappa if hg.kappa else None,
        n_cross=hg.n_cross_total, secs=round(_time.time() - t0, 1),
    )

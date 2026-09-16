"""Cross-hyperedge four-cycles.

rho(C4) = 2 and the Fichtenberger-Peng decomposition of C4 is a perfect matching of two
edges, so BOTH elementary draws are base-edge draws against the global normaliser.  There
is no closing-vertex step and therefore no localization term: every gain measured here is
attributable to the predictor alone, which makes C4 the clean test of the permutation
control.
"""
from collections import defaultdict

import numpy as np

__all__ = ['CrossC4']


class CrossC4:
    """Cross-hyperedge four-cycles on the projection of a hypergraph.

    A 4-cycle a-x-b-y-a is *cross* when no single hyperedge contains all four vertices.
    rho(C4) = 2 and the Fichtenberger-Peng decomposition is a perfect matching of two
    edges, so BOTH elementary draws are base-edge draws against the global normalizer W.
    There is no closing-vertex step, hence no localization term: every gain measured here
    is attributable to the predictor alone.  That makes C4 the clean complement to K3.
    """

    def __init__(self, hg, max_c4=15_000_000, max_sumdeg2=5e7, max_n=3500, verbose=True):
        self.hg = hg
        self.ok = True
        self.reason = ''
        sumdeg2 = float((hg.deg.astype(float) ** 2).sum())
        self.n_c4_all = None
        if hg.n > max_n:
            self.ok = False
            self.reason = f'n = {hg.n} exceeds budget {max_n} (codegree table too large)'
            return
        if sumdeg2 > max_sumdeg2:
            self.ok = False
            self.reason = f'sum deg^2 = {sumdeg2:.2e} exceeds budget {max_sumdeg2:.1e}'
            return

        # adjacency lists from the CSR of hg
        nbr = [hg.indices[hg.indptr[v]:hg.indptr[v + 1]] for v in range(hg.n)]
        cod = defaultdict(list)
        for w in range(hg.n):
            a = np.sort(nbr[w])
            for i in range(len(a)):
                ai = int(a[i])
                for j in range(i + 1, len(a)):
                    cod[(ai, int(a[j]))].append(w)

        total = sum(len(v) * (len(v) - 1) // 2 for v in cod.values()) // 2
        self.n_c4_all = total
        if total > max_c4:
            self.ok = False
            self.reason = f'#C4 = {total:.3g} exceeds budget {max_c4:.3g}'
            return

        e1, e2 = [], []
        n_cross = 0
        eid = hg.eid
        memb = hg.memb
        for (a, b), L in cod.items():
            if len(L) < 2:
                continue
            L = sorted(L)
            mab = memb[a] & memb[b]
            for i in range(len(L)):
                x = L[i]
                for j in range(i + 1, len(L)):
                    y = L[j]
                    if (a, b) > (x, y):          # canonical: generate each C4 once
                        continue
                    if mab & memb[x] & memb[y]:  # covered by a single hyperedge
                        continue
                    n_cross += 1
                    ax = (a, x) if a < x else (x, a)
                    by = (b, y) if b < y else (y, b)
                    xb = (x, b) if x < b else (b, x)
                    ya = (y, a) if y < a else (a, y)
                    # canonical matching: the one holding the lexicographically least edge
                    if min(ax, by) <= min(xb, ya):
                        e1.append(eid[ax]); e2.append(eid[by])
                    else:
                        e1.append(eid[xb]); e2.append(eid[ya])
        self.e1 = np.array(e1, dtype=np.int32)
        self.e2 = np.array(e2, dtype=np.int32)
        self.n_cross = n_cross
        self.t = (np.bincount(self.e1, minlength=hg.m)
                  + np.bincount(self.e2, minlength=hg.m)).astype(np.int64)
        if verbose:
            print(f'    #C4(all)={self.n_c4_all}  #C4(cross)={self.n_cross}')

    # -- weightings
    def w_perfect(self):
        return self.t.astype(float)

    def w_uniform(self):
        return np.zeros(self.hg.m)

    def w_mindeg(self):
        return np.minimum(self.hg.deg[self.hg.eu], self.hg.deg[self.hg.ev]).astype(float)

    def w_permuted(self, w, seed):
        return np.random.default_rng(seed).permutation(np.asarray(w, dtype=float))

    def p_succ(self, w, first_edge_only=False):
        wp = np.asarray(w, dtype=float) + 1.0
        W = wp.sum()
        if first_edge_only:
            return float(wp[self.e1].sum()) / (W * self.hg.m)
        return float((wp[self.e1] * wp[self.e2]).sum()) / (W * W)

    def evaluate(self, n_perm=50, seed=0):
        p_u = self.p_succ(self.w_uniform())
        p_p = self.p_succ(self.w_perfect())
        p_m = self.p_succ(self.w_mindeg())
        p_f = self.p_succ(self.w_perfect(), first_edge_only=True)
        perms = np.array([self.p_succ(self.w_permuted(self.w_perfect(), seed + b))
                          for b in range(n_perm)])
        s_p, s_perm = p_p / p_u, perms / p_u
        gain = s_p - 1.0
        shares = (s_p - s_perm) / gain if abs(gain) > 1e-12 else np.full(n_perm, np.nan)
        return dict(
            m=self.hg.m, n_c4_all=self.n_c4_all, n_cross=self.n_cross,
            copy_density=self.n_cross / max(self.hg.m, 1),
            perfect_over_unif=s_p,
            permuted_over_unif_mean=float(s_perm.mean()),
            permuted_over_unif_lo=float(np.percentile(s_perm, 2.5)),
            permuted_over_unif_hi=float(np.percentile(s_perm, 97.5)),
            mindeg_over_unif=p_m / p_u,
            firstedge_over_unif=p_f / p_u,
            structural_share=float(np.mean(shares)),
            structural_share_lo=float(np.percentile(shares, 2.5)),
            structural_share_hi=float(np.percentile(shares, 97.5)),
            perm_pvalue=float((1 + (s_perm >= s_p).sum()) / (1 + n_perm)),
        )

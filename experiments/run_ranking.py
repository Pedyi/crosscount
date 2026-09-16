"""Tables 14 and 15: ranking vertices by higher-order closure, exact and from a stream."""
import time

import numpy as np
import pandas as pd
from _common import build_all, save, DOMAIN
from crosscount.bench import _covbits
from crosscount.ranking import (vertex_scores_exact, simulate_vertex_scores,
                                precision_at_k, rank_correlation)

MAX_M_SIM = 150_000
K_GRID = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
REPS = 8
TOPK = 100


def main():
    _, HG = build_all()
    EX, rows = {}, []
    for name, hg in HG.items():
        c = vertex_scores_exact(hg)
        assert abs(c.sum() - 3 * hg.n_cross_total) < 1e-6 * max(1, hg.n_cross_total), name
        EX[name] = c
        top = np.sort(c)[::-1]
        rows.append(dict(label=name, domain=DOMAIN[name], n=hg.n, m=hg.m,
                         n_cross=hg.n_cross_total,
                         frac_vertices_with_copies=float((c > 0).mean()),
                         top100_share=float(top[:100].sum() / c.sum()) if c.sum() else np.nan))
    print('identity sum_v c(v) = 3 #cross holds on every graph')
    save(pd.DataFrame(rows), 'table14_ranking_stats.csv')

    names = [n for n, h in HG.items() if h.m <= MAX_M_SIM]
    COV = {n: _covbits(HG[n]) for n in names}
    rows = []
    for name in names:
        hg, ex = HG[name], EX[name]
        for method, w in [('localized', hg.w_uniform()), ('weighted', hg.w_perfect())]:
            for k in K_GRID:
                t0 = time.time()
                est = simulate_vertex_scores(hg, w, k=k, reps=REPS, seed=3, cov=COV[name])
                rows.append(dict(label=name, domain=DOMAIN[name], method=method, k=k,
                                 prec_mean=np.mean([precision_at_k(e, ex, TOPK) for e in est]),
                                 spearman=np.nanmean([rank_correlation(e, ex) for e in est]),
                                 secs=round(time.time() - t0, 1)))
            print(f"{name:24s} {method:10s} P@{TOPK}={rows[-1]['prec_mean']:.3f} "
                  f"rho={rows[-1]['spearman']:.3f}")
    df = pd.DataFrame(rows)
    agg = df.groupby(['method', 'k'])[['prec_mean', 'spearman']].mean()
    print('\n', agg.round(3).to_string())
    save(df, 'table15_ranking_from_stream.csv')


if __name__ == '__main__':
    main()

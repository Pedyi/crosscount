"""Table 8: words of state to reach 10% relative error, against prediction-free methods.

Every estimator is unbiased, so the cost is set by its variance.  Methods that keep O(1)
state per instance are charged one word per instance; methods that count t(e) exactly are
charged deg(u) + deg(v) words per sampled edge, since that is what a local count needs.
"""
import pandas as pd
from _common import build_all, save, DOMAIN
from crosscount.bench import (baseline_cost, weighted_cost, wedge_cost,
                              edge_local_cost, heavy_storage_cost)


def main():
    _, HG = build_all()
    rows = []
    for name, hg in HG.items():
        r = dict(label=name, domain=DOMAIN[name], m=hg.m, n_cross=hg.n_cross_total,
                 copy_density=hg.n_cross_total / hg.m,
                 fp=baseline_cost(hg)['words'],
                 localized=weighted_cost(hg, hg.w_uniform())['words'],
                 weighted=weighted_cost(hg, hg.w_perfect())['words'],
                 wedge=wedge_cost(hg)['words'],
                 edge_local=edge_local_cost(hg)['words'])
        h = heavy_storage_cost(hg)
        r['heavy'], r['heavy_tau'] = h['words'], h['tau']
        rows.append(r)
        print(f"{name:24s} fp={r['fp']:.2e} loc={r['localized']:.2e} "
              f"wgt={r['weighted']:.2e} wedge={r['wedge']:.2e} "
              f"edge+loc={r['edge_local']:.2e} heavy={r['heavy']:.2e}")
    df = pd.DataFrame(rows)
    cols = ['fp', 'localized', 'weighted', 'wedge', 'edge_local', 'heavy']
    df['best'] = df[cols].idxmin(axis=1)
    print('\nwins per method:')
    print(df.best.value_counts().to_string())
    save(df, 'table8_space_comparison.csv')


if __name__ == '__main__':
    main()

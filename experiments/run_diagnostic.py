"""Table 1: the diagnostic alpha_H / kappa, exact, on all ten hypergraph projections."""
import pandas as pd
from _common import build_all, save, DOMAIN
from crosscount import diagnostic_crossK3

TIME_BUDGET = 1200.0    # seconds per graph for the exact alpha


def main():
    _, HG = build_all()
    rows = []
    for name, hg in HG.items():
        r = diagnostic_crossK3(hg, time_budget=TIME_BUDGET)
        r['label'], r['domain'] = name, DOMAIN[name]
        rows.append(r)
        a = r['alpha'] if r['alpha'] is not None else f"[{r['alpha_lb']},{r['alpha_ub']}]"
        print(f"{name:24s} kappa={r['kappa']:>4d} kappa_copy={r['kappa_copy']:>4d} "
              f"alpha={a} kappa_copy/kappa={r['kappa_copy_over_kappa']:.3f}")
    df = pd.DataFrame(rows)
    ok = df.dropna(subset=['alpha_over_kappa'])
    if len(ok):
        print(f"\nalpha/kappa {ok.alpha_over_kappa.min():.3f}-{ok.alpha_over_kappa.max():.3f} "
              f"(mean {ok.alpha_over_kappa.mean():.3f})")
    print(f"kappa_copy/kappa {df.kappa_copy_over_kappa.min():.3f}-"
          f"{df.kappa_copy_over_kappa.max():.3f}")
    save(df, 'table1_diagnostic.csv')


if __name__ == '__main__':
    main()

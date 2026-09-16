"""Table 10: predictions on the wedge estimator.

The prediction-free wedge sampler is often cheaper than the Fichtenberger-Peng one, so the
question of what a predictor buys has to be asked there too.  Setting w = 0 in the weighted
form recovers prediction-free uniform wedge sampling exactly, which is asserted below.
"""
import numpy as np
import pandas as pd
from _common import build_all, save, DOMAIN
from crosscount.bench import wedge_cost, wedge_weighted_cost, wedge_report


def main():
    _, HG = build_all()
    for name, hg in HG.items():
        a, b = wedge_cost(hg)['words'], wedge_weighted_cost(hg, hg.w_uniform())['words']
        assert abs(a - b) / a < 1e-9, (name, a, b)
    print('weighted wedge at w=0 matches the closed form on every graph\n')

    rows = []
    for name, hg in HG.items():
        r = wedge_report(hg, n_perm=1000, seed=0)
        if r['perfect_over_unif'] <= 1.0:       # gain undefined, do not report a share
            r['structural_share'] = np.nan
        r['label'], r['domain'] = name, DOMAIN[name]
        rows.append(r)
        print(f"{name:24s} pred={r['perfect_over_unif']:5.2f}x "
              f"perm={r['permuted_over_unif_mean']:5.3f}x "
              f"mdeg={r['mindeg_over_unif']:5.3f}x share={r['structural_share']:.3f}")
    df = pd.DataFrame(rows)
    helps = df.perfect_over_unif > 1.0
    print(f"\na perfect predictor helps on {int(helps.sum())}/{len(df)} graphs")
    print(f"permuted below 1 on {int((df.permuted_over_unif_mean < 1).sum())}/{len(df)}")
    save(df, 'table10_wedge_predictions.csv')


if __name__ == '__main__':
    main()

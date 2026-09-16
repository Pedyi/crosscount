"""Table 9: the estimator run for real; relative error against the number of instances."""
import numpy as np
import pandas as pd
from _common import build_all, save
from crosscount.bench import simulate_estimator, weighted_cost, _covbits

GRAPHS = ['email-Enron', 'NDC-classes', 'contact-high-school', 'contact-primary-school']
K_GRID = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
REPS = 25


def main():
    _, HG = build_all(GRAPHS)
    rows = []
    for name, hg in HG.items():
        cov = _covbits(hg)
        truth = hg.n_cross_total
        for tag, w in [('localized', hg.w_uniform()), ('weighted', hg.w_perfect())]:
            var = weighted_cost(hg, w)['var']
            for k in K_GRID:
                est = simulate_estimator(hg, w, k=k, reps=REPS, seed=11, cov=cov)
                rows.append(dict(label=name, method=tag, k=k, truth=truth,
                                 mean=est.mean(), rel_bias=est.mean() / truth - 1,
                                 rel_rmse=float(np.sqrt(((est - truth) ** 2).mean())) / truth,
                                 predicted_cov=np.sqrt(var / k) / truth))
            print(f"{name:24s} {tag:10s} bias {rows[-1]['rel_bias']:+.4f} "
                  f"rmse {rows[-1]['rel_rmse']:.4f} predicted {rows[-1]['predicted_cov']:.4f}")
    df = pd.DataFrame(rows)
    print(f"\nlargest relative bias: {df.rel_bias.abs().max():.4f}")
    save(df, 'table9_simulation.csv')


if __name__ == '__main__':
    main()

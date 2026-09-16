"""Appendix: Monte-Carlo check that the analytic success probabilities are the right ones.

Runs the actual weighted sampler -- draw a base edge, draw a closing vertex, verify the
cross triangle -- and compares the hit rate with the analytic value.
"""
import pandas as pd
from _common import build_all, save
from crosscount import monte_carlo_check

N_SAMPLES = 40_000
N_GRAPHS = 4


def main():
    _, HG = build_all()
    rows = []
    for name in list(HG)[:N_GRAPHS]:
        hg = HG[name]
        for tag, w in [('perfect', hg.w_perfect()), ('uniform', hg.w_uniform())]:
            an = hg.p_succ(w)
            mc = monte_carlo_check(hg, w, n_samples=N_SAMPLES, seed=7)
            rows.append(dict(label=name, predictor=tag, analytic=an, monte_carlo=mc,
                             rel_err=abs(mc - an) / an if an else float('nan')))
            print(f"{name:24s} {tag:8s} analytic={an:.6f} MC={mc:.6f} "
                  f"rel.err={rows[-1]['rel_err']:.4f}")
    save(pd.DataFrame(rows), 'validation_montecarlo.csv')


if __name__ == '__main__':
    main()

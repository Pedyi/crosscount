"""Table 3: a ridge predictor, strict leave-one-graph-out over all ordered pairs."""
import itertools

import numpy as np
import pandas as pd
from _common import build_all, save
from crosscount import ridge_fit, ridge_predict_weights, FEATURE_BLOCKS

LAMBDAS = [0.1, 1.0, 10.0]


def boot_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan
    s = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    _, HG = build_all()
    names = list(HG)
    pairs = []
    for blk in FEATURE_BLOCKS:
        for lam in LAMBDAS:
            for tr, te in itertools.permutations(names, 2):
                mdl = ridge_fit(HG[tr].F[blk], HG[tr].y, lam=lam)
                w = ridge_predict_weights(mdl, HG[te].F[blk])
                hg = HG[te]
                p_loc = hg.p_succ(hg.w_uniform())
                s_learn = hg.p_succ(w) / p_loc
                s_perf = hg.p_succ(hg.w_perfect()) / p_loc
                gain = s_perf - 1.0
                pairs.append(dict(
                    block=blk, lam=lam, train=tr, test=te,
                    learned_over_loc=s_learn, perfect_over_loc=s_perf,
                    capture=(s_learn - 1) / gain if abs(gain) > 1e-12 else np.nan,
                    pred_corr=float(np.corrcoef(np.log1p(w), hg.y)[0, 1])))
    P = pd.DataFrame(pairs)
    summ = []
    for blk in FEATURE_BLOCKS:
        for lam in LAMBDAS:
            d = P[(P.block == blk) & (P.lam == lam)]
            lo, hi = boot_ci(d.capture)
            summ.append(dict(block=blk, lam=lam, n_pairs=len(d),
                             capture_mean=d.capture.mean(), capture_lo=lo, capture_hi=hi,
                             corr_mean=d['pred_corr'].mean(),
                             learned_over_loc_mean=d.learned_over_loc.mean()))
    T = pd.DataFrame(summ)
    print(T.round(4).to_string(index=False))
    best = T.sort_values('capture_mean').iloc[-1]
    print(f"\nbest: block {best.block} recovers {best.capture_mean:.3f} "
          f"[{best.capture_lo:.3f}, {best.capture_hi:.3f}] over {int(best.n_pairs)} pairs")
    save(P, 'table3_learned_pairs.csv')
    save(T, 'table3_learned_summary.csv')


if __name__ == '__main__':
    main()

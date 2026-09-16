"""Tables 17 and 18: a stronger learner, and confidence intervals for the copy-density rule.

The paired comparison is the right one here: ridge and gradient-boosted trees see identical
features, targets and splits, so the per-pair difference is the statistic to bootstrap.
"""
import itertools

import numpy as np
import pandas as pd
from _common import build_all, save, DOMAIN
from crosscount import (ridge_fit, ridge_predict_weights, gbt_fit, gbt_predict_weights,
                        FEATURE_BLOCKS)
from crosscount.bench import wedge_weighted_cost


def boot_ci(x, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if not len(x):
        return np.nan, np.nan
    s = rng.choice(x, size=(B, len(x)), replace=True).mean(1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def main():
    _, HG = build_all()
    names = list(HG)
    rows = []
    for blk in FEATURE_BLOCKS:
        for tr, te in itertools.permutations(names, 2):
            hg = HG[te]
            p_loc = hg.p_succ(hg.w_uniform())
            s_perf = hg.p_succ(hg.w_perfect()) / p_loc
            gain = s_perf - 1.0
            rec = dict(block=blk, train=tr, test=te, perfect_over_loc=s_perf)
            for tag, fit, pred in [('ridge', ridge_fit, ridge_predict_weights),
                                   ('gbt', gbt_fit, gbt_predict_weights)]:
                w = pred(fit(HG[tr].F[blk], HG[tr].y), hg.F[blk])
                s = hg.p_succ(w) / p_loc
                rec[f'{tag}_capture'] = (s - 1) / gain if abs(gain) > 1e-12 else np.nan
                rec[f'{tag}_corr'] = float(np.corrcoef(np.log1p(w), hg.y)[0, 1])
            rows.append(rec)
        print(f'block {blk} done')
    L = pd.DataFrame(rows)
    L['paired_diff'] = L.gbt_capture - L.ridge_capture
    save(L, 'table17_learner_pairs.csv')

    summ = []
    for blk in FEATURE_BLOCKS:
        d = L[L.block == blk]
        lo_r, hi_r = boot_ci(d.ridge_capture)
        lo_g, hi_g = boot_ci(d.gbt_capture)
        lo_d, hi_d = boot_ci(d.paired_diff)
        summ.append(dict(block=blk, n_pairs=len(d),
                         ridge=d.ridge_capture.mean(), ridge_lo=lo_r, ridge_hi=hi_r,
                         gbt=d.gbt_capture.mean(), gbt_lo=lo_g, gbt_hi=hi_g,
                         diff=d.paired_diff.mean(), diff_lo=lo_d, diff_hi=hi_d))
    S = pd.DataFrame(summ)
    print(S.round(4).to_string(index=False))
    save(S, 'table17_learner_summary.csv')

    rows = []
    for name, hg in HG.items():
        p_loc = hg.p_succ(hg.w_uniform())
        rows.append(dict(label=name, domain=DOMAIN[name],
                         copy_density=hg.n_cross_total / hg.m,
                         fp_gain=hg.p_succ(hg.w_perfect()) / p_loc,
                         wedge_gain=(wedge_weighted_cost(hg, hg.w_uniform())['words'] /
                                     wedge_weighted_cost(hg, hg.w_perfect())['words'])))
    D = pd.DataFrame(rows)
    for tag, col in [('augmented', 'fp_gain'), ('wedge', 'wedge_gain')]:
        ok = D[col] > 1
        if ok.sum() < 4:
            print(f'{tag}: too few graphs with a gain to fit'); continue
        x, y = np.log(D.copy_density[ok].values), np.log(D[col][ok].values - 1)
        sl = np.polyfit(x, y, 1)[0]
        rng = np.random.default_rng(0)
        sls = [np.polyfit(x[i], y[i], 1)[0]
               for i in (rng.choice(len(x), size=len(x), replace=True) for _ in range(4000))
               if len(np.unique(x[i])) > 1]
        print(f'{tag}: slope {sl:+.2f} '
              f'[{np.percentile(sls, 2.5):+.2f}, {np.percentile(sls, 97.5):+.2f}] '
              f'over {int(ok.sum())} graphs')
    save(D, 'table18_density_rule.csv')


if __name__ == '__main__':
    main()

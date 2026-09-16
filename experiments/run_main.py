"""Table 2: localization, perfect predictor, permutation null, cheap heuristic."""
import numpy as np
import pandas as pd
from _common import build_all, save, DOMAIN
from crosscount import evaluate

N_PERM = 1000


def main():
    _, HG = build_all()
    rows = []
    for name, hg in HG.items():
        r = evaluate(hg, n_perm=N_PERM, seed=0)
        r['label'], r['domain'] = name, DOMAIN[name]
        rows.append(r)
        print(f"{name:24s} loc={r['loc_x']:5.2f}x pred={r['perfect_over_loc']:5.2f}x "
              f"perm={r['permuted_over_loc_mean']:.3f}x share={r['structural_share']:.3f} "
              f"p={r['perm_pvalue']:.4f}")
    df = pd.DataFrame(rows)
    print(f"\nlocalization {df.loc_x.min():.2f}-{df.loc_x.max():.2f}x, "
          f"log-log slope in m {np.polyfit(np.log(df.m), np.log(df.loc_x), 1)[0]:+.3f}")
    print(f"predictor    {df.perfect_over_loc.min():.2f}-{df.perfect_over_loc.max():.2f}x")
    print(f"struct share {df.structural_share.min():.3f}-{df.structural_share.max():.3f}")
    x, y = np.log(df.copy_density), np.log(df.perfect_over_loc - 1)
    print(f"log(pred-1) vs log(copy density): slope {np.polyfit(x, y, 1)[0]:.3f}, "
          f"r={np.corrcoef(x, y)[0, 1]:.3f}")
    save(df, 'table2_main.csv')


if __name__ == '__main__':
    main()

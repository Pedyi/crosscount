"""Table 4: cross-hyperedge four-cycles.

C4 has no closing-vertex draw, hence no localization term; everything is measured against
the unweighted sampler and is attributable to the predictor alone.  Enumeration cost is the
binding constraint, so graphs over budget are skipped with the reason printed.
"""
import pandas as pd
from _common import build_all, save, DOMAIN
from crosscount import CrossC4

MAX_C4 = 15_000_000
MAX_SUMDEG2 = 5e7
MAX_N = 3500


def main():
    _, HG = build_all()
    rows = []
    for name, hg in HG.items():
        c = CrossC4(hg, max_c4=MAX_C4, max_sumdeg2=MAX_SUMDEG2, max_n=MAX_N)
        if not c.ok:
            print(f'{name:24s} skipped: {c.reason}')
            continue
        r = c.evaluate(n_perm=1000, seed=0)
        r['label'], r['domain'] = name, DOMAIN[name]
        rows.append(r)
        print(f"{name:24s} pred={r['perfect_over_unif']:.3f}x "
              f"perm={r['permuted_over_unif_mean']:.3f}x "
              f"share={r['structural_share']:.3f} p={r['perm_pvalue']:.4f}")
    if not rows:
        print('every graph was over budget')
        return
    df = pd.DataFrame(rows)
    print(f"\nstructural share {df.structural_share.min():.3f}-"
          f"{df.structural_share.max():.3f} (mean {df.structural_share.mean():.4f})")
    save(df, 'table4_crossc4.csv')


if __name__ == '__main__':
    main()

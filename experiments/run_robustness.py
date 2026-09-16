"""Table 5: sensitivity to dataset size and to triple subsampling."""
import pandas as pd
from _common import build_all, save, MAX_TRIPLES
from crosscount import CrossHG, evaluate

BIG = 'congress-bills'
CAPS = [3_000, 10_000, 30_000, None]          # None = all simplices
TRIPLE_SETTINGS = [(1_000_000, 1), (2_000_000, 2), (4_000_000, 3)]


def main():
    hyper, _ = build_all([BIG])
    H = hyper[BIG]
    rows = []
    for cap in CAPS:
        sub = H if cap is None else H[:cap]
        hg = CrossHG(sub, max_size=25, max_triples=MAX_TRIPLES, seed=0, verbose=False)
        r = evaluate(hg, n_perm=15, seed=0)
        r['cap'] = len(sub)
        rows.append(r)
        print(f"cap={len(sub):>7d} m={r['m']:>7d} loc={r['loc_x']:.2f}x "
              f"pred={r['perfect_over_loc']:.3f}x share={r['structural_share']:.3f}")
    save(pd.DataFrame(rows), 'table5_cap_sensitivity.csv')

    rows = []
    for mt, sd in TRIPLE_SETTINGS:
        hg = CrossHG(H, max_size=25, max_triples=mt, seed=sd, verbose=False)
        r = evaluate(hg, n_perm=10, seed=0)
        r['max_triples'], r['seed'] = mt, sd
        rows.append(r)
        print(f"triples={mt:>9d} frac={r['triple_fraction']:.4f} loc={r['loc_x']:.2f}x "
              f"pred={r['perfect_over_loc']:.3f}x")
    save(pd.DataFrame(rows), 'table5_triple_sensitivity.csv')


if __name__ == '__main__':
    main()

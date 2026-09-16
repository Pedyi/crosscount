"""Table 6: the separable synthetic family, where localization does grow with m."""
import numpy as np
import pandas as pd
from _common import save
from crosscount import CrossHG, evaluate, cliques_plus_cross

SIZES = [20, 30, 40, 60, 80]


def main():
    rows = []
    for cs in SIZES:
        hg = CrossHG(cliques_plus_cross(6, cs, 6 * cs, seed=1), max_size=200, verbose=False)
        r = evaluate(hg, n_perm=15, seed=0)
        r['label'] = f'cliques(r={cs})'
        rows.append(r)
        print(f"r={cs:>3d} m={r['m']:>6d} loc={r['loc_x']:7.1f}x "
              f"pred={r['perfect_over_loc']:.3f}x")
    df = pd.DataFrame(rows)
    b = np.polyfit(np.log(df.m), np.log(df.loc_x), 1)[0]
    print(f"\nlocalization grows as m^{b:.3f}  (prediction-free exponent rho(K3)-1 = 0.5)")
    save(df, 'table6_synthetic.csv')


if __name__ == '__main__':
    main()

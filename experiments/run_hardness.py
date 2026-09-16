"""Table 16: our bound on the instances used to prove the known lower bounds."""
import numpy as np
import pandas as pd
from _common import save
from crosscount.hardness import hubs_graph, bera_chakrabarti, instance_report
from crosscount.synthetic import friendship_plus_decoy

S = 1.5


def main():
    rows = []
    for r, d in [(1, 64), (1, 256), (4, 64), (8, 64)]:
        rows.append(instance_report(hubs_graph(r, d), f'hubs r={r} d={d}'))
    for s in [64, 256, 1024]:
        rows.append(instance_report(friendship_plus_decoy(s), f'friendship+decoy s={s}'))
    for N in [6, 9, 12, 16, 20]:
        b = max(2, int(round(N ** S)))
        d = max(1, int(round(N ** (S - 1))))
        ones = max(1, N // 3)
        x = np.zeros(N, int); y = np.zeros(N, int)
        x[:ones] = 1; y[ones - 1:2 * ones - 1] = 1
        r = instance_report(bera_chakrabarti(b, d, N, x, y), f'BC N={N} b={b} d={d}')
        r['N'], r['b'] = N, b
        rows.append(r)
    df = pd.DataFrame(rows)
    print(df[['label', 'm', 'n_triangles', 'kappa', 'alpha', 'alpha_over_kappa',
              'ours', 'bera_seshadhri', 'bera_chakrabarti_lb']].round(3).to_string(index=False))
    bc = df.dropna(subset=['N'])
    print(f"\nBC family: ours/N in "
          f"[{(bc.ours / bc.N).min():.2f}, {(bc.ours / bc.N).max():.2f}]; "
          f"log-log slope of ours/LB in N = "
          f"{np.polyfit(np.log(bc.N), np.log(bc.ours_over_lb), 1)[0]:+.3f}")
    save(df, 'table16_hard_instances.csv')


if __name__ == '__main__':
    main()

"""Tables 12 and 13: the higher-order triadic profile, exact and from a stream."""
import time

import numpy as np
import pandas as pd
from _common import build_all, save, DOMAIN
from crosscount.bench import _covbits
from crosscount.profile import exact_profile, simulate_profile, domain_separation

MAX_M_SIM = 150_000
K_GRID = [200, 500, 1000, 2000, 5000, 10000, 20000]
REPS = 10


def main():
    _, HG = build_all()

    prof = []
    for name, hg in HG.items():
        p = exact_profile(hg)
        p['label'], p['domain'], p['m'] = name, DOMAIN[name], hg.m
        prof.append(p)
    PROF = pd.DataFrame(prof)
    print(PROF[['label', 'domain', 'n_triangles', 'n_cross', 'f_cross',
                'f_persistent']].round(4).to_string(index=False))
    acc = domain_separation(PROF[['f_cross', 'f_persistent']].values, PROF.domain.values)
    print(f'\nexact profile, leave-one-out domain accuracy: {acc:.2f}')
    save(PROF, 'table12_profile_exact.csv')

    names = [n for n, h in HG.items() if h.m <= MAX_M_SIM]
    print(f'\nsimulating on {len(names)} of {len(HG)} graphs')
    COV = {n: _covbits(HG[n]) for n in names}
    exact_pts = PROF.set_index('label').loc[names, ['f_cross', 'f_persistent']].values

    rows = []
    for method, wfun in [('localized', lambda h: h.w_uniform()),
                         ('weighted', lambda h: h.w_perfect())]:
        for k in K_GRID:
            t0 = time.time()
            ests = {}
            for n in names:
                out = simulate_profile(HG[n], wfun(HG[n]), k=k, reps=REPS, seed=5, cov=COV[n])
                with np.errstate(invalid='ignore', divide='ignore'):
                    ests[n] = np.column_stack([out[:, 1] / out[:, 0], out[:, 2] / out[:, 1]])
            accs = []
            for r in range(REPS):
                pts = np.array([ests[n][r] for n in names])
                if np.isfinite(pts).all():
                    accs.append(domain_separation(pts, [DOMAIN[n] for n in names]))
            errs = [np.nanmean(np.abs(ests[n] - exact_pts[i]) / np.abs(exact_pts[i]))
                    for i, n in enumerate(names)]
            rows.append(dict(method=method, k=k, sep_mean=np.mean(accs),
                             sep_std=np.std(accs), rel_err=np.mean(errs),
                             secs=round(time.time() - t0, 1)))
            print(f'{method:10s} k={k:>6d} separation {np.mean(accs):.2f} '
                  f'rel.err {np.mean(errs):.4f}')
    save(pd.DataFrame(rows), 'table13_profile_from_stream.csv')


if __name__ == '__main__':
    main()

"""Why the learned predictor transfers to some graphs and not others."""
import os

import numpy as np
import pandas as pd
from _common import build_all, save, RESULTS
from crosscount import feature_diagnosis


def main():
    _, HG = build_all()
    F = pd.DataFrame([feature_diagnosis(hg, name) for name, hg in HG.items()])
    path = os.path.join(RESULTS, 'table3_learned_pairs.csv')
    if os.path.exists(path):
        raw = pd.read_csv(path)
        d = raw[(raw.block == 'ABC') & (raw.lam == 0.1)]
        F['capture'] = F.label.map(d.groupby('test')['capture'].mean())
        G = F.dropna(subset=['capture'])
        for col in ['y_std', 't_cv', 'gini', 'best_feat_corr_ABC', 'R2_in_sample_ABC']:
            print(f'corr(capture, {col:>18s}) = '
                  f'{np.corrcoef(G[col], G.capture)[0, 1]:+.3f}')
    else:
        print('run run_learned.py first to attach the capture column')
    save(F, 'table7_feature_diagnosis.csv')


if __name__ == '__main__':
    main()

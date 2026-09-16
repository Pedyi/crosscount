"""Table 11: which normalizer a skewed weighting inflates.

W is linear and unchanged; D_x is linear but local and shrinks; T is quadratic and inflates.
That is why the augmented estimator profits from a predictor and the wedge estimator does not.
"""
import numpy as np
import pandas as pd
from _common import build_all, save
from crosscount.bench import normalizer_inflation


def main():
    _, HG = build_all()
    df = pd.DataFrame([dict(label=n, **normalizer_inflation(h)) for n, h in HG.items()])
    print(df[['label', 'weight_cv', 'W_inflation', 'D_mean_inflation', 'D_max_inflation',
              'T_inflation', 'weighted_gain', 'wedge_gain']].round(3).to_string(index=False))
    print(f"\nW unchanged on {int((df.W_inflation.round(6) == 1).sum())}/{len(df)} graphs")
    print(f"D_x shrinks on {int((df.D_mean_inflation < 1).sum())}/{len(df)}; "
          f"T inflates on {int((df.T_inflation > 1).sum())}/{len(df)}")
    ok = df.weighted_gain > 1
    print(f"corr(log gain, log D_mean_inflation) = "
          f"{np.corrcoef(np.log(df.weighted_gain), np.log(df.D_mean_inflation))[0, 1]:+.3f}")
    print(f"corr(log(gain-1), log weight_cv) over the {int(ok.sum())} graphs with a gain = "
          f"{np.corrcoef(np.log(df.loc[ok, 'weighted_gain'] - 1), np.log(df.loc[ok, 'weight_cv']))[0, 1]:+.3f}")
    save(df, 'table11_normalizer_inflation.csv')


if __name__ == '__main__':
    main()

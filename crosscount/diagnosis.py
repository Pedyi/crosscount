"""Why a learned predictor transfers to some graphs and not others.

Reports, per graph, the spread of the target and how well each feature block fits it
in sample.  Used for the analysis of the `contact-high-school` outlier in the paper.
"""
import numpy as np

__all__ = ['feature_diagnosis']


def feature_diagnosis(hg, label=''):
    """Signal available to a predictor on this graph: target spread and feature strength."""
    y = hg.y                       # log1p(t_cross) per edge
    t = hg.t
    out = dict(label=label, m=hg.m, n=hg.n,
               frac_zero=float((t == 0).mean()),
               t_mean=float(t.mean()), t_std=float(t.std()),
               t_cv=float(t.std() / t.mean()) if t.mean() else np.nan,
               y_std=float(y.std()),
               max_hyperedge=int(hg.hsize.max()),
               gini=float(_gini(t.astype(float))))
    for blk in ['A', 'AB', 'ABC']:
        X = hg.F[blk]
        cors = [abs(np.corrcoef(X[:, j], y)[0, 1]) if X[:, j].std() > 0 else 0.0
                for j in range(X.shape[1])]
        out[f'best_feat_corr_{blk}'] = float(np.nanmax(cors))
        out[f'R2_in_sample_{blk}'] = float(_r2(X, y))
    return out


def _gini(x):
    x = np.sort(x)
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def _r2(X, y):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    return 1 - resid.var() / (y.var() + 1e-12)

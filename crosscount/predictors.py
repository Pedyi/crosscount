"""Predictors: the perfect oracle, the permutation null, the cheap heuristic, and a
transferable ridge model over three feature blocks.

Feature blocks (all standardised per graph before fitting):
  A    endpoint degrees and core numbers      -- the block used for ordinary triangles
  AB   A + cheap higher-order features        -- hyperedge membership of the endpoints
  ABC  AB + the covered-common-neighbour count
"""
import numpy as np

__all__ = ['ridge_fit', 'ridge_predict_weights', 'gbt_fit', 'gbt_predict_weights',
           'cluster_bootstrap_ci', 'FEATURE_BLOCKS']

FEATURE_BLOCKS = ['A', 'AB', 'ABC']


def ridge_fit(X, y, lam=1.0):
    mx, sx = X.mean(0), X.std(0) + 1e-12
    Z = (X - mx) / sx
    my, sy = y.mean(), y.std() + 1e-12
    yz = (y - my) / sy
    Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    A = Z1.T @ Z1 + lam * np.eye(Z1.shape[1])
    A[-1, -1] -= lam                       # do not penalise the intercept
    beta = np.linalg.solve(A, Z1.T @ yz)
    return dict(beta=beta, mx=mx, sx=sx, my=my, sy=sy)


def ridge_predict_weights(model, X):
    Z = (X - model['mx']) / model['sx']
    Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    yz = Z1 @ model['beta']
    y = yz * model['sy'] + model['my']
    return np.clip(np.expm1(np.clip(y, 0, 30)), 0, None)




# ------------------------------------------------- a stronger learner, for the ablation

def gbt_fit(X, y, **kw):
    """Gradient-boosted trees on the same standardised features and log target as ridge.

    Included to test whether the informational reading of the permutation control survives a
    model that is not linear: if a stronger learner recovers much more of the achievable gain,
    the limit was the model; if it recovers about the same, the limit is the signal.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    mx, sx = X.mean(0), X.std(0) + 1e-12
    my, sy = y.mean(), y.std() + 1e-12
    params = dict(max_iter=200, learning_rate=0.1, max_depth=None,
                  early_stopping=False, random_state=0)
    params.update(kw)
    model = HistGradientBoostingRegressor(**params).fit((X - mx) / sx, (y - my) / sy)
    return dict(model=model, mx=mx, sx=sx, my=my, sy=sy)


def gbt_predict_weights(model, X):
    z = (X - model['mx']) / model['sx']
    y = model['model'].predict(z) * model['sy'] + model['my']
    return np.clip(np.expm1(np.clip(y, 0, 30)), 0, None)



def cluster_bootstrap_ci(df, col, train_col='train', test_col='test', B=4000, seed=0):
    """Bootstrap that resamples GRAPHS, not train/test pairs.

    With G graphs the leave-one-graph-out design yields G(G-1) pairs, but each graph appears in
    2(G-1) of them, so the pairs are far from independent and a pair-level bootstrap
    understates the uncertainty --- on our data by roughly a factor of two and a half.  Drawing
    graphs with replacement and weighting each pair by the product of its endpoints'
    multiplicities respects the dependence.
    """
    import pandas as pd
    graphs = sorted(set(df[train_col]) | set(df[test_col]))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        cnt = pd.Series(rng.choice(graphs, size=len(graphs), replace=True)).value_counts()
        w = (df[train_col].map(cnt).fillna(0).values
             * df[test_col].map(cnt).fillna(0).values)
        v = df[col].values
        keep = np.isfinite(v) & (w > 0)
        if keep.any():
            out.append(np.average(v[keep], weights=w[keep]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

"""Known hard instances for streaming triangle counting, measured in our parameters.

The point of this module is to check whether the upper bound of the prediction-augmented
estimator, O~(m * alpha_H / #H), is tight on the instances the literature uses to prove
lower bounds.  If it is, alpha_H is the right parameter and a matching lower bound is worth
attempting; if it is not, the upper bound is loose somewhere and the gap says where.

Three families:

  hubs_graph          Kallaugher and Price's instance, used by Chen et al. for their
                      one-pass lower bound with a heavy-edge oracle.  Every edge lies on at
                      most one triangle, so a heaviness oracle is vacuous on it.
  bera_chakrabarti    the set-disjointness construction behind the constant-pass bound
                      Omega(min{m^{3/2}/T, m/sqrt(T)}).
  friendship_decoy    the separation instance of the PredCount paper (in synthetic.py).

For each we report m, kappa, alpha_H, #K3, and three space bounds: ours, the
degeneracy-based prediction-free bound of Bera and Seshadhri, and the prediction-free
lower bound of Bera and Chakrabarti.  The comparison with Bera and Seshadhri matters,
because that algorithm is also constant-pass: an instance only witnesses a separation if
predictions beat it too, which is exactly what the diagnostic alpha_H / kappa measures.
"""
import numpy as np

__all__ = ['hubs_graph', 'bera_chakrabarti', 'instance_report']


def hubs_graph(r, d):
    """Hub vertex with 2rd incident edges; d disjoint neighbour pairs joined into triangles."""
    E = []
    hub = ('h',)
    for i in range(2 * r * d):
        E.append((hub, ('leaf', i)))
    for j in range(d):                    # pair up 2j and 2j+1 to close a triangle
        E.append((('leaf', 2 * j), ('leaf', 2 * j + 1)))
    return E


def bera_chakrabarti(b, d, n_blocks, x, y):
    """The set-disjointness instance: K_{b,b} plus blocks attached by Alice and Bob.

    A triangle exists for every index i with x_i = y_i = 1 and every (u, v, w) with u in A,
    v in B, w in V_i, so #K3 = b^2 * d * |x AND y|.
    """
    E = []
    A = [('A', i) for i in range(b)]
    B = [('B', i) for i in range(b)]
    for u in A:
        for v in B:
            E.append((u, v))
    for i in range(n_blocks):
        V = [('V', i, t) for t in range(d)]
        if x[i]:
            for u in A:
                for w in V:
                    E.append((u, w))
        if y[i]:
            for v in B:
                for w in V:
                    E.append((v, w))
    return E


def instance_report(edges, label='', exact_alpha=True):
    """Measure the instance and evaluate the three bounds on it."""
    from .core import canon, build_adj, degeneracy, oracle_width
    from .synthetic import incidence_edges  # noqa: F401  (kept for symmetry of imports)

    E = canon(edges)
    adj = build_adj(E)
    m = len(E)
    kappa, _ = degeneracy(adj)

    cb, tri = [], 0
    for u in adj:
        for v in adj[u]:
            if u < v:
                c = len(adj[u] & adj[v])
                if c:
                    cb.append((u, v))
                    tri += c
    n_tri = tri // 3
    kappa_copy, _ = degeneracy(build_adj(cb)) if cb else (0, {})
    if cb and exact_alpha:
        eu = np.array([e[0] for e in cb]); ev = np.array([e[1] for e in cb])
        alpha = oracle_width(eu, ev)
    else:
        alpha = 0

    # floor degree: the largest, over triangles, of the least degree of its vertices
    deg = {v: len(adj[v]) for v in adj}
    tau = 0
    for (u, v) in cb:
        for w in adj[u] & adj[v]:
            tau = max(tau, min(deg[u], deg[v], deg[w]))

    ours = (m * alpha / n_tri) if n_tri else np.inf          # O~(m alpha_H / #H)
    bs = (m * kappa / n_tri) if n_tri else np.inf            # Bera and Seshadhri
    bc = (min(m ** 1.5 / n_tri, m / np.sqrt(n_tri))          # Bera and Chakrabarti
          if n_tri else np.inf)
    return dict(
        label=label, m=m, n_triangles=n_tri, kappa=kappa, kappa_copy=kappa_copy,
        alpha=alpha, tau=tau, separable=(tau <= 4 * max(alpha, 1)),
        alpha_over_kappa=alpha / kappa if kappa else np.nan,
        ours=ours, bera_seshadhri=bs, bera_chakrabarti_lb=bc,
        ours_over_lb=ours / bc if bc else np.nan,
        ours_over_bs=ours / bs if bs else np.nan,
    )

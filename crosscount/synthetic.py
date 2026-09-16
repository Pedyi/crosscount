"""Synthetic families.

Two of these realise the separation regime (Proposition 4 of the paper) and two are
negative controls that do not.
"""
import numpy as np

__all__ = ['friendship_plus_decoy', 'affine_plane_incidence', 'steiner_plus_butterflies',
           'cliques_plus_cross', 'incidence_edges', 'random_uniform_hypergraph']


def friendship_plus_decoy(s):
    """The separation instance of the PredCount paper: F_s together with K_{d,d}.

    kappa = Theta(sqrt(m)) from the decoy, alpha_{K3} = 2 from the friendship graph.
    """
    d = int(np.ceil(np.sqrt(s)))
    E = []
    for i in range(s):
        E += [('h', ('a', i)), ('h', ('b', i)), (('a', i), ('b', i))]
    for i in range(d):
        for j in range(d):
            E.append((('L', i), ('R', j)))
    return E


def affine_plane_incidence(q):
    """Incidence graph of AG(2, q) for prime q.

    Any two points lie on exactly one line, so the incidence graph is C4-free while its
    degeneracy is Theta(q) = Theta(sqrt(m)): a dense region bearing no copies, from
    geometry rather than construction.
    """
    E = []
    for a in range(q):
        for b in range(q):
            lid = ('L', a, b)
            for x in range(q):
                E.append((('P', x, (a * x + b) % q), lid))
    for c in range(q):
        lid = ('V', c)
        for y in range(q):
            E.append((('P', c, y), lid))
    return E


def steiner_plus_butterflies(q, n_gadgets, r=3):
    """AG(2, q) as a copy-free dense core, plus shallow butterfly gadgets.

    C4-freeness alone is vacuous for counting, so pairs co-occurring in two hyperedges are
    added; they are vertex-disjoint, so the copy-bearing subgraph has constant
    pseudoarboricity.
    """
    E = affine_plane_incidence(q)
    for g in range(n_gadgets):
        a, b = ('G', g, 0), ('G', g, 1)
        for t in range(2):
            h = ('H', g, t)
            E.append((a, h))
            E.append((b, h))
            for j in range(r - 2):
                E.append((('F', g, t, j), h))
    return E


def cliques_plus_cross(n_cliques, clique_size, n_gadgets, seed=0):
    """Disjoint large hyperedges (copy-free cliques) plus shallow cross-triangle gadgets.

    A hyperedge of size r projects to K_r, so kappa >= r - 1, but no triple inside it is
    cross: the dense region bears no copies.  This is the hypergraph analogue of the
    friendship-plus-decoy instance, and the family on which localization grows as
    m^{rho(K3) - 1} = m^{0.5}.
    """
    rng = np.random.default_rng(seed)
    H, nxt, cliques = [], 0, []
    for _ in range(n_cliques):
        c = list(range(nxt, nxt + clique_size))
        nxt += clique_size
        H.append(tuple(c))
        cliques.append(c)
    for _ in range(n_gadgets):
        c = cliques[rng.integers(len(cliques))]
        u, v = rng.choice(c, size=2, replace=False)
        w = nxt
        nxt += 1
        H.append((int(u), w))
        H.append((int(v), w))
    return H


def random_uniform_hypergraph(n, m_h, r, seed=0):
    rng = np.random.default_rng(seed)
    return [tuple(rng.choice(n, size=r, replace=False)) for _ in range(m_h)]


def incidence_edges(hyperedges):
    """Bipartite incidence graph of a hyperedge list."""
    E = []
    for i, h in enumerate(hyperedges):
        for v in set(h):
            E.append((('v', v), ('e', i)))
    return E

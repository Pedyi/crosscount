"""Core graph primitives.

Degeneracy by peeling, exact oracle-width (pseudoarboricity) by binary search over
max-flow feasibility, and the structural bracket

    ceil(kappa_H / 2)  <=  alpha_H  <=  kappa_H  <=  kappa(G)

of Lemma 1 in the paper.
"""
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

__all__ = ['bits_to_idx', 'popcount', 'relabel', 'canon', 'build_adj',
           'degeneracy', 'degeneracy_arrays', 'orientation_feasible', 'oracle_width',
           'bracket']


# ----------------------------------------------------------------- bit helpers

def bits_to_idx(x, nbytes):
    """Indices of the set bits of a Python int."""
    if x == 0:
        return np.empty(0, dtype=np.int32)
    b = np.frombuffer(x.to_bytes(nbytes, 'little'), dtype=np.uint8)
    return np.flatnonzero(np.unpackbits(b, bitorder='little')).astype(np.int32)


def popcount(x):
    try:
        return x.bit_count()
    except AttributeError:              # Python < 3.10
        return bin(x).count('1')


# ------------------------------------------------------------- edge-list helpers

def relabel(edges):
    """Map arbitrary hashable node labels to consecutive integers."""
    ids, out = {}, []
    for u, v in edges:
        for x in (u, v):
            if x not in ids:
                ids[x] = len(ids)
        out.append((ids[u], ids[v]))
    return out, ids


def canon(edges):
    """De-duplicate, drop self-loops, relabel to ints, return sorted (u, v) with u < v."""
    edges, _ = relabel(edges)
    s = set()
    for u, v in edges:
        if u == v:
            continue
        s.add((u, v) if u < v else (v, u))
    return sorted(s)


def build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        if u == v:
            continue
        adj[u].add(v)
        adj[v].add(u)
    return adj


# ------------------------------------------------------------------- degeneracy

def degeneracy(adj):
    """Exact degeneracy of an adjacency dict, by peeling. Returns (kappa, core numbers)."""
    if not adj:
        return 0, {}
    deg = {v: len(adj[v]) for v in adj}
    maxdeg = max(deg.values())
    buckets = [set() for _ in range(maxdeg + 1)]
    for v, d in deg.items():
        buckets[d].add(v)
    core, k, removed = {}, 0, set()
    i = 0
    for _ in range(len(deg)):
        while i <= maxdeg and not buckets[i]:
            i += 1
        if i > maxdeg:
            break
        v = buckets[i].pop()
        k = max(k, i)
        core[v] = k
        removed.add(v)
        for w in adj[v]:
            if w in removed:
                continue
            d = deg[w]
            buckets[d].discard(w)
            deg[w] = d - 1
            buckets[d - 1].add(w)
            if d - 1 < i:
                i = d - 1
    return k, core


def degeneracy_arrays(eu, ev):
    """Degeneracy of the graph given by parallel edge arrays."""
    if len(eu) == 0:
        return 0, 0
    nodes = np.unique(np.concatenate([eu, ev]))
    remap = {int(v): i for i, v in enumerate(nodes)}
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for a, b in zip(eu, ev):
        a, b = remap[int(a)], remap[int(b)]
        adj[a].append(b)
        adj[b].append(a)
    deg = np.array([len(a) for a in adj])
    maxd = int(deg.max())
    buckets = [set() for _ in range(maxd + 1)]
    for v in range(n):
        buckets[deg[v]].add(v)
    removed = np.zeros(n, dtype=bool)
    k, i = 0, 0
    for _ in range(n):
        while i <= maxd and not buckets[i]:
            i += 1
        if i > maxd:
            break
        v = buckets[i].pop()
        k = max(k, i)
        removed[v] = True
        for w in adj[v]:
            if removed[w]:
                continue
            d = int(deg[w])
            buckets[d].discard(w)
            deg[w] = d - 1
            buckets[d - 1].add(w)
            if d - 1 < i:
                i = d - 1
    return k, n


# ------------------------------------------------- oracle-width (pseudoarboricity)

def orientation_feasible(eu, ev, k):
    """Is there an orientation of these edges with maximum out-degree at most k?

    Max-flow feasibility (Hakimi; Frank and Gyarfas):
    s -> edge (cap 1); edge -> each endpoint (cap 1); vertex -> t (cap k).
    Feasible iff the flow saturates all m edges.
    """
    m = len(eu)
    if m == 0:
        return True
    nodes = np.unique(np.concatenate([eu, ev]))
    pos = {int(v): i for i, v in enumerate(nodes)}
    n = len(nodes)
    S, E0, V0 = 0, 1, 1 + m
    T = 1 + m + n
    rows = np.empty(3 * m + n, dtype=np.int64)
    cols = np.empty(3 * m + n, dtype=np.int64)
    data = np.empty(3 * m + n, dtype=np.int32)
    idx = np.arange(m)
    rows[:m] = S
    cols[:m] = E0 + idx
    data[:m] = 1
    rows[m:2 * m] = E0 + idx
    cols[m:2 * m] = V0 + np.array([pos[int(x)] for x in eu])
    data[m:2 * m] = 1
    rows[2 * m:3 * m] = E0 + idx
    cols[2 * m:3 * m] = V0 + np.array([pos[int(x)] for x in ev])
    data[2 * m:3 * m] = 1
    rows[3 * m:] = V0 + np.arange(n)
    cols[3 * m:] = T
    data[3 * m:] = int(k)
    g = csr_matrix((data, (rows, cols)), shape=(T + 1, T + 1))
    return int(maximum_flow(g, S, T).flow_value) == m


def oracle_width(eu, ev, time_budget=None):
    """Exact pseudoarboricity of the edge set; None if the time budget runs out."""
    if len(eu) == 0:
        return 0
    hi, _ = degeneracy_arrays(eu, ev)
    hi = max(hi, 1)
    lo = 1
    t0 = time.time()
    while lo < hi:
        if time_budget is not None and time.time() - t0 > time_budget:
            return None
        mid = (lo + hi) // 2
        if orientation_feasible(eu, ev, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def bracket(kappa_copy):
    """The bracket of Lemma 1: (lower bound, upper bound) on alpha_H."""
    return int(np.ceil(kappa_copy / 2)), int(kappa_copy)

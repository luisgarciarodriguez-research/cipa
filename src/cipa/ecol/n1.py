"""ECoL N1: Fraction of Borderline Points (MST-based). See §3.1 of García Rodríguez et al. (2026).

This module is part of the CIPA software package, companion implementation to:

    García Rodríguez, L., Neme Castillo, J. A., Gómez Adorno, H. M., & Fuentes Pineda, G. (2026).
    CIPA: A Multi-Domain Statistical Framework for Characterizing Imbalanced
    Datasets and Computing a Difficulty Score.
    COMIA 2026 — XVIII Congreso Mexicano de Inteligencia Artificial.
    DOI: TODO (pending publication)

Instituto de Investigaciones en Matemáticas Aplicadas y en Sistemas (IIMAS)
Universidad Nacional Autónoma de México (UNAM)

Development supported by SECIHTI (researcher ID (CVU) 905206, Luis García Rodríguez).

License: MIT — see LICENSE file for full terms.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import BallTree, KDTree, NearestNeighbors

from cipa._constants import DEFAULT_N1_NEIGHBORS, DEFAULT_QUERY_CHUNK_SIZE

logger = logging.getLogger(__name__)

# Components with at most this many unresolved points are searched by brute
# force; larger ones get a KD-tree over the points outside the component.
_BRUTE_FORCE_MAX_QUERIES = 64
# Upper bound on entries of each brute-force distance block.
_BRUTE_FORCE_BLOCK = 1 << 22


def compute_n1(
    X: np.ndarray,
    y: np.ndarray,
    n_neighbors: int = DEFAULT_N1_NEIGHBORS,
    n_jobs: int | None = None,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
) -> float:
    """Compute N1: Fraction of Borderline Points via the MST (ECoL measure).

    Builds the exact Euclidean minimum spanning tree of all instances and
    identifies the borderline set B: all instances incident to at least one
    MST edge that crosses the class boundary (connects instances of different
    classes). N1 = |B| / N.

    The tree is built without an N×N distance matrix (see
    ``euclidean_minimum_spanning_tree``). Edges of length 0 between exact
    duplicates are valid edges, so two identical rows with different labels
    are both borderline. No subsampling happens here: the pipeline decides
    which rows N1 sees (C6).

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Feature matrix.
    y : np.ndarray, shape (N,)
        Binary label vector.
    n_neighbors : int
        Neighbours precomputed per instance for the MST search.
    n_jobs : int or None
        Parallel jobs for the neighbour queries.
    chunk_size : int
        Maximum number of queries sent to the tree at once.

    Returns
    -------
    float
        N1 ∈ [0, 1]. Higher = more borderline instances = more overlap.
    """
    rows, cols, _ = euclidean_minimum_spanning_tree(
        X, n_neighbors=n_neighbors, n_jobs=n_jobs, chunk_size=chunk_size
    )
    y = np.asarray(y)
    cross = y[rows] != y[cols]
    borderline = np.zeros(len(X), dtype=bool)
    borderline[rows[cross]] = True
    borderline[cols[cross]] = True
    return float(borderline.sum() / len(X))


def euclidean_minimum_spanning_tree(
    X: np.ndarray,
    n_neighbors: int = DEFAULT_N1_NEIGHBORS,
    n_jobs: int | None = None,
    chunk_size: int = DEFAULT_QUERY_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Euclidean minimum spanning tree in O(N·n_neighbors) memory (C4).

    Borůvka's algorithm. Each round finds, for every component, its lightest
    edge to another component and merges along those edges, so the number of
    components at least halves per round.

    The lightest outgoing edge of each point is searched first among its
    ``n_neighbors`` precomputed nearest neighbours (self excluded by index).
    A point whose list holds no neighbour from another component, or whose
    best such neighbour is as far as the end of its list, is unresolved: its
    true lightest edge is at least as long as its last listed neighbour. An
    unresolved point is searched exactly only if that lower bound does not
    rule it out against its component's best resolved edge. The exact search
    uses brute force for components with few such points and a KD-tree (or
    ball tree above 15 features) over the points outside the component
    otherwise. Distances are computed with
    the same Euclidean kernel in every path, so equal edges compare equal.

    Tie-breaking rule
    -----------------
    Edges are totally ordered by the key (length, min(i, j), max(i, j)), with
    i and j row indices of X. Under this order the MST is unique and Borůvka
    never forms a cycle; merges still go through a union-find as a guard.
    Among spanning trees of equal total length, the one returned is the
    minimum under that key. Another implementation that breaks ties
    differently (e.g. scipy on a dense matrix) can return a different tree of
    the same total length, and therefore a different borderline set for N1.

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
        Feature matrix.
    n_neighbors : int
        Neighbours precomputed per point (clamped to N-1).
    n_jobs : int or None
        Parallel jobs for the neighbour queries.
    chunk_size : int
        Maximum number of queries sent to the tree at once.

    Returns
    -------
    rows, cols : np.ndarray of int, shape (N-1,)
        Endpoints of each MST edge, with rows < cols.
    weights : np.ndarray of float, shape (N-1,)
        Euclidean length of each edge.
    """
    from cipa._knn import kneighbors_excluding_self

    X = np.ascontiguousarray(X, dtype=np.float64)
    n = len(X)
    if n < 2:
        return np.empty(0, np.intp), np.empty(0, np.intp), np.empty(0)

    k = min(n_neighbors, n - 1)
    algorithm = "kd_tree" if X.shape[1] <= 15 else "ball_tree"
    nn = NearestNeighbors(algorithm=algorithm, n_jobs=n_jobs).fit(X)
    knn_dist, knn_idx = kneighbors_excluding_self(nn, X, np.arange(n), k, chunk_size)
    # Every point farther than the last listed neighbour is at least this far
    radius = np.full(n, np.inf) if k == n - 1 else knn_dist[:, -1]

    comp = np.arange(n)
    parent = np.arange(n)
    point_rows = np.arange(n)
    edges_a: list[np.ndarray] = []
    edges_b: list[np.ndarray] = []
    edges_w: list[np.ndarray] = []
    n_components = n
    n_rounds = 0

    while n_components > 1:
        n_rounds += 1
        # 1. Lightest foreign edge of each point from its neighbour list
        foreign = comp[knn_idx] != comp[:, None]
        has_foreign = foreign.any(axis=1)
        cand_w = np.where(has_foreign, knn_dist[point_rows, foreign.argmax(axis=1)], np.inf)
        tied = foreign & (knn_dist == cand_w[:, None])
        cand_j = np.where(tied, knn_idx, n).min(axis=1)
        resolved = has_foreign & (cand_w < radius)

        # 2. Best resolved edge per component, then exact search where needed
        res_pts = np.flatnonzero(resolved)
        best_w, _, _ = _best_per_component(
            comp, res_pts, cand_w[res_pts], cand_j[res_pts], n
        )
        unresolved = np.flatnonzero(~resolved)
        need = unresolved[radius[unresolved] <= best_w[comp[unresolved]]]
        exact_w, exact_j = _exact_foreign_neighbours(X, comp, need, chunk_size)

        pts = np.concatenate([res_pts, need])
        best_w, best_a, best_b = _best_per_component(
            comp, pts,
            np.concatenate([cand_w[res_pts], exact_w]),
            np.concatenate([cand_j[res_pts], exact_j]),
            n,
        )

        # 3. Merge along the selected edges in key order
        chosen = np.flatnonzero(np.isfinite(best_w))
        a, b, w = best_a[chosen], best_b[chosen], best_w[chosen]
        # Two components may select the same edge
        _, unique = np.unique(a * n + b, return_index=True)
        a, b, w = a[unique], b[unique], w[unique]
        order = np.lexsort((b, a, w))
        for e in order:
            ra = _find(parent, comp[a[e]])
            rb = _find(parent, comp[b[e]])
            if ra == rb:
                continue
            parent[max(ra, rb)] = min(ra, rb)
            edges_a.append(a[e])
            edges_b.append(b[e])
            edges_w.append(w[e])
            n_components -= 1

        roots = parent.copy()
        while True:
            nxt = roots[roots]
            if np.array_equal(nxt, roots):
                break
            roots = nxt
        parent = roots
        comp = roots[comp]

    logger.debug("EMST: N=%d, k=%d, %d Borůvka rounds", n, k, n_rounds)
    return (
        np.asarray(edges_a, dtype=np.intp),
        np.asarray(edges_b, dtype=np.intp),
        np.asarray(edges_w, dtype=np.float64),
    )


def _find(parent: np.ndarray, i: int) -> int:
    """Union-find root of i with path halving."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return int(i)


def _best_per_component(
    comp: np.ndarray,
    points: np.ndarray,
    weights: np.ndarray,
    partners: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lightest edge per component under the key (w, min(i, j), max(i, j)).

    Returns arrays indexed by component label; components without a
    candidate have weight inf and endpoints -1.
    """
    best_w = np.full(n, np.inf)
    best_a = np.full(n, -1, dtype=np.intp)
    best_b = np.full(n, -1, dtype=np.intp)
    finite = np.isfinite(weights)
    points, weights, partners = points[finite], weights[finite], partners[finite]
    if len(points) == 0:
        return best_w, best_a, best_b
    lo = np.minimum(points, partners)
    hi = np.maximum(points, partners)
    labels = comp[points]
    order = np.lexsort((hi, lo, weights, labels))
    first = np.ones(len(order), dtype=bool)
    first[1:] = labels[order][1:] != labels[order][:-1]
    sel = order[first]
    best_w[labels[sel]] = weights[sel]
    best_a[labels[sel]] = lo[sel]
    best_b[labels[sel]] = hi[sel]
    return best_w, best_a, best_b


def _exact_foreign_neighbours(
    X: np.ndarray,
    comp: np.ndarray,
    points: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest point outside its own component for each of ``points``.

    Among equidistant candidates the smallest row index is returned, which
    realises the (w, min, max) key for a fixed endpoint.

    Returns
    -------
    weights : np.ndarray, shape (len(points),)
    partners : np.ndarray, shape (len(points),)
    """
    weights = np.empty(len(points))
    partners = np.empty(len(points), dtype=np.intp)
    if len(points) == 0:
        return weights, partners

    n = len(X)
    metric = DistanceMetric.get_metric("euclidean")
    labels = comp[points]
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    for group in np.split(order, bounds):
        label = labels[group[0]]
        queries = points[group]
        if len(queries) <= _BRUTE_FORCE_MAX_QUERIES:
            block = max(1, _BRUTE_FORCE_BLOCK // n)
            outside = comp != label
            for start in range(0, len(queries), block):
                sub = group[start:start + block]
                dist = metric.pairwise(X[points[sub]], X)
                dist[:, ~outside] = np.inf
                w = dist.min(axis=1)
                weights[sub] = w
                partners[sub] = (dist == w[:, None]).argmax(axis=1)
        else:
            others = np.flatnonzero(comp != label)
            tree = (KDTree if X.shape[1] <= 15 else BallTree)(X[others])
            k = min(2, len(others))
            for start in range(0, len(queries), chunk_size):
                sub = group[start:start + chunk_size]
                q = X[points[sub]]
                dist, idx = tree.query(q, k=k)
                weights[sub] = dist[:, 0]
                partners[sub] = others[idx[:, 0]]
                if k == 2:
                    ties = np.flatnonzero(dist[:, 0] == dist[:, 1])
                    for t in ties:
                        # The tree compares squared distances, so widen the
                        # radius and keep the exact ties with the shared kernel
                        r = dist[t, 0] * (1 + 1e-9) + 1e-300
                        hits = tree.query_radius(q[t:t + 1], r=r)[0]
                        exact = metric.pairwise(q[t:t + 1], X[others[hits]])[0]
                        partners[sub[t]] = others[hits[exact == dist[t, 0]]].min()
    return weights, partners

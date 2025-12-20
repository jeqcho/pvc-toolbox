"""Internal max-flow implementation with scipy optimization.

This module provides max-flow computation using scipy when available,
with a pure Python Dinic fallback for environments without scipy.
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple

# Try to import scipy for optimized max-flow
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    import warnings

    warnings.warn(
        "scipy not found. Install scipy for faster max-flow computation: "
        "pip install pvc-toolbox[fast]",
        ImportWarning,
        stacklevel=2,
    )


class Dinic:
    """
    Dinic's algorithm for maximum flow with integer capacities.

    This is a pure Python implementation used as a fallback when scipy
    is not available. When scipy is installed, the scipy-based implementation
    is used instead for better performance.
    """

    def __init__(self, n_vertices: int) -> None:
        if n_vertices <= 1:
            raise ValueError("Dinic graph must have at least 2 vertices")
        self.n = n_vertices
        self.graph: List[List[Tuple[int, int, int]]] = [[] for _ in range(n_vertices)]

    def add_edge(self, u: int, v: int, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        # forward edge
        self.graph[u].append((v, capacity, len(self.graph[v])))
        # reverse edge (initial capacity 0)
        self.graph[v].append((u, 0, len(self.graph[u]) - 1))

    def _bfs_levels(self, s: int, t: int) -> List[int]:
        level = [-1] * self.n
        q = deque([s])
        level[s] = 0
        while q:
            u = q.popleft()
            for idx, (v, cap, rev) in enumerate(self.graph[u]):
                if cap > 0 and level[v] == -1:
                    level[v] = level[u] + 1
                    q.append(v)
        return level

    def _dfs_block(
        self, u: int, t: int, f: int, level: List[int], it: List[int]
    ) -> int:
        if u == t:
            return f
        adj = self.graph[u]
        i = it[u]
        while i < len(adj):
            v, cap, rev = adj[i]
            if cap > 0 and level[u] + 1 == level[v]:
                pushed = self._dfs_block(v, t, min(f, cap), level, it)
                if pushed > 0:
                    # update forward (u->v)
                    adj[i] = (v, cap - pushed, rev)
                    # update reverse (v->u)
                    vr, vcap, vrev = self.graph[v][rev]
                    self.graph[v][rev] = (vr, vcap + pushed, vrev)
                    return pushed
            i += 1
        it[u] = i
        return 0

    def max_flow(self, s: int, t: int) -> int:
        if not (0 <= s < self.n) or not (0 <= t < self.n):
            raise ValueError("s and t must be valid vertex indices")
        flow = 0
        INF = 10**18  # large sentinel
        while True:
            level = self._bfs_levels(s, t)
            if level[t] == -1:
                break
            it = [0] * self.n
            while True:
                pushed = self._dfs_block(s, t, INF, level, it)
                if pushed == 0:
                    break
                flow += pushed
        return flow


class FlowNetwork:
    """
    Flow network abstraction that uses scipy when available, otherwise Dinic.

    This provides a unified interface for max-flow computation regardless
    of whether scipy is installed.
    """

    def __init__(self, n_vertices: int) -> None:
        self.n = n_vertices
        self._edges: List[Tuple[int, int, int]] = []

    def add_edge(self, u: int, v: int, capacity: int) -> None:
        """Add a directed edge from u to v with given capacity."""
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        self._edges.append((u, v, capacity))

    def max_flow(self, source: int, sink: int) -> int:
        """Compute maximum flow from source to sink."""
        if _HAS_SCIPY:
            return self._max_flow_scipy(source, sink)
        else:
            return self._max_flow_dinic(source, sink)

    def _max_flow_scipy(self, source: int, sink: int) -> int:
        """Compute max flow using scipy's C-optimized implementation."""
        # Build sparse matrix representation
        row = []
        col = []
        data = []

        for u, v, cap in self._edges:
            row.append(u)
            col.append(v)
            data.append(cap)

        # Create sparse matrix
        graph = csr_matrix((data, (row, col)), shape=(self.n, self.n), dtype=int)

        # Compute max flow
        result = maximum_flow(graph, source, sink)
        return result.flow_value

    def _max_flow_dinic(self, source: int, sink: int) -> int:
        """Compute max flow using pure Python Dinic implementation."""
        dinic = Dinic(self.n)
        for u, v, cap in self._edges:
            dinic.add_edge(u, v, cap)
        return dinic.max_flow(source, sink)


def has_scipy() -> bool:
    """Check if scipy is available for optimized max-flow computation."""
    return _HAS_SCIPY


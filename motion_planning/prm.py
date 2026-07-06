# prm.py
import heapq
import numpy as np
from typing import List, Tuple, Optional, Dict

from utils import reconstruct_path, euclid
from lines import line_is_free

def build_prm(mask: np.ndarray, N: int, k: int, radius: int, use8: bool,
              start_xy: Tuple[int,int], goal_xy: Tuple[int,int]):
    """Return (nodes_xy [M,2], graph {i:[(j,w)]}, edges_draw) or (None,None,None) if invalid."""
    H, W = mask.shape
    if not (0 <= start_xy[0] < W and 0 <= start_xy[1] < H):
        return (None, None, None)
    if not (0 <= goal_xy[0]  < W and 0 <= goal_xy[1]  < H):
        return (None, None, None)
    if mask[start_xy[1], start_xy[0]] or mask[goal_xy[1], goal_xy[0]]:
        return (None, None, None)

    rng = np.random.default_rng()
    samples, seen = [], set()
    attempts, target = 0, max(0, N)
    while len(samples) < target and attempts < 20 * target:
        x = int(rng.integers(0, W)); y = int(rng.integers(0, H)); attempts += 1
        if mask[y, x] or (x, y) in seen:
            continue
        seen.add((x, y))
        samples.append((x, y))

    nodes = np.array([start_xy] + samples + [goal_xy], dtype=np.int32)  # [M,2]
    M = nodes.shape[0]
    graph: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(M)}
    edges_draw = []

    for i in range(M):
        xi, yi = nodes[i]
        dx = nodes[:, 0] - xi
        dy = nodes[:, 1] - yi
        dists = np.sqrt(dx*dx + dy*dy)
        order = np.argsort(dists)
        added = 0
        for j in order[1:]:
            if added >= k:
                break
            d = float(dists[j])
            if d > radius:
                continue
            if line_is_free(mask, (xi, yi), (int(nodes[j,0]), int(nodes[j,1])), use8):
                graph[i].append((j, d))
                graph[j].append((i, d))
                edges_draw.append(((xi, yi), (int(nodes[j,0]), int(nodes[j,1]))))
                added += 1
    return nodes, graph, edges_draw

def graph_astar(nodes_xy: np.ndarray, graph: Dict[int, List[Tuple[int,float]]], start_idx: int, goal_idx: int):
    def h(i: int) -> float:
        return euclid(tuple(nodes_xy[i]), tuple(nodes_xy[goal_idx]))
    pq, g, came, visited = [], {start_idx: 0.0}, {}, []
    heapq.heappush(pq, (h(start_idx), start_idx))
    while pq:
        f, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.append(u)
        if u == goal_idx:
            return reconstruct_path(came, start_idx, goal_idx), visited
        for v, w in graph.get(u, []):
            ng = g[u] + w
            if v not in g or ng < g[v]:
                g[v] = ng
                came[v] = u
                heapq.heappush(pq, (ng + h(v), v))
    return [], visited

def graph_dijkstra(nodes_xy: np.ndarray, graph: Dict[int, List[Tuple[int,float]]], start_idx: int, goal_idx: int):
    pq, dist, came, visited = [], {start_idx: 0.0}, {}, []
    heapq.heappush(pq, (0.0, start_idx))
    while pq:
        g, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.append(u)
        if u == goal_idx:
            return reconstruct_path(came, start_idx, goal_idx), visited
        for v, w in graph.get(u, []):
            ng = g + w
            if v not in dist or ng < dist[v]:
                dist[v] = ng
                came[v] = u
                heapq.heappush(pq, (ng, v))
    return [], visited

# --- PRM* (radius-based, asymptotically optimal) -----------------------------
import math

def _estimate_gamma_prm(mask: np.ndarray, d: int = 2, safety: float = 1.1) -> float:
    """
    Heuristic γ_PRM for PRM*:
      γ > 2 * ( (1 + 1/d) ** (1/d) ) * ( (μ(X_free) / ζ_d) ** (1/d) )
    Here μ(X_free) is free-space area (in px^2), ζ_d is volume of the unit ball in d-D.
    We estimate μ(X_free) from the mask and multiply by a small safety factor.
    """
    H, W = mask.shape
    mu_free = float((~mask).sum())            # number of free pixels
    if mu_free <= 0:
        return 0.0
    # ζ_d for d=2 is π (area of unit disk)
    zeta_d = math.pi if d == 2 else (math.pi ** (d/2)) / math.gamma(d/2 + 1.0)
    base = (1.0 + 1.0/d) ** (1.0/d)
    gamma = 2.0 * base * (mu_free / zeta_d) ** (1.0/d)
    return safety * gamma

def build_prm_star(mask: np.ndarray, N: int, radius_cap: float, use8: bool,
                   start_xy: Tuple[int,int], goal_xy: Tuple[int,int],
                   gamma: Optional[float] = None, min_radius: float = 5.0):
    """
    PRM* (radius version). Connect all pairs within r_n where:
        r_n = min( radius_cap, max(min_radius, gamma * sqrt(log n / n)) ),  d=2
    Returns (nodes_xy [M,2], graph {i:[(j,w)]}, edges_draw) or (None,None,None) if invalid.
    """
    H, W = mask.shape
    if not (0 <= start_xy[0] < W and 0 <= start_xy[1] < H): return (None, None, None)
    if not (0 <= goal_xy[0]  < W and 0 <= goal_xy[1]  < H): return (None, None, None)
    if mask[start_xy[1], start_xy[0]] or mask[goal_xy[1], goal_xy[0]]: return (None, None, None)

    rng = np.random.default_rng()
    samples, seen = [], set()
    target = max(0, int(N))
    attempts = 0
    while len(samples) < target and attempts < 20 * target:
        x = int(rng.integers(0, W)); y = int(rng.integers(0, H)); attempts += 1
        if mask[y, x] or (x, y) in seen: continue
        seen.add((x, y)); samples.append((x, y))

    nodes = np.array([start_xy] + samples + [goal_xy], dtype=np.int32)  # [M,2]
    M = nodes.shape[0]
    if M < 2: return (None, None, None)

    # Compute PRM* radius
    if gamma is None:
        gamma = _estimate_gamma_prm(mask, d=2, safety=1.15)  # small safety bump
    rn = gamma * math.sqrt(max(1.0, math.log(M)) / float(M))
    rn = max(min_radius, min(float(radius_cap), float(rn)))

    graph: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(M)}
    edges_draw: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []

    # Connect within rn (symmetric). We add each undirected edge once (j>i).
    for i in range(M):
        xi, yi = nodes[i]
        dx = nodes[:, 0] - xi
        dy = nodes[:, 1] - yi
        dists = np.hypot(dx, dy)
        nbrs = np.where((dists > 0.0) & (dists <= rn))[0]
        for j in nbrs:
            if j <= i:  # avoid duplicates; we'll add both directions here
                continue
            xj, yj = int(nodes[j, 0]), int(nodes[j, 1])
            if line_is_free(mask, (xi, yi), (xj, yj), use8):
                w = float(dists[j])
                graph[i].append((j, w))
                graph[j].append((i, w))
                edges_draw.append(((xi, yi), (xj, yj)))

    return nodes, graph, edges_draw



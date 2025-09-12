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

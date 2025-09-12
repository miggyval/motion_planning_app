# grid_search.py
import heapq
import math
from typing import List, Tuple, Optional, Dict

from utils import reconstruct_path

def heuristic_grid(a: Tuple[int, int], b: Tuple[int, int], diag: bool) -> float:
    (r1, c1), (r2, c2) = a, b
    dr, dc = abs(r1 - r2), abs(c1 - c2)
    if diag:
        # Octile distance (Chebyshev + diagonal weight sqrt(2))
        return (max(dr, dc) - min(dr, dc)) + (math.sqrt(2.0) * min(dr, dc))
    else:
        return float(dr + dc)

def neighbors_grid(r: int, c: int, grid, allow_diag: bool, costmap: Optional):
    R, C = grid.shape
    out = []
    # 4-neigh
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < R and 0 <= cc < C and grid[rr, cc] == 0:
            base = 1.0
            extra = 0.0 if costmap is None else float(costmap[rr, cc])
            out.append(((rr, cc), base * (1.0 + extra)))
    # diagonals with corner-cut check
    if allow_diag:
        for dr, dc in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C and grid[rr, cc] == 0:
                if grid[r, c+dc] == 0 and grid[r+dr, c] == 0:
                    base = math.sqrt(2.0)
                    extra = 0.0 if costmap is None else float(costmap[rr, cc])
                    out.append(((rr, cc), base * (1.0 + extra)))
    return out

def dijkstra_grid(grid, start: Tuple[int,int], goal: Tuple[int,int],
                  allow_diag: bool, costmap: Optional):
    pq = []
    heapq.heappush(pq, (0.0, start))
    dist, came, visited = {start: 0.0}, {}, []
    while pq:
        g, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.append(u)
        if u == goal:
            return reconstruct_path(came, start, goal), visited
        ur, uc = u
        for v, cost in neighbors_grid(ur, uc, grid, allow_diag, costmap):
            ng = g + cost
            if v not in dist or ng < dist[v]:
                dist[v] = ng
                came[v] = u
                heapq.heappush(pq, (ng, v))
    return [], visited

def astar_grid(grid, start: Tuple[int,int], goal: Tuple[int,int],
               allow_diag: bool, costmap: Optional):
    pq = []
    h0 = heuristic_grid(start, goal, allow_diag)
    heapq.heappush(pq, (h0, 0.0, start))
    g, came, visited = {start: 0.0}, {}, []
    while pq:
        f, cg, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.append(u)
        if u == goal:
            return reconstruct_path(came, start, goal), visited
        ur, uc = u
        for v, cost in neighbors_grid(ur, uc, grid, allow_diag, costmap):
            ng = g[u] + cost
            if v not in g or ng < g[v]:
                g[v] = ng
                came[v] = u
                fs = ng + heuristic_grid(v, goal, allow_diag)
                heapq.heappush(pq, (fs, ng, v))
    return [], visited

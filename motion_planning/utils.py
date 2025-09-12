# utils.py
from typing import Dict, Tuple, List

def reconstruct_path(came_from: Dict, start, goal):
    """Reconstruct path using predecessor map."""
    if goal not in came_from:
        return []
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    return list(reversed(path))

def euclid(a_xy: Tuple[int, int], b_xy: Tuple[int, int]) -> float:
    ax, ay = a_xy
    bx, by = b_xy
    # Using hypot would require math import here; keep it simple
    dx = float(ax - bx)
    dy = float(ay - by)
    return (dx*dx + dy*dy) ** 0.5

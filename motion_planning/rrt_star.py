from typing import List, Tuple, Optional, Dict, Callable, Iterable
import numpy as np

from utils import euclid
from lines import line_is_free

# ------------------------------
# Helpers for kinodynamic (diff)
# ------------------------------

def _in_bounds(mask: np.ndarray, x: int, y: int) -> bool:
    H, W = mask.shape
    return 0 <= x < W and 0 <= y < H

def _trajectory_is_free(mask: np.ndarray,
                        pts: Iterable[Tuple[float, float]],
                        allow_diag: bool) -> bool:
    """Check collision along a polyline by testing segment-by-segment."""
    it = iter(pts)
    try:
        x0f, y0f = next(it)
    except StopIteration:
        return True
    x0, y0 = int(round(x0f)), int(round(y0f))
    if not _in_bounds(mask, x0, y0) or mask[y0, x0]:
        return False
    for x1f, y1f in it:
        x1, y1 = int(round(x1f)), int(round(y1f))
        if not _in_bounds(mask, x1, y1):
            return False
        if not line_is_free(mask, (x0, y0), (x1, y1), allow_diag):
            return False
        x0, y0 = x1, y1
    return True

def _path_length(pts: Iterable[Tuple[float, float]]) -> float:
    total = 0.0
    it = iter(pts)
    try:
        x0, y0 = next(it)
    except StopIteration:
        return 0.0
    for x1, y1 in it:
        dx, dy = (x1 - x0), (y1 - y0)
        total += (dx*dx + dy*dy) ** 0.5
        x0, y0 = x1, y1
    return total

def _simulate_diff(from_state: Tuple[float, float, float],
                   v: float,
                   omega: float,
                   T: float,
                   dt: float) -> Tuple[List[Tuple[float,float]], Tuple[float,float,float]]:
    x, y, th = from_state
    pts = [(x, y)]
    N = max(1, int(np.ceil(T / max(1e-6, dt))))
    dt_ = T / N
    for _ in range(N):
        if abs(omega) < 1e-9:
            x += v * dt_ * np.cos(th)
            y += v * dt_ * np.sin(th)
        else:
            th_new = th + omega * dt_
            R = v / omega
            x += R * (np.sin(th_new) - np.sin(th))
            y -= R * (np.cos(th_new) - np.cos(th))
            th = th_new
        pts.append((x, y))
    th = (th + np.pi) % (2*np.pi) - np.pi
    return pts, (x, y, th)

def _best_steer_diff(from_state, sample_xy, omegas, v, T, dt, mask, allow_diag):
    sx, sy = sample_xy
    best = None
    best_d2 = float("inf")
    for w in omegas:
        pts, end_state = _simulate_diff(from_state, v, w, T, dt)
        if not _trajectory_is_free(mask, pts, allow_diag):
            continue
        ex, ey, eth = end_state
        d2 = (ex - sx)*(ex - sx) + (ey - sy)*(ey - sy)
        if d2 < best_d2:
            best_d2 = d2
            best = (pts, (int(round(ex)), int(round(ey)), eth), _path_length(pts))
    return best

def _can_reach_diff(from_state, target_xy, omegas, v, T, dt, mask, allow_diag, reach_eps=2.0):
    cand = _best_steer_diff(from_state, target_xy, omegas, v, T, dt, mask, allow_diag)
    if cand is None:
        return None
    pts, end_state, length = cand
    ex, ey, _ = end_state
    if euclid((ex, ey), target_xy) <= reach_eps:
        return pts, end_state, length
    return None

# ------------------------------
# Holonomic (original) steering
# ------------------------------

def _steer_euclid(p_from: Tuple[int,int], p_to: Tuple[int,int], step: float) -> Tuple[int,int]:
    x0, y0 = p_from
    x1, y1 = p_to
    vx, vy = (x1 - x0), (y1 - y0)
    d2 = float(vx*vx + vy*vy)
    if d2 <= 1e-18:
        return (x0, y0)
    d = d2 ** 0.5
    if d <= step:
        return (int(round(x1)), int(round(y1)))
    s = step / d
    return (int(round(x0 + s * vx)), int(round(y0 + s * vy)))

# --------------------------------
# Unified RRT / RRT* with dynamics
# --------------------------------

def rrt_plan(mask, start_xy, goal_xy, allow_diag,
             iters=4000, step=20, radius=35, goal_bias=0.05,
             animate_cb=None, should_cancel=None,
             *, variant="rrt*", dynamics="diff", start_theta=0.0,
             goal_radius=15.0, omegas=None, v=None, T=0.5, dt=1/30.0,
             reach_eps=2.0):
    H, W = mask.shape
    rng = np.random.default_rng()
    if dynamics not in ("diff", "euclid"):
        raise ValueError("dynamics must be 'diff' or 'euclid'")
    if variant not in ("rrt", "rrt*"):
        raise ValueError("variant must be 'rrt' or 'rrt*'")

    if dynamics == "diff":
        if omegas is None:
            omegas = [-1.2, -0.6, 0.0, 0.6, 1.2]
        if v is None:
            v = float(step) / max(1e-6, T)

    # If no heading given, face roughly toward the goal
    if start_theta is None:
        start_theta = 0.0
    if dynamics == "diff":
        start_theta = float(np.arctan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0]))

    nodes_state: List[Tuple[float,float,float]] = [(float(start_xy[0]), float(start_xy[1]), float(start_theta))]
    parent: List[int] = [-1]
    cost: List[float] = [0.0]
    children: List[set] = [set()]
    edge_traj: List[Optional[List[Tuple[float,float]]]] = [None]  # polyline from parent->node

    goal_idx: Optional[int] = None
    goal_best_cost = float("inf")
    edges_draw: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []

    def nodes_xy_list():
        return [(int(round(x)), int(round(y))) for (x, y, _) in nodes_state]

    def add_child(pidx, cidx):
        while len(children) <= max(pidx, cidx):
            children.append(set())
        children[pidx].add(cidx)

    def set_parent(child_idx: int, new_parent_idx: int, traj_pts: List[Tuple[float,float]], edge_cost: float):
        # update parent
        old_parent = parent[child_idx]
        if old_parent != -1 and child_idx in children[old_parent]:
            children[old_parent].remove(child_idx)
        parent[child_idx] = new_parent_idx
        add_child(new_parent_idx, child_idx)
        # set trajectory and cost
        edge_traj[child_idx] = list(traj_pts)
        cost[child_idx] = cost[new_parent_idx] + float(edge_cost)
        # paint edge
        px = [(int(round(ax)), int(round(ay))) for (ax, ay) in traj_pts]
        for a, b in zip(px[:-1], px[1:]):
            edges_draw.append((a, b))

    def nearest_idx_xy(p: Tuple[int,int]) -> int:
        px, py = p
        best_i, best_d2 = 0, float("inf")
        for i, (x, y, _) in enumerate(nodes_state):
            d2 = (x - px)*(x - px) + (y - py)*(y - py)
            if d2 < best_d2:
                best_d2 = d2; best_i = i
        return best_i

    def neighbors(i_new: int) -> List[int]:
        x0, y0, _ = nodes_state[i_new]
        out = []
        r2 = radius * radius
        for i, (x, y, _) in enumerate(nodes_state[:-1]):
            if (x - x0)*(x - x0) + (y - y0)*(y - y0) <= r2:
                out.append(i)
        return out

    def sample_free() -> Tuple[int,int]:
        for _ in range(2000):
            if rng.random() < goal_bias:
                x, y = goal_xy
            else:
                x = int(rng.integers(0, W))
                y = int(rng.integers(0, H))
            if _in_bounds(mask, x, y) and (not mask[y, x]):
                return (int(x), int(y))
        return (int(W//2), int(H//2))

    def steer_from(i_from: int, target_xy: Tuple[int,int]):
        """Return (traj_pts, end_state, edge_cost) from node i_from toward target."""
        if dynamics == "euclid":
            from_xy = (int(round(nodes_state[i_from][0])), int(round(nodes_state[i_from][1])))
            p_new = _steer_euclid(from_xy, target_xy, float(step))
            if p_new == from_xy:
                return None
            if not _in_bounds(mask, p_new[0], p_new[1]) or mask[p_new[1], p_new[0]]:
                return None
            if not line_is_free(mask, from_xy, p_new, allow_diag):
                return None
            traj = [from_xy, p_new]
            end_state = (float(p_new[0]), float(p_new[1]), 0.0)
            return traj, end_state, euclid(from_xy, p_new)
        else:
            from_state = nodes_state[i_from]
            cand = _best_steer_diff(from_state, target_xy, omegas, v, T, dt, mask, allow_diag)
            if cand is None:
                return None
            traj_pts, end_state, traj_len = cand
            p_new_xy = (int(round(end_state[0])), int(round(end_state[1])))
            if not _in_bounds(mask, p_new_xy[0], p_new_xy[1]) or mask[p_new_xy[1], p_new_xy[0]]:
                return None
            return traj_pts, (float(end_state[0]), float(end_state[1]), float(end_state[2])), float(traj_len)

    # --- main loop ---
    for it in range(iters):
        if should_cancel and should_cancel():
            break
        rnd = sample_free()
        i_near = nearest_idx_xy(rnd)

        steer_res = steer_from(i_near, rnd)
        if steer_res is None:
            continue
        traj_pts, end_state, edge_cost = steer_res
        # add node
        nodes_state.append(end_state)
        parent.append(-1)
        cost.append(float("inf"))
        children.append(set())
        edge_traj.append(None)
        i_new = len(nodes_state) - 1

        # choose parent
        best_parent = i_near
        best_traj = traj_pts
        best_cost = cost[i_near] + edge_cost

        if variant == "rrt*":
            # RRT*: try better parent among neighbors (holonomic only for safety)
            if dynamics == "euclid":
                for i_nb in neighbors(i_new):
                    sr = steer_from(i_nb, (int(round(end_state[0])), int(round(end_state[1]))))
                    if sr is None:
                        continue
                    traj_nb, _, ec_nb = sr
                    cand_cost = cost[i_nb] + ec_nb
                    if cand_cost + 1e-9 < best_cost:
                        best_cost = cand_cost
                        best_parent = i_nb
                        best_traj = traj_nb

        set_parent(i_new, best_parent, best_traj, best_cost - cost[best_parent])

        # rewiring (holonomic only; diff rewiring is non-trivial)
        if variant == "rrt*" and dynamics == "euclid":
            for i_nb in neighbors(i_new):
                if i_nb == best_parent:
                    continue
                # cost through i_new to neighbor
                sr = steer_from(i_new, (int(round(nodes_state[i_nb][0])), int(round(nodes_state[i_nb][1]))))
                if sr is None:
                    continue
                traj_nb, _, ec_nb = sr
                cand_cost = cost[i_new] + ec_nb
                if cand_cost + 1e-9 < cost[i_nb]:
                    # rewire
                    set_parent(i_nb, i_new, traj_nb, ec_nb)

        # goal check
        gx, gy = goal_xy
        ex, ey = int(round(end_state[0])), int(round(end_state[1]))
        if (ex - gx)*(ex - gx) + (ey - gy)*(ey - gy) <= goal_radius*goal_radius:
            if cost[i_new] < goal_best_cost:
                goal_best_cost = cost[i_new]
                goal_idx = i_new

        # animate
        if animate_cb and (it % 10 == 0 or it < 50):
            if not animate_cb(nodes_xy_list(), edges_draw, goal_idx):
                break

    # If we never hit the goal region, try nearest to goal as fallback
    if goal_idx is None and len(nodes_state) > 0:
        # pick nearest to goal
        best_i, best_d2 = 0, float("inf")
        gx, gy = goal_xy
        for i, (x, y, _) in enumerate(nodes_state):
            d2 = (x - gx)*(x - gx) + (y - gy)*(y - gy)
            if d2 < best_d2:
                best_d2 = d2; best_i = i
        if best_d2 <= (goal_radius*goal_radius):
            goal_idx = best_i

    # Recover path (x,y ints) and per-edge polylines
    path_px: List[Tuple[int,int]] = []
    path_edges: List[List[Tuple[int,int]]] = []
    if goal_idx is not None:
        cur = goal_idx
        while parent[cur] != -1:
            pidx = parent[cur]
            x, y, _ = nodes_state[cur]
            path_px.append((int(round(x)), int(round(y))))
            traj = edge_traj[cur] if edge_traj[cur] is not None else [
                (nodes_state[pidx][0], nodes_state[pidx][1]), (nodes_state[cur][0], nodes_state[cur][1])
            ]
            traj_i = [(int(round(ax)), int(round(ay))) for (ax, ay) in traj]
            path_edges.append(traj_i)
            cur = pidx
        x0, y0, _ = nodes_state[cur]
        path_px.append((int(round(x0)), int(round(y0))))
        path_px = list(reversed(path_px))
        path_edges = list(reversed(path_edges))

    nodes_xy = np.array(nodes_xy_list(), dtype=np.int32)
    return path_px, edges_draw, nodes_xy, path_edges

# ------------------------------
# Wrappers
# ------------------------------

def rrt_star(mask, start_xy, goal_xy, allow_diag,
             iters=4000, step=20, radius=35, goal_bias=0.05,
             animate_cb=None, should_cancel=None,
             *, dynamics="diff", start_theta=0.0, goal_radius=15.0,
             omegas=None, v=None, T=0.5, dt=1/30.0, reach_eps=2.0):
    return rrt_plan(mask, start_xy, goal_xy, allow_diag,
                    iters, step, radius, goal_bias,
                    animate_cb, should_cancel,
                    variant="rrt*", dynamics=dynamics,
                    start_theta=start_theta,
                    goal_radius=goal_radius,
                    omegas=omegas, v=v, T=T, dt=dt, reach_eps=reach_eps)

def rrt(mask, start_xy, goal_xy, allow_diag,
        iters=4000, step=20, radius=35, goal_bias=0.05,
        animate_cb=None, should_cancel=None,
        *, dynamics="diff", start_theta=0.0, goal_radius=15.0,
        omegas=None, v=None, T=0.5, dt=1/30.0):
    return rrt_plan(mask, start_xy, goal_xy, allow_diag,
                    iters, step, radius, goal_bias,
                    animate_cb, should_cancel,
                    variant="rrt", dynamics=dynamics,
                    start_theta=start_theta,
                    goal_radius=goal_radius,
                    omegas=omegas, v=v, T=T, dt=dt)

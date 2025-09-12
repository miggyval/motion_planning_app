# rrt_star.py
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
        # Stepwise collision check using your existing segment tester
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
    """
    Forward-simulate a diff-drive (unicycle) for fixed control (v, omega) over time T.
    Returns (polyline points including the start, final_state).
    from_state = (x, y, theta) with pixels + radians.
    """
    x, y, th = from_state
    pts = [(x, y)]
    t = 0.0
    # Integrate with simple Euler (fine for viz/teaching)
    N = max(1, int(np.ceil(T / max(1e-6, dt))))
    dt_ = T / N
    for _ in range(N):
        if abs(omega) < 1e-9:
            # Straight
            x += v * dt_ * np.cos(th)
            y += v * dt_ * np.sin(th)
        else:
            # Constant curvature arc
            th_new = th + omega * dt_
            R = v / omega  # signed radius
            x += R * (np.sin(th_new) - np.sin(th))
            y -= R * (np.cos(th_new) - np.cos(th))
            th = th_new
        pts.append((x, y))
    # Normalize heading to [-pi, pi)
    th = (th + np.pi) % (2*np.pi) - np.pi
    return pts, (x, y, th)

def _best_steer_diff(from_state: Tuple[float,float,float],
                     sample_xy: Tuple[int,int],
                     omegas: List[float],
                     v: float,
                     T: float,
                     dt: float,
                     mask: np.ndarray,
                     allow_diag: bool) -> Optional[Tuple[List[Tuple[float,float]], Tuple[int,int,float], float]]:
    """
    Try all discrete omegas; pick the reachable endpoint closest to the random sample.
    Returns (traj_pts, end_state_int, traj_length) or None if all collide.
    """
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

def _can_reach_diff(from_state: Tuple[float,float,float],
                    target_xy: Tuple[int,int],
                    omegas: List[float],
                    v: float,
                    T: float,
                    dt: float,
                    mask: np.ndarray,
                    allow_diag: bool,
                    reach_eps: float = 2.0) -> Optional[Tuple[List[Tuple[float,float]], Tuple[int,int,float], float]]:
    """
    Try to connect to a *specific* (x,y) using diff-drive controls in one T step.
    Returns a traj if endpoint within reach_eps of target.
    """
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

def rrt_plan(mask: np.ndarray,
             start_xy: Tuple[int,int],
             goal_xy: Tuple[int,int],
             allow_diag: bool,
             iters: int = 4000,
             step: int = 20,
             radius: int = 35,
             goal_bias: float = 0.05,
             animate_cb: Optional[Callable] = None,
             should_cancel: Optional[Callable] = None,
             *,
             variant: str = "rrt*",            # "rrt" or "rrt*"
             dynamics: str = "diff",           # "diff" or "euclid"
             start_theta: float = 0.0,         # radians (used if dynamics="diff")
             goal_radius: float = 15.0,        # px; goal region radius (diff)
             omegas: Optional[List[float]] = None,  # list of angular velocities (rad/s)
             v: Optional[float] = None,        # px/s; if None, uses step/T
             T: float = 0.5,                   # s per control
             dt: float = 1/30.0,              # s integration step
             reach_eps: float = 2.0           # px, tolerance for rewiring reachability in diff
             ) -> Tuple[List[Tuple[int,int]], List[Tuple[Tuple[int,int], Tuple[int,int]]], np.ndarray]:
    """
    Build an RRT or RRT* tree with either:
      - holonomic steering ('euclid'), or
      - differential-drive/unicycle steering ('diff') with constant v and discrete omegas over fixed T.

    Returns (path_px, edges_draw, nodes_xy).
    - path_px: polyline from start to goal if connected (x,y int)
    - edges_draw: list of small line segments [(p0,p1), (p1,p2), ...] to draw tree edges (works for curves)
    - nodes_xy: array of (x,y) ints for all nodes (theta is internal if dynamics='diff')

    Notes:
    - For dynamics='diff', 'step' is only used to set default v if v is None (v ≈ step / T).
    - For variant='rrt', there is no rewiring.
    - For variant='rrt*' + dynamics='diff', rewiring is conservative: a neighbor can re-parent a node only
      if it can reach that node within 'reach_eps' using one T-step from its own state.
    """
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
            v = float(step) / max(1e-6, T)  # so average spatial reach ~ step per expansion

    # Internal node state: (x, y, theta) if diff; (x, y, 0) if euclid
    
    start_theta = np.arctan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
    
    nodes_state: List[Tuple[float,float,float]] = [(float(start_xy[0]), float(start_xy[1]), float(start_theta))]
    parent: List[int] = [-1]
    cost: List[float] = [0.0]
    children: List[set] = [set()]

    goal_idx: Optional[int] = None
    goal_best_cost = float("inf")
    edges_draw: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []

    def nodes_xy_list() -> List[Tuple[int,int]]:
        return [(int(round(x)), int(round(y))) for (x, y, _) in nodes_state]

    def add_child(pidx: int, cidx: int):
        while len(children) <= max(pidx, cidx):
            children.append(set())
        children[pidx].add(cidx)

    def change_parent(child_idx: int, new_parent_idx: int, edge_pts: List[Tuple[float,float]], edge_cost: float):
        """Re-parent and propagate cost deltas; draw the new edge curve segments."""
        old_parent = parent[child_idx]
        if old_parent != -1 and child_idx in children[old_parent]:
            children[old_parent].remove(child_idx)
        parent[child_idx] = new_parent_idx
        add_child(new_parent_idx, child_idx)
        delta = (cost[new_parent_idx] + edge_cost) - cost[child_idx]
        if abs(delta) > 1e-12:
            stack = [child_idx]
            while stack:
                n = stack.pop()
                cost[n] += delta
                for ch in children[n]:
                    stack.append(ch)
        # Add segments to drawing
        px = [(int(round(ax)), int(round(ay))) for (ax, ay) in edge_pts]
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
        for i, (x, y, _) in enumerate(nodes_state[:-1]):  # exclude just-added
            if (x - x0)*(x - x0) + (y - y0)*(y - y0) <= r2:
                out.append(i)
        return out

    def sample_free() -> Tuple[int,int]:
        for _ in range(1000):
            if rng.random() < goal_bias:
                x, y = goal_xy
            else:
                x = int(rng.integers(0, W))
                y = int(rng.integers(0, H))
            if _in_bounds(mask, x, y) and (not mask[y, x]):
                return (x, y)
        # Fallback (center)
        return (int(W//2), int(H//2))

    for it in range(iters):
        if should_cancel and should_cancel():
            break

        rnd = sample_free()
        i_near = nearest_idx_xy(rnd)

        # --- Extend ---
        if dynamics == "euclid":
            from_xy = (int(round(nodes_state[i_near][0])), int(round(nodes_state[i_near][1])))
            p_new_xy = _steer_euclid(from_xy, rnd, float(step))
            if p_new_xy == from_xy:
                continue
            if not _in_bounds(mask, p_new_xy[0], p_new_xy[1]) or mask[p_new_xy[1], p_new_xy[0]]:
                continue
            if not line_is_free(mask, from_xy, p_new_xy, allow_diag):
                continue
            traj_pts = [from_xy, p_new_xy]
            end_state = (float(p_new_xy[0]), float(p_new_xy[1]), 0.0)
            edge_cost = euclid(from_xy, p_new_xy)
        else:
            # diff-drive: choose omega that gets closest to rnd in one T-step
            from_state = nodes_state[i_near]
            cand = _best_steer_diff(from_state, rnd, omegas, v, T, dt, mask, allow_diag)
            if cand is None:
                continue
            traj_pts_f, end_state_f, traj_len = cand
            end_state = (float(end_state_f[0]), float(end_state_f[1]), float(end_state_f[2]))
            p_new_xy = (int(round(end_state[0])), int(round(end_state[1])))
            # Guard
            if not _in_bounds(mask, p_new_xy[0], p_new_xy[1]) or mask[p_new_xy[1], p_new_xy[0]]:
                continue
            traj_pts = traj_pts_f
            edge_cost = traj_len

        # Append new node tentatively
        nodes_state.append(end_state)
        parent.append(-1)
        cost.append(float("inf"))
        children.append(set())
        i_new = len(nodes_state) - 1

        # Draw edge (polyline segments)
        px = [(int(round(ax)), int(round(ay))) for (ax, ay) in traj_pts]
        for a, b in zip(px[:-1], px[1:]):
            edges_draw.append((a, b))

        # --- Choose parent ---
        best_parent = i_near
        best_cost = cost[i_near] + edge_cost
        if variant == "rrt*":
            neigh = neighbors(i_new)
            if dynamics == "euclid":
                for j in neigh:
                    d = euclid((int(round(nodes_state[j][0])), int(round(nodes_state[j][1]))), p_new_xy)
                    if d < 1e-9:
                        continue
                    if (cost[j] + d < best_cost and
                        line_is_free(mask,
                                     (int(round(nodes_state[j][0])), int(round(nodes_state[j][1]))),
                                     p_new_xy, allow_diag)):
                        best_parent = j
                        best_cost = cost[j] + d
                        # Re-draw edge for chosen parent
                        # (No need to erase; we just add segments for viz)
                        edges_draw.append(((int(round(nodes_state[j][0])), int(round(nodes_state[j][1]))), p_new_xy))
            else:
                # diff: a neighbor must be able to land sufficiently close to *this same node*
                for j in neigh:
                    if j == i_near:
                        continue
                    try_conn = _can_reach_diff(nodes_state[j], p_new_xy, omegas, v, T, dt, mask, allow_diag, reach_eps)
                    if try_conn is None:
                        continue
                    edge_pts_j, _, edge_cost_j = try_conn
                    if cost[j] + edge_cost_j < best_cost - 1e-9:
                        best_parent = j
                        best_cost = cost[j] + edge_cost_j
                        # draw those segments too (viz only)
                        pxj = [(int(round(ax)), int(round(ay))) for (ax, ay) in edge_pts_j]
                        for a, b in zip(pxj[:-1], pxj[1:]):
                            edges_draw.append((a, b))

        parent[i_new] = best_parent
        add_child(best_parent, i_new)
        cost[i_new] = best_cost

        # --- Rewire (RRT*) ---
        if variant == "rrt*":
            if dynamics == "euclid":
                for j in neigh:
                    if j == best_parent or j == i_new:
                        continue
                    pj = (int(round(nodes_state[j][0])), int(round(nodes_state[j][1])))
                    d = euclid(p_new_xy, pj)
                    if d < 1e-9:
                        continue
                    new_cost = cost[i_new] + d
                    if new_cost + 1e-9 < cost[j] and line_is_free(mask, p_new_xy, pj, allow_diag):
                        # Re-parent j under i_new
                        change_parent(j, i_new, [p_new_xy, pj], d)
            else:
                for j in neigh:
                    if j == best_parent or j == i_new:
                        continue
                    try_conn = _can_reach_diff(nodes_state[i_new],
                                               (int(round(nodes_state[j][0])), int(round(nodes_state[j][1]))),
                                               omegas, v, T, dt, mask, allow_diag, reach_eps)
                    if try_conn is None:
                        continue
                    edge_pts, _, edge_cost_j = try_conn
                    new_cost = cost[i_new] + edge_cost_j
                    if new_cost + 1e-9 < cost[j]:
                        change_parent(j, i_new, edge_pts, edge_cost_j)

        # --- Goal handling ---
        if goal_idx is None:
            if dynamics == "euclid":
                if (euclid(p_new_xy, goal_xy) <= step and
                        line_is_free(mask, p_new_xy, goal_xy, allow_diag)):
                    nodes_state.append((float(goal_xy[0]), float(goal_xy[1]), 0.0))
                    parent.append(i_new)
                    add_child(i_new, len(nodes_state)-1)
                    gcost = cost[i_new] + euclid(p_new_xy, goal_xy)
                    cost.append(gcost)
                    edges_draw.append((p_new_xy, goal_xy))
                    goal_idx = len(nodes_state) - 1
                    goal_best_cost = gcost
            else:
                # diff: attempt a single-T connection into the goal region
                if euclid(p_new_xy, goal_xy) <= max(goal_radius*2, step*1.5):
                    try_goal = _best_steer_diff(nodes_state[i_new], goal_xy, omegas, v, T, dt, mask, allow_diag)
                    if try_goal is not None:
                        pts_g, end_g, len_g = try_goal
                        if euclid((end_g[0], end_g[1]), goal_xy) <= goal_radius:
                            nodes_state.append((float(goal_xy[0]), float(goal_xy[1]), end_g[2]))
                            parent.append(i_new)
                            add_child(i_new, len(nodes_state)-1)
                            gcost = cost[i_new] + len_g
                            cost.append(gcost)
                            # draw arc segments
                            pg = [(int(round(ax)), int(round(ay))) for (ax, ay) in pts_g]
                            for a, b in zip(pg[:-1], pg[1:]):
                                edges_draw.append((a, b))
                            goal_idx = len(nodes_state) - 1
                            goal_best_cost = gcost
        else:
            if dynamics == "euclid":
                if (euclid(p_new_xy, goal_xy) <= step and
                        line_is_free(mask, p_new_xy, goal_xy, allow_diag)):
                    candidate = cost[i_new] + euclid(p_new_xy, goal_xy)
                    if candidate + 1e-9 < goal_best_cost:
                        # Reconnect goal under i_new
                        change_parent(goal_idx, i_new, [p_new_xy, goal_xy], euclid(p_new_xy, goal_xy))
                        goal_best_cost = cost[goal_idx]
            else:
                if euclid(p_new_xy, goal_xy) <= max(goal_radius*2, step*1.5):
                    try_goal = _best_steer_diff(nodes_state[i_new], goal_xy, omegas, v, T, dt, mask, allow_diag)
                    if try_goal is not None:
                        pts_g, end_g, len_g = try_goal
                        if euclid((end_g[0], end_g[1]), goal_xy) <= goal_radius:
                            candidate = cost[i_new] + len_g
                            if candidate + 1e-9 < goal_best_cost:
                                change_parent(goal_idx, i_new, pts_g, len_g)
                                goal_best_cost = cost[goal_idx]

        # Animate occasionally
        if animate_cb and (it % 10 == 0 or it < 50):
            if not animate_cb(nodes_xy_list(), edges_draw, goal_idx):
                break

    # Recover path (x,y ints) if goal connected
    path_px: List[Tuple[int,int]] = []
    if goal_idx is not None:
        cur = goal_idx
        while cur != -1:
            x, y, _ = nodes_state[cur]
            path_px.append((int(round(x)), int(round(y))))
            cur = parent[cur]
        path_px = list(reversed(path_px))

    nodes_xy = np.array(nodes_xy_list(), dtype=np.int32)
    return path_px, edges_draw, nodes_xy

# ------------------------------
# Backwards-compatible wrappers
# ------------------------------

def rrt_star(mask,
             start_xy: Tuple[int,int],
             goal_xy: Tuple[int,int],
             allow_diag: bool,
             iters: int = 4000,
             step: int = 20,
             radius: int = 35,
             goal_bias: float = 0.05,
             animate_cb: Optional[Callable] = None,
             should_cancel: Optional[Callable] = None,
             # New knobs (all optional / defaulted)
             *,
             dynamics: str = "diff",
             start_theta: float = 0.0,
             goal_radius: float = 15.0,
             omegas: Optional[List[float]] = None,
             v: Optional[float] = None,
             T: float = 0.5,
             dt: float = 1/30.0,
             reach_eps: float = 2.0):
    """RRT* wrapper. Defaults to diff-drive dynamics with constant v and discrete omegas."""
    return rrt_plan(mask, start_xy, goal_xy, allow_diag,
                    iters, step, radius, goal_bias,
                    animate_cb, should_cancel,
                    variant="rrt*",
                    dynamics=dynamics,
                    start_theta=start_theta,
                    goal_radius=goal_radius,
                    omegas=omegas, v=v, T=T, dt=dt, reach_eps=reach_eps)

def rrt(mask,
        start_xy: Tuple[int,int],
        goal_xy: Tuple[int,int],
        allow_diag: bool,
        iters: int = 4000,
        step: int = 20,
        radius: int = 35,
        goal_bias: float = 0.05,
        animate_cb: Optional[Callable] = None,
        should_cancel: Optional[Callable] = None,
        *,
        dynamics: str = "diff",
        start_theta: float = 0.0,
        goal_radius: float = 15.0,
        omegas: Optional[List[float]] = None,
        v: Optional[float] = None,
        T: float = 0.5,
        dt: float = 1/30.0):
    """Classic RRT wrapper (no rewiring)."""
    return rrt_plan(mask, start_xy, goal_xy, allow_diag,
                    iters, step, radius, goal_bias,
                    animate_cb, should_cancel,
                    variant="rrt",
                    dynamics=dynamics,
                    start_theta=start_theta,
                    goal_radius=goal_radius,
                    omegas=omegas, v=v, T=T, dt=dt)

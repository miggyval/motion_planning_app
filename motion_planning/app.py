# app.py
# Main application that wires together the modular components:
# - constants.py
# - lines.py
# - grid_search.py
# - prm.py
# - rrt_star.py
# - utils.py
#
# Requirements: pip install opencv-python numpy
#
# Notes:
# * This file intentionally keeps UI/state, drawing, and event handling here,
#   while search/geometry utilities live in the modules above.

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict

# --- Local modules ---
from constants import (
    INIT_ROWS, INIT_COLS, CANVAS_H, CANVAS_W, MARGIN, ANIM_DELAY, WINDOW,
    MIN_ROWS, MIN_COLS, MAX_ROWS, MAX_COLS,
    COL_BG, COL_GRID, COL_OBS, COL_START, COL_GOAL, COL_PATH, COL_EXPLORED,
    COL_TEXT, COL_HINT, COL_PREVIEW, COL_PRM_NODE, COL_PRM_EDGE, COL_PRM_VIS,
    COL_COST_CYAN, COL_COST_PINK, COL_COST_WHITE,
    PEN_START, PEN_GOAL, PEN_DRAW, PEN_ERASE, PEN_LINEDRAW, PEN_LINEERASE, PEN_NAMES,
)
from lines import line_points_8, line_points_4, line_is_free
from grid_search import astar_grid, dijkstra_grid
from prm import build_prm, graph_astar, graph_dijkstra
from rrt_star import rrt_star, rrt


# ------------------------
# App
# ------------------------

class App:
    def __init__(self):
        self.grid = np.zeros((INIT_ROWS, INIT_COLS), dtype=np.uint8)  # 0 free, 1 obstacle
        self.start: Optional[Tuple[int,int]] = None
        self.goal: Optional[Tuple[int,int]] = None

        # Drawing state
        self.pen = PEN_DRAW
        self.allow_diag = False
        self.show_grid = True
        self.show_hud = False
        self.dragging = False
        self.last_rc = None

        # Line tool state
        self.line_active = False
        self.line_anchor: Optional[Tuple[int,int]] = None
        self.line_current: Optional[Tuple[int,int]] = None
        self.preview_points: List[Tuple[int,int]] = []
        self.line_is_erase = False

        # Planner selection
        self.planner = "RRT"   # "GRID" or "PRM" or "RRT"
        self.algorithm = "A*"   # "A*" or "Dijkstra" (GRID/PRM)

        # Overlays
        self.path_cells: List[Tuple[int,int]] = []
        self.visited_cells: List[Tuple[int,int]] = []
        self.path_px: List[Tuple[int,int]] = []
        self.visited_idx: List[int] = []

        # PRM data
        self.prm_dirty = True
        self.prm_N = 600
        self.prm_k = 12
        self.prm_radius = 120
        self.mask: Optional[np.ndarray] = None
        self.nodes_xy: Optional[np.ndarray] = None
        self.graph: Optional[Dict[int, List[Tuple[int,float]]]] = None
        self.edges_draw: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []

        # RRT* data/params
        self.rrt_iters = 4000
        self.rrt_step = 20
        self.rrt_radius = 35
        self.rrt_goal_bias = 0.05
        self.rrt_edges: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []
        self.rrt_nodes_xy: Optional[np.ndarray] = None
        self.rrt_goal_connected = False

        # Run/build state
        self.is_running = False       # locks edits
        self.is_building = False      # distinguishes building vs search
        self.cancel_requested = False

        # Costmap state (GRID only)
        self.costmap_enabled = False
        self.costmap_dirty = True
        self.costmap: Optional[np.ndarray] = None
        self.cost_R = 2               # dilation radius in CELLS
        self.cost_high = 1e6          # "almost infinite" multiplier
        self.cost_smooth_max = 2.0    # peak shoulder multiplier at boundary
        
        # RRT mode + dynamics
        self.rrt_variant = "RRT"          # or "RRT*"
        self.rrt_dynamics = "euclid"      # "diff" (unicycle) or "euclid" (straight-line)
        self.rrt_start_theta = 0.0
        self.rrt_goal_radius = 15.0
        self.rrt_T = 0.5
        self.rrt_dt = 1/30.0
        self.rrt_omegas = [-1.2, -0.6, 0.0, 0.6, 1.2]
        self.rrt_v = None                # None → auto v ≈ step / T

        self.rrt_path_edges: List[List[Tuple[int,int]]] = []

        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

    # ---------- Geometry helpers (fixed canvas; variable grid) ----------

    def rc_to_rect(self, r: int, c: int) -> Tuple[int, int, int, int]:
        R, C = self.grid.shape
        x1 = (c    * CANVAS_W) // C
        x2 = ((c+1)* CANVAS_W) // C - 1
        y1 = (r    * CANVAS_H) // R
        y2 = ((r+1)* CANVAS_H) // R - 1
        return x1, y1, x2, y2

    def rc_center_px(self, r: int, c: int) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.rc_to_rect(r, c)
        return (x1 + x2) // 2, (y1 + y2) // 2

    def pix_to_rc(self, x: int, y: int) -> Tuple[int, int]:
        R, C = self.grid.shape
        r = int(np.clip((y * R) // CANVAS_H, 0, R - 1))
        c = int(np.clip((x * C) // CANVAS_W, 0, C - 1))
        return r, c

    def draw_cell(self, img, r: int, c: int, color, fill=True):
        x1, y1, x2, y2 = self.rc_to_rect(r, c)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1 if fill else 1)

    # ---------- Mask & PRM/RRT* ----------

    def grid_to_mask(self) -> np.ndarray:
        mask = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
        rs, cs = np.where(self.grid == 1)
        for r, c in zip(rs, cs):
            x1, y1, x2, y2 = self.rc_to_rect(r, c)
            mask[y1:y2+1, x1:x2+1] = True
        return mask

    def ensure_mask(self):
        self.mask = self.grid_to_mask()

    def ensure_prm(self):
        if not self.prm_dirty and self.nodes_xy is not None and self.graph is not None:
            return True
        if self.start is None or self.goal is None:
            return False
        self.ensure_mask()
        sx, sy = self.rc_center_px(*self.start)
        gx, gy = self.rc_center_px(*self.goal)
        nodes, graph, edges = build_prm(
            self.mask, self.prm_N, self.prm_k, self.prm_radius,
            use8=self.allow_diag, start_xy=(sx, sy), goal_xy=(gx, gy)
        )
        if nodes is None:
            self.nodes_xy = None; self.graph = None; self.edges_draw = []
            return False
        self.nodes_xy, self.graph, self.edges_draw = nodes, graph, edges
        self.prm_dirty = False
        return True

    # ---------- Costmap (GRID only) ----------

    def mark_costmap_dirty(self):
        self.costmap_dirty = True
        self.costmap = None

    def ensure_costmap(self):
        """Compute/refresh costmap if enabled and dirty, for GRID planner."""
        if not self.costmap_enabled:
            return False
        if not self.costmap_dirty and self.costmap is not None:
            return True

        R, C = self.grid.shape
        free = (self.grid == 0).astype(np.uint8) * 255
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 3).astype(np.float32)

        cost = np.zeros((R, C), dtype=np.float32)
        R_d = float(self.cost_R)
        R_s = max(1.0, R_d / 2.0)
        sigma = max(1.0, R_s / 2.0)

        free_mask = (self.grid == 0)

        high_zone = free_mask & (dist > 0.0) & (dist <= R_d)
        cost[high_zone] = float(self.cost_high)

        smooth_zone = free_mask & (dist > R_d) & (dist <= R_d + R_s)
        d2 = dist - R_d
        gvals = np.exp(-0.5 * ((d2 / sigma) ** 2))
        cost[smooth_zone] += float(self.cost_smooth_max) * gvals[smooth_zone]

        self.costmap = cost
        self.costmap_dirty = False
        return True

    # ---------- Overlays / lifecycle ----------

    def clear_path_overlay(self):
        self.path_cells.clear()
        self.visited_cells.clear()
        self.path_px.clear()
        self.visited_idx.clear()
        self.rrt_edges = []
        self.rrt_nodes_xy = None
        self.rrt_goal_connected = False
        self.rrt_path_edges = []

    def mark_prm_dirty(self):
        self.prm_dirty = True
        self.nodes_xy = None
        self.graph = None
        self.edges_draw = []

    def _clear_line_preview(self):
        self.line_active = False
        self.line_anchor = None
        self.line_current = None
        self.preview_points = []

    # ---------- Edit lock ----------

    def editing_allowed(self) -> bool:
        return not self.is_running

    # ---------- Mouse ----------

    def on_mouse(self, event, x, y, flags, param):
        if not self.editing_allowed():
            return
        r, c = self.pix_to_rc(x, y)
        R, C = self.grid.shape
        if not (0 <= r < R and 0 <= c < C):
            return

        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        pen_is_line = (self.pen in (PEN_LINEDRAW, PEN_LINEERASE))
        shift_line_with_brush = (shift and self.pen in (PEN_DRAW, PEN_ERASE))
        want_line = pen_is_line or shift_line_with_brush

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_rc = (r, c)
            if want_line:
                erase = (self.pen in (PEN_LINEERASE, PEN_ERASE))
                self._begin_line(r, c, erase=erase)
            else:
                self._apply_pen(r, c)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if want_line and self.line_active:
                if self.line_current != (r, c):
                    self.line_current = (r, c)
                    self._update_preview()
            else:
                if self.last_rc != (r, c):
                    self._apply_pen(r, c)
                    self.last_rc = (r, c)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.last_rc = None
            if want_line and self.line_active:
                self._apply_line()

    def _begin_line(self, r: int, c: int, erase: bool):
        self.clear_path_overlay()
        self.line_active = True
        self.line_anchor = (r, c)
        self.line_current = (r, c)
        self.line_is_erase = erase
        self._update_preview()
        self.mark_prm_dirty()
        self.mark_costmap_dirty()

    def _apply_line(self):
        if not (self.line_active and self.line_anchor and self.line_current):
            return
        r0, c0 = self.line_anchor
        r1, c1 = self.line_current
        pts = (line_points_8 if self.allow_diag else line_points_4)(r0, c0, r1, c1)
        val = 0 if self.line_is_erase else 1
        for rr, cc in pts:
            if 0 <= rr < self.grid.shape[0] and 0 <= cc < self.grid.shape[1]:
                self.grid[rr, cc] = val
                if val == 1:
                    if self.start == (rr, cc): self.start = None
                    if self.goal  == (rr, cc): self.goal  = None
        self._clear_line_preview()
        self.clear_path_overlay()
        self.mark_prm_dirty()
        self.mark_costmap_dirty()

    def _apply_pen(self, r, c):
        self.clear_path_overlay()
        if self.pen == PEN_DRAW:
            self.grid[r, c] = 1
            if self.start == (r, c): self.start = None
            if self.goal  == (r, c): self.goal  = None
        elif self.pen == PEN_ERASE:
            self.grid[r, c] = 0
        elif self.pen == PEN_START:
            if self.grid[r, c] == 1: self.grid[r, c] = 0
            self.start = (r, c)
        elif self.pen == PEN_GOAL:
            if self.grid[r, c] == 1: self.grid[r, c] = 0
            self.goal = (r, c)
        self.mark_prm_dirty()
        self.mark_costmap_dirty()

    def _update_preview(self):
        if self.line_anchor and self.line_current:
            r0, c0 = self.line_anchor
            r1, c1 = self.line_current
            pts = (line_points_8 if self.allow_diag else line_points_4)(r0, c0, r1, c1)
            R, C = self.grid.shape
            self.preview_points = [(rr, cc) for (rr, cc) in pts if 0 <= rr < R and 0 <= cc < C]

    # ---------- PRM build animation ----------

    def build_prm_animated(self):
        if self.start is None or self.goal is None:
            return

        self.is_running = True
        self.is_building = True
        self.clear_path_overlay()

        self.ensure_mask()
        sx, sy = self.rc_center_px(*self.start)
        gx, gy = self.rc_center_px(*self.goal)

        H, W = self.mask.shape
        if not (0 <= sx < W and 0 <= sy < H and 0 <= gx < W and 0 <= gy < H):
            self.is_running = False; self.is_building = False; return
        if self.mask[sy, sx] or self.mask[gy, gy if False else gx]:  # keep shape safe
            self.is_running = False; self.is_building = False; return
        if self.mask[gy, gx]:
            self.is_running = False; self.is_building = False; return

        rng = np.random.default_rng()

        # Sampling phase (animate in batches)
        samples: List[Tuple[int,int]] = []
        seen = {(sx, sy), (gx, gy)}
        target = max(0, self.prm_N)
        batch = max(10, target // 20)  # ~20 frames during sampling
        animate = True

        # Show start/goal as initial nodes
        self.nodes_xy = np.array([ (sx, sy), (gx, gy) ], dtype=np.int32)
        self.edges_draw = []
        self.graph = {0: [], 1: []}
        cv2.imshow(WINDOW, self.draw()); cv2.waitKey(1)

        while len(samples) < target:
            if animate:
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('b'), ord('B')):
                    animate = False
                    break

            accepted = 0
            tries = 0
            while accepted < batch and len(samples) < target and tries < batch*50:
                x = int(rng.integers(0, W)); y = int(rng.integers(0, H)); tries += 1
                if self.mask[y, x] or (x, y) in seen: continue
                seen.add((x, y)); samples.append((x, y)); accepted += 1

            if animate:
                self.nodes_xy = np.array([ (sx, sy) ] + samples + [ (gx, gy) ], dtype=np.int32)
                cv2.imshow(WINDOW, self.draw())
                cv2.waitKey(1)

        if not animate:
            nodes, graph, edges = build_prm(self.mask, self.prm_N, self.prm_k, self.prm_radius,
                                            use8=self.allow_diag, start_xy=(sx, sy), goal_xy=(gx, gy))
            self.nodes_xy, self.graph, self.edges_draw = nodes, graph, edges
            self.prm_dirty = False
            self.is_running = False
            self.is_building = False
            cv2.imshow(WINDOW, self.draw()); cv2.waitKey(1)
            return

        nodes_np = np.array([ (sx, sy) ] + samples + [ (gx, gy) ], dtype=np.int32)
        M = nodes_np.shape[0]
        graph: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(M)}
        edges_draw: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []

        show_every = max(1, M // 40)
        for i in range(M):
            xi, yi = nodes_np[i]
            dx = nodes_np[:, 0] - xi; dy = nodes_np[:, 1] - yi
            dists = np.sqrt(dx*dx + dy*dy)
            order = np.argsort(dists)
            added = 0
            for j in order[1:]:
                if added >= self.prm_k: break
                d = dists[j]
                if d > self.prm_radius: continue
                if line_is_free(self.mask, (xi, yi), (int(nodes_np[j,0]), int(nodes_np[j,1])), self.allow_diag):
                    graph[i].append((j, float(d)))
                    graph[j].append((i, float(d)))
                    edges_draw.append(((xi, yi), (int(nodes_np[j,0]), int(nodes_np[j,1]))))
                    added += 1

            if i % show_every == 0:
                self.nodes_xy = nodes_np
                self.graph = graph
                self.edges_draw = edges_draw
                cv2.imshow(WINDOW, self.draw())
                k = cv2.waitKey(1) & 0xFF
                if k in (ord('b'), ord('B')):
                    nodes, graph2, edges2 = build_prm(self.mask, self.prm_N, self.prm_k, self.prm_radius,
                                                      use8=self.allow_diag, start_xy=(sx, sy), goal_xy=(gx, gy))
                    self.nodes_xy, self.graph, self.edges_draw = nodes, graph2, edges2
                    break

        self.nodes_xy = self.nodes_xy if self.nodes_xy is not None else nodes_np
        self.graph = self.graph if self.graph is not None else graph
        self.edges_draw = self.edges_draw if self.edges_draw else edges_draw
        self.prm_dirty = False
        self.is_running = False
        self.is_building = False
        cv2.imshow(WINDOW, self.draw()); cv2.waitKey(1)

    # ---------- Search & animation with cancel ----------

    def run_search(self):
        if self.start is None or self.goal is None:
            return

        self.is_running = True
        self.is_building = False
        self.cancel_requested = False
        self.clear_path_overlay()  # start clean

        # Prepare costmap if needed (GRID only)
        active_costmap = None
        if self.planner == "GRID" and self.costmap_enabled:
            self.ensure_costmap()
            active_costmap = self.costmap

        if self.planner == "GRID":
            if self.algorithm == "A*":
                path, visited = astar_grid(self.grid, self.start, self.goal, self.allow_diag, active_costmap)
            else:
                path, visited = dijkstra_grid(self.grid, self.start, self.goal, self.allow_diag, active_costmap)

            self.visited_cells = []
            for node in visited:
                if self.cancel_requested:
                    self.is_running = False
                    self.clear_path_overlay()
                    cv2.imshow(WINDOW, self.draw())
                    return
                self.visited_cells.append(node)
                cv2.imshow(WINDOW, self.draw())
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):  # escape early
                    self.is_running = False
                    return
                if key == 32:  # space -> cancel
                    self.cancel_requested = True
                time.sleep(ANIM_DELAY)

            self.path_cells = path

        elif self.planner == "PRM":
            if not self.ensure_prm():
                self.is_running = False
                cv2.imshow(WINDOW, self.draw())
                return

            start_idx = 0
            goal_idx = self.nodes_xy.shape[0] - 1
            if self.algorithm == "A*":
                node_path, visited = graph_astar(self.nodes_xy, self.graph, start_idx, goal_idx)
            else:
                node_path, visited = graph_dijkstra(self.nodes_xy, self.graph, start_idx, goal_idx)

            self.visited_idx = []
            for nid in visited:
                if self.cancel_requested:
                    self.is_running = False
                    self.clear_path_overlay()
                    cv2.imshow(WINDOW, self.draw())
                    return
                self.visited_idx.append(nid)
                cv2.imshow(WINDOW, self.draw())
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    self.is_running = False
                    return
                if key == 32:
                    self.cancel_requested = True
                time.sleep(ANIM_DELAY)

            self.path_px = [tuple(map(int, self.nodes_xy[i])) for i in node_path] if node_path else []
            
        else:  # RRT (RRT or RRT*)
            self.ensure_mask()
            sx, sy = self.rc_center_px(*self.start)
            gx, gy = self.rc_center_px(*self.goal)

            def animate_cb(nodes_xy, edges_draw, goal_idx):
                # Store for draw() during animation (cast to int for OpenCV)
                if nodes_xy is not None:
                    self.rrt_nodes_xy = np.array(nodes_xy, dtype=np.int32)
                if edges_draw is not None:
                    self.rrt_edges = [
                        ((int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])))
                        for (p0, p1) in edges_draw
                    ]
                self.rrt_goal_connected = (goal_idx is not None)
                cv2.imshow(WINDOW, self.draw())
                k = cv2.waitKey(1) & 0xFF
                if k in (27, ord('q'), ord('Q')):
                    self.cancel_requested = True
                    return False
                if k == 32:
                    self.cancel_requested = True
                # If we reached goal and user asked to cancel, stop early
                if goal_idx is not None and self.cancel_requested:
                    return False
                return True

            def should_cancel():
                return self.cancel_requested

            func = rrt_star if self.rrt_variant == "RRT*" else rrt
            path_px, edges, nodes_xy, path_edges = func(
                self.mask, (sx, sy), (gx, gy), self.allow_diag,
                iters=self.rrt_iters, step=self.rrt_step,
                radius=self.rrt_radius, goal_bias=self.rrt_goal_bias,
                animate_cb=animate_cb, should_cancel=should_cancel,
                dynamics=self.rrt_dynamics,
                start_theta=self.rrt_start_theta,
                goal_radius=self.rrt_goal_radius,
                omegas=self.rrt_omegas,
                v=self.rrt_v,
                T=self.rrt_T,
                dt=self.rrt_dt
            )

            # --- Make everything int for drawing ---
            self.path_px = [(int(x), int(y)) for (x, y) in (path_px or [])]
            self.rrt_edges = [
                ((int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])))
                for (p0, p1) in (edges or [])
            ]
            self.rrt_nodes_xy = np.array(nodes_xy, dtype=np.int32) if nodes_xy is not None else None
            self.rrt_path_edges = [
                [(int(px), int(py)) for (px, py) in seg]
                for seg in (path_edges or [])
            ]
            self.rrt_goal_connected = len(self.path_px) >= 2

        img = self.draw()
        no_path = ((self.planner == "GRID" and not self.path_cells) or
                   (self.planner in ("PRM","RRT") and not self.path_px))
        if no_path:
            cv2.putText(img, "No path found.", (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, img)
        self.is_running = False

    # ---------- Grid resize (rows ±2, cols ±3; fixed canvas; preserves drawings) ----------

    def resize_grid(self, drows: int, dcols: int):
        if not self.editing_allowed():
            return
        oldR, oldC = self.grid.shape
        newR = int(np.clip(oldR + drows, MIN_ROWS, MAX_ROWS))
        newC = int(np.clip(oldC + dcols, MIN_COLS, MAX_COLS))
        if newR == oldR and newC == oldC:
            return

        old_mask = self.grid_to_mask().astype(np.uint8)
        sat = old_mask.cumsum(axis=0).cumsum(axis=1)

        def rect_sum(y1, x1, y2, x2):
            A = sat[y2, x2]
            B = sat[y1-1, x2] if y1 > 0 else 0
            C = sat[y2, x1-1] if x1 > 0 else 0
            D = sat[y1-1, x1-1] if (y1 > 0 and x1 > 0) else 0
            return A - B - C + D

        new_grid = np.zeros((newR, newC), dtype=np.uint8)
        for r in range(newR):
            for c in range(newC):
                x1 = (c    * CANVAS_W) // newC
                x2 = ((c+1)* CANVAS_W) // newC - 1
                y1 = (r    * CANVAS_H) // newR
                y2 = ((r+1)* CANVAS_H) // newR - 1
                if rect_sum(y1, x1, y2, x2) > 0:
                    new_grid[r, c] = 1

        def remap_point(rc: Optional[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
            if rc is None: return None
            sx, sy = self.rc_center_px(*rc)
            nr = int((sy * newR) // CANVAS_H)
            nc = int((sx * newC) // CANVAS_W)
            if 0 <= nr < newR and 0 <= nc < newC and new_grid[nr, nc] == 0:
                return (nr, nc)
            return None

        self.grid = new_grid
        self.start = remap_point(self.start)
        self.goal  = remap_point(self.goal)
        self.clear_path_overlay()
        self._clear_line_preview()
        self.mark_prm_dirty()
        self.mark_costmap_dirty()

    # ---------- Save/Load ----------

    def save_grid(self, path="grid.npy"):
        np.save(path, self.grid)
        print(f"[saved] {path}")

    def load_grid(self, path="grid.npy"):
        if not self.editing_allowed():
            return
        try:
            g = np.load(path)
            assert g.ndim == 2
            self.grid = g.astype(np.uint8)
            self.start = None
            self.goal = None
            self.clear_path_overlay()
            self._clear_line_preview()
            self.mark_prm_dirty()
            self.mark_costmap_dirty()
            print(f"[loaded] {path}")
        except Exception as e:
            print(f"[load failed] {e}")

    # ---------- Render ----------

    def _draw_costmap_overlay(self, img):
        if self.costmap is None: return
        R, C = self.grid.shape
        for r in range(R):
            for c in range(C):
                if self.grid[r, c] != 0:
                    continue
                v = float(self.costmap[r, c])
                if v <= 0.0:
                    continue
                x1, y1, x2, y2 = self.rc_to_rect(r, c)
                if v >= self.cost_high * 0.9:
                    color = COL_COST_CYAN
                else:
                    t = max(0.0, min(1.0, v / float(self.cost_smooth_max)))
                    b = int((1.0 - t) * COL_COST_WHITE[0] + t * COL_COST_PINK[0])
                    g = int((1.0 - t) * COL_COST_WHITE[1] + t * COL_COST_PINK[1])
                    rC = int((1.0 - t) * COL_COST_WHITE[2] + t * COL_COST_PINK[2])
                    color = (b, g, rC)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    def _hud_left(self) -> int:
        return 8 + 44

    def draw_tool_icon(self, img):
        x0, y0 = 8, 8
        s = 36  # icon box size
        cv2.rectangle(img, (x0-3, y0-3), (x0+s+3, y0+s+3), (255, 255, 255), -1)
        cv2.rectangle(img, (x0-3, y0-3), (x0+s+3, y0+s+3), (200, 200, 200), 1)

        pen = self.pen
        if pen == PEN_START:
            cv2.circle(img, (x0 + s//2, y0 + s//2), s//3, COL_START, -1, cv2.LINE_AA)
        elif pen == PEN_GOAL:
            cv2.circle(img, (x0 + s//2, y0 + s//2), s//3, COL_GOAL, -1, cv2.LINE_AA)
        elif pen == PEN_DRAW:
            cv2.circle(img, (x0 + s//2, y0 + s//2), s//3, (0, 0, 0), -1, cv2.LINE_AA)
        elif pen == PEN_ERASE:
            body_col = (220, 220, 220)
            tip_col  = COL_COST_PINK
            bx1, by1, bx2, by2 = x0 + 6, y0 + 10, x0 + s - 6, y0 + s - 8 - 2
            tx2 = bx1 + 8
            cv2.rectangle(img, (bx1, by1), (bx2, by2), body_col, -1, cv2.LINE_AA)
            cv2.rectangle(img, (bx1, by1), (tx2, by2), tip_col, -1, cv2.LINE_AA)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), (120, 120, 120), 1, cv2.LINE_AA)
        elif pen == PEN_LINEDRAW:
            cv2.line(img, (x0 + 5, y0 + s - 5), (x0 + s - 5, y0 + 5), (40, 40, 40), 3, cv2.LINE_AA)
        elif pen == PEN_LINEERASE:
            cv2.line(img, (x0 + 5, y0 + s - 5), (x0 + s - 5, y0 + 5), (120, 120, 120), 2, cv2.LINE_AA)
            body_col = (220, 220, 220)
            tip_col  = COL_COST_PINK
            bx2, by2 = x0 + s - 6 - 4, y0 + 8 + 4
            bx1, by1 = bx2 - 16 - 4, by2
            cv2.rectangle(img, (bx1, by1), (bx2, by2 + 10), body_col, -1, cv2.LINE_AA)
            cv2.rectangle(img, (bx1, by1), (bx1 + 8, by2 + 10), tip_col, -1, cv2.LINE_AA)
            cv2.rectangle(img, (bx1, by1), (bx2, by2 + 10), (120, 120, 120), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (x0 + s//2, y0 + s//2), s//6, (50, 50, 50), -1, cv2.LINE_AA)

    def draw_connectivity_icon(self, img):
        x0, y0 = 8, 8 + 36 + 8
        s = 36
        cv2.rectangle(img, (x0-3, y0-3), (x0+s+3, y0+s+3), (255, 255, 255), -1)
        cv2.rectangle(img, (x0-3, y0-3), (x0+s+3, y0+s+3), (200, 200, 200), 1)
        pad = 3
        cv2.rectangle(img, (x0+pad, y0+pad), (x0+s-pad, y0+s-pad), (180, 180, 180), 1)

        cx, cy = x0 + s//2, y0 + s//2
        col = (50, 50, 50)
        thickness = 1
        tip = 0.35
        cv2.arrowedLine(img, (cx, cy), (cx, cy - 11), col, thickness, cv2.LINE_AA, tipLength=tip)
        cv2.arrowedLine(img, (cx, cy), (cx, cy + 11), col, thickness, cv2.LINE_AA, tipLength=tip)
        cv2.arrowedLine(img, (cx, cy), (cx - 11, cy), col, thickness, cv2.LINE_AA, tipLength=tip)
        cv2.arrowedLine(img, (cx, cy), (cx + 11, cy), col, thickness, cv2.LINE_AA, tipLength=tip)
        if self.allow_diag:
            d = 8
            cv2.arrowedLine(img, (cx, cy), (cx - d, cy - d), col, thickness, cv2.LINE_AA, tipLength=tip)
            cv2.arrowedLine(img, (cx, cy), (cx + d, cy - d), col, thickness, cv2.LINE_AA, tipLength=tip)
            cv2.arrowedLine(img, (cx, cy), (cx - d, cy + d), col, thickness, cv2.LINE_AA, tipLength=tip)
            cv2.arrowedLine(img, (cx, cy), (cx + d, cy + d), col, thickness, cv2.LINE_AA, tipLength=tip)

    def draw(self):
        img = np.full((CANVAS_H, CANVAS_W, 3), COL_BG, dtype=np.uint8)

        # Costmap underlay (GRID only)
        if self.planner == "GRID" and self.costmap_enabled:
            self.ensure_costmap()
            self._draw_costmap_overlay(img)

        # Obstacles
        rs, cs = np.where(self.grid == 1)
        for r, c in zip(rs, cs):
            self.draw_cell(img, r, c, COL_OBS, fill=True)

        # GRID overlays
        if self.planner == "GRID":
            for (r, c) in self.visited_cells:
                if self.grid[r, c] == 0:
                    self.draw_cell(img, r, c, COL_EXPLORED, fill=True)
            for (r, c) in self.path_cells:
                self.draw_cell(img, r, c, COL_PATH, fill=True)

        # PRM overlays
        if self.planner == "PRM":
            for (p0, p1) in self.edges_draw:
                cv2.line(img, p0, p1, COL_PRM_EDGE, 1, cv2.LINE_AA)
            if self.nodes_xy is not None:
                for (x, y) in self.nodes_xy:
                    cv2.circle(img, (int(x), int(y)), 2, COL_PRM_NODE, -1, cv2.LINE_AA)
            for idx in self.visited_idx:
                if self.nodes_xy is None: break
                x, y = self.nodes_xy[idx]
                cv2.circle(img, (int(x), int(y)), 3, COL_PRM_VIS, -1, cv2.LINE_AA)
            if len(self.path_px) >= 2:
                cv2.polylines(img, [np.array(self.path_px, np.int32)], False, COL_PATH, 2, cv2.LINE_AA)

        # RRT overlays
        if self.planner == "RRT":
            # During animation we still draw the growing tree (already int-cast)
            if self.is_running:
                for (p0, p1) in self.rrt_edges:
                    cv2.line(img, p0, p1, COL_PRM_EDGE, 1, cv2.LINE_AA)
                if self.rrt_nodes_xy is not None:
                    for (x, y) in self.rrt_nodes_xy:
                        cv2.circle(img, (int(x), int(y)), 2, (90, 90, 255), -1, cv2.LINE_AA)

            # Draw only the chosen path (works for euclid or diff)
            if len(self.path_px) >= 2:
                if self.rrt_dynamics == "diff" and self.rrt_path_edges:
                    for traj in self.rrt_path_edges:
                        if len(traj) >= 2:
                            cv2.polylines(img, [np.array(traj, np.int32)], False, COL_PATH, 2, cv2.LINE_AA)
                else:
                    cv2.polylines(img, [np.array(self.path_px, np.int32)], False, COL_PATH, 2, cv2.LINE_AA)

        # Line preview
        for (r, c) in self.preview_points:
            self.draw_cell(img, r, c, COL_PREVIEW, fill=True)

        # Start/Goal
        if self.start is not None:
            self.draw_cell(img, self.start[0], self.start[1], COL_START, fill=True)
        if self.goal is not None:
            self.draw_cell(img, self.goal[0], self.goal[1], COL_GOAL, fill=True)

        # Grid lines
        if self.show_grid:
            R, C = self.grid.shape
            for r in range(R + 1):
                y = (r * CANVAS_H) // R
                cv2.line(img, (0, y), (CANVAS_W, y), COL_GRID, MARGIN)
            for c in range(C + 1):
                x = (c * CANVAS_W) // C
                cv2.line(img, (x, 0), (x, CANVAS_H), COL_GRID, MARGIN)

        # Tool & connectivity icons
        self.draw_tool_icon(img)
        self.draw_connectivity_icon(img)

        # HUD
        if self.show_hud:
            R, C = self.grid.shape
            prm_status = f"N={self.prm_N} k={self.prm_k} R={self.prm_radius}px"
            rrt_status = f"Iters={self.rrt_iters} step={self.rrt_step}px rad={self.rrt_radius}px bias={int(self.rrt_goal_bias*100)}%"
            cost_status = f"Costmap: {'ON' if (self.costmap_enabled and self.planner=='GRID') else 'OFF'}"
            hud1 = f"Pen: {PEN_NAMES[self.pen]}  |  Algo: {self.algorithm if self.planner!='RRT' else '—'}  |  Moves: {'8-way' if self.allow_diag else '4-way'}  |  {cost_status}"
            spec = prm_status if self.planner=='PRM' else ('RRT ' + rrt_status if self.planner=='RRT' else 'Grid search')
            hud2 = f"Planner: {self.planner}{'*' if (self.planner == 'RRT' and self.rrt_variant == 'RRT*') else ''} |  Grid: {R}x{C}  |  Canvas: {CANVAS_H}x{CANVAS_W}  |  {spec}"
            hud3 = "1:Start 2:Goal 3:Draw 4:Erase 5:Line-Draw 6:Line-Erase (Shift+3/4=Line)  A:A*  D:Dijkstra  8:Toggle 8-way  H:Hide Help"
            if self.planner == "RRT":
                hud4 = "RRT: T/t iters  U/u step  Y/y radius  J/j bias%   Space: Run/Cancel   P:Planner  C:RstPath  V:RRT/RRT*  F:euclid/diff"
            else:
                hud4 = "c:Costmap  C:RstPath  0/9: finer/coarser grid (+2,+3 / -2,-3)  P:Planner  B:Build PRM (= skip)  =/-:N  ]/[ :k  ./, :radius  R:Reset  G:Grid  S:Save  L:Load  Esc/Q:Quit"
            for i, line in enumerate([hud1, hud2, hud3, hud4]):
                cv2.putText(img, line, (self._hud_left(), 22 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_TEXT, 1, cv2.LINE_AA)

            # Hints
            if self.start is None or self.goal is None:
                cv2.putText(img, "Set Start (1) and Goal (2), then press Space.", (8, CANVAS_H-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_HINT, 1, cv2.LINE_AA)

            if (self.planner in ("PRM","RRT")) and self.costmap_enabled:
                cv2.putText(img, "Note: Costmap ignored in PRM/RRT.", (8, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,90), 1, cv2.LINE_AA)

        # Warnings and status
        if self.planner in ("PRM","RRT") and self.start and self.goal:
            self.ensure_mask()
            sx, sy = self.rc_center_px(*self.start)
            gx, gy = self.rc_center_px(*self.goal)
            if self.mask is not None:
                if self.mask[sy, sx]:
                    cv2.putText(img, "Start is inside an obstacle.", (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                elif self.mask[gy, gx]:
                    cv2.putText(img, "Goal is inside an obstacle.", (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)

        if self.is_running:
            if self.planner == "PRM" and self.is_building:
                msg = "BUILDING PRM (B to skip)"
            elif self.planner == "RRT":
                msg = "RUNNING RRT*" if self.rrt_variant == "RRT*" else "RUNNING RRT"
                msg += " (Space to cancel)"
            else:
                msg = "RUNNING (Space to cancel)"
            cv2.putText(img, msg, (8, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,50), 1, cv2.LINE_AA)

        return img

    # ---------- Main loop ----------

    def loop(self):
        while True:
            cv2.imshow(WINDOW, self.draw())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            # While running/building, only allow limited controls
            if self.is_running and not (key in (32, 27, ord('q'), ord('Q')) or
                                        (self.is_building and key in (ord('b'), ord('B')))):
                continue

            if key in (27, ord('q'), ord('Q')):
                break

            elif key == 32:  # Space: run or cancel (search)
                if self.is_running and not self.is_building:
                    self.cancel_requested = True
                elif not self.is_running:
                    if self.line_active: self._apply_line()
                    self.run_search()

            elif key in (ord('b'), ord('B')):
                if not self.is_running and self.planner == "PRM":
                    self.mark_prm_dirty(); self.clear_path_overlay()
                    self.build_prm_animated()

            elif key == ord('1'): self.pen = PEN_START;  self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord('2'): self.pen = PEN_GOAL;   self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord('3'): self.pen = PEN_DRAW;   self.clear_path_overlay(); self.mark_prm_dirty(); self.mark_costmap_dirty()
            elif key == ord('4'): self.pen = PEN_ERASE;  self.clear_path_overlay(); self.mark_prm_dirty(); self.mark_costmap_dirty()
            elif key == ord('5'): self.pen = PEN_LINEDRAW
            elif key == ord('6'): self.pen = PEN_LINEERASE

            elif key in (ord('a'), ord('A')):
                self.algorithm = "A*"; self.clear_path_overlay()
            elif key in (ord('d'), ord('D')):
                self.algorithm = "Dijkstra"; self.clear_path_overlay()

            elif key == ord('8'):
                self.allow_diag = not self.allow_diag
                self.clear_path_overlay(); self.mark_prm_dirty()

            elif key in (ord('g'), ord('G')):
                self.show_grid = not self.show_grid  # visual only

            elif key in (ord('h'), ord('H')):
                self.show_hud = not self.show_hud
                if not self.show_hud:
                    print("[help] HUD hidden. Press 'h' to show it again.")
                else:
                    print("[help] HUD shown. Press 'h' to hide it again.")

            elif key == ord('C'):  # uppercase C: reset overlays
                self.clear_path_overlay()

            elif key == ord('c'):  # lowercase c: toggle costmap
                if self.planner == "GRID":
                    self.costmap_enabled = not self.costmap_enabled
                    if self.costmap_enabled:
                        self.ensure_costmap()
                else:
                    self.costmap_enabled = not self.costmap_enabled
                cv2.imshow(WINDOW, self.draw())

            elif key in (ord('r'), ord('R')):
                if not self.editing_allowed(): continue
                self.grid[:] = 0; self.start = None; self.goal = None
                self.clear_path_overlay(); self._clear_line_preview()
                self.mark_prm_dirty(); self.mark_costmap_dirty()

            elif key in (ord('s'), ord('S')):
                self.save_grid()
            elif key in (ord('l'), ord('L')):
                self.load_grid()

            elif key in (ord('p'), ord('P')):
                # Cycle planners: GRID -> PRM -> RRT -> GRID
                self.planner = {"GRID":"PRM", "PRM":"RRT", "RRT":"GRID"}[self.planner]
                self.clear_path_overlay()

            # PRM params — also clear overlays
            elif key == ord('='): self.prm_N = min(self.prm_N + 50, 5000); self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord('-'): self.prm_N = max(self.prm_N - 50, 50);  self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord(']'): self.prm_k = min(self.prm_k + 1, 64);   self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord('['): self.prm_k = max(self.prm_k - 1, 1);    self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord('.'): self.prm_radius = min(self.prm_radius + 10, 2000); self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord(','): self.prm_radius = max(self.prm_radius - 10, 10);   self.clear_path_overlay(); self.mark_prm_dirty()

            # RRT / RRT* params
            elif key == ord('T'): self.rrt_iters = min(self.rrt_iters + 500, 20000)
            elif key == ord('t'): self.rrt_iters = max(self.rrt_iters - 500, 200)
            elif key == ord('U'): self.rrt_step  = min(self.rrt_step  + 5, 200)
            elif key == ord('u'): self.rrt_step  = max(self.rrt_step  - 5, 5)
            elif key == ord('Y'): self.rrt_radius = min(self.rrt_radius + 5, 300)
            elif key == ord('y'): self.rrt_radius = max(self.rrt_radius - 5, 5)
            elif key == ord('J'): self.rrt_goal_bias = min(self.rrt_goal_bias + 0.01, 0.50)
            elif key == ord('j'): self.rrt_goal_bias = max(self.rrt_goal_bias - 0.01, 0.00)
            elif key == ord('V'): self.rrt_variant = "RRT" if self.rrt_variant == "RRT*" else "RRT*"
            elif key == ord('F'): self.rrt_dynamics = "euclid" if self.rrt_dynamics == "diff" else "diff"

            # Grid discretization: keep 2:3 aspect by stepping +2 rows / +3 cols
            elif key == ord('0'):   # finer
                self.resize_grid(+2, +3)
            elif key == ord('9'):   # coarser
                self.resize_grid(-2, -3)

        cv2.destroyAllWindows()

def main():
    App().loop()

if __name__ == "__main__":
    main()

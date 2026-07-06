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
# * UI/state, drawing, and event handling live here; planners & geometry live in modules.

import cv2
import numpy as np
import time
import math
from typing import List, Tuple, Optional, Dict, Callable

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
from prm import build_prm, graph_astar, graph_dijkstra, build_prm_star
from rrt_star import rrt_star, rrt


# ================================
# Lightweight, mode-aware Controls
# ================================

class Slider:
    """A simple horizontal slider rendered into a custom OpenCV window."""
    def __init__(
        self,
        label: str,
        get_value: Callable[[], float],
        set_value: Callable[[float], None],
        vmin: float,
        vmax: float,
        step: float,
        fmt: str = "{:.0f}",
        note: Optional[str] = None,
    ):
        self.label = label
        self.get = get_value
        self.set = set_value
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.step = float(step)
        self.fmt = fmt
        self.note = note
        # Layout (filled by panel during render)
        self.track_rect = (0, 0, 0, 0)  # (x1,y1,x2,y2) expanded click area
        self.track_line = (0, 0, 0, 0)  # (x1,y, x2,y)
        self.handle_xy = (0, 0)

    def clamp_step(self, v: float) -> float:
        v = max(self.vmin, min(self.vmax, v))
        # Quantize to step grid
        k = round((v - self.vmin) / self.step)
        return self.vmin + k * self.step

    def value_to_pos(self, val: float, x1: int, x2: int) -> int:
        if self.vmax == self.vmin:
            return x1
        t = (val - self.vmin) / (self.vmax - self.vmin)
        t = max(0.0, min(1.0, t))
        return int(round(x1 + t * (x2 - x1)))

    def pos_to_value(self, x: int, x1: int, x2: int) -> float:
        if x2 == x1:
            return self.vmin
        t = (x - x1) / float(x2 - x1)
        t = max(0.0, min(1.0, t))
        v = self.vmin + t * (self.vmax - self.vmin)
        return self.clamp_step(v)

    def value_str(self, v: float) -> str:
        try:
            return self.fmt.format(v)
        except Exception:
            # fallback if fmt mismatched
            return f"{v:.3g}"


class ControlPanel:
    """
    A minimal immediate-mode panel with headings & sliders.
    - Rebuilds the visible slider set based on App state (planner, costmap, dynamics).
    - Renders to a standalone window ("Controls").
    - Captures mouse to drag sliders and set values live.
    """
    def __init__(self, app: "App"):
        self.app = app
        self.win = "Controls"
        cv2.namedWindow(self.win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.win, self._on_mouse)
        # Visuals
        self.W = 360
        self.pad = 12
        self.line_h = 22
        self.section_gap = 16
        self.block_gap = 10
        self.slider_block_h = 48  # label + slider + value
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_PLAIN
        # Interaction
        self.sliders: List[Slider] = []
        self.slider_layout: List[Tuple[Slider, Tuple[int, int, int, int]]] = []  # slider -> block rect
        self.dragging: Optional[Slider] = None
        self.drag_off = (0, 0)  # unused (kept for future)
        # Style
        self.col_bg = (245, 245, 245)
        self.col_fg = (30, 30, 30)
        self.col_muted = (110, 110, 110)
        self.col_section = (225, 225, 225)
        self.col_track = (210, 210, 210)
        self.col_track_fill = (170, 170, 170)
        self.col_handle = (60, 60, 60)
        self.col_outline = (200, 200, 200)
        self.col_lock = (20, 20, 80)

    # ---------- Build visible slider set ----------

    def _build_for_app(self):
        self.sliders.clear()
        a = self.app

        # Top "context" is display-only; discrete choices stay on keys.
        # Sections below show only numeric, mode-relevant parameters.

        if a.planner == "GRID":
            if a.costmap_enabled:
                # Costmap tuning
                self._section("Costmap Tuning")
                self._slider(
                    "Inflation radius (cells)",
                    get=lambda: float(a.cost_R),
                    setf=lambda v: self._set_attr_int("cost_R", int(v), minv=1, maxv=20, touch_costmap=True),
                    vmin=1, vmax=20, step=1, fmt="{:.0f}"
                )
                self._slider(
                    "Shoulder strength (×)",
                    get=lambda: float(a.cost_smooth_max),
                    setf=lambda v: self._set_attr_float("cost_smooth_max", float(v), minv=0.0, maxv=5.0, touch_costmap=True),
                    vmin=0.0, vmax=5.0, step=0.05, fmt="{:.2f}"
                )
            # else: GRID with no costmap — no numeric tuning needed

        elif a.planner == "PRM":
            self._section("PRM Parameters")
            self._slider(
                "Samples N",
                get=lambda: float(a.prm_N),
                setf=lambda v: self._set_attr_int("prm_N", int(v), minv=50, maxv=5000, touch_prm=True),
                vmin=50, vmax=5000, step=50, fmt="{:.0f}"
            )
            self._slider(
                "Neighbors k",
                get=lambda: float(a.prm_k),
                setf=lambda v: self._set_attr_int("prm_k", int(v), minv=1, maxv=64, touch_prm=True),
                vmin=1, vmax=64, step=1, fmt="{:.0f}"
            )
            self._slider(
                "Connection radius (px)",
                get=lambda: float(a.prm_radius),
                setf=lambda v: self._set_attr_int("prm_radius", int(v), minv=10, maxv=2000, touch_prm=True),
                vmin=10, vmax=2000, step=10, fmt="{:.0f}"
            )

        else:  # RRT / RRT*
            self._section("RRT Parameters" + (" (RRT*)" if a.rrt_variant == "RRT*" else ""))
            self._slider(
                "Iterations",
                get=lambda: float(a.rrt_iters),
                setf=lambda v: self._set_attr_int("rrt_iters", int(v), minv=200, maxv=20000),
                vmin=200, vmax=20000, step=100, fmt="{:.0f}"
            )
            self._slider(
                "Step (px)",
                get=lambda: float(a.rrt_step),
                setf=lambda v: self._set_attr_int("rrt_step", int(v), minv=5, maxv=200),
                vmin=5, vmax=200, step=1, fmt="{:.0f}"
            )
            self._slider(
                "Connection radius (px)",
                get=lambda: float(a.rrt_radius),
                setf=lambda v: self._set_attr_int("rrt_radius", int(v), minv=5, maxv=300),
                vmin=5, vmax=300, step=1, fmt="{:.0f}"
            )
            self._slider(
                "Goal bias (%)",
                get=lambda: float(a.rrt_goal_bias * 100.0),
                setf=lambda v: self._set_attr_float("rrt_goal_bias", float(v) / 100.0, minv=0.0, maxv=0.50),
                vmin=0.0, vmax=50.0, step=1.0, fmt="{:.0f}%"
            )
            self._slider(
                "Goal radius (px)",
                get=lambda: float(a.rrt_goal_radius),
                setf=lambda v: self._set_attr_float("rrt_goal_radius", float(v), minv=1.0, maxv=150.0),
                vmin=1.0, vmax=150.0, step=1.0, fmt="{:.0f}"
            )

            if a.rrt_dynamics == "diff":
                self._subsection("Diff-Drive Dynamics")
                self._slider(
                    "Horizon T (s)",
                    get=lambda: float(a.rrt_T),
                    setf=lambda v: self._set_attr_float("rrt_T", float(v), minv=0.05, maxv=2.0),
                    vmin=0.05, vmax=2.0, step=0.05, fmt="{:.2f}"
                )
                self._slider(
                    "dt (s)",
                    get=lambda: float(a.rrt_dt),
                    setf=lambda v: self._set_attr_float("rrt_dt", float(v), minv=0.005, maxv=0.1),
                    vmin=0.005, vmax=0.1, step=0.005, fmt="{:.3f}"
                )
                # v: 0 -> auto
                self._slider(
                    "v (px/s) — 0 = auto",
                    get=lambda: 0.0 if (a.rrt_v is None) else float(a.rrt_v),
                    setf=lambda v: self._set_rrt_v(v),
                    vmin=0.0, vmax=500.0, step=10.0, fmt="{:.0f}"
                )

    def _section(self, title: str):
        # Section placeholder; rendering handles the visual. We just mark a break.
        self.sliders.append(Slider(f"__SECTION__:{title}", lambda: 0, lambda v: None, 0, 1, 1))

    def _subsection(self, title: str):
        self.sliders.append(Slider(f"__SUBSECTION__:{title}", lambda: 0, lambda v: None, 0, 1, 1))

    def _slider(self, label, get, setf, vmin, vmax, step, fmt="{:.0f}", note=None):
        self.sliders.append(Slider(label, get, setf, vmin, vmax, step, fmt, note))

    # ---------- Safe setters + dirty flags ----------

    def _set_attr_int(self, name: str, value: int, minv=None, maxv=None, touch_prm=False, touch_costmap=False):
        if self.app.is_running:
            return
        if minv is not None: value = max(minv, value)
        if maxv is not None: value = min(maxv, value)
        setattr(self.app, name, int(value))
        if touch_prm: self.app.mark_prm_dirty()
        if touch_costmap: self.app.mark_costmap_dirty()
        # Clear overlays where it makes sense
        self.app.clear_path_overlay()

    def _set_attr_float(self, name: str, value: float, minv=None, maxv=None, touch_prm=False, touch_costmap=False):
        if self.app.is_running:
            return
        if minv is not None: value = max(minv, value)
        if maxv is not None: value = min(maxv, value)
        setattr(self.app, name, float(value))
        if touch_prm: self.app.mark_prm_dirty()
        if touch_costmap: self.app.mark_costmap_dirty()
        self.app.clear_path_overlay()

    def _set_rrt_v(self, v: float):
        if self.app.is_running:
            return
        v = max(0.0, min(500.0, v))
        if v < 1.0:
            self.app.rrt_v = None
        else:
            self.app.rrt_v = float(v)
        # no overlay clear required

    # ---------- Mouse handling ----------

    def _on_mouse(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            # find which slider rect we clicked
            for sl, rect in self.slider_layout:
                if sl.label.startswith("__"):  # headings are non-interactive
                    continue
                x1, y1, x2, y2 = sl.track_rect
                if (x1 <= x <= x2) and (y1 <= y <= y2):
                    self.dragging = sl
                    # apply immediate update
                    vx1, vy, vx2, _ = sl.track_line
                    new_v = sl.pos_to_value(x, vx1, vx2)
                    sl.set(new_v)
                    break

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            sl = self.dragging
            vx1, vy, vx2, _ = sl.track_line
            new_v = sl.pos_to_value(x, vx1, vx2)
            sl.set(new_v)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

    # ---------- Render ----------

    def render(self) -> np.ndarray:
        # Rebuild what to show each frame (mode-aware)
        self._build_for_app()
        # Compute height
        h = self.pad
        blocks: List[Tuple[Slider, Tuple[int, int, int, int]]] = []
        # First pass: layout
        for sl in self.sliders:
            if sl.label.startswith("__SECTION__"):
                h += 30 + self.section_gap
                blocks.append((sl, (0, 0, 0, 0)))
            elif sl.label.startswith("__SUBSECTION__"):
                h += 24 + self.section_gap
                blocks.append((sl, (0, 0, 0, 0)))
            else:
                h += self.slider_block_h
                blocks.append((sl, (self.pad, h - self.slider_block_h, self.W - self.pad, h)))

        h += self.pad
        keys = ["Keys:", "- Space=Run", "- P=Planner", "- A=Algorithm", "- 8=Connectivity", '- R=Reset', '- G=Grid']
        keys_h = len(keys) * 18 + 50
        H = max(260, h + keys_h)
        img = np.full((H, self.W, 3), self.col_bg, dtype=np.uint8)

        # Top context
        cx = self.pad
        cy = 20
        mode_txt = f"Planner: {self.app.planner}"
        alg_txt = "" if self.app.planner == "RRT" else f"   |   Algorithm: {self.app.algorithm}"
        cv2.putText(img, mode_txt + alg_txt, (cx, cy), self.font, 0.45, self.col_muted, 1, cv2.LINE_AA)
        if self.app.is_running:
            cv2.putText(img, "Running... edits locked", (cx, cy + 18), self.font, 0.45, self.col_lock, 1, cv2.LINE_AA)

        # Second pass: draw
        y = self.pad + 44
        self.slider_layout = []
        for sl, rect in blocks:
            if sl.label.startswith("__SECTION__"):
                
                title = sl.label.split(":", 1)[1]
                # Section header
                y0 = y - 4
                cv2.rectangle(img, (self.pad, y0), (self.W - self.pad, y0 + 30), self.col_section, -1, cv2.LINE_AA)  # was +26
                cv2.rectangle(img, (self.pad, y0), (self.W - self.pad, y0 + 30), self.col_outline, 1, cv2.LINE_AA)  # was +26
                cv2.putText(img, title, (self.pad + 8, y0 + 22), self.font, 0.55, self.col_fg, 1, cv2.LINE_AA)      # was +18
                y += 40 + self.section_gap                                                                            # was 30 + self.section_gap

                continue

            if sl.label.startswith("__SUBSECTION__"):
                title = sl.label.split(":", 1)[1]
                cv2.putText(img, title, (self.pad + 2, y), self.font, 0.5, self.col_muted, 1, cv2.LINE_AA)
                y += 24 + self.section_gap
                continue

            x1, y1, x2, y2 = rect
            # Label
            cv2.putText(img, sl.label, (x1 + 2, y1 + 18 + 40), self.font, 0.5, self.col_fg, 1, cv2.LINE_AA)

            # Slider track
            track_left = x1 + 4
            track_right = x2 - 70
            track_y = y1 + 70
            cv2.line(img, (track_left, track_y), (track_right, track_y), self.col_track, 6, cv2.LINE_AA)

            # Filled portion
            val = sl.get()
            hx = sl.value_to_pos(val, track_left, track_right)
            cv2.line(img, (track_left, track_y), (hx, track_y), self.col_track_fill, 6, cv2.LINE_AA)

            # Handle
            cv2.circle(img, (hx, track_y), 7, self.col_handle, -1, cv2.LINE_AA)
            cv2.circle(img, (hx, track_y), 7, (255, 255, 255), 1, cv2.LINE_AA)

            # Value text (right)
            vtxt = sl.value_str(val)
            (tw, th), _ = cv2.getTextSize(vtxt, self.font, 0.5, 1)
            cv2.putText(img, vtxt, (track_right + 8, track_y + th // 2 - 2), self.font, 0.5, self.col_fg, 1, cv2.LINE_AA)

            # Click region (expand vertically)
            sl.track_rect = (track_left, track_y - 10, track_right, track_y + 10)
            sl.track_line = (track_left, track_y, track_right, track_y)
            sl.handle_xy = (hx, track_y)
            self.slider_layout.append((sl, rect))

            y += self.slider_block_h

        # Footer keys (stacked, with a separator line)
        base_y = H - keys_h + 50
        cv2.line(img, (self.pad, base_y - 20), (self.W - self.pad, base_y - 20), self.col_outline, 1, cv2.LINE_AA)
        for i, k in enumerate(keys):
            cv2.putText(img, k, (self.pad, base_y + i * 18), self.font, 0.45, self.col_muted, 1, cv2.LINE_AA)


        return img


# ==========
# Main App
# ==========


 # --- Icon helpers -------------------------------------------------------------
def _icon_box(img, x0, y0, s, fill=(255,255,255), border=(200,200,200)):
    cv2.rectangle(img, (x0-3, y0-3), (x0+s+3, y0+s+3), fill, -1)
    cv2.rectangle(img, (x0-3, y0-3), (x0+s+3, y0+s+3), border, 1)

def draw_icon_grid(img, x0, y0, s=36):
    _icon_box(img, x0, y0, s)
    pad=6; x1,y1=x0+pad,y0+pad; x2,y2=x0+s-pad,y0+s-pad
    for t in np.linspace(x1, x2, 4, dtype=int):
        cv2.line(img, (t,y1), (t,y2), (160,160,160), 1, cv2.LINE_AA)
    for t in np.linspace(y1, y2, 4, dtype=int):
        cv2.line(img, (x1,t), (x2,t), (160,160,160), 1, cv2.LINE_AA)

def draw_icon_prm(img, x0, y0, s=36, starred=False):
    _icon_box(img, x0, y0, s)
    r = 12
    pts_list  = [[x0 + s / 2 + r * np.cos(2*np.pi*t), y0 + s / 2 + r * np.sin(2*np.pi*t)] for t in np.arange(0.0, 1.0, 1/6)]
    pts_list.append([x0 + s / 2, y0 + s / 2])
    pts = np.array(pts_list, np.int32)
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            cv2.line(img, tuple(pts[i]), tuple(pts[j]),
                     (170,170,170), 1, cv2.LINE_AA)
    for (x,y) in pts:
        cv2.circle(img, (x,y), 2, (90,90,90), -1, cv2.LINE_AA)
    # optional RRT* star
    if starred:
        cv2.putText(img, "*", (x0 + s - 10, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 1, cv2.LINE_AA)

def draw_icon_rrt(img, x0, y0, s=36, starred=False):
    _icon_box(img, x0, y0, s)

    pad = 6
    root = (x0 + s//2, y0 + s - pad)

    # main trunk
    trunk_top = (x0 + s//2, y0 + pad + 2)
    cv2.line(img, root, trunk_top, (0,0,0), 1, cv2.LINE_AA)

    # simple branches along the trunk
    branches = [
        (x0 + s//2, y0 + s - pad - 7, -7, -7),
        (x0 + s//2, y0 + s - pad - 7, +7, -7),
    ]
    for (bx, by, dx, dy) in branches:
        cv2.line(img, (bx, by), (bx + dx, by + dy), (0,0,0), 1, cv2.LINE_AA)

    # nodes as small dots
    cv2.circle(img, root, 2, (0,0,0), -1, cv2.LINE_AA)
    cv2.circle(img, trunk_top, 2, (0,0,0), -1, cv2.LINE_AA)
    for (bx, by, dx, dy) in branches:
        cv2.circle(img, (bx + dx, by + dy), 2, (0,0,0), -1, cv2.LINE_AA)

    # optional RRT* star
    if starred:
        cv2.putText(img, "*", (x0 + s - 13, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 1, cv2.LINE_AA)



def draw_icon_astar(img, x0, y0, s=36):
    _icon_box(img, x0, y0, s)
    cx,cy=x0+s//2, y0+s//2
    cv2.putText(img, "A*", (cx - 14, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)

def draw_icon_dijkstra(img, x0, y0, s=36):
    _icon_box(img, x0, y0, s)
    cx,cy=x0+s//2, y0+s//2
    cv2.putText(img, "D", (cx - 10, cy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
# ---------------------------------------------------------------------------



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
        self.prm_variant = "PRM" # PRM or PRM*
        self.mask: Optional[np.ndarray] = None
        self.nodes_xy: Optional[np.ndarray] = None
        self.graph: Optional[Dict[int, List[Tuple[int,float]]]] = None
        self.edges_draw: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []

        # RRT / RRT* params
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

        # Controls panel (custom, mode-aware)
        self.controls = ControlPanel(self)

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
        if self.prm_variant == "PRM*":
            nodes, graph, edges = build_prm_star(
                self.mask, self.prm_N, self.prm_radius,  # radius here is a cap
                use8=self.allow_diag, start_xy=(sx, sy), goal_xy=(gx, gy)
            )
        else:
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
    
    def draw_mode_icons(self, img):
        # stack below existing icons (tool at y=8, connectivity at y=8+36+8)
        x0 = 8
        s = 36
        y0_planner = 8 + 36 + 8 + 36 + 8 + 36 + 8    # under connectivity
        y0_algo    = y0_planner + s + 8     # under planner

        # Planner icon
        if self.planner == "GRID":
            draw_icon_grid(img, x0, y0_planner, s)
        elif self.planner == "PRM":
            draw_icon_prm(img, x0, y0_planner, s, starred=(self.prm_variant == "PRM*"))
        else:
            draw_icon_rrt(img, x0, y0_planner, s, starred=(self.rrt_variant == "RRT*"))

        # Algorithm icon (only for GRID/PRM)
        if self.planner != "RRT":
            if self.algorithm == "A*":
                draw_icon_astar(img, x0, y0_algo, s)
            else:
                draw_icon_dijkstra(img, x0, y0_algo, s)

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

    # ---------- Mouse (canvas) ----------

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
            if self.prm_variant == "PRM*":
                nodes, graph, edges = build_prm_star(
                    self.mask, self.prm_N, self.prm_radius,
                    use8=self.allow_diag, start_xy=(sx, sy), goal_xy=(gx, gy)
                )
            else:
                nodes, graph, edges = build_prm(
                    self.mask, self.prm_N, self.prm_k, self.prm_radius,
                    use8=self.allow_diag, start_xy=(sx, sy), goal_xy=(gx, gy)
                )
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

    # ---------- Render (canvas) ----------

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
        x0, y0 = 8, 8 + 36 + 8
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
        x0, y0 = 8, 8 + 36 + 8 + 36 + 8
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

    def run_icon_rect(self) -> Tuple[int,int,int,int]:
        # Third small icon, same size/spacing as the others
        x0 = 8
        y0 = 8 + 0 * (36 + 8)
        s = 36
        return (x0, y0, x0 + s, y0 + s)

    def draw_run_icon(self, img):
        x1, y1, x2, y2 = self.run_icon_rect()
        s = x2 - x1
        # Panel box
        cv2.rectangle(img, (x1-3, y1-3), (x2+3, y2+3), (255,255,255), -1)
        cv2.rectangle(img, (x1-3, y1-3), (x2+3, y2+3), (200,200,200), 1)

        # Subtle inner glow when running (inside the content box)
        if self.is_running:
            glow_inset = 0
            cv2.rectangle(img, (x1+glow_inset, y1+glow_inset),
                               (x2-glow_inset, y2-glow_inset),
                               (110, 200, 110), 2, cv2.LINE_AA)

        pad = 6
        cx, cy = (x1 + x2)//2, (y1 + y2)//2

        if not self.is_running:
            # Green PLAY ►
            pts = np.array([[x1+pad, y1+pad], [x2-pad, cy], [x1+pad, y2-pad]], np.int32)
            cv2.fillConvexPoly(img, pts, (80,200,80))
        elif self.planner == "RRT" and self.rrt_goal_connected:
            # Green SKIP ►|
            pts = np.array([[x1+pad, y1+pad], [x2-pad-6, cy], [x1+pad, y2-pad]], np.int32)
            cv2.fillConvexPoly(img, pts, (200,80,80))
            cv2.line(img, (x2-pad-3, y1+pad), (x2-pad-3, y2-pad), (200,80,80), 3, cv2.LINE_AA)
        else:
            # Red STOP ■ (instead of Pause)
            inset = 9
            cv2.rectangle(img, (x1+inset, y1+inset), (x2-inset, y2-inset), (0,0,200), -1, cv2.LINE_AA)


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
            # In RRT mode, draw faint circle for goal radius
            if self.planner == "RRT":
                gx, gy = self.rc_center_px(*self.goal)
                cv2.circle(img, (gx, gy), int(self.rrt_goal_radius), (100, 100, 200), 1, cv2.LINE_AA)


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
        self.draw_run_icon(img)
        self.draw_mode_icons(img)



        # HUD
        if self.show_hud:
            R, C = self.grid.shape
            prm_status = f"N={self.prm_N} k={self.prm_k} R={self.prm_radius}px"
            rrt_status = f"Iters={self.rrt_iters} step={self.rrt_step}px rad={self.rrt_radius}px bias={int(self.rrt_goal_bias*100)}%"
            cost_status = f"Costmap: {'ON' if (self.costmap_enabled and self.planner=='GRID') else 'OFF'}"
            hud1 = f"Pen: {PEN_NAMES[self.pen]}  |  Algo: {self.algorithm if self.planner!='RRT' else '—'}  |  {cost_status}"
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

        return img

    # ---------- Main loop ----------

    def loop(self):
        while True:
            # Render controls (mode-aware) and canvas
            ctrl_img = self.controls.render()
            cv2.imshow(self.controls.win, ctrl_img)

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

            elif key == ord('c'):  # lowercase c: toggle costmap (only affects GRID planner)
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

            # PRM params (keyboard still supported)
            elif key == ord('='): self.prm_N = min(self.prm_N + 50, 5000); self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord('-'): self.prm_N = max(self.prm_N - 50, 50);  self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord(']'): self.prm_k = min(self.prm_k + 1, 64);   self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord('['): self.prm_k = max(self.prm_k - 1, 1);    self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord('.'): self.prm_radius = min(self.prm_radius + 10, 2000); self.clear_path_overlay(); self.mark_prm_dirty()
            elif key == ord(','): self.prm_radius = max(self.prm_radius - 10, 10);   self.clear_path_overlay(); self.mark_prm_dirty()

            # RRT / RRT* toggles (keyboard)
            elif key == ord('T'): self.rrt_iters = min(self.rrt_iters + 500, 20000)
            elif key == ord('t'): self.rrt_iters = max(self.rrt_iters - 500, 200)
            elif key == ord('U'): self.rrt_step  = min(self.rrt_step  + 5, 200)
            elif key == ord('u'): self.rrt_step  = max(self.rrt_step  - 5, 5)
            elif key == ord('Y'): self.rrt_radius = min(self.rrt_radius + 5, 300)
            elif key == ord('y'): self.rrt_radius = max(self.rrt_radius - 5, 5)
            elif key == ord('J'): self.rrt_goal_bias = min(self.rrt_goal_bias + 0.01, 0.50)
            elif key == ord('j'): self.rrt_goal_bias = max(self.rrt_goal_bias - 0.01, 0.00)
            elif key == ord('V'):
                self.rrt_variant = "RRT" if self.rrt_variant == "RRT*" else "RRT*"
                self.prm_variant = "PRM" if self.prm_variant == "PRM*" else "PRM*"
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
